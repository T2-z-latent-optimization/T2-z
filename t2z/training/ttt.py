"""Causal write-then-predict TTT training."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import accelerate
from accelerate.utils import InitProcessGroupKwargs
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from scripts.train import WanTrainingModule, wan_parser
from t2z.data import build_dataset
from t2z.training.sampling import build_ttt_sampler
from t2z.training.checkpoints import build_logger, AllocationFinished
from wan_video_action.methods.baselines.ttt_kvb.wan_adapter import install_ttt_kvb
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed

def add_ttt_kvb_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group('ttt_kvb_causal')
    group.add_argument('--ttt_active_environment_manifest', type=str, required=False)
    group.add_argument('--ttt_environment_key', type=str, default='mu_index')
    group.add_argument('--ttt_action_key', type=str, default='action_id')
    group.add_argument('--ttt_window_sampling_mode', choices=('uniform_episode_then_window',), default='uniform_episode_then_window', help='Sample episodes and metadata-selected chunks; action_key identifies episodes.')
    group.add_argument('--ttt_window_kind_field', default='sampling_kind')
    group.add_argument('--ttt_preferred_window_kind', default='precise')
    group.add_argument('--ttt_preferred_window_probability', type=float, default=0.5)
    group.add_argument('--ttt_distinct_actions', action=argparse.BooleanOptionalAction, default=False)
    group.add_argument('--ttt_sequence_length', type=int, default=6)
    group.add_argument('--ttt_streams_per_environment', type=int, default=1)
    group.add_argument('--ttt_detach_every_chunks', type=int, default=0, help='Truncated outer gradients: backward every N chunks, then carry detached fast-state values; 0 keeps the full stream graph.')
    group.add_argument('--ttt_environments_per_rank', type=int, default=1)
    group.add_argument('--ttt_layers', type=str, default='uniform:8')
    group.add_argument('--ttt_expansion', type=int, default=4)
    group.add_argument('--ttt_base_inner_lr', type=float, default=0.1)
    group.add_argument('--ttt_inner_batch_size', type=int, default=64)
    group.add_argument('--ttt_write_token_budget', type=int, default=0)
    group.add_argument('--ttt_gate_init', type=float, default=0.1)
    group.add_argument('--ttt_slow_learning_rate', type=float, default=0.0001)
    group.add_argument('--ttt_start_step', type=int, default=0, help='Logical completed-update offset for model-only checkpoint continuation.')
    group.add_argument('--ttt_max_updates', type=int, default=50000)
    group.add_argument('--ttt_protocol', type=str, choices=('write_then_predict',), default='write_then_predict')
    group.add_argument('--ttt_gate_vector', action=argparse.BooleanOptionalAction, default=True)
    group.add_argument('--ttt_serial_after_attention', action=argparse.BooleanOptionalAction, default=True)
    group.add_argument('--ttt_saved_tensor_cpu_offload', action=argparse.BooleanOptionalAction, default=False, help='Offload tensors retained by the differentiable fast-weight scan to CPU. This preserves the full TTT graph while reducing GPU peak memory.')
    group.add_argument('--ttt_saved_tensor_policy', choices=('default', 'selective'), default='default')
    group.add_argument('--ttt_gpu_saved_tensor_budget_gib', type=float, default=12.0)
    group.add_argument('--ttt_host_saved_tensor_budget_gib', type=float, default=None)
    group.add_argument('--ttt_pin_saved_tensor_memory', action=argparse.BooleanOptionalAction, default=True)
    group.add_argument('--ttt_scan_checkpoint_updates', type=int, default=0)
    group.add_argument('--ttt_backward_per_stream', action=argparse.BooleanOptionalAction, default=False)
    group.add_argument('--ttt_sync_per_update', action=argparse.BooleanOptionalAction, default=False)
    return parser

class CausalTTTWanTrainingModule(WanTrainingModule):
    """Wan SFT with one diffusion timestep shared by a configurable chunk stream."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ttt_timestep_index = None
        self.task_to_loss['sft'] = self._ttt_flow_loss
        self.task_to_loss['sft:train'] = self._ttt_flow_loss

    def set_timestep_index(self, index: int) -> None:
        self._ttt_timestep_index = int(index)

    def _ttt_flow_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        if self._ttt_timestep_index is None:
            raise RuntimeError('TTT timestep index was not set for this stream.')
        if self.history_anchor_windows:
            from t2z.training.losses import flow_matching_loss
            return flow_matching_loss(pipe, inputs_shared, inputs_posi, inputs_nega, timestep_index=self._ttt_timestep_index)
        inputs = {**inputs_shared, **inputs_posi}
        timestep = pipe.scheduler.timesteps[self._ttt_timestep_index].reshape(1).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise = torch.randn_like(inputs['input_latents'])
        inputs['latents'] = pipe.scheduler.add_noise(inputs['input_latents'], noise, timestep)
        target = pipe.scheduler.training_target(inputs['input_latents'], noise, timestep)
        if 'first_frame_latents' in inputs:
            inputs['latents'][:, :, 0:1] = inputs['first_frame_latents'][:, :, 0:1]
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
        if 'first_frame_latents' in inputs:
            prediction = prediction[:, :, 1:]
            target = target[:, :, 1:]
        return F.mse_loss(prediction.float(), target.float()) * pipe.scheduler.training_weight(timestep)

def main() -> None:
    parser = add_ttt_kvb_config(wan_parser())
    parser.add_argument('--frame_stride', type=int, default=1)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    if args.ttt_protocol != 'write_then_predict':
        raise ValueError('Only write_then_predict is supported')
    if not args.ttt_active_environment_manifest:
        raise ValueError('ttt_active_environment_manifest is required.')
    if args.use_gradient_checkpointing or args.use_gradient_checkpointing_offload:
        print('[ttt_kvb] disabling activation checkpointing for state-consistent fast-weight query replay', flush=True)
        args.use_gradient_checkpointing = False
        args.use_gradient_checkpointing_offload = False
    if int(args.ttt_detach_every_chunks) < 0:
        raise ValueError('ttt_detach_every_chunks must be nonnegative')
    start_step = int(args.ttt_start_step)
    if start_step < 0 or start_step >= int(args.ttt_max_updates):
        raise ValueError('ttt_start_step must be in [0, ttt_max_updates)')
    if int(args.ttt_detach_every_chunks):
        if not args.ttt_backward_per_stream:
            raise ValueError('Chunk-boundary detach requires causal write-then-predict and backward_per_stream')
    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1, mixed_precision=args.mixed_precision, kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters, gradient_as_bucket_view=True), InitProcessGroupKwargs(timeout=timedelta(hours=1))])
    dataset = build_dataset(args, runtime_config)
    sampler = build_ttt_sampler(args)
    model_class = CausalTTTWanTrainingModule
    model = model_class(model_paths=json.dumps(runtime_config['model_paths_list']), model_id_with_origin_paths=args.model_id_with_origin_paths, tokenizer_path=runtime_config['tokenizer_path'], enable_text=args.enable_text, trainable_models=args.trainable_models, lora_base_model=args.lora_base_model, lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank, lora_checkpoint=args.lora_checkpoint, preset_lora_path=args.preset_lora_path, preset_lora_model=args.preset_lora_model, use_gradient_checkpointing=args.use_gradient_checkpointing, use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload, allow_disabled_gradient_checkpointing=True, extra_inputs=args.extra_inputs, modules=runtime_config['modules'], fp8_models=args.fp8_models, offload_models=args.offload_models, ckpt_path=args.ckpt_path, task=args.task, device='cpu' if args.initialize_model_on_cpu else accelerator.device, max_timestep_boundary=args.max_timestep_boundary, min_timestep_boundary=args.min_timestep_boundary, num_history_frames=args.num_history_frames, args=args)
    installation = install_ttt_kvb(model.pipe.dit, layer_spec=args.ttt_layers, expansion=args.ttt_expansion, base_inner_lr=args.ttt_base_inner_lr, inner_batch_size=args.ttt_inner_batch_size, write_token_budget=args.ttt_write_token_budget, gate_init=args.ttt_gate_init, gate_vector=args.ttt_gate_vector, serial_after_attention=args.ttt_serial_after_attention, scan_checkpoint_updates=args.ttt_scan_checkpoint_updates)
    model.use_gradient_checkpointing = False
    model.use_gradient_checkpointing_offload = False
    controller = installation.controller
    ttt_parameters = list(installation.ttt_parameters())
    ttt_parameter_ids = {id(parameter) for parameter in ttt_parameters}
    wan_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in ttt_parameter_ids]
    optimizer = torch.optim.AdamW([{'name': 'wan', 'params': wan_parameters, 'lr': float(args.learning_rate), 'weight_decay': float(args.weight_decay)}, {'name': 'ttt_slow', 'params': ttt_parameters, 'lr': float(args.ttt_slow_learning_rate), 'weight_decay': 0.01}])
    warmup_steps = int(getattr(args, 'stage1_warmup_steps', 0) or 0)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, float(start_step + step + 1) / float(warmup_steps)) if warmup_steps > 0 else 1.0)
    model.to(accelerator.device)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    model_logger = build_logger(args, 'ttt')
    model_logger.num_steps = start_step
    saved_storage = None
    if args.ttt_saved_tensor_policy == 'selective':
        from wan_video_action.methods.baselines.ttt_kvb.activation_storage import SelectiveSavedTensorStorage
        saved_storage = SelectiveSavedTensorStorage(accelerator.unwrap_model(model), gpu_budget_gib=args.ttt_gpu_saved_tensor_budget_gib, host_budget_gib=args.ttt_host_saved_tensor_budget_gib, pin_memory=args.ttt_pin_saved_tensor_memory)
    diagnostic_callback = getattr(model_logger, 'on_step_diagnostics', None)
    updates_per_chunk = f'all_tokens/{args.ttt_inner_batch_size}'
    if accelerator.is_main_process:
        print(f'[ttt_kvb_causal] layers={installation.layer_indices} sequence_length={args.ttt_sequence_length} start_step={start_step} detach_every_chunks={args.ttt_detach_every_chunks} environments_per_rank={args.ttt_environments_per_rank} protocol={args.ttt_protocol} updates_per_chunk={updates_per_chunk} layer_local_updates_per_stream=dynamic gate_init={args.ttt_gate_init} base_inner_lr={args.ttt_base_inner_lr} gradient_checkpointing={accelerator.unwrap_model(model).use_gradient_checkpointing} saved_tensor_cpu_offload={args.ttt_saved_tensor_cpu_offload} saved_tensor_policy={args.ttt_saved_tensor_policy} scan_checkpoint_updates={args.ttt_scan_checkpoint_updates} gpu_saved_tensor_budget_gib={args.ttt_gpu_saved_tensor_budget_gib} host_saved_tensor_budget_gib={args.ttt_host_saved_tensor_budget_gib} pin_saved_tensor_memory={args.ttt_pin_saved_tensor_memory}', flush=True)
    iterator = tqdm(range(start_step, int(args.ttt_max_updates)), disable=not accelerator.is_local_main_process)
    sequence_length = int(args.ttt_sequence_length)
    backprop_segment_length = int(args.ttt_detach_every_chunks) or sequence_length
    for update_index in iterator:
        step = update_index + 1
        update_started_wall = time.time()
        update_started_monotonic = time.monotonic()
        if diagnostic_callback is not None:
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        if saved_storage is not None:
            saved_storage.begin_update()
        episodes = sampler.sample(step=step, process_index=accelerator.process_index, num_processes=accelerator.num_processes, environments_per_rank=args.ttt_environments_per_rank)
        optimizer.zero_grad(set_to_none=True)
        local_position_losses = torch.zeros(sequence_length, device=accelerator.device, dtype=torch.float32)
        for episode_number, episode in enumerate(episodes):
            controller.reset(batch_size=1)
            if saved_storage is not None:
                saved_storage.begin_stream()
            stream_losses = []
            unwrapped_model = accelerator.unwrap_model(model)
            lower = int(float(args.min_timestep_boundary) * len(unwrapped_model.pipe.scheduler.timesteps))
            upper = int(float(args.max_timestep_boundary) * len(unwrapped_model.pipe.scheduler.timesteps))
            unwrapped_model.set_timestep_index(int(torch.randint(lower, upper, (1,)).item()))
            for position, sample_index in enumerate(episode.indices):
                chunk = dataset[sample_index]
                is_final_query = position == sequence_length - 1
                is_segment_end = (position + 1) % backprop_segment_length == 0 or is_final_query
                synchronize = is_segment_end and (not args.ttt_sync_per_update or (is_final_query and episode_number == len(episodes) - 1))
                if args.ttt_detach_every_chunks and step == 1:
                    print(f'[ttt_segment_chunk] rank={accelerator.process_index} step={step} environment={episode.environment_id} chunk={position + 1}/{sequence_length} segment={position // backprop_segment_length + 1} cuda_allocated_gib={torch.cuda.memory_allocated(accelerator.device) / 1024 ** 3:.3f}', flush=True)
                sync_context = nullcontext() if synchronize else accelerator.no_sync(model)
                with sync_context:
                    mode_context = controller.causal_scan(differentiable=True)
                    with mode_context:
                        saved_tensor_context = saved_storage.scope() if saved_storage is not None else torch.autograd.graph.save_on_cpu(pin_memory=False) if args.ttt_saved_tensor_cpu_offload else nullcontext()
                        with saved_tensor_context:
                            loss = model(chunk)
                        if args.ttt_backward_per_stream:
                            stream_losses.append(loss)
                            if is_segment_end:
                                stream_loss = torch.stack(stream_losses).sum() / float(sequence_length * len(episodes))
                                accelerator.backward(stream_loss)
                                stream_losses.clear()
                                del stream_loss
                        else:
                            accelerator.backward(loss / float(sequence_length * len(episodes)), retain_graph=not is_final_query)
                local_position_losses[position] += loss.detach().float() / float(len(episodes))
                if args.ttt_detach_every_chunks and is_segment_end:
                    if not is_final_query:
                        controller.detach_fast_state()
                        if saved_storage is not None:
                            saved_storage.begin_stream()
                    if step == 1 or (int(args.log_steps) > 0 and step % int(args.log_steps) == 0):
                        print(f'[ttt_segment_backward] rank={accelerator.process_index} step={step} environment={episode.environment_id} end_chunk={position + 1} state_carried={not is_final_query} cuda_allocated_gib={torch.cuda.memory_allocated(accelerator.device) / 1024 ** 3:.3f}', flush=True)
                    del loss
            if int(args.log_steps) > 0 and step % int(args.log_steps) == 0:
                state_norms = controller.state_norms()
                mean_state_norm = sum(state_norms.values()) / float(len(state_norms)) if state_norms else 0.0
                print(f'[ttt_stream] rank={accelerator.process_index} step={step} environment={episode.environment_id} actions={episode.action_ids} mean_state_norm={mean_state_norm:.6f}', flush=True)
            controller.clear()
            if saved_storage is not None:
                saved_storage.end_stream()
        gradient_norm = None
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            gradient_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        gathered_losses = accelerator.gather(local_position_losses)
        gathered_losses = gathered_losses.reshape(-1, sequence_length).mean(dim=0)
        detached_loss = gathered_losses.mean()
        if accelerator.is_main_process and int(args.log_steps) > 0:
            if step % int(args.log_steps) == 0:
                position_text = ' '.join((f'p{position + 1}={value:.6f}' for position, value in enumerate(gathered_losses.tolist())))
                print(f'[ttt_train] step={step} loss={detached_loss.item():.6f} {position_text}', flush=True)
        if diagnostic_callback is not None:
            diagnostic_callback(accelerator=accelerator, model=model, step=step, loss=detached_loss, position_losses=gathered_losses, gradient_norm=gradient_norm, started_wall=update_started_wall, started_monotonic=update_started_monotonic, storage_statistics=dict(saved_storage.statistics) if saved_storage is not None else None)
        try:
            model_logger.on_step_end(accelerator, model, args.save_steps, loss=detached_loss)
        except AllocationFinished:
            return
    model_logger.on_training_end(accelerator, model, args.save_steps)
if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
