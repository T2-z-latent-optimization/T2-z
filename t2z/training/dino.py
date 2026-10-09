"""Support-to-query training for the DINOv2 baseline."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import accelerate
from accelerate.utils import InitProcessGroupKwargs
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from scripts.train import WanTrainingModule, wan_parser
from t2z.data import build_dataset
from t2z.training.sampling import build_dino_sampler
from t2z.training.checkpoints import build_logger, AllocationFinished
from wan_video_action.methods.baselines.dinov2_amortized import DINOv2AmortizedContextEncoder
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed

def add_dinov2_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group('dino')
    group.add_argument('--dinov2_model_path', type=str, required=False)
    group.add_argument('--dinov2_active_environment_manifest', type=str, required=False)
    group.add_argument('--dinov2_environment_key', type=str, default='mu_index')
    group.add_argument('--dinov2_action_key', type=str, default='action_id')
    group.add_argument('--dinov2_window_sampling_mode', choices=('uniform_episode_then_window',), default='uniform_episode_then_window', help='Sample episodes and metadata-selected chunks; action_key identifies episodes.')
    group.add_argument('--dinov2_window_kind_field', default='sampling_kind')
    group.add_argument('--dinov2_preferred_window_kind', default='precise')
    group.add_argument('--dinov2_preferred_window_probability', type=float, default=0.5)
    group.add_argument('--dinov2_support_light_augmentation_enabled', action=argparse.BooleanOptionalAction, default=False, help='Apply the existing training-only Wan light augmentation to supports.')
    group.add_argument('--dinov2_distinct_actions', action=argparse.BooleanOptionalAction, default=False)
    group.add_argument('--dinov2_sampled_frames', type=int, default=11)
    group.add_argument('--dinov2_temporal_stride', type=int, default=4)
    group.add_argument('--dinov2_hidden_dim', type=int, default=256)
    group.add_argument('--dinov2_action_hidden_dim', type=int, default=128)
    group.add_argument('--dinov2_output_dim', type=int, default=32)
    group.add_argument('--dinov2_aggregation_mode', choices=('concat_mlp',), default='concat_mlp')
    group.add_argument('--dinov2_mlp_hidden_dim', type=int, default=1024)
    group.add_argument('--dinov2_support_k', type=int, default=1)
    group.add_argument('--dinov2_support_k_choices', type=str, default='1,2')
    group.add_argument('--dinov2_trajectories_per_environment', type=int, default=0)
    group.add_argument('--dinov2_environments_per_rank', type=int, default=2)
    group.add_argument('--dinov2_head_learning_rate', type=float, default=0.0001)
    group.add_argument('--dinov2_max_updates', type=int, default=50000)
    return parser

class DINOv2WanTrainingModule(nn.Module):

    def __init__(self, *, wan: WanTrainingModule, support_encoder: DINOv2AmortizedContextEncoder, support_light_augmentation_enabled: bool=False):
        super().__init__()
        self.wan = wan
        self.support_encoder = support_encoder
        self.support_light_augmentation_enabled = bool(support_light_augmentation_enabled)

    def extract_support_visual(self, video):
        if self.support_light_augmentation_enabled and self.training and self.wan.training and self.wan.video_light_augmentation_enabled:
            device = next(self.support_encoder.dino.parameters()).device
            video = torch.as_tensor(video, dtype=torch.float32, device=device).clone()
            if video.ndim == 4:
                video = video.unsqueeze(0)
            shared, _, _ = self.wan.apply_video_light_augmentation(({'input_video': video}, {}, {}))
            video = shared['input_video']
        return self.support_encoder.extract_visual_features(video)

    def forward(self, *, query_data, support_visual_features, support_action, support_frame_indices, support_frame_count):
        if isinstance(support_visual_features, tuple):
            code = self.support_encoder.project_supports(visual_features=support_visual_features, actions=support_action, frame_indices=support_frame_indices, frame_counts=support_frame_count)
        else:
            code = self.support_encoder.project_support(visual_features=support_visual_features, action=support_action, frame_indices=support_frame_indices, frame_count=support_frame_count)
        conditioned_query = query_data.copy()
        conditioned_query['physical_context'] = code
        return self.wan(conditioned_query)

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        wan_state = {key[len('wan.'):]: value for key, value in state_dict.items() if key.startswith('wan.')}
        exported_wan = self.wan.export_trainable_state_dict(wan_state, remove_prefix=remove_prefix)
        exported = {f'wan.{key}': value for key, value in exported_wan.items()}
        for key, value in state_dict.items():
            if key.startswith('support_encoder.') and (not key.startswith('support_encoder.dino.')):
                exported[key] = value
        return exported

def main() -> None:
    parser = add_dinov2_config(wan_parser())
    parser.add_argument('--frame_stride', type=int, default=1)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    if int(args.dinov2_support_k) < 1:
        raise ValueError('dinov2_support_k must be positive.')
    support_k_choices = None
    if args.dinov2_support_k_choices:
        support_k_choices = tuple((int(value.strip()) for value in str(args.dinov2_support_k_choices).split(',') if value.strip()))
        if not support_k_choices or min(support_k_choices) < 1:
            raise ValueError('dinov2_support_k_choices must be a comma-separated list of positive integers.')
    if not args.dinov2_model_path or not args.dinov2_active_environment_manifest:
        raise ValueError('DINO model path and active-environment manifest are required.')
    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1, mixed_precision=args.mixed_precision, kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters), InitProcessGroupKwargs(timeout=timedelta(hours=1))])
    dataset = build_dataset(args, runtime_config)
    sampler = build_dino_sampler(args, support_k_choices)
    wan = WanTrainingModule(model_paths=json.dumps(runtime_config['model_paths_list']), model_id_with_origin_paths=args.model_id_with_origin_paths, tokenizer_path=runtime_config['tokenizer_path'], enable_text=args.enable_text, trainable_models=args.trainable_models, lora_base_model=args.lora_base_model, lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank, lora_checkpoint=args.lora_checkpoint, preset_lora_path=args.preset_lora_path, preset_lora_model=args.preset_lora_model, use_gradient_checkpointing=args.use_gradient_checkpointing, use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload, extra_inputs=args.extra_inputs, modules=runtime_config['modules'], fp8_models=args.fp8_models, offload_models=args.offload_models, ckpt_path=args.ckpt_path, task=args.task, device='cpu' if args.initialize_model_on_cpu else accelerator.device, max_timestep_boundary=args.max_timestep_boundary, min_timestep_boundary=args.min_timestep_boundary, num_history_frames=args.num_history_frames, args=args)
    support_encoder = DINOv2AmortizedContextEncoder(model_path=args.dinov2_model_path, sampled_frames=args.dinov2_sampled_frames, temporal_stride=args.dinov2_temporal_stride, action_dim=args.action_dim, hidden_dim=args.dinov2_hidden_dim, action_hidden_dim=args.dinov2_action_hidden_dim, output_dim=args.dinov2_output_dim, aggregation_mode=args.dinov2_aggregation_mode, mlp_hidden_dim=args.dinov2_mlp_hidden_dim)
    model = DINOv2WanTrainingModule(wan=wan, support_encoder=support_encoder, support_light_augmentation_enabled=args.dinov2_support_light_augmentation_enabled)
    wan_parameters = [parameter for parameter in model.wan.parameters() if parameter.requires_grad]
    head_parameters = [parameter for parameter in model.support_encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([{'name': 'wan', 'params': wan_parameters, 'lr': float(args.learning_rate), 'weight_decay': float(args.weight_decay)}, {'name': 'amortized_head', 'params': head_parameters, 'lr': float(args.dinov2_head_learning_rate), 'weight_decay': 0.01}])
    warmup_steps = int(getattr(args, 'stage1_warmup_steps', 0) or 0)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)) if warmup_steps > 0 else 1.0)
    model.to(accelerator.device)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    model_logger = build_logger(args, 'dino')
    if accelerator.is_main_process:
        query_label = str(args.dinov2_trajectories_per_environment) if args.dinov2_trajectories_per_environment else 'all_sampled'
        global_environments = int(accelerator.num_processes) * int(args.dinov2_environments_per_rank)
        print(f'[dinov2_grouped_batch] global_environments={global_environments} environments_per_rank={args.dinov2_environments_per_rank} support_k={args.dinov2_support_k} support_k_choices={support_k_choices} trajectories_per_environment={args.dinov2_trajectories_per_environment} queries_per_environment={query_label}', flush=True)
    iterator = tqdm(range(int(args.dinov2_max_updates)), disable=not accelerator.is_local_main_process)
    for update_index in iterator:
        step = update_index + 1
        episodes = sampler.sample(step=step, process_index=accelerator.process_index, num_processes=accelerator.num_processes, environments_per_rank=args.dinov2_environments_per_rank)
        total_queries = sum((len(episode.query_indices) for episode in episodes))
        environment_count = len(episodes)
        if environment_count < 1 or total_queries < 1:
            raise RuntimeError('Every rank must receive at least one environment and query.')
        optimizer.zero_grad(set_to_none=True)
        detached_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        query_counter = 0
        for episode in episodes:
            query_count = len(episode.query_indices)
            if query_count < 1:
                raise RuntimeError(f'Environment {episode.environment_id} has no query.')
            query_weight = 1.0 / float(environment_count * query_count)
            supports = tuple((dataset[index] for index in episode.support_indices))
            unwrapped = accelerator.unwrap_model(model)
            support_visual = tuple((unwrapped.extract_support_visual(support['video']) for support in supports))
            visual_features = tuple((item[0] for item in support_visual))
            frame_indices = tuple((item[1] for item in support_visual))
            frame_counts = tuple((item[2] for item in support_visual))
            support_actions = tuple((support['action'] for support in supports))
            for query_index in episode.query_indices:
                query_counter += 1
                sync_context = accelerator.no_sync(model) if query_counter < total_queries else nullcontext()
                with sync_context:
                    loss = model(query_data=dataset[query_index], support_visual_features=visual_features, support_action=support_actions, support_frame_indices=frame_indices, support_frame_count=frame_counts)
                    accelerator.backward(loss * query_weight)
                detached_loss += loss.detach().float() * query_weight
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        detached_loss = accelerator.reduce(detached_loss, reduction='mean')
        try:
            model_logger.on_step_end(accelerator, model, args.save_steps, loss=detached_loss)
        except AllocationFinished:
            return
    model_logger.on_training_end(accelerator, model, args.save_steps)
if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
