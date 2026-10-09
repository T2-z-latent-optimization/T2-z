"""Standard world-model training."""
from __future__ import annotations
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import accelerate
from accelerate.utils import InitProcessGroupKwargs
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
import torch
from torch.optim.lr_scheduler import LambdaLR
from scripts.train import WanTrainingModule, wan_parser
from t2z.data import build_dataset
from t2z.training.sampling import episode_sampler
from t2z.training.checkpoints import build_logger, AllocationFinished
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config
from wan_video_action.utils import set_global_seed



def main():
    parser = wan_parser()
    if '--frame_stride' not in parser._option_string_actions:
        parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--standard_max_updates', type=int, default=5500)
    parser.add_argument('--standard_environments_per_rank', type=int, default=3)
    parser.add_argument('--standard_episodes_per_environment', type=int, default=8)
    parser.add_argument('--standard_window_kind_field', default='sampling_kind')
    parser.add_argument('--standard_preferred_window_kind', default='precise')
    parser.add_argument('--standard_preferred_window_probability', type=float, default=0.5)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1, mixed_precision=args.mixed_precision, kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters), InitProcessGroupKwargs(timeout=timedelta(hours=1))])
    sampler = episode_sampler(args, 'standard')
    dataset = build_dataset(args, runtime_config)
    wan = WanTrainingModule(model_paths=json.dumps(runtime_config['model_paths_list']), model_id_with_origin_paths=args.model_id_with_origin_paths, tokenizer_path=runtime_config['tokenizer_path'], enable_text=args.enable_text, trainable_models=args.trainable_models, lora_base_model=args.lora_base_model, lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank, lora_checkpoint=args.lora_checkpoint, preset_lora_path=args.preset_lora_path, preset_lora_model=args.preset_lora_model, use_gradient_checkpointing=args.use_gradient_checkpointing, use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload, extra_inputs=args.extra_inputs, modules=runtime_config['modules'], fp8_models=args.fp8_models, offload_models=args.offload_models, ckpt_path=args.ckpt_path, task=args.task, device='cpu' if args.initialize_model_on_cpu else accelerator.device, max_timestep_boundary=args.max_timestep_boundary, min_timestep_boundary=args.min_timestep_boundary, num_history_frames=args.num_history_frames, args=args)
    model = wan
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    warmup = int(getattr(args, 'stage1_warmup_steps', 100) or 0)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda index: min(1.0, (index + 1) / warmup) if warmup else 1.0)
    model.to(accelerator.device)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    model.train()
    model_logger = build_logger(args, 'standard')
    for step in range(1, int(args.standard_max_updates) + 1):
        indices = sampler.sample(step, accelerator.process_index)
        optimizer.zero_grad(set_to_none=True)
        detached_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        for position, index in enumerate(indices):
            sync = nullcontext() if position == len(indices) - 1 else accelerator.no_sync(model)
            with sync:
                loss = model(dataset[index])
                accelerator.backward(loss / len(indices))
            detached_loss += loss.detach().float() / len(indices)
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        loss_value = accelerator.reduce(detached_loss, reduction='mean')
        try:
            model_logger.on_step_end(accelerator, model, args.save_steps, loss=loss_value)
        except AllocationFinished:
            return
    model_logger.on_training_end(accelerator, model, args.save_steps)
if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
