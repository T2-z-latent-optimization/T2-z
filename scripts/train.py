import glob
import random
import time
from contextlib import nullcontext
import torch, os, argparse, accelerate, warnings, json
from accelerate.utils import broadcast
from wan_video_action.data import RoboTwinUnifiedDataset
from wan_video_action.data.operators import LoadCobotAction, ResolvePromptEmbPath, create_video_operator
from wan_video_action.data.data_utils import pack_paths
from diffsynth.core import ModelConfig
from wan_video_action.pipelines.wan_video_action import build_wan_video_action_pipeline
from diffsynth.diffusion import *
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from wan_video_action.parsers import merge_yaml_and_args, prepare_runtime_config, add_general_config
from wan_video_action.utils import set_global_seed
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def _parse_view_indices(value):
    text = str(value or '').strip().lower()
    if text in ('', 'all', '*'):
        return None
    indices = tuple(sorted({int(token.strip()) for token in text.split(',') if token.strip()}))
    if not indices or any((index < 0 for index in indices)):
        raise ValueError("video_light_augmentation_view_indices must be 'all' or comma-separated non-negative indices.")
    return indices

class TimedRetentionModelLogger(ModelLogger):

    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x, save_minutes: float=0.0, keep_last: int=0, log_steps: int=10):
        super().__init__(output_path, remove_prefix_in_ckpt=remove_prefix_in_ckpt, state_dict_converter=state_dict_converter)
        self.save_seconds = float(save_minutes) * 60.0 if save_minutes else 0.0
        self.keep_last = int(keep_last or 0)
        self.log_steps = int(log_steps or 0)
        self.last_save_time = time.monotonic()

    def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
        self.num_steps += 1
        loss = kwargs.get('loss')
        if self.log_steps > 0 and self.num_steps % self.log_steps == 0 and accelerator.is_main_process:
            loss_value = None
            if loss is not None:
                loss_value = float(loss.detach().to(dtype=torch.float32).cpu())
            if loss_value is None:
                print(f'[train] step={self.num_steps}', flush=True)
            else:
                print(f'[train] step={self.num_steps} loss={loss_value:.6f}', flush=True)
        should_save = False
        file_name = None
        if save_steps is not None and self.num_steps % int(save_steps) == 0:
            should_save = True
            file_name = f'step-{self.num_steps}.safetensors'
        timed_save = False
        if self.save_seconds > 0 and accelerator.is_main_process:
            timed_save = time.monotonic() - self.last_save_time >= self.save_seconds
        if self.save_seconds > 0 and accelerator.num_processes > 1:
            timed_save_tensor = torch.tensor(int(timed_save), device=accelerator.device)
            timed_save = bool(broadcast(timed_save_tensor, from_process=0).item())
        if timed_save:
            should_save = True
            file_name = f'step-{self.num_steps}.safetensors'
        if should_save and file_name is not None:
            if accelerator.is_main_process:
                print(f'[checkpoint] saving {os.path.join(self.output_path, file_name)}', flush=True)
            self.save_model(accelerator, model, file_name)
            self.last_save_time = time.monotonic()

    def on_epoch_end(self, accelerator, model, epoch_id):
        self.save_model(accelerator, model, f'epoch-{epoch_id}.safetensors')

    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % int(save_steps) == 0:
            return
        self.save_model(accelerator, model, f'step-{self.num_steps}.safetensors')

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            print(f'[checkpoint] saved {path}', flush=True)
            self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        if self.keep_last <= 0:
            return
        paths = sorted(glob.glob(os.path.join(self.output_path, '*.safetensors')), key=lambda path: os.path.getmtime(path))
        for path in paths[:-self.keep_last]:
            try:
                os.remove(path)
                print(f'[checkpoint] pruned {path}', flush=True)
            except FileNotFoundError:
                pass

class WanTrainingModule(DiffusionTrainingModule):

    def __init__(self, model_paths=None, model_id_with_origin_paths=None, tokenizer_path=None, trainable_models=None, lora_base_model=None, lora_target_modules='', lora_rank=32, lora_checkpoint=None, preset_lora_path=None, preset_lora_model=None, use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False, allow_disabled_gradient_checkpointing=False, extra_inputs=None, enable_text=True, modules=('dit', 'text', 'vae', 'image', 'action'), fp8_models=None, offload_models=None, ckpt_path=None, device='cpu', task='sft', max_timestep_boundary=1.0, min_timestep_boundary=0.0, num_history_frames=1, args=None):
        super().__init__()
        if not use_gradient_checkpointing and (not allow_disabled_gradient_checkpointing):
            warnings.warn('Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.')
            use_gradient_checkpointing = True
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id='Wan-AI/Wan2.1-T2V-1.3B', origin_file_pattern='google/umt5-xxl/') if enable_text and tokenizer_path is None else ModelConfig(tokenizer_path) if enable_text and tokenizer_path else None
        self.pipe = build_wan_video_action_pipeline(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, ckpt_path=ckpt_path, action_dim=getattr(args, 'action_dim', 14), action_mode=getattr(args, 'action_mode', 'adaln'), args=args)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.switch_pipe_to_training_mode(self.pipe, trainable_models, lora_base_model, lora_target_modules, lora_rank, lora_checkpoint, preset_lora_path, preset_lora_model, task=task)
        self.freeze_unused_action_modules()
        if not enable_text:
            self.freeze_text_modules()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(',') if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {'sft:data_process': lambda pipe, *args: args, 'direct_distill:data_process': lambda pipe, *args: args, 'sft': lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi), 'sft:train': lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi), 'direct_distill': lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi), 'direct_distill:train': lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi)}
        self.history_anchor_windows = bool(getattr(args, 'history_anchor_windows', False))
        if self.history_anchor_windows:
            from t2z.training.losses import flow_matching_loss
            self.task_to_loss['sft'] = flow_matching_loss
            self.task_to_loss['sft:train'] = flow_matching_loss
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.num_history_frames = num_history_frames
        self.video_light_augmentation_enabled = bool(getattr(args, 'video_light_augmentation_enabled', False))
        self.video_light_augmentation_probability = float(getattr(args, 'video_light_augmentation_probability', 0.7))
        if not 0.0 <= self.video_light_augmentation_probability <= 1.0:
            raise ValueError('video_light_augmentation_probability must lie in [0, 1].')
        self.video_light_augmentation_view_indices = _parse_view_indices(getattr(args, 'video_light_augmentation_view_indices', 'all'))
        self.video_light_augmentation_ranges = {'gain': (float(getattr(args, 'video_light_augmentation_gain_min', 0.88)), float(getattr(args, 'video_light_augmentation_gain_max', 1.12))), 'contrast': (float(getattr(args, 'video_light_augmentation_contrast_min', 0.9)), float(getattr(args, 'video_light_augmentation_contrast_max', 1.1))), 'gamma': (float(getattr(args, 'video_light_augmentation_gamma_min', 0.9)), float(getattr(args, 'video_light_augmentation_gamma_max', 1.1))), 'tint': (float(getattr(args, 'video_light_augmentation_tint_min', 0.97)), float(getattr(args, 'video_light_augmentation_tint_max', 1.03))), 'offset': (float(getattr(args, 'video_light_augmentation_offset_min', -0.015)), float(getattr(args, 'video_light_augmentation_offset_max', 0.015)))}
        for name, (lower, upper) in self.video_light_augmentation_ranges.items():
            if lower > upper:
                raise ValueError(f'video_light_augmentation_{name}_min must not exceed max.')
        self.video_light_augmentation_gradient_abs_max = float(getattr(args, 'video_light_augmentation_gradient_abs_max', 0.08))
        self.video_light_augmentation_noise_std = float(getattr(args, 'video_light_augmentation_noise_std', 0.004))
        if self.video_light_augmentation_gradient_abs_max < 0.0:
            raise ValueError('video_light_augmentation_gradient_abs_max must be non-negative.')
        if self.video_light_augmentation_noise_std < 0.0:
            raise ValueError('video_light_augmentation_noise_std must be non-negative.')
        augmentation_seed = int(getattr(args, 'seed', 0)) + 104729 * (int(os.environ.get('RANK', '0')) + 1)
        self._video_light_augmentation_rng = random.Random(augmentation_seed)
        self._video_light_augmentation_grid_cache = {}
        if self.video_light_augmentation_enabled:
            augmentation_views = 'all' if self.video_light_augmentation_view_indices is None else ','.join((str(index) for index in self.video_light_augmentation_view_indices))
            print(f"[video_light_augmentation] probability={self.video_light_augmentation_probability:g} views={augmentation_views} temporal_parameters=shared temporal_noise=fixed gain={self.video_light_augmentation_ranges['gain']} contrast={self.video_light_augmentation_ranges['contrast']} gamma={self.video_light_augmentation_ranges['gamma']} tint={self.video_light_augmentation_ranges['tint']} offset={self.video_light_augmentation_ranges['offset']} gradient_abs_max={self.video_light_augmentation_gradient_abs_max:g} noise_std={self.video_light_augmentation_noise_std:g}", flush=True)

    def apply_video_light_augmentation(self, inputs):
        """Apply temporally coherent, online photometric augmentation before VAE encoding."""
        if not self.video_light_augmentation_enabled or not self.training:
            return inputs
        rng = self._video_light_augmentation_rng
        if rng.random() >= self.video_light_augmentation_probability:
            return inputs
        inputs_shared, inputs_posi, inputs_nega = inputs
        video = inputs_shared.get('input_video')
        if not torch.is_tensor(video) or video.ndim != 5:
            raise ValueError('Online video light augmentation expects input_video shaped (V,C,T,H,W).')
        if int(video.shape[1]) != 3:
            raise ValueError(f'Online video light augmentation requires RGB video, got C={video.shape[1]}.')
        if self.video_light_augmentation_view_indices is None:
            view_indices = tuple(range(int(video.shape[0])))
        else:
            view_indices = self.video_light_augmentation_view_indices
        invalid = [index for index in view_indices if index >= int(video.shape[0])]
        if invalid:
            raise IndexError(f'video_light_augmentation_view_indices={invalid} outside num_views={video.shape[0]}.')

        def sample(name):
            lower, upper = self.video_light_augmentation_ranges[name]
            return rng.uniform(lower, upper)
        gain = sample('gain')
        contrast = sample('contrast')
        gamma = sample('gamma')
        tint_values = [sample('tint') for _ in range(3)]
        offset = sample('offset')
        gradient_limit = self.video_light_augmentation_gradient_abs_max
        gx = rng.uniform(-gradient_limit, gradient_limit)
        gy = rng.uniform(-gradient_limit, gradient_limit)
        height, width = (int(video.shape[-2]), int(video.shape[-1]))
        grid_key = (video.device.type, video.device.index, height, width)
        cached_grid = self._video_light_augmentation_grid_cache.get(grid_key)
        if cached_grid is None:
            yy = torch.linspace(-1.0, 1.0, height, device=video.device, dtype=torch.float32)
            xx = torch.linspace(-1.0, 1.0, width, device=video.device, dtype=torch.float32)
            cached_grid = (xx.view(1, 1, 1, 1, width), yy.view(1, 1, 1, height, 1))
            self._video_light_augmentation_grid_cache[grid_key] = cached_grid
        xx, yy = cached_grid
        with torch.no_grad():
            x = video[list(view_indices)].float().add(1.0).mul(0.5).clamp_(0.0, 1.0)
            x = x.pow(gamma)
            x = (x - 0.5) * contrast + 0.5
            tint = torch.tensor(tint_values, device=video.device, dtype=torch.float32).view(1, 3, 1, 1, 1)
            illumination_field = 1.0 + gx * xx + gy * yy
            x = x * gain * tint * illumination_field + offset
            if self.video_light_augmentation_noise_std > 0.0:
                noise = torch.randn((len(view_indices), 3, 1, height, width), device=video.device, dtype=torch.float32) * self.video_light_augmentation_noise_std
                x = x + noise
            x = x.clamp_(0.0, 1.0).mul_(255.0).round_().div_(255.0)
            video[list(view_indices)] = x.mul(2.0).sub(1.0).to(dtype=video.dtype)
        return (inputs_shared, inputs_posi, inputs_nega)

    def flow_mse(self, prediction, target):
        return torch.nn.functional.mse_loss(prediction.float(), target.float())

    def freeze_text_modules(self):
        self.pipe.dit.text_embedding.requires_grad_(False)
        self.pipe.dit.text_embedding.eval()

    def freeze_unused_action_modules(self):
        if getattr(self.pipe, 'action_injection_mode', 'none') != 'adaln':
            return
        action_encoder = getattr(self.pipe, 'action_encoder', None)
        action_embedding = getattr(action_encoder, 'action_embedding', None)
        if action_embedding is not None:
            action_embedding.requires_grad_(False)
            action_embedding.eval()

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == 'input_image':
                inputs_shared['input_image'] = data['video'][0]
            elif extra_input == 'end_image':
                inputs_shared['end_image'] = data['video'][-1]
            elif extra_input == 'reference_image' or extra_input == 'vace_reference_image':
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def _video_geometry(self, video):
        if torch.is_tensor(video):
            if video.ndim == 5:
                num_views = int(video.shape[0])
                return (int(video.shape[-2]) * num_views, int(video.shape[-1]), int(video.shape[2]), num_views)
            if video.ndim == 4:
                return (int(video.shape[-2]), int(video.shape[-1]), int(video.shape[1]), 1)
            raise ValueError(f'Unsupported tensor video shape: {tuple(video.shape)}')
        if isinstance(video[0], (list, tuple)):
            num_views = len(video)
            return (video[0][0].size[1] * num_views, video[0][0].size[0], len(video[0]), num_views)
        return (video[0].size[1], video[0].size[0], len(video), 1)

    def get_pipeline_inputs(self, data):
        height, width, num_frames, num_views = self._video_geometry(data['video'])
        inputs_posi = {'prompt': data.get('prompt'), 'prompt_emb': data.get('prompt_emb')}
        inputs_nega = {'negative_prompt': data.get('negative_prompt'), 'prompt_emb': data.get('negative_prompt_emb')}
        inputs_shared = {'input_video': data['video'], 'action': data.get('action'), 'height': height, 'width': width, 'num_frames': num_frames, 'num_views': num_views, 'num_history_frames': self.num_history_frames, 'cfg_scale': 1, 'tiled': False, 'rand_device': self.pipe.device, 'use_gradient_checkpointing': self.use_gradient_checkpointing, 'use_gradient_checkpointing_offload': self.use_gradient_checkpointing_offload, 'cfg_merge': False, 'vace_scale': 1, 'max_timestep_boundary': self.max_timestep_boundary, 'min_timestep_boundary': self.min_timestep_boundary}
        if getattr(self.pipe, 'physical_context_mode', 'none') != 'none' or 'physical_context' in data:
            inputs_shared['physical_context'] = data.get('physical_context')
        if getattr(self.pipe, 'background_context_enabled', False) or 'background_context' in data:
            inputs_shared['background_context'] = data.get('background_context')
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return (inputs_shared, inputs_posi, inputs_nega)

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        inputs = self.apply_video_light_augmentation(inputs)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

def wan_parser():
    parser = argparse.ArgumentParser(description='WAN Video Action training script.')
    parser = add_general_config(parser)
    parser.add_argument('--stage1_warmup_steps', type=int, default=0, help='Linearly warm up the outer optimizer LR for vanilla stage1 SFT training.')
    return parser

def launch_training_task_with_optional_warmup(accelerator, dataset, model, model_logger, args):
    local_batch_size = int(args.batch_size)
    if local_batch_size < 1:
        raise ValueError(f'batch_size must be positive, got {local_batch_size}.')
    accumulation_steps = int(args.gradient_accumulation_steps)
    if accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be positive.')
    local_batch_size *= accumulation_steps
    accelerator.gradient_accumulation_steps = 1
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = int(getattr(args, 'stage1_warmup_steps', 0) or 0)
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)))
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=local_batch_size, collate_fn=lambda items: items, num_workers=args.dataset_num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    if accelerator.is_main_process:
        print(f'[sequential_batch] local_batch_size={local_batch_size} updates_per_epoch={len(dataloader)} epochs={args.num_epochs}', flush=True)
    for epoch_id in range(args.num_epochs):
        iterator = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for data_batch in iterator:
            optimizer.zero_grad()
            batch_loss = None
            microbatch_count = len(data_batch)
            for microbatch_id, data in enumerate(data_batch):
                sync_context = accelerator.no_sync(model) if microbatch_id + 1 < microbatch_count else nullcontext()
                with sync_context:
                    loss = model(data)
                    accelerator.backward(loss / float(microbatch_count))
                detached_loss = loss.detach().to(dtype=torch.float32)
                batch_loss = detached_loss if batch_loss is None else batch_loss + detached_loss
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model_logger.on_step_end(accelerator, model, args.save_steps, loss=batch_loss / float(microbatch_count))
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)
if __name__ == '__main__':
    parser = wan_parser()
    parser.add_argument('--frame_stride', type=int, default=1, help='Temporal stride applied identically to video frames and actions.')
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    set_global_seed(args.seed)
    runtime_config = prepare_runtime_config(args)
    loggers = [name for name in ('wandb', 'swanlab') if getattr(args, f'use_{name}', False)]
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with=loggers or None, kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)])
    special_operator_map = {}
    if runtime_config['text_enabled'] and 'prompt_emb' in runtime_config['data_file_keys']:
        special_operator_map['prompt_emb'] = ResolvePromptEmbPath(base_path=args.dataset_base_path)
    with open(args.action_stat_path, 'r') as f:
        stats = json.load(f)
    stat = {args.action_type: stats[args.action_type]} if args.action_type in stats else stats
    dataset = RoboTwinUnifiedDataset(base_path=args.dataset_base_path, metadata_path=args.dataset_metadata_path, repeat=args.dataset_repeat, data_file_keys=runtime_config['data_file_keys'], main_data_operator=create_video_operator(base_path=args.dataset_base_path, max_pixels=args.max_pixels, height=args.height, width=args.width, height_division_factor=16, width_division_factor=16, num_frames=args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder, resize_mode=args.resize_mode, pad_short=args.pad_short_chunks, frame_stride=args.frame_stride), special_operator_map=special_operator_map)
    pack_paths(dataset.data, ('video', 'start_frame', 'end_frame'), ('action', 'start_frame', 'end_frame'))
    if 'action' in runtime_config['data_file_keys']:
        dataset.special_operator_map['action'] = LoadCobotAction(base_path=args.dataset_base_path, action_type=args.action_type, stat=stat, num_frames=args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder, pad_short=args.pad_short_chunks, output_dim=args.action_dim, frame_stride=args.frame_stride)
    model = WanTrainingModule(model_paths=json.dumps(runtime_config['model_paths_list']), model_id_with_origin_paths=args.model_id_with_origin_paths, tokenizer_path=runtime_config['tokenizer_path'], enable_text=args.enable_text, trainable_models=args.trainable_models, lora_base_model=args.lora_base_model, lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank, lora_checkpoint=args.lora_checkpoint, preset_lora_path=args.preset_lora_path, preset_lora_model=args.preset_lora_model, use_gradient_checkpointing=args.use_gradient_checkpointing, use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload, extra_inputs=args.extra_inputs, modules=runtime_config['modules'], fp8_models=args.fp8_models, offload_models=args.offload_models, ckpt_path=args.ckpt_path, task=args.task, device='cpu' if args.initialize_model_on_cpu else accelerator.device, max_timestep_boundary=args.max_timestep_boundary, min_timestep_boundary=args.min_timestep_boundary, num_history_frames=args.num_history_frames, args=args)
    model_logger = TimedRetentionModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, save_minutes=args.checkpoint_save_minutes, keep_last=args.checkpoint_keep_last, log_steps=args.log_steps)
    launcher_map = {'sft:data_process': launch_data_process_task, 'direct_distill:data_process': launch_data_process_task, 'sft': launch_training_task, 'sft:train': launch_training_task, 'direct_distill': launch_training_task, 'direct_distill:train': launch_training_task}
    if (int(getattr(args, 'stage1_warmup_steps', 0) or 0) > 0 or int(args.batch_size) > 1) and args.task in {'sft', 'sft:train', 'direct_distill', 'direct_distill:train'}:
        launch_training_task_with_optional_warmup(accelerator, dataset, model, model_logger, args=args)
    else:
        launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
