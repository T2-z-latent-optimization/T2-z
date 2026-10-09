import argparse
import glob
import json
import os
import sys
from omegaconf import OmegaConf

def merge_yaml_and_args(yaml_path, parser, args):
    if not yaml_path or not os.path.exists(yaml_path):
        return args
    yaml_dict = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True) or {}
    explicit_cli_keys = set()
    argv = sys.argv[1:]
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith('--') and (option in argv or any((arg.startswith(f'{option}=') for arg in argv))):
                explicit_cli_keys.add(action.dest)
    cli_overrides = {}
    for key, value in vars(args).items():
        if key in explicit_cli_keys or value != parser.get_default(key):
            cli_overrides[key] = value
    for section in yaml_dict.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if key not in cli_overrides and hasattr(args, key):
                    setattr(args, key, value)
    return args

def prepare_runtime_config(args):
    model_config_path = getattr(args, 'model_config_path', '')
    cfg = OmegaConf.to_container(OmegaConf.load(model_config_path), resolve=True)
    enabled_mods = [m for m in ['dit', 'vae', 'image'] if getattr(args, f'enable_{m}', True)]
    text_mode = getattr(args, 'text_mode', 'emb')
    text_enabled = getattr(args, 'enable_text', True) and text_mode != 'none'
    if text_enabled:
        enabled_mods.append('text' if text_mode == 't5' else f'text:{text_mode}')
    action_mode = getattr(args, 'action_mode', 'none')
    if action_mode != 'none':
        enabled_mods.append(f'action:{action_mode}')
    model_paths = getattr(args, 'model_paths', '')
    yaml_modules_map = cfg.get('modules', {})

    def resolve_module_path(module_entry):
        if isinstance(module_entry, (list, tuple)):
            return [os.path.join(model_paths, path) for path in module_entry]
        path = os.path.join(model_paths, module_entry)
        if isinstance(module_entry, str) and module_entry.endswith('.safetensors.index.json'):
            with open(path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            weight_files = sorted(set(index.get('weight_map', {}).values()))
            if weight_files:
                return [os.path.join(os.path.dirname(path), file_name) for file_name in weight_files]
        if isinstance(module_entry, str) and glob.has_magic(path):
            matches = sorted(glob.glob(path))
            return matches if len(matches) != 1 else matches[0]
        return path
    paths_list = [resolve_module_path(yaml_modules_map[m.split(':')[0]]) for m in enabled_mods if m.split(':')[0] in yaml_modules_map]
    tokenizer_path = None
    if text_enabled and text_mode == 't5' and model_paths:
        subdir = getattr(args, 'tokenizer_subdir', cfg.get('tokenizer_subdir', 'tokenizer'))
        tokenizer_path = os.path.join(model_paths, subdir)
    raw_keys = getattr(args, 'data_file_keys', 'image,video')
    data_keys = [k.strip() for k in raw_keys.split(',') if k.strip()]
    return {'modules': enabled_mods, 'model_paths_list': paths_list, 'tokenizer_path': tokenizer_path, 'data_file_keys': data_keys, 'text_enabled': text_enabled, 'action_enabled': action_mode != 'none'}

def add_dataset_base_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('dataset')
    group.add_argument('--dataset_base_path', type=str, default='', help='[REQUIRED] Base path of the dataset.')
    group.add_argument('--dataset_metadata_path', type=str, default=None, help='[OPTIONAL] Path to the metadata file of the dataset.')
    group.add_argument('--dataset_repeat', type=int, default=1, help='[TUNABLE] Number of times to repeat the dataset per epoch.')
    group.add_argument('--dataset_num_workers', type=int, default=0, help='[OPTIONAL] Number of workers for data loading.')
    group.add_argument('--data_file_keys', type=str, default='image,video', help='[OPTIONAL] Data file keys in the metadata. Comma-separated.')
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('video')
    group.add_argument('--height', type=int, default=None, help='[KEY] Height of images. Leave `height` and `width` empty to enable dynamic resolution.')
    group.add_argument('--width', type=int, default=None, help='[KEY] Width of images. Leave `height` and `width` empty to enable dynamic resolution.')
    group.add_argument('--max_pixels', type=int, default=1048576, help='[OPTIONAL] Maximum number of pixels per frame, used for dynamic resolution.')
    group.add_argument('--num_frames', type=int, default=81, help='[KEY] Number of frames per video. Frames are sampled from the video prefix.')
    group.add_argument('--resize_mode', type=str, default='fit', choices=['crop', 'fit', 'letterbox'], help='[OPTIONAL] Resize behavior: crop (center crop), fit (dynamic aspect ratio), letterbox (fixed size with padding).')
    group.add_argument('--num_history_frames', type=int, default=1, help='[KEY] Number of conditioning history frames. Must satisfy 1 <= num_history_frames < num_frames.')
    group.add_argument('--time_division_factor', type=int, default=4, help='[OPTIONAL] Temporal frame divisor used to align video/action frame counts.')
    group.add_argument('--time_division_remainder', type=int, default=1, help='[OPTIONAL] Temporal frame remainder used with time_division_factor.')
    group.add_argument('--spatial_division_factor', type=int, default=32, help='[OPTIONAL] Spatial size divisor used to align frame height and width.')
    group.add_argument('--chunk_mode', type=str, default='static', choices=['static', 'dynamic'], help='[OPTIONAL] Sampling mode for video chunks, static uses dataset bounds and dynamic uses random crop.')
    group.add_argument('--pad_short_chunks', action='store_true', default=False, help='[KEY] Repeat-pad video/action chunks that run past episode end instead of shortening them.')
    group.add_argument('--video_light_augmentation_enabled', action='store_true', default=False, help='[OPTIONAL] Apply temporally coherent online light/shadow augmentation before VAE encoding.')
    group.add_argument('--video_light_augmentation_probability', type=float, default=0.7, help='[OPTIONAL] Probability of augmenting a training video; the remainder stays exactly original.')
    group.add_argument('--video_light_augmentation_view_indices', type=str, default='all', help="[OPTIONAL] Views to augment: 'all' or comma-separated zero-based indices.")
    group.add_argument('--video_light_augmentation_gain_min', type=float, default=0.88)
    group.add_argument('--video_light_augmentation_gain_max', type=float, default=1.12)
    group.add_argument('--video_light_augmentation_contrast_min', type=float, default=0.9)
    group.add_argument('--video_light_augmentation_contrast_max', type=float, default=1.1)
    group.add_argument('--video_light_augmentation_gamma_min', type=float, default=0.9)
    group.add_argument('--video_light_augmentation_gamma_max', type=float, default=1.1)
    group.add_argument('--video_light_augmentation_tint_min', type=float, default=0.97)
    group.add_argument('--video_light_augmentation_tint_max', type=float, default=1.03)
    group.add_argument('--video_light_augmentation_offset_min', type=float, default=-0.015)
    group.add_argument('--video_light_augmentation_offset_max', type=float, default=0.015)
    group.add_argument('--video_light_augmentation_gradient_abs_max', type=float, default=0.08)
    group.add_argument('--video_light_augmentation_noise_std', type=float, default=0.004)
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('model')
    group.add_argument('--model_paths', type=str, default=None, help='[REQUIRED] Paths to load models. In JSON format, comma-separated, or a single model root.')
    group.add_argument('--model_id_with_origin_paths', type=str, default=None, help='[OPTIONAL] Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.')
    group.add_argument('--extra_inputs', type=str, default=None, help='[OPTIONAL] Additional model inputs, comma-separated.')
    group.add_argument('--fp8_models', type=str, default=None, help='[OPTIONAL] Models with FP8 precision, comma-separated.')
    group.add_argument('--offload_models', type=str, default=None, help='[OPTIONAL] Models with offload, comma-separated. Only used in splited training.')
    group.add_argument('--model_config_path', type=str, default='configs/model/wan2_2_ti2v_5b.yaml', help='[KEY] Path to model config YAML.')
    group.add_argument('--initialize_model_on_cpu', action='store_true', default=False, help='[OPTIONAL] Whether to initialize models on CPU.')
    group.add_argument('--text_mode', type=str, default='emb', choices=['t5', 'emb', 'none'], help="[KEY] Execution mode for the text module: 't5' (full inference), 'emb' (use pre-extracted embeddings), 'none' (disabled).")
    group.add_argument('--enable_text', action='store_true', default=True, help='[OPTIONAL] Enable text encoder.')
    group.add_argument('--enable_dit', action='store_true', default=True, help='[OPTIONAL] Enable DiT diffusion model.')
    group.add_argument('--enable_vae', action='store_true', default=True, help='[OPTIONAL] Enable VAE model.')
    group.add_argument('--enable_image', action='store_true', default=True, help='[OPTIONAL] Enable image encoder (CLIP).')
    group.add_argument('--action_mode', type=str, default='noise', choices=['noise', 'adaln', 'none'], help='[KEY] Action injection mode: noise (per-frame), adaln (global), none (disabled).')
    return parser

def add_action_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('action')
    group.add_argument('--action_type', type=str, choices=['joint_abs', 'eef_abs', 'joint_delta', 'eef_delta', 'joint_state_action', 'eef_observed_state', 'eef_state_action', 'eef_swing_angle', 'joint_target', 'joint_target_export', 'eef_target'], default='eef_delta', help='[KEY] Action/state representation. joint_target reads recorded commanded joints; joint_target_export reads target_joint_action[:7]; eef_target reads an explicitly target-EEF action export. ')
    group.add_argument('--action_stat_path', type=str, default=None, help='[OPTIONAL] Path to robot normalization stats (stat.json).')
    group.add_argument('--action_dim', type=int, default=14, help='[OPTIONAL] Action dimension.')
    group.add_argument('--canonical_target_eef', action='store_true', default=False)
    return parser

def add_physical_context_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('physical_context')
    group.add_argument('--physical_context_mode', type=str, default='none', choices=['none', 'token', 'modulation', 'both'], help='[KEY] How latent physical code C conditions the world model.')
    group.add_argument('--physical_context_dim', type=int, default=128, help='[TUNABLE] Latent physical code dimension.')
    group.add_argument('--physical_context_tokens', type=int, default=1, help='[TUNABLE] Number of C tokens inserted into cross-attention context.')
    group.add_argument('--physical_context_hidden_dim', type=int, default=0, help='[TUNABLE] Hidden width for C projection MLPs. 0 selects a small default.')
    group.add_argument('--physical_context_init_std', type=float, default=0.0, help='[OPTIONAL] Std for non-zero default C initialization.')
    group.add_argument('--physical_context_init_value', type=float, default=0.0, help='[OPTIONAL] Mean value for default latent C initialization.')
    group.add_argument('--physical_context_input_norm', type=str, default='layernorm', choices=['layernorm', 'none'], help='[TUNABLE] Normalization before projecting latent C.')
    group.add_argument('--physical_context_projection', type=str, default='mlp', choices=['mlp'], help='Project the context with shared MLPs.')
    group.add_argument('--physical_context_temporal_position', type=str, default='none', choices=['none', 'learned'], help='[TUNABLE] Temporal position embedding for per-time latent C tokens.')
    group.add_argument('--background_context_enabled', action='store_true', default=False, help='[KEY] Add a separately encoded context shared by all friction groups in one visual background.')
    group.add_argument('--background_context_dim', type=int, default=32, help='[TUNABLE] Latent background-context dimension.')
    group.add_argument('--background_context_tokens', type=int, default=1, help='[TUNABLE] Number of background-context cross-attention tokens.')
    group.add_argument('--background_context_hidden_dim', type=int, default=0, help='[TUNABLE] Hidden width for the independent background-context projection MLPs.')
    group.add_argument('--background_context_init_std', type=float, default=0.0, help='[OPTIONAL] Std for non-zero default background-context initialization.')
    group.add_argument('--background_context_init_value', type=float, default=0.0, help='[OPTIONAL] Mean value for default background-context initialization.')
    group.add_argument('--background_context_input_norm', type=str, default='layernorm', choices=['layernorm', 'none'], help='[TUNABLE] Normalization before projecting background context.')
    group.add_argument('--background_context_temporal_position', type=str, default='none', choices=['none', 'learned'], help='[TUNABLE] Temporal position embedding for per-time background-context tokens.')
    group.add_argument('--physical_adapter_mode', type=str, default='none', choices=['none', 'residual'], help='[KEY] Lightweight residual-adapter parameter adaptation mode.')
    group.add_argument('--physical_adapter_rank', type=int, default=16, help='[TUNABLE] Low-rank width for residual physical adapters.')
    group.add_argument('--physical_adapter_layers', type=str, default='all', help='[TUNABLE] Adapter layer set: "all", "uniform:N", "last_quarter", "last:N", or comma-separated layer indices.')
    group.add_argument('--physical_adapter_gate_init', type=float, default=0.0, help='[OPTIONAL] Initial residual adapter gate value.')
    group.add_argument('--physical_support_weight', type=float, default=1.0, help='[PLANNED] Weight for support-trajectory adaptation loss.')
    group.add_argument('--physical_query_weight', type=float, default=1.0, help='[PLANNED] Weight for same-property query loss.')
    group.add_argument('--physical_specificity_weight', type=float, default=0.0, help='[PLANNED] Penalty against overfitting only the support trajectory.')
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('training')
    group.add_argument('--learning_rate', type=float, default=0.0001, help='[TUNABLE] Learning rate.')
    group.add_argument('--num_epochs', type=int, default=1, help='[TUNABLE] Number of epochs.')
    group.add_argument('--trainable_models', type=str, default=None, help='[KEY] Models to train, e.g., dit, vae, text_encoder.')
    group.add_argument('--find_unused_parameters', action='store_true', default=False, help='[OPTIONAL] Whether to find unused parameters in DDP.')
    group.add_argument('--weight_decay', type=float, default=0.01, help='[TUNABLE] Weight decay.')
    group.add_argument('--task', type=str, default='sft', help='[OPTIONAL] Task type.')
    group.add_argument('--seed', type=int, default=42, help='[OPTIONAL] Random seed for python/numpy/torch.')
    group.add_argument('--mixed_precision', type=str, default='bf16', choices=['no', 'fp16', 'bf16'], help='[OPTIONAL] Mixed precision mode.')
    group.add_argument('--max_timestep_boundary', type=float, default=1.0, help='[OPTIONAL] Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).')
    group.add_argument('--min_timestep_boundary', type=float, default=0.0, help='[OPTIONAL] Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).')
    group.add_argument('--batch_size', type=int, default=1, help='[TUNABLE] Batch size per GPU.')
    group.add_argument('--history_anchor_windows', action='store_true', default=False)
    group.add_argument('--environment_sampling_scope', choices=('global_batch', 'independent_rank'), default='global_batch')
    group.add_argument('--checkpoint_policy', choices=('periodic', 'protected'), default='periodic')
    group.add_argument('--protected_checkpoint_steps', type=str, default='')
    group.add_argument('--checkpoint_deadline_env', type=str, default='T2Z_TRAINING_DEADLINE_EPOCH')
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('output')
    group.add_argument('--output_path', type=str, default='./models', help='[KEY] Output save path.')
    group.add_argument('--remove_prefix_in_ckpt', type=str, default='pipe.dit.', help='[OPTIONAL] Remove prefix in ckpt. (default: "pipe.dit.")')
    group.add_argument('--save_steps', type=int, default=None, help='[OPTIONAL] Number of checkpoint saving intervals. If None, checkpoints will be saved every epoch.')
    group.add_argument('--checkpoint_save_minutes', type=float, default=0.0, help='[OPTIONAL] Save checkpoints by wall-clock interval. 0 disables timed checkpoints.')
    group.add_argument('--checkpoint_keep_last', type=int, default=0, help='[OPTIONAL] Keep only the newest N .safetensors checkpoints in output_path. 0 disables pruning.')
    group.add_argument('--log_steps', type=int, default=10, help='[OPTIONAL] Print training loss every N optimizer steps on the main process.')
    group.add_argument('--ckpt_path', type=str, default=None, help='[OPTIONAL] Path to model checkpoint (.safetensors) used to initialize training weights (model-only resume).')
    group.add_argument('--resume_from', type=str, default=None, help='[OPTIONAL] Path to a checkpoint directory saved by accelerator (e.g., output_path/epoch-0).')
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('lora')
    group.add_argument('--lora_base_model', type=str, default=None, help='[OPTIONAL] Which model LoRA is added to.')
    group.add_argument('--lora_target_modules', type=str, default='q,k,v,o,ffn.0,ffn.2', help='[OPTIONAL] Which layers LoRA is added to.')
    group.add_argument('--lora_rank', type=int, default=32, help='[TUNABLE] Rank of LoRA.')
    group.add_argument('--lora_checkpoint', type=str, default=None, help='[OPTIONAL] Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.')
    group.add_argument('--preset_lora_path', type=str, default=None, help='[OPTIONAL] Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.')
    group.add_argument('--preset_lora_model', type=str, default=None, help='[OPTIONAL] Which model the preset LoRA is fused to.')
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('gradient')
    group.add_argument('--use_gradient_checkpointing', action='store_true', default=False, help='[KEY] Whether to use gradient checkpointing.')
    group.add_argument('--use_gradient_checkpointing_offload', action='store_true', default=False, help='[KEY] Whether to offload gradient checkpointing to CPU memory.')
    group.add_argument('--gradient_accumulation_steps', type=int, default=1, help='[TUNABLE] Gradient accumulation steps.')
    group.add_argument('--max_grad_norm', type=float, default=0.5, help='[OPTIONAL] Maximum gradient norm for clipping. (default: 0.5)')
    return parser

def add_tracking_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('tracking')
    group.add_argument('--use_wandb', action='store_true', default=False, help='[OPTIONAL] Enable Weights & Biases tracking.')
    group.add_argument('--use_swanlab', action='store_true', default=False, help='[OPTIONAL] Enable SwanLab tracking.')
    group.add_argument('--swanlab_experiment_name', type=str, default=None, help='[OPTIONAL] SwanLab experiment name. Defaults to output_path.')
    return parser

def add_infer_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('infer')
    group.add_argument('--cfg_scale', type=float, default=5.0, help='[OPTIONAL] CFG scale for generation.')
    group.add_argument('--num_inference_steps', type=int, default=50, help='[OPTIONAL] Number of inference steps.')
    group.add_argument('--negative_prompt', type=str, default='The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion', help='[OPTIONAL] Negative prompt for generation.')
    group.add_argument('--negative_prompt_emb', type=str, default=None, help='[OPTIONAL] Path to the pre-extracted negative prompt embedding.')
    group.add_argument('--quality', type=int, default=5, help='[OPTIONAL] Output video quality.')
    group.add_argument('--disable_chunk_infer', dest='chunk_infer', action='store_false', default=True, help='[OPTIONAL] Disable chunked inference with 81-frame segments.')
    group.add_argument('--fps', type=int, default=24, help='[OPTIONAL] Output video FPS.')
    group.add_argument('--disable_metrics', dest='enable_metrics', action='store_false', default=True, help='[OPTIONAL] Disable evaluation metrics.')
    group.add_argument('--start_index', type=int, default=0, help='[OPTIONAL] First metadata row index to process.')
    group.add_argument('--max_samples', type=int, default=0, help='[OPTIONAL] Maximum number of metadata rows to process. 0 means all.')
    return parser

def add_config_support(parser: argparse.ArgumentParser):
    parser.add_argument('--config', type=str, default=None, help='Path to a YAML config file. CLI args override YAML config.')
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_config_support(parser)
    parser = add_dataset_base_config(parser)
    parser = add_video_size_config(parser)
    parser = add_model_config(parser)
    parser = add_action_config(parser)
    parser = add_physical_context_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_tracking_config(parser)
    parser = add_infer_config(parser)
    return parser
