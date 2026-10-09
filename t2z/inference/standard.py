import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np
import torch
from diffsynth.core import ModelConfig
from wan_video_action.data import LoadCobotAction, RoboTwinUnifiedDataset, create_video_operator
from wan_video_action.data.data_utils import pack_paths
from wan_video_action.parsers import add_general_config, merge_yaml_and_args, prepare_runtime_config
from wan_video_action.pipelines.wan_video_action import build_wan_video_action_pipeline
from wan_video_action.utils import align_num_frames, resolve_model_path, save_video
from t2z.data import configure_dataset as _configure_dataset

def parse_args():
    parser = argparse.ArgumentParser('World-model prediction')
    parser = add_general_config(parser)
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--skip_existing', action='store_true', default=False)
    parser.add_argument('--sample_indices', type=str, default=None, help='[OPTIONAL] Comma/range list of metadata row indices to process, e.g. 2,10,20-22.')
    parser.add_argument('--physical_context_override', type=str, default=None, help='[OPTIONAL] Comma-separated physical C override for all samples, e.g. 0.5 or 0.1,0.2.')
    parser.add_argument('--physical_context_table_path', type=str, default=None, help='[OPTIONAL] Grouped-C context_table.json. If set, lookup C by sample friction_mu.')
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args

def _parse_sample_indices(value: str | None):
    if not value:
        return None
    indices = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices

def _parse_physical_context_override(value: str | None):
    if value is None or str(value).strip() == '':
        return None
    return [float(part.strip()) for part in str(value).split(',') if part.strip()]

def _load_physical_context_table(path: str | None):
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    table = {}
    for record in data.get('records', []):
        mu = round(float(record['friction_mu']), 10)
        table[mu] = record['context']
    if not table:
        raise ValueError(f'No records found in physical context table: {path}')
    return table

def _lookup_physical_context(table: dict | None, sample: Dict):
    if table is None:
        return None
    if 'friction_mu' not in sample:
        raise KeyError('Sample has no friction_mu for physical_context_table lookup.')
    mu = round(float(sample['friction_mu']), 10)
    if mu in table:
        return table[mu]
    best_mu = min(table, key=lambda key: abs(key - mu))
    if abs(best_mu - mu) > 1e-06:
        raise KeyError(f'No context for friction_mu={mu}; nearest={best_mu}')
    return table[best_mu]

def _build_autoregressive_history_indices(num_generated_frames: int, history_frames: int) -> List[int]:
    if history_frames <= 0:
        return []
    if num_generated_frames <= 0:
        raise ValueError('Autoregressive history requires at least one generated frame.')
    if num_generated_frames < history_frames:
        return [0] * (history_frames - num_generated_frames) + list(range(num_generated_frames))
    if history_frames == 1:
        return [num_generated_frames - 1]
    return [0] + list(range(num_generated_frames - (history_frames - 1), num_generated_frames))

def _build_autoregressive_action_condition(action: torch.Tensor, history_indices: List[int], future_start: int, future_count: int, infer_frames: int) -> torch.Tensor:
    history_action = action[:, history_indices, :]
    future_action = action[:, future_start:future_start + future_count, :]
    action_condition = torch.cat([history_action, future_action], dim=1)
    current_frames = int(action_condition.shape[1])
    if current_frames < infer_frames:
        pad = action_condition[:, -1:, :].repeat(1, infer_frames - current_frames, 1)
        action_condition = torch.cat([action_condition, pad], dim=1)
    return action_condition

def _run_autoregressive(pipe, sample: Dict, args):
    sample_index = int(sample['sample_index'])
    episode_index = int(sample['episode_index'])
    total_frames = int(sample['total_frames'])
    input_video = sample['video']
    action = sample['action']
    output_path = sample['output_path']
    history_frames = int(args.num_history_frames)
    chunk_size = int(args.num_frames)
    future_frames = chunk_size - history_frames
    if future_frames <= 0:
        raise ValueError(f'Invalid rollout setting: num_frames={chunk_size}, num_history_frames={history_frames}')
    generated_frames: List[torch.Tensor] = [input_video[:, :, 0].clone()]
    chunk_idx = 0
    while len(generated_frames) < total_frames:
        future_start = len(generated_frames)
        remaining_future = total_frames - future_start
        current_future = min(future_frames, remaining_future)
        requested_frames = history_frames + current_future
        infer_frames = requested_frames
        history_indices = _build_autoregressive_history_indices(future_start, history_frames)
        chunk_input_video = torch.stack([generated_frames[index] for index in history_indices], dim=2)
        chunk_action = _build_autoregressive_action_condition(action=action, history_indices=history_indices, future_start=future_start, future_count=current_future, infer_frames=infer_frames)
        print(f'[window] sample_index={sample_index} episode_index={episode_index} window={chunk_idx} future_start={future_start} future_count={current_future} infer_frames={infer_frames}')
        seed = None if args.seed is None else int(args.seed) + chunk_idx
        chunk_video = pipe(input_video=chunk_input_video, action=chunk_action, physical_context=sample.get('physical_context'), seed=seed, rand_device='cpu', tiled=False, height=int(args.height) * int(chunk_input_video.shape[0]), width=int(args.width), num_frames=infer_frames, num_history_frames=int(args.num_history_frames), cfg_scale=float(args.cfg_scale), num_inference_steps=int(args.num_inference_steps), use_history_condition_noise_in_inference=True, progress_bar_cmd=lambda iterable, *args, **kwargs: iterable, output_type='floatpoint')
        future_video = chunk_video[:, :, history_frames:].detach().cpu()
        append_count = min(current_future, int(future_video.shape[2]))
        if append_count <= 0:
            raise RuntimeError('Autoregressive inference produced no future frames.')
        for frame_idx in range(append_count):
            generated_frames.append(future_video[:, :, frame_idx].clone())
        chunk_idx += 1
    predicted_video = torch.stack(generated_frames, dim=2)[:, :, :total_frames]
    final_output = Path(output_path)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = final_output.with_name(f'.{final_output.stem}.partial{final_output.suffix}')
    temporary_output.unlink(missing_ok=True)
    try:
        save_video(predicted_video, output_path=str(temporary_output), fps=int(args.fps), quality=int(args.quality))
        temporary_output.replace(final_output)
    finally:
        temporary_output.unlink(missing_ok=True)
    return str(final_output)

def build_pipeline(args):
    runtime_config = prepare_runtime_config(args)
    model_configs = []
    for model_path in runtime_config['model_paths_list']:
        if isinstance(model_path, (list, tuple)):
            model_configs.append(ModelConfig(path=[resolve_model_path(path) for path in model_path]))
        else:
            resolved_path = resolve_model_path(model_path)
            model_configs.append(ModelConfig(path=resolved_path))
    print('[resolved_models] model_configs:', [config.path for config in model_configs])
    dtype = torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = build_wan_video_action_pipeline(torch_dtype=dtype, device=device, model_configs=model_configs, tokenizer_config=None, ckpt_path=args.ckpt_path, action_dim=args.action_dim, action_mode=args.action_mode, args=args)
    pipe.use_gradient_checkpointing = False
    pipe.use_gradient_checkpointing_offload = False
    pipe.eval()
    return pipe

def build_infer_dataset(args):
    with open(args.action_stat_path, 'r') as f:
        stats = json.load(f)
    stat = {args.action_type: stats[args.action_type]} if args.action_type in stats else stats
    dataset = RoboTwinUnifiedDataset(base_path=args.dataset_base_path, metadata_path=args.dataset_metadata_path, repeat=1, data_file_keys=('video', 'action'), main_data_operator=create_video_operator(base_path=args.dataset_base_path, max_pixels=args.max_pixels, height=args.height, width=args.width, height_division_factor=16, width_division_factor=16, num_frames=args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder, resize_mode=args.resize_mode, pad_short=args.pad_short_chunks, frame_stride=int(getattr(args, 'frame_stride', 1))), special_operator_map={})
    pack_paths(dataset.data, ('video', 'start_frame', 'end_frame'), ('action', 'start_frame', 'end_frame'))
    dataset.special_operator_map['action'] = LoadCobotAction(base_path=args.dataset_base_path, action_type=args.action_type, stat=stat, num_frames=args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder, pad_short=args.pad_short_chunks, output_dim=args.action_dim, frame_stride=int(getattr(args, 'frame_stride', 1)))
    return _configure_dataset(dataset, args)

def prepare_sample_for_rollout(sample: Dict, sample_index: int, pipe, args) -> Dict:
    raw_total_frames = int(sample['length'])
    total_frames = align_num_frames(raw_total_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder)
    if total_frames <= 0:
        raise ValueError(f'Invalid aligned total_frames={total_frames} from raw length={raw_total_frames}')
    episode_index = int(sample['episode_index'])
    start_frame = int(sample.get('start_frame', 0))
    end_frame = int(sample.get('end_frame', start_frame + raw_total_frames - 1))
    raw_action = sample['action']
    raw_action_shape = tuple(raw_action.shape) if torch.is_tensor(raw_action) else tuple(np.asarray(raw_action).shape)
    action = torch.as_tensor(raw_action, dtype=pipe.torch_dtype, device=pipe.device)
    action_frames = int(action.shape[-2])
    if action_frames != raw_total_frames:
        raise ValueError(f"Action/video temporal mismatch after loading: sample={sample_index}, action_frames={action_frames}, metadata_frames={raw_total_frames}, frame_stride={int(getattr(args, 'frame_stride', 1))}, raw_action_shape={raw_action_shape}.")
    sample['sample_index'] = sample_index
    sample['raw_total_frames'] = raw_total_frames
    sample['total_frames'] = total_frames
    sample['start_frame'] = start_frame
    sample['end_frame'] = end_frame
    sample['raw_action_shape'] = raw_action_shape
    sample['output_path'] = os.path.join(args.output_path, f'sample{sample_index:04d}_episode{episode_index:06d}_frames{start_frame:04d}-{end_frame:04d}.mp4')
    sample['action'] = action[:, :total_frames]
    return sample

def main():
    args = parse_args()
    print('[resolved_config] model_paths:', args.model_paths)
    print('[resolved_config] model_config_path:', args.model_config_path)
    print('[resolved_config] dataset_base_path:', args.dataset_base_path)
    print('[resolved_config] dataset_metadata_path:', args.dataset_metadata_path)
    print('[resolved_config] action_stat_path:', args.action_stat_path)
    print('[resolved_config] ckpt_path:', args.ckpt_path)
    print('[resolved_config] output_path:', args.output_path)
    print('[resolved_config] profile: Wan2.2 TI2V, text off, image off, VAE fused latent, action adaln, first-frame rollout')
    print('[resolved_config] height:', args.height)
    print('[resolved_config] width:', args.width)
    print('[resolved_config] num_frames:', args.num_frames)
    print('[resolved_config] frame_stride:', int(getattr(args, 'frame_stride', 1)))
    print('[resolved_config] num_history_frames:', args.num_history_frames)
    print('[resolved_config] time_division_factor:', args.time_division_factor)
    print('[resolved_config] time_division_remainder:', args.time_division_remainder)
    print('[resolved_config] action_type:', args.action_type)
    print('[resolved_config] cfg_scale:', args.cfg_scale)
    print('[resolved_config] num_inference_steps:', args.num_inference_steps)
    print('[resolved_config] fps:', args.fps)
    physical_context_override = _parse_physical_context_override(getattr(args, 'physical_context_override', None))
    physical_context_table = _load_physical_context_table(getattr(args, 'physical_context_table_path', None))
    print('[resolved_config] physical_context_override:', physical_context_override)
    print('[resolved_config] physical_context_table_path:', getattr(args, 'physical_context_table_path', None))
    os.makedirs(args.output_path, exist_ok=True)
    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    sample_indices = _parse_sample_indices(getattr(args, 'sample_indices', None))
    if sample_indices is None:
        sample_indices = range(args.start_index, len(dataset))
    processed = 0
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f'sample_index={sample_index} is outside dataset length {len(dataset)}')
        if args.max_samples and processed >= args.max_samples:
            break
        sample = dataset[sample_index]
        sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
        if physical_context_override is not None:
            sample['physical_context'] = torch.tensor(physical_context_override, dtype=pipe.torch_dtype, device=pipe.device)
        elif physical_context_table is not None:
            sample['physical_context'] = torch.tensor(_lookup_physical_context(physical_context_table, sample), dtype=pipe.torch_dtype, device=pipe.device)
        print(f"[sample] sample_index={sample['sample_index']} episode_index={sample['episode_index']} range=[{sample['start_frame']},{sample['end_frame']}] video_shape={tuple(sample['video'].shape)} action_shape={sample['raw_action_shape']}")
        print(f"[sample_window_target] sample_index={sample['sample_index']} episode_index={sample['episode_index']} range=[0,{sample['total_frames'] - 1}] output={sample['output_path']}")
        if args.skip_existing and Path(sample['output_path']).is_file():
            print(f"[skip] existing prediction {sample['output_path']}")
            processed += 1
            continue
        predicted_path = _run_autoregressive(pipe=pipe, sample=sample, args=args)
        print(f"[done] sample_index={sample_index} episode_index={sample['episode_index']} output={predicted_path}")
        processed += 1
if __name__ == '__main__':
    main()
