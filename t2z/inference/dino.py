"""Grouped-support DINOv2 amortized-context inference."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from t2z.inference.standard import _parse_sample_indices, _run_autoregressive, build_infer_dataset, build_pipeline, prepare_sample_for_rollout
from t2z.training.dino import add_dinov2_config
from wan_video_action.methods.baselines.dinov2_amortized import DINOv2AmortizedContextEncoder
from wan_video_action.parsers import add_general_config, merge_yaml_and_args

def parse_args():
    parser = add_dinov2_config(add_general_config(argparse.ArgumentParser('DINOv2 support/query inference')))
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--skip_existing', action='store_true', default=False)
    parser.add_argument('--sample_indices', type=str, default=None)
    parser.add_argument('--support_indices', type=str, default=None)
    parser.add_argument('--expected_supports_per_environment', type=int, default=0, help='Require this many pooled support trajectories per environment when positive.')
    parser.add_argument('--dinov2_checkpoint_path', type=str, default=None)
    parser.add_argument('--wan_checkpoint_output', type=str, default=None)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    for key in ('sample_indices', 'support_indices', 'dinov2_checkpoint_path', 'wan_checkpoint_output'):
        if not getattr(args, key):
            parser.error(f'{key} is required in the config or CLI')
    return args

def materialize_wan_checkpoint(source_path: str, destination_path: str) -> None:
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    tensors = {}
    with safe_open(source_path, framework='pt', device='cpu') as handle:
        for key in handle.keys():
            if key.startswith('wan.'):
                tensors[key[len('wan.'):]] = handle.get_tensor(key)
    if not tensors:
        raise ValueError(f'No wan.* tensors found in {source_path}.')
    temporary = destination.with_name(f'.{destination.name}.partial')
    temporary.unlink(missing_ok=True)
    save_file(tensors, str(temporary))
    os.replace(temporary, destination)

def load_support_encoder(args, device: torch.device):
    encoder = DINOv2AmortizedContextEncoder(model_path=args.dinov2_model_path, sampled_frames=args.dinov2_sampled_frames, temporal_stride=args.dinov2_temporal_stride, action_dim=args.action_dim, hidden_dim=args.dinov2_hidden_dim, action_hidden_dim=args.dinov2_action_hidden_dim, output_dim=args.dinov2_output_dim, aggregation_mode=args.dinov2_aggregation_mode, mlp_hidden_dim=args.dinov2_mlp_hidden_dim)
    state = {}
    with safe_open(args.dinov2_checkpoint_path, framework='pt', device='cpu') as handle:
        for key in handle.keys():
            if key.startswith('support_encoder.'):
                state[key[len('support_encoder.'):]] = handle.get_tensor(key)
    incompatible = encoder.load_state_dict(state, strict=False)
    bad_missing = [key for key in incompatible.missing_keys if not key.startswith('dino.')]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(f'Invalid DINO head checkpoint: missing={bad_missing}, unexpected={incompatible.unexpected_keys}')
    encoder.to(device=device)
    encoder.eval()
    return encoder

def canonical_environment_id(value):
    """Keep continuous environment values distinct while making scalar keys stable."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f'Environment key must be scalar, got {tuple(value.shape)}.')
        value = value.detach().cpu().item()
    elif not isinstance(value, (str, int, float, bool)) and hasattr(value, 'item'):
        value = value.item()
    if not isinstance(value, (str, int, float, bool)):
        value = str(value)
    return value

def main() -> None:
    args = parse_args()
    materialize_wan_checkpoint(args.dinov2_checkpoint_path, args.wan_checkpoint_output)
    args.ckpt_path = args.wan_checkpoint_output
    os.makedirs(args.output_path, exist_ok=True)
    dataset = build_infer_dataset(args)
    pipe = build_pipeline(args)
    encoder = load_support_encoder(args, pipe.device)
    grouped_supports = defaultdict(list)
    support_records = []
    with torch.no_grad():
        for support_index in _parse_sample_indices(args.support_indices):
            support = dataset[support_index]
            environment_id = canonical_environment_id(support[args.dinov2_environment_key])
            visual_features, frame_indices, frame_count = encoder.extract_visual_features(support['video'])
            grouped_supports[environment_id].append({'support_index': int(support_index), 'visual_features': visual_features, 'action': support['action'], 'frame_indices': frame_indices, 'frame_count': frame_count})
        support_codes = {}
        for environment_id, records in grouped_supports.items():
            code = encoder.project_supports(visual_features=tuple((row['visual_features'] for row in records)), actions=tuple((row['action'] for row in records)), frame_indices=tuple((row['frame_indices'] for row in records)), frame_counts=tuple((row['frame_count'] for row in records)))[0].detach().cpu()
            support_indices = [row['support_index'] for row in records]
            support_codes[environment_id] = code
            support_records.append({'environment_id': environment_id, 'support_index': support_indices[0], 'support_indices': support_indices, 'support_count': len(support_indices), 'aggregation': 'mean_support_summary_before_output_head', 'context': code.float().tolist()})
    with (Path(args.output_path).parent / 'support_codes.json').open('w', encoding='utf-8') as handle:
        json.dump({'aggregation': 'mean_support_summary_before_output_head', 'expected_supports_per_environment': int(args.expected_supports_per_environment), 'records': support_records}, handle, indent=2)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for sample_index in _parse_sample_indices(args.sample_indices):
        sample = dataset[sample_index]
        sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
        environment_id = canonical_environment_id(sample[args.dinov2_environment_key])
        if environment_id not in support_codes:
            raise KeyError(f'No DINO support code for environment {environment_id}.')
        sample['physical_context'] = support_codes[environment_id].to(device=pipe.device, dtype=pipe.torch_dtype)
        if args.skip_existing and Path(sample['output_path']).is_file():
            print(f"[skip] existing prediction {sample['output_path']}")
            continue
        predicted_path = _run_autoregressive(pipe=pipe, sample=sample, args=args)
        print(f'[done] sample_index={sample_index} environment={environment_id} output={predicted_path}', flush=True)
if __name__ == '__main__':
    main()
