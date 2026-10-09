from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image, ImageDraw
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from t2z.inference.standard import _parse_sample_indices, _run_autoregressive, build_infer_dataset, build_pipeline, prepare_sample_for_rollout
from make_gt_pred_comparison import _default_pred_name, _read_gt_video, _read_pred_video, _resize_rgb

def add_adaptation_config(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('stage2_ttt')
    group.add_argument('--stage2_group_keys', type=str, default='environment_index')
    group.add_argument('--stage2_inner_steps', type=int, default=1)
    group.add_argument('--stage2_inner_lr', type=float, default=0.05)
    group.add_argument('--stage2_inner_lr_schedule', type=str, default='', help='Optional inner-loop schedule, e.g. "0.6:5,0.3:5,0.1:5,0.03:5". Overrides stage2_inner_steps/lr.')
    group.add_argument('--stage2_inner_grad_clip', type=float, default=1.0)
    group.add_argument('--stage2_context_clamp_min', type=float, default=None)
    group.add_argument('--stage2_context_clamp_max', type=float, default=None)
    group.add_argument('--stage2_context_reg_weight', type=float, default=0.001)
    group.add_argument('--stage2_known_context_keys', type=str, default='', help='Comma-separated metadata keys that should be directly injected into the tail dimensions of physical_context during stage2. These dimensions are per-sample oracle conditions and are not learned as shared TTT context.')
    return parser
from wan_video_action.data import LoadCobotAction, RoboTwinUnifiedDataset, create_video_operator
from wan_video_action.parsers import add_general_config, merge_yaml_and_args, prepare_runtime_config
from wan_video_action.pipelines.wan_video_action import load_checkpoint_weights
from wan_video_action.utils import set_global_seed
from t2z.data import configure_dataset as _configure_dataset

def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'.{path.name}.partial')
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + '\n')
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

def _load_grouped_context_table(path: str | Path | None):
    if not path:
        return None
    with Path(path).open('r', encoding='utf-8') as f:
        data = json.load(f)
    records = data.get('records', [])
    if not records:
        raise ValueError(f'No context table records found in {path}.')
    friction_values = np.asarray([float(record['friction_mu']) for record in records], dtype=np.float32)
    contexts = np.asarray([record['context'] for record in records], dtype=np.float32)
    if contexts.ndim == 2:
        contexts = contexts[:, None, :]
    if contexts.ndim != 3:
        raise ValueError(f'Expected context table shape (groups,tokens,dim), got {contexts.shape}.')
    global_context = data.get('global_context')
    if global_context is not None:
        global_context = np.asarray(global_context, dtype=np.float32)
        if global_context.ndim == 1:
            global_context = global_context[None, :]
        if global_context.shape != contexts.shape[1:]:
            raise ValueError(f'Expected global context shape {contexts.shape[1:]}, got {global_context.shape}.')
    return {'path': str(path), 'friction_values': friction_values, 'contexts': contexts, 'mean_context': contexts.mean(axis=0), 'global_context': global_context}

def _context_table_mean_tensor(table, *, device, dtype):
    if table is None:
        return None
    return torch.tensor(table['mean_context'], device=device, dtype=dtype)

def _nearest_table_context(table, friction_mu, *, device, dtype):
    if table is None or friction_mu is None:
        return None
    mu = float(friction_mu)
    idx = int(np.argmin(np.abs(table['friction_values'] - mu)))
    return torch.tensor(table['contexts'][idx], device=device, dtype=dtype)

def _context_record(*, step: int, context: torch.Tensor, context0: torch.Tensor, prev_context: torch.Tensor | None, target_context: torch.Tensor | None, inner_lr: float | None, loss: float | None, support_loss: float | None, context_reg: float | None, meta: dict) -> dict:
    current = context.detach().float().cpu()
    initial = context0.detach().float().cpu()
    previous = None if prev_context is None else prev_context.detach().float().cpu()
    target = None if target_context is None else target_context.detach().float().cpu()
    flat = current.reshape(-1)
    row = {**meta, 'inner_step': int(step), 'inner_lr': None if inner_lr is None else float(inner_lr), 'loss': None if loss is None else float(loss), 'support_loss': None if support_loss is None else float(support_loss), 'context_reg': None if context_reg is None else float(context_reg), 'context_mean': float(current.mean()), 'context_norm': float(current.norm()), 'context_total_delta_l2': float((current - initial).norm()), 'context_step_delta_l2': 0.0 if previous is None else float((current - previous).norm()), 'target_context_l2': None if target is None else float((current - target).norm()), 'context_flat': [float(value) for value in flat.tolist()]}
    return row

def _fit_context_pca(contexts: np.ndarray):
    flat = contexts.reshape(contexts.shape[0], -1).astype(np.float64)
    center = flat.mean(axis=0, keepdims=True)
    centered = flat - center
    (_, singular_values, vt) = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] == 1:
        components = np.concatenate([vt, np.zeros_like(vt)], axis=0)
    else:
        components = vt[:2]
    total_var = float(np.sum(singular_values ** 2))
    explained = singular_values[:2] ** 2 / total_var if total_var > 0 else np.zeros(2)
    if explained.shape[0] < 2:
        explained = np.pad(explained, (0, 2 - explained.shape[0]))
    return (center.reshape(-1), components, explained)

def _project_contexts(values: np.ndarray, center: np.ndarray, components: np.ndarray):
    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    return (flat - center[None, :]) @ components.T

def _write_context_pca_plot(path: Path, table, trajectory_rows: list[dict], active_values: list[float] | None=None, metadata_path: Path | None=None) -> None:
    if table is None or not trajectory_rows:
        return
    contexts = table['contexts']
    table_values = np.asarray(table['friction_values'], dtype=np.float64)
    if active_values:
        requested = np.asarray(active_values, dtype=np.float64)
        active_mask = np.any(np.isclose(table_values[:, None], requested[None, :], atol=1e-06, rtol=0.0), axis=1)
        if int(active_mask.sum()) < 2:
            raise ValueError(f'Need at least two active C-table entries for PCA, found {int(active_mask.sum())}.')
        contexts = contexts[active_mask]
        table_values = table_values[active_mask]
    (center, components, explained) = _fit_context_pca(contexts)
    table_xy = _project_contexts(contexts, center, components)
    pc1_correlation = np.corrcoef(table_values, table_xy[:, 0])[0, 1]
    if np.isfinite(pc1_correlation) and pc1_correlation < 0:
        components[0] *= -1
        table_xy[:, 0] *= -1
    if metadata_path is not None and metadata_path.is_file():
        physical_mu_by_group: dict[float, float] = {}
        consistent = True
        with metadata_path.open(encoding='utf-8') as metadata_file:
            for line in metadata_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get('context_group_id') is None or record.get('physical_friction_mu') is None:
                    continue
                group = float(record['context_group_id'])
                physical_mu = float(record['physical_friction_mu'])
                previous = physical_mu_by_group.get(group)
                if previous is not None and (not np.isclose(previous, physical_mu, atol=1e-06, rtol=0.0)):
                    consistent = False
                    break
                physical_mu_by_group[group] = physical_mu
        if consistent and all((float(value) in physical_mu_by_group for value in table_values)):
            table_values = np.asarray([physical_mu_by_group[float(value)] for value in table_values], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_sample: dict[int, list[dict]] = defaultdict(list)
    for row in trajectory_rows:
        by_sample[int(row['sample_index'])].append(row)
    projected_by_sample = {}
    for (sample_index, rows) in sorted(by_sample.items()):
        rows = sorted(rows, key=lambda item: int(item['inner_step']))
        final_value = np.asarray(rows[-1]['context_flat'], dtype=np.float32)[None, None, :]
        final_xy = _project_contexts(final_value, center, components)[0]
        projected_by_sample[sample_index] = (rows, final_xy)
    endpoint_xy = np.stack([item[1] for item in projected_by_sample.values()])
    all_xy = np.concatenate([table_xy, endpoint_xy], axis=0)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-06)
    (width, height) = (1240, 820)
    (margin_left, margin_right, margin_top, margin_bottom) = (105, 80, 120, 145)
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def sx(x):
        return margin_left + float((x - min_xy[0]) / span[0]) * plot_w

    def sy(y):
        return margin_top + (1.0 - float((y - min_xy[1]) / span[1])) * plot_h

    def esc(text):
        return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    palette = ['#2463eb', '#226bd7', '#2173c4', '#1f7ab0', '#1d829d', '#1a8e7e', '#179e57', '#25a346', '#4fa23a', '#7aa12e', '#a4a022', '#ce9f16', '#f59c0b', '#f08511', '#e7591b', '#dc2626']
    value_label = 'friction' if all(rows[-1].get('friction_mu') is not None for rows, _ in projected_by_sample.values()) else 'environment'

    def endpoint_value(row):
        value = row.get('friction_mu')
        if value is None:
            value = row.get('environment_id')
        if value is None:
            value = row['group_key'][0]
        return float(value)

    endpoint_values = [endpoint_value(rows[-1]) for (rows, _) in projected_by_sample.values()]
    color_min = min(float(table_values.min()), min(endpoint_values))
    color_max = max(float(table_values.max()), max(endpoint_values))

    def color_for(value):
        ratio = (float(value) - color_min) / max(color_max - color_min, 1e-12)
        index = round(min(1.0, max(0.0, ratio)) * (len(palette) - 1))
        return palette[index]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<defs><linearGradient id="friction" x1="0" y1="0" x2="1" y2="0">' + ''.join((f'<stop offset="{100 * idx / (len(palette) - 1):.1f}%" stop-color="{color}"/>' for (idx, color) in enumerate(palette))) + '</linearGradient></defs>', '<rect width="100%" height="100%" fill="#fbfaf6"/>', f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" rx="7" fill="#fffdf8" stroke="#d8d2c7"/>', '<style>text{font-family:DejaVu Sans,sans-serif}.axis{stroke:#4b5359;stroke-width:1}.grid{stroke:#ded9ce;stroke-width:1;opacity:.72}.label{font-size:15px;fill:#25313a}.small{font-size:11px;fill:#746f66}.title{font-size:24px;font-weight:650;fill:#17212b}.subtitle{font-size:14px;fill:#6c675f}</style>', f'<text class="title" x="{width / 2:.1f}" y="39" text-anchor="middle">Training-Time and Inference-Time Z (PCA)</text>', f'<text class="subtitle" x="{width / 2:.1f}" y="72" text-anchor="middle">PCA fitted on {len(table_values)} active training-time Z entries | {len(projected_by_sample)} final inference-time Z values | update paths omitted</text>']
    for i in range(6):
        tx = min_xy[0] + span[0] * i / 5
        ty = min_xy[1] + span[1] * i / 5
        x = sx(tx)
        y = sy(ty)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}"/>')
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small" x="{x:.2f}" y="{height - margin_bottom + 22}" text-anchor="middle">{tx:.3f}</text>')
        parts.append(f'<text class="small" x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end">{ty:.3f}</text>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>')
    parts.append(f'<text class="label" x="{margin_left + plot_w / 2:.1f}" y="{height - 77}" text-anchor="middle">PC1 ({100.0 * float(explained[0]):.1f}% active-table variance)</text>')
    parts.append(f'<text class="label" transform="translate(26 {margin_top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">PC2 ({100.0 * float(explained[1]):.1f}% active-table variance)</text>')
    training_order = np.argsort(table_values)
    training_points = ' '.join((f'{sx(table_xy[index, 0]):.2f},{sy(table_xy[index, 1]):.2f}' for index in training_order))
    parts.append(f'<polyline points="{training_points}" fill="none" stroke="#8b857b" stroke-width="1.5" stroke-dasharray="5 5" opacity=".58"/>')
    for (xy, mu) in zip(table_xy, table_values):
        (x, y) = (sx(xy[0]), sy(xy[1]))
        color = color_for(mu)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}" fill-opacity=".88" stroke="#fffdf8" stroke-width="1.4"><title>training time Z, {value_label}={float(mu):.6g}</title></circle>')
    for (sample_index, (rows, xy)) in sorted(projected_by_sample.items()):
        mu = endpoint_value(rows[-1])
        color = color_for(mu)
        (x, y) = (sx(xy[0]), sy(xy[1]))
        points = f'{x:.2f},{y - 9:.2f} {x - 8.2:.2f},{y + 7.2:.2f} {x + 8.2:.2f},{y + 7.2:.2f}'
        parts.append(f'''<polygon points="{points}" fill="{color}" stroke="#111827" stroke-width="1.8" stroke-linejoin="round"><title>inference time final Z, sample={sample_index}, {value_label}={mu:.6g}, step={int(rows[-1]['inner_step'])}</title></polygon>''')
    parts.append('<circle cx="120" cy="778" r="6" fill="#4fa23a" stroke="#fffdf8" stroke-width="1.2"/>')
    parts.append('<text class="small" x="136" y="783">training time active Z-table</text>')
    parts.append('<polygon points="385,769 377,785 393,785" fill="#e2441f" stroke="#111827" stroke-width="1.8" stroke-linejoin="round"/>')
    parts.append('<text class="small" x="403" y="783">inference time final Z</text>')
    parts.append('<rect x="910" y="772" width="210" height="14" rx="2" fill="url(#friction)" stroke="#8b857b" stroke-width=".6"/>')
    parts.append(f'<text class="small" x="900" y="783" text-anchor="end">low {value_label}</text>')
    parts.append(f'<text class="small" x="1130" y="783">high {value_label}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')

def _build_support_dataset(args, runtime_config):
    with open(args.action_stat_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    stat = {args.action_type: stats[args.action_type]} if args.action_type in stats else stats
    special_operator_map = {}
    if 'action' in runtime_config['data_file_keys']:
        special_operator_map['action'] = LoadCobotAction(base_path=args.dataset_base_path, action_type=args.action_type, stat=stat, num_frames=args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder, pad_short=args.pad_short_chunks, output_dim=args.action_dim, frame_stride=int(getattr(args, 'frame_stride', 1)))
    return _configure_dataset(RoboTwinUnifiedDataset(base_path=args.dataset_base_path, metadata_path=args.support_metadata_path, repeat=1, data_file_keys=runtime_config['data_file_keys'], main_data_operator=create_video_operator(base_path=args.dataset_base_path, max_pixels=args.max_pixels, height=args.height, width=args.width, height_division_factor=16, width_division_factor=16, num_frames=args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder, resize_mode=args.resize_mode, pad_short=args.pad_short_chunks, frame_stride=int(getattr(args, 'frame_stride', 1))), special_operator_map=special_operator_map), args)

def _group_rows(rows: list[dict], group_keys: str) -> dict[tuple, list[int]]:
    keys = tuple((part.strip() for part in group_keys.split(',') if part.strip()))
    groups: dict[tuple, list[int]] = defaultdict(list)
    for (index, row) in enumerate(rows):
        groups[tuple((row.get(key) for key in keys))].append(index)
    return groups

def _support_sort_key(row: dict):
    push_start = int(row.get('push_start', 60))
    start = int(row.get('start_frame', 0))
    is_push_focus = row.get('chunk_type') == 'push_focus'
    return (0 if is_push_focus else 1, abs(start - max(0, push_start - 10)), int(row.get('episode_index', 0)), str(row.get('sample_id', '')))

def _select_support_indices(*, support_rows: list[dict], support_groups: dict[tuple, list[int]], group_key: tuple, support_count: int, seed: int) -> list[int]:
    candidates = list(support_groups[group_key])
    stable_group_id = int(hashlib.sha1(str(group_key).encode('utf-8')).hexdigest()[:8], 16)
    rng = random.Random(seed + stable_group_id)
    candidates.sort(key=lambda index: _support_sort_key(support_rows[index]))
    if support_count <= 0:
        return candidates
    selected = []
    used_episodes = set()
    for index in candidates:
        episode = support_rows[index].get('episode_index')
        if episode in used_episodes:
            continue
        selected.append(index)
        used_episodes.add(episode)
        if len(selected) >= support_count:
            return selected
    remaining = [index for index in candidates if index not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, support_count - len(selected))])
    return selected

def _select_support_episode_batch(*, support_rows: list[dict], support_groups: dict[tuple, list[int]], group_key: tuple, episode_count: int, excluded_episode: int, seed: int) -> tuple[list[int], list[int]]:
    if group_key not in support_groups:
        raise KeyError(f'No support rows for group={group_key}.')
    by_episode: dict[int, list[int]] = defaultdict(list)
    for index in support_groups[group_key]:
        episode_index = int(support_rows[index]['episode_index'])
        if episode_index == int(excluded_episode):
            continue
        by_episode[episode_index].append(index)
    episode_indices = sorted(by_episode)
    stable_group_id = int(hashlib.sha1(str(group_key).encode('utf-8')).hexdigest()[:8], 16)
    rng = random.Random(int(seed) + stable_group_id)
    rng.shuffle(episode_indices)
    selected_episodes = episode_indices[:int(episode_count)]
    if len(selected_episodes) < int(episode_count):
        raise ValueError(f'Requested {episode_count} support episodes for group={group_key} after excluding episode={excluded_episode}, but only found {len(selected_episodes)}.')
    selected_indices = []
    for episode_index in selected_episodes:
        selected_indices.extend(sorted(by_episode[episode_index], key=lambda index: (int(support_rows[index].get('start_frame', 0)), int(support_rows[index].get('window_index', 0)))))
    return (selected_indices, selected_episodes)

def _select_lightswitch_balanced_support(*, support_rows: list[dict], support_groups: dict[tuple, list[int]], group_key: tuple, per_condition: int, excluded_episode: int, dataset_base_path: str, seed: int) -> tuple[list[int], dict[str, int]]:
    if group_key not in support_groups:
        raise KeyError(f'No support rows for group={group_key}.')
    candidates = list(support_groups[group_key])
    stable_group_id = int(hashlib.sha1(str(group_key).encode('utf-8')).hexdigest()[:8], 16)
    rng = random.Random(int(seed) + stable_group_id)
    rng.shuffle(candidates)
    lamp_cache: dict[Path, list] = {}
    candidate_records = []
    for index in candidates:
        row = support_rows[index]
        episode_index = int(row['episode_index'])
        if episode_index == int(excluded_episode):
            continue
        colors = list(row.get('covered_button_colors') or [])
        if len(colors) != 1 or colors[0] not in ('red', 'blue'):
            continue
        parquet_path = Path(dataset_base_path) / str(row['action'])
        if parquet_path not in lamp_cache:
            lamp_cache[parquet_path] = pq.read_table(parquet_path, columns=['observation.button_lamp_state']).column(0).to_pylist()
        state = lamp_cache[parquet_path][int(row.get('start_frame', 0))]
        initial_lamp_on = bool(float(state[-1]) >= 0.5)
        candidate_records.append((index, episode_index, colors[0], initial_lamp_on))
    bucket_order = (('blue', False), ('blue', True), ('red', False), ('red', True))
    selected = []
    used_indices = set()
    used_episodes = set()
    counts = {bucket: 0 for bucket in bucket_order}

    def select_matching(predicate, limit: int, *, allow_reused_episodes: bool=False) -> None:
        for (index, episode_index, color, initial_lamp_on) in candidate_records:
            if limit <= 0:
                return
            if index in used_indices:
                continue
            if not allow_reused_episodes and episode_index in used_episodes:
                continue
            if not predicate(color, initial_lamp_on):
                continue
            selected.append(index)
            used_indices.add(index)
            used_episodes.add(episode_index)
            counts[color, initial_lamp_on] += 1
            limit -= 1
    for bucket in bucket_order:
        select_matching(lambda color, initial_lamp_on, bucket=bucket: (color, initial_lamp_on) == bucket, int(per_condition))
    target_per_button = int(per_condition) * 2
    for color in ('blue', 'red'):
        current = sum((count for ((bucket_color, _), count) in counts.items() if bucket_color == color))
        select_matching(lambda candidate_color, _lamp, color=color: candidate_color == color, target_per_button - current)
        current = sum((count for ((bucket_color, _), count) in counts.items() if bucket_color == color))
        select_matching(lambda candidate_color, _lamp, color=color: candidate_color == color, target_per_button - current, allow_reused_episodes=True)
    select_matching(lambda _color, _lamp: True, int(per_condition) * 4 - len(selected))
    select_matching(lambda _color, _lamp: True, int(per_condition) * 4 - len(selected), allow_reused_episodes=True)
    expected = int(per_condition) * 4
    if len(selected) != expected:
        raise ValueError(f'Could not build an {expected}-chunk balanced LightSwitch support batch for group={group_key}; selected={len(selected)} counts={counts}.')
    summary = {f"{color}_{('on' if initial_lamp_on else 'off')}": counts[color, initial_lamp_on] for (color, initial_lamp_on) in bucket_order}
    return (selected, summary)

def _freeze_pipe(pipe) -> None:
    for name in ('dit', 'vae', 'action_encoder', 'physical_context_encoder', 'physical_adapter_bank'):
        module = getattr(pipe, name, None)
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    pipe.eval()

def _video_geometry(video: torch.Tensor) -> tuple[int, int, int, int]:
    if video.ndim != 5:
        raise ValueError(f'Expected video tensor (V,C,T,H,W), got {tuple(video.shape)}')
    num_views = int(video.shape[0])
    return (int(video.shape[-2]) * num_views, int(video.shape[-1]), int(video.shape[2]), num_views)

def _known_context_keys(args) -> list[str]:
    return [item.strip() for item in str(getattr(args, 'stage2_known_context_keys', '') or '').split(',') if item.strip()]

def _compose_known_physical_context(data: dict, physical_context: torch.Tensor, args) -> torch.Tensor:
    keys = _known_context_keys(args)
    if not keys:
        return physical_context
    if len(keys) > int(physical_context.shape[-1]):
        raise ValueError(f'stage2_known_context_keys has {len(keys)} keys, but physical_context dim is {int(physical_context.shape[-1])}.')
    values = []
    for key in keys:
        if key not in data:
            raise KeyError(f'Missing known physical context key {key!r} in stage2 metadata row.')
        value = data[key]
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=physical_context.device, dtype=physical_context.dtype).flatten()[0]
        else:
            tensor = torch.tensor(float(value), device=physical_context.device, dtype=physical_context.dtype)
        values.append(tensor)
    known = torch.stack(values)
    start_dim = int(physical_context.shape[-1]) - len(keys)
    context = physical_context.clone()
    if context.ndim == 1:
        context[start_dim:] = known
    else:
        view_shape = [1] * (context.ndim - 1) + [len(keys)]
        expand_shape = list(context.shape[:-1]) + [len(keys)]
        context[..., start_dim:] = known.view(*view_shape).expand(*expand_shape)
    return context

def _prepare_loss_inputs(pipe, data: dict, physical_context: torch.Tensor, args):
    (height, width, num_frames, num_views) = _video_geometry(data['video'])
    inputs_shared = {'input_video': data['video'], 'action': data.get('action'), 'height': height, 'width': width, 'num_frames': num_frames, 'num_views': num_views, 'num_history_frames': int(args.num_history_frames), 'cfg_scale': 1, 'tiled': False, 'rand_device': pipe.device, 'physical_context': _compose_known_physical_context(data, physical_context, args), 'use_gradient_checkpointing': bool(args.use_gradient_checkpointing), 'use_gradient_checkpointing_offload': bool(args.use_gradient_checkpointing_offload), 'max_timestep_boundary': float(args.max_timestep_boundary), 'min_timestep_boundary': float(args.min_timestep_boundary)}
    inputs_posi = {'prompt': data.get('prompt'), 'prompt_emb': data.get('prompt_emb')}
    inputs_nega = {'negative_prompt': data.get('negative_prompt'), 'prompt_emb': data.get('negative_prompt_emb')}
    inputs = (inputs_shared, inputs_posi, inputs_nega)
    for unit in pipe.units:
        inputs = pipe.unit_runner(unit, pipe, *inputs)
    return inputs

def _shared_inputs(inputs):
    return inputs[0] if isinstance(inputs, tuple) else inputs

def _sample_timestep(pipe, inputs: dict, args) -> torch.Tensor:
    inputs = _shared_inputs(inputs)
    max_idx = int(float(args.max_timestep_boundary) * len(pipe.scheduler.timesteps))
    min_idx = int(float(args.min_timestep_boundary) * len(pipe.scheduler.timesteps))
    max_idx = max(min_idx + 1, max_idx)
    fixed_index = getattr(args, 'stage2_fixed_timestep_index', None)
    if fixed_index is None:
        timestep_id = torch.randint(min_idx, max_idx, (1,))
    else:
        fixed_index = int(fixed_index)
        if not min_idx <= fixed_index < max_idx:
            raise ValueError(f'stage2_fixed_timestep_index={fixed_index} is outside the active scheduler index range [{min_idx}, {max_idx}).')
        timestep_id = torch.tensor([fixed_index], dtype=torch.long)
    return pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

def _flow_mse(prediction: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    return torch.nn.functional.mse_loss(prediction.float(), target.float())

def _flow_match_loss(pipe, inputs, args) -> torch.Tensor:
    inputs = dict(_shared_inputs(inputs))
    timestep = _sample_timestep(pipe, inputs, args)
    noise = torch.randn_like(inputs['input_latents'])
    inputs['latents'] = pipe.scheduler.add_noise(inputs['input_latents'], noise, timestep)
    target = pipe.scheduler.training_target(inputs['input_latents'], noise, timestep)
    if 'first_frame_latents' in inputs:
        inputs['latents'][:, :, 0:1] = inputs['first_frame_latents']
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    if 'first_frame_latents' in inputs:
        pred = pred[:, :, 1:]
        target = target[:, :, 1:]
    loss = _flow_mse(pred, target, args)
    return loss * pipe.scheduler.training_weight(timestep)

def _context_reg(pipe, context: torch.Tensor, target_context: torch.Tensor | None=None) -> torch.Tensor:
    if target_context is None:
        target = pipe.physical_context_encoder.default_context.detach().to(dtype=context.dtype, device=context.device)
    else:
        target = target_context.detach().to(dtype=context.dtype, device=context.device)
    return torch.mean((context.float() - target.float()) ** 2)

def _parse_inner_lr_schedule(args) -> list[float]:
    raw = str(getattr(args, 'stage2_inner_lr_schedule', '') or '').strip()
    if not raw:
        return [float(args.stage2_inner_lr)] * int(args.stage2_inner_steps)
    schedule = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            (lr_text, count_text) = item.split(':', 1)
        elif 'x' in item:
            (lr_text, count_text) = item.split('x', 1)
        elif '*' in item:
            (lr_text, count_text) = item.split('*', 1)
        else:
            (lr_text, count_text) = (item, '1')
        lr = float(lr_text.strip())
        count = int(count_text.strip())
        if count <= 0:
            raise ValueError(f'Invalid non-positive inner LR schedule count in {item!r}.')
        schedule.extend([lr] * count)
    if not schedule:
        raise ValueError(f'Empty stage2_inner_lr_schedule={raw!r}.')
    return schedule

def _clamp_context(context: torch.Tensor, args) -> torch.Tensor:
    clamp_min = getattr(args, 'stage2_context_clamp_min', None)
    clamp_max = getattr(args, 'stage2_context_clamp_max', None)
    if clamp_min is None and clamp_max is None:
        return context
    clamp_min = None if clamp_min is None else float(clamp_min)
    clamp_max = None if clamp_max is None else float(clamp_max)
    return torch.clamp(context, min=clamp_min, max=clamp_max)

def _adapt_ttt_state(pipe, support_items: list[dict], args, *, initial_context: torch.Tensor | None=None, target_context: torch.Tensor | None=None, trajectory_meta: dict | None=None) -> tuple[torch.Tensor, list[float], dict[str, float], list[dict]]:
    if pipe.physical_context_encoder is None:
        raise ValueError("Stage2 TTT inference requires physical_context_mode != 'none'.")
    scope = 'context'
    inner_lrs = _parse_inner_lr_schedule(args)
    pipe.scheduler.set_timesteps(1000, training=True)
    context_dtype = torch.float32 if bool(getattr(args, 'ttt_context_fp32', True)) else pipe.torch_dtype
    if initial_context is None:
        context0 = pipe.physical_context_encoder.default_context.detach().clone().to(dtype=context_dtype, device=pipe.device)
    else:
        context0 = initial_context.detach().clone().to(dtype=context_dtype, device=pipe.device)
    context = context0.clone()
    context = context.requires_grad_(True)
    target_context = None if target_context is None else target_context.detach().clone().to(dtype=context_dtype, device=pipe.device)
    meta = {} if trajectory_meta is None else dict(trajectory_meta)
    losses = []
    trajectory_rows = [_context_record(step=0, context=context, context0=context0, prev_context=None, target_context=target_context, inner_lr=None, loss=None, support_loss=None, context_reg=None, meta=meta)]
    support_loss_reduction = str(getattr(args, 'ttt_support_loss_reduction', 'mean')).lower()
    if support_loss_reduction not in {'mean', 'sum'}:
        raise ValueError(f'Unsupported ttt_support_loss_reduction={support_loss_reduction!r}.')
    for (inner_idx, inner_lr) in enumerate(inner_lrs):
        prev_context = context.detach()
        grad_targets: list[torch.Tensor] = []
        grad_targets.append(context)
        if not support_items:
            raise ValueError('Stage2 TTT received an empty support batch.')
        context_reg_value = 0.0
        if bool(getattr(args, 'ttt_support_gradient_accumulation', False)):
            grads: list[torch.Tensor | None] = [None] * len(grad_targets)
            support_loss_value = 0.0
            support_scale = 1.0 / float(len(support_items)) if support_loss_reduction == 'mean' else 1.0
            for item in support_items:
                inputs = _prepare_loss_inputs(pipe, item, context, args)
                item_loss = _flow_match_loss(pipe, inputs, args)
                support_loss_value += float(item_loss.detach().float().cpu()) * support_scale
                item_grads = torch.autograd.grad(item_loss, grad_targets, create_graph=False, allow_unused=True)
                for (grad_index, item_grad) in enumerate(item_grads):
                    if item_grad is None:
                        continue
                    scaled_grad = item_grad * support_scale
                    if grads[grad_index] is None:
                        grads[grad_index] = scaled_grad
                    else:
                        grads[grad_index] = grads[grad_index] + scaled_grad
            regularization_loss = None
            regularization_value = 0.0
            if float(args.stage2_context_reg_weight) > 0:
                context_reg = _context_reg(pipe, context, context0)
                context_reg_value = float(context_reg.detach().float().cpu())
                weighted_context_reg = float(args.stage2_context_reg_weight) * context_reg
                regularization_loss = weighted_context_reg
                regularization_value += float(weighted_context_reg.detach().float().cpu())
            if regularization_loss is not None:
                regularization_grads = torch.autograd.grad(regularization_loss, grad_targets, create_graph=False, allow_unused=True)
                for (grad_index, regularization_grad) in enumerate(regularization_grads):
                    if regularization_grad is None:
                        continue
                    if grads[grad_index] is None:
                        grads[grad_index] = regularization_grad
                    else:
                        grads[grad_index] = grads[grad_index] + regularization_grad
            loss_value = support_loss_value + regularization_value
        else:
            support_losses = []
            for item in support_items:
                inputs = _prepare_loss_inputs(pipe, item, context, args)
                support_losses.append(_flow_match_loss(pipe, inputs, args))
            support_loss = getattr(torch.stack(support_losses), support_loss_reduction)()
            support_loss_value = float(support_loss.detach().float().cpu())
            loss = support_loss
            if float(args.stage2_context_reg_weight) > 0:
                context_reg = _context_reg(pipe, context, context0)
                context_reg_value = float(context_reg.detach().float().cpu())
                loss = loss + float(args.stage2_context_reg_weight) * context_reg
            grads = list(torch.autograd.grad(loss, grad_targets, create_graph=False, allow_unused=True))
            loss_value = float(loss.detach().float().cpu())
        context_grad = grads[0]
        if context_grad is None:
            context_grad = torch.zeros_like(context)
        if float(args.stage2_inner_grad_clip) > 0:
            grad_norm = context_grad.float().norm().clamp_min(1e-06)
            scale = min(1.0, float(args.stage2_inner_grad_clip) / float(grad_norm.detach().cpu()))
            context_grad = context_grad * scale
        context = _clamp_context(context - float(inner_lr) * context_grad, args).detach().requires_grad_(True)
        losses.append(loss_value)
        trajectory_rows.append(_context_record(step=inner_idx + 1, context=context, context0=context0, prev_context=prev_context, target_context=target_context, inner_lr=float(inner_lr), loss=losses[-1], support_loss=support_loss_value, context_reg=context_reg_value, meta=meta))
        if not bool(getattr(args, 'ttt_quiet_inner', False)):
            print(f"[inner] step={inner_idx + 1} scope={scope} loss={losses[-1]:.6f} support={support_loss_value:.6f} support_count={len(support_items)} context_reg={context_reg_value:.6f} inner_lr={float(inner_lr):.6f} c_mean={float(context.detach().float().mean().cpu()):.6f} step_delta_l2={trajectory_rows[-1]['context_step_delta_l2']:.6f} target_l2={trajectory_rows[-1]['target_context_l2']}", flush=True)
    metrics = {'context_initial_mean': float(context0.detach().float().mean().cpu()), 'context_adapted_mean': float(context.detach().float().mean().cpu()), 'context_delta_mean': float((context.detach().float() - context0.detach().float()).mean().cpu()), 'inner_steps': float(len(inner_lrs))}
    return (context.detach(), losses, metrics, trajectory_rows)

def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame).convert('RGB')
    draw = ImageDraw.Draw(image)
    width = max(88, 9 * len(label) + 12)
    draw.rectangle((0, 0, width, 22), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)

def _write_three_way_comparison(*, row: dict, sample_index: int, dataset_base_path: Path, stage1_pred_path: Path, stage2_pred_path: Path, output_dir: Path, width: int, height: int, fps: int, quality: int) -> Path:
    num_views = len(row['video'])
    target_width = width * num_views
    total_frames = int(row.get('length', row['end_frame'] - row['start_frame'] + 1))
    gt_frames = _read_gt_video(row, dataset_base_path, width, height, total_frames)
    stage1_frames = _read_pred_video(stage1_pred_path, target_width, height)
    stage2_frames = _read_pred_video(stage2_pred_path, target_width, height)
    total = min(len(gt_frames), len(stage1_frames), len(stage2_frames))
    if total <= 0:
        raise RuntimeError(f'No frames to compare for sample_index={sample_index}.')
    output_dir.mkdir(parents=True, exist_ok=True)
    name = _default_pred_name(sample_index, row).replace('.mp4', '_gt_stage1_stage2ttt.mp4')
    output_path = output_dir / name
    with imageio.get_writer(str(output_path), fps=fps, codec='libx264', quality=quality) as writer:
        for frame_id in range(total):
            gt = _draw_label(_resize_rgb(gt_frames[frame_id], target_width, height), 'GT')
            stage1 = _draw_label(_resize_rgb(stage1_frames[frame_id], target_width, height), 'STAGE1')
            stage2 = _draw_label(_resize_rgb(stage2_frames[frame_id], target_width, height), 'STAGE2+TTT')
            writer.append_data(np.concatenate([gt, stage1, stage2], axis=0))
    return output_path

def _write_two_way_comparison(*, row: dict, sample_index: int, dataset_base_path: Path, pred_path: Path, output_dir: Path, width: int, height: int, fps: int, quality: int, pred_label: str) -> Path:
    num_views = len(row['video'])
    target_width = width * num_views
    total_frames = int(row.get('length', row['end_frame'] - row['start_frame'] + 1))
    gt_frames = _read_gt_video(row, dataset_base_path, width, height, total_frames)
    pred_frames = _read_pred_video(pred_path, target_width, height)
    total = min(len(gt_frames), len(pred_frames))
    if total <= 0:
        raise RuntimeError(f'No frames to compare for sample_index={sample_index}.')
    output_dir.mkdir(parents=True, exist_ok=True)
    name = _default_pred_name(sample_index, row).replace('.mp4', f'_gt_{pred_label.lower()}.mp4')
    output_path = output_dir / name
    with imageio.get_writer(str(output_path), fps=fps, codec='libx264', quality=quality) as writer:
        for frame_id in range(total):
            gt = _draw_label(_resize_rgb(gt_frames[frame_id], target_width, height), 'GT')
            pred = _draw_label(_resize_rgb(pred_frames[frame_id], target_width, height), pred_label)
            writer.append_data(np.concatenate([gt, pred], axis=0))
    return output_path

def parse_args():
    parser = argparse.ArgumentParser('Latent adaptation and prediction')
    parser = add_adaptation_config(add_general_config(parser))
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--stage2_ckpt_path', type=str, required=True)
    parser.add_argument('--support_metadata_path', type=str, default='data/push_box_bwm_calibrated_v2_100pairs/train.jsonl')
    parser.add_argument('--support_count', type=int, default=2)
    parser.add_argument('--sample_indices', type=str, default=None)
    parser.add_argument('--baseline_pred_dir', type=str, default=None)
    parser.add_argument('--comparison_output_path', type=str, default=None)
    parser.add_argument('--render_support', action='store_true', default=False)
    parser.add_argument('--support_output_path', type=str, default=None)
    parser.add_argument('--support_comparison_output_path', type=str, default=None)
    parser.add_argument('--skip_existing', action='store_true', default=False)
    parser.add_argument('--dry_run', action='store_true', default=False)
    parser.add_argument('--ttt_support_same_as_query', action='store_true', default=False, help='Use each query chunk itself as the support chunk and adapt a separate C per query sample.')
    parser.add_argument('--ttt_support_same_episode', action='store_true', default=False, help='Adapt one C per query episode using every metadata chunk from that episode as the joint support batch. Queries from the same episode share the adapted C.')
    parser.add_argument('--ttt_support_episodes_per_group', type=int, default=0, help="Select this many episodes from the query's environment group, excluding the query episode, and use every metadata chunk from the selected episodes as one support batch.")
    parser.add_argument('--ttt_support_lightswitch_per_condition', type=int, default=0, help='Build a LightSwitch support batch with this many chunks for each button-color and initial-lamp-state condition. Missing lamp-state buckets are redistributed within the same button color without crossing environment groups.')
    parser.add_argument('--ttt_support_gradient_accumulation', action='store_true', default=False, help='Compute the support-batch mean gradient one chunk at a time, then apply one update. This is mathematically a mean support loss without retaining every DiT graph at once.')
    parser.add_argument('--ttt_support_loss_reduction', choices=('mean', 'sum'), default='mean', help='Aggregate support flow-matching losses by mean (default) or sum. With gradient accumulation, sum removes the per-support 1/K scale.')
    parser.add_argument('--ttt_initial_context_table_path', type=str, default=None, help='If set, initialize test-time C from the mean of this grouped context_table.json.')
    parser.add_argument('--ttt_initial_context_table_index', type=int, default=None, help='Optional zero-based record index in --ttt_initial_context_table_path. When set, initialize every test-time rollout from that learned context instead of the table mean.')
    parser.add_argument('--ttt_initial_context_global', action='store_true', default=False, help='Initialize every test-time rollout from the global_context stored in --ttt_initial_context_table_path instead of an endpoint or endpoint mean.')
    parser.add_argument('--ttt_context_table_path', type=str, default=None, help='Grouped context_table.json used as target dictionary and PCA basis for trajectory analysis.')
    parser.add_argument('--ttt_context_active_values', type=str, default=None, help='Comma-separated active table values used exclusively for PCA fitting and display.')
    parser.add_argument('--ttt_context_trajectory_path', type=str, default=None)
    parser.add_argument('--ttt_context_pca_output_path', type=str, default=None)
    parser.add_argument('--ttt_context_fp32', action=argparse.BooleanOptionalAction, default=True, help='Keep optimized Z and its updates in FP32; cast only the encoder input to the model dtype.')
    parser.add_argument('--ttt_initial_context_random_uniform', action='store_true', default=False, help='Initialize a deterministic independent Uniform(0,1) context for every sample.')
    parser.add_argument('--ttt_initial_context_random_shared', action='store_true', default=False, help='Initialize every sample from one shared deterministic Uniform(0,1) context.')
    parser.add_argument('--ttt_initial_context_random_seed', type=int, default=None, help='Optional seed dedicated to random context initialization.')
    parser.add_argument('--stage2_fixed_timestep_index', type=int, default=None, help='Use one fixed training scheduler index for every support chunk and inner step.')
    parser.add_argument('--ttt_quiet_inner', action='store_true', default=False)
    parser.add_argument('--adapt_only', action='store_true', default=False)
    parser.add_argument('--resume_context_trajectory', action='store_true', default=False)
    parser.add_argument('--ttt_transfer_targets_by_source', type=str, default=None, help='Optional SOURCE:TARGET,TARGET;SOURCE:TARGET,TARGET mapping. After adapting each listed source, generate all mapped queries with the same adapted state in this process so the model is loaded only once.')
    parser.add_argument('--ttt_transfer_output_path', type=str, default=None)
    parser.add_argument('--ttt_transfer_plan_output_path', type=str, default=None)
    parser.add_argument('--ttt_transfer_skip_existing', action='store_true', default=False)
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, parser, args)
    return args

def main() -> None:
    args = parse_args()
    if args.ttt_initial_context_random_uniform and args.ttt_initial_context_random_shared:
        raise ValueError('Choose only one of --ttt_initial_context_random_uniform and --ttt_initial_context_random_shared.')
    if args.ttt_initial_context_table_index is not None and (not args.ttt_initial_context_table_path):
        raise ValueError('--ttt_initial_context_table_index requires --ttt_initial_context_table_path.')
    if args.ttt_initial_context_global and (not args.ttt_initial_context_table_path):
        raise ValueError('--ttt_initial_context_global requires --ttt_initial_context_table_path.')
    explicit_initializers = sum((args.ttt_initial_context_table_index is not None, bool(args.ttt_initial_context_global), bool(args.ttt_initial_context_random_uniform), bool(args.ttt_initial_context_random_shared)))
    if explicit_initializers > 1:
        raise ValueError('Choose only one explicit test-time context initializer: table row, global context, independent random, or shared random.')
    set_global_seed(int(args.seed))
    query_rows = _read_jsonl(args.dataset_metadata_path)
    support_rows = _read_jsonl(args.support_metadata_path)
    support_mode_count = sum((bool(args.ttt_support_same_as_query), bool(args.ttt_support_same_episode), int(args.ttt_support_episodes_per_group) > 0, int(args.ttt_support_lightswitch_per_condition) > 0))
    if support_mode_count > 1:
        raise ValueError('Choose only one of --ttt_support_same_as_query, --ttt_support_same_episode, --ttt_support_episodes_per_group, or --ttt_support_lightswitch_per_condition.')
    support_groups = _group_rows(support_rows, args.stage2_group_keys)
    support_episodes: dict[int, list[int]] = defaultdict(list)
    for (support_index, support_row) in enumerate(support_rows):
        support_episodes[int(support_row['episode_index'])].append(support_index)
    initial_context_table = _load_grouped_context_table(args.ttt_initial_context_table_path)
    context_table = _load_grouped_context_table(args.ttt_context_table_path or args.ttt_initial_context_table_path)
    sample_indices = _parse_sample_indices(args.sample_indices)
    if sample_indices is None:
        start = int(args.start_index)
        end = len(query_rows) if not args.max_samples else min(len(query_rows), start + int(args.max_samples))
        sample_indices = list(range(start, end))
    if args.max_samples and args.sample_indices is not None:
        sample_indices = sample_indices[:int(args.max_samples)]
    transfer_targets_by_source: dict[int, list[int]] = {}
    if args.ttt_transfer_targets_by_source:
        if not args.ttt_transfer_output_path or not args.ttt_transfer_plan_output_path:
            raise ValueError('--ttt_transfer_targets_by_source requires --ttt_transfer_output_path and --ttt_transfer_plan_output_path.')
        if args.adapt_only:
            raise ValueError('In-process transfer is incompatible with --adapt_only.')
        for entry in str(args.ttt_transfer_targets_by_source).split(';'):
            entry = entry.strip()
            if not entry:
                continue
            if ':' not in entry:
                raise ValueError(f'Invalid transfer mapping entry: {entry!r}')
            (source_raw, targets_raw) = entry.split(':', 1)
            source_index = int(source_raw.strip())
            targets = _parse_sample_indices(targets_raw)
            if not targets:
                raise ValueError(f'No transfer targets supplied for source={source_index}.')
            if source_index in transfer_targets_by_source:
                raise ValueError(f'Duplicate transfer source={source_index}.')
            transfer_targets_by_source[source_index] = [int(index) for index in targets]
        missing_sources = sorted(set(transfer_targets_by_source) - set(map(int, sample_indices)))
        if missing_sources:
            raise ValueError(f'Every in-process transfer source must be included in --sample_indices; missing={missing_sources}.')
    support_plan = {}
    sample_cache_keys = {}
    for sample_index in sample_indices:
        row = query_rows[int(sample_index)]
        group_key = tuple((row.get(name.strip()) for name in args.stage2_group_keys.split(',') if name.strip()))
        if args.ttt_support_same_as_query:
            cache_key = ('sample', int(sample_index))
            support_plan[cache_key] = [int(sample_index)]
        elif args.ttt_support_same_episode:
            episode_index = int(row['episode_index'])
            cache_key = ('episode', episode_index)
            if cache_key not in support_plan:
                episode_indices = support_episodes.get(episode_index, [])
                if not episode_indices:
                    raise KeyError(f'No support chunks for episode={episode_index}.')
                support_plan[cache_key] = sorted(episode_indices, key=lambda index: (int(support_rows[index].get('start_frame', 0)), int(support_rows[index].get('window_index', 0))))
        elif int(args.ttt_support_episodes_per_group) > 0:
            episode_index = int(row['episode_index'])
            cache_key = ('group_episode_batch',) + tuple(group_key) + ('exclude', episode_index)
            if cache_key not in support_plan:
                (selected_indices, selected_episodes) = _select_support_episode_batch(support_rows=support_rows, support_groups=support_groups, group_key=group_key, episode_count=int(args.ttt_support_episodes_per_group), excluded_episode=episode_index, seed=int(args.seed))
                support_plan[cache_key] = selected_indices
                print(f'[support_episodes] query={sample_index} group={group_key} excluded_episode={episode_index} selected_episodes={selected_episodes} support_count={len(selected_indices)}', flush=True)
        elif int(args.ttt_support_lightswitch_per_condition) > 0:
            episode_index = int(row['episode_index'])
            cache_key = ('lightswitch_balanced',) + tuple(group_key) + ('exclude', episode_index)
            if cache_key not in support_plan:
                (selected_indices, condition_counts) = _select_lightswitch_balanced_support(support_rows=support_rows, support_groups=support_groups, group_key=group_key, per_condition=int(args.ttt_support_lightswitch_per_condition), excluded_episode=episode_index, dataset_base_path=args.dataset_base_path, seed=int(args.seed))
                support_plan[cache_key] = selected_indices
                print(f'[lightswitch_balanced_support] query={sample_index} group={group_key} excluded_episode={episode_index} support={selected_indices} condition_counts={condition_counts}', flush=True)
        elif group_key not in support_groups:
            raise KeyError(f'No support rows for group={group_key}.')
        elif group_key not in support_plan:
            cache_key = group_key
            support_plan[cache_key] = _select_support_indices(support_rows=support_rows, support_groups=support_groups, group_key=group_key, support_count=int(args.support_count), seed=int(args.seed))
        else:
            cache_key = group_key
        sample_cache_keys[int(sample_index)] = cache_key
        print(f'[plan] query={sample_index} group={group_key} cache_key={cache_key} support={support_plan[cache_key]}', flush=True)
    output_path = Path(args.output_path)
    comparison_path = Path(args.comparison_output_path) if args.comparison_output_path else output_path.parent / 'comparison_videos'
    support_output_path = Path(args.support_output_path) if args.support_output_path else output_path.parent / 'show_raw'
    support_comparison_path = Path(args.support_comparison_output_path) if args.support_comparison_output_path else output_path.parent / 'show_comparison_videos'
    plan_rows = []
    for (group_key, indices) in sorted(support_plan.items(), key=lambda item: str(item[0])):
        plan_rows.append({'group_key': list(group_key), 'support_indices': indices, 'support_sample_ids': [support_rows[index].get('sample_id') for index in indices], 'support_episode_indices': [support_rows[index].get('episode_index') for index in indices]})
    _write_jsonl(output_path.parent / 'support_plan.jsonl', plan_rows)
    if args.dry_run:
        return
    runtime_config = prepare_runtime_config(args)
    support_dataset = _build_support_dataset(args, runtime_config)
    query_dataset = None if args.adapt_only else build_infer_dataset(args)
    pipe = build_pipeline(args)
    load_checkpoint_weights(pipe, args.stage2_ckpt_path)
    _freeze_pipe(pipe)
    context_dtype = torch.float32 if bool(args.ttt_context_fp32) else pipe.torch_dtype
    initial_context_table_index = args.ttt_initial_context_table_index
    initial_context_kind = 'table_mean'
    if args.ttt_initial_context_global:
        if initial_context_table is None or initial_context_table['global_context'] is None:
            raise ValueError('--ttt_initial_context_global was requested, but the context table has no saved global_context.')
        initial_context_tensor = torch.tensor(initial_context_table['global_context'], device=pipe.device, dtype=context_dtype)
        initial_context_kind = 'global_context'
    elif initial_context_table_index is None:
        initial_context_tensor = _context_table_mean_tensor(initial_context_table, device=pipe.device, dtype=context_dtype)
    else:
        num_initial_contexts = int(initial_context_table['contexts'].shape[0])
        if not 0 <= int(initial_context_table_index) < num_initial_contexts:
            raise ValueError(f'--ttt_initial_context_table_index={initial_context_table_index} is outside [0, {num_initial_contexts}).')
        initial_context_tensor = torch.tensor(initial_context_table['contexts'][int(initial_context_table_index)], device=pipe.device, dtype=context_dtype)
        initial_context_kind = f'table_index_{int(initial_context_table_index)}'
    if initial_context_tensor is not None:
        print(f"[initial_context] source={initial_context_kind} path={initial_context_table['path']} shape={tuple(initial_context_tensor.shape)} mean={float(initial_context_tensor.float().mean().cpu()):.6f}", flush=True)
    shared_random_context = None
    shared_random_context_seed = None
    if args.ttt_initial_context_random_shared:
        template = initial_context_tensor
        if template is None:
            template = pipe.physical_context_encoder.default_context.detach().to(device=pipe.device, dtype=context_dtype)
        shared_random_context_seed = int(args.ttt_initial_context_random_seed) if args.ttt_initial_context_random_seed is not None else int(args.seed)
        generator = torch.Generator(device=pipe.device)
        generator.manual_seed(shared_random_context_seed)
        shared_random_context = torch.rand(template.shape, generator=generator, device=pipe.device, dtype=context_dtype)
        print(f'[initial_context] source=shared_random seed={shared_random_context_seed} shape={tuple(shared_random_context.shape)} mean={float(shared_random_context.float().mean().cpu()):.6f}', flush=True)
    context_cache: dict[tuple, tuple[torch.Tensor, list[float], dict[str, float], list[dict]]] = {}
    rendered_support = set()
    result_rows = []
    trajectory_rows = []
    resumed_trajectory_rows_by_sample: dict[int, list[dict]] = {}
    trajectory_path = Path(args.ttt_context_trajectory_path) if args.ttt_context_trajectory_path else output_path.parent / 'context_trajectory.jsonl'
    pca_output_path = Path(args.ttt_context_pca_output_path) if args.ttt_context_pca_output_path else output_path.parent / 'context_trajectory_pca.svg'
    completed_context_samples = set()
    if args.resume_context_trajectory and trajectory_path.exists():
        trajectory_rows = _read_jsonl(trajectory_path)
        expected_steps = len(_parse_inner_lr_schedule(args))
        trajectory_complete_samples = {int(row['sample_index']) for row in trajectory_rows if row.get('sample_index') is not None and int(row.get('inner_step', -1)) >= expected_steps}
        completed_context_samples = {sample_index for sample_index in trajectory_complete_samples if (output_path / _default_pred_name(sample_index, query_rows[sample_index])).is_file()}
        resumed_trajectory_rows_by_sample = {sample_index: sorted([row for row in trajectory_rows if row.get('sample_index') is not None and int(row['sample_index']) == sample_index], key=lambda row: int(row['inner_step'])) for sample_index in completed_context_samples}
        trajectory_rows = [row for row in trajectory_rows if row.get('sample_index') is not None and int(row['sample_index']) in completed_context_samples]
        _write_jsonl(trajectory_path, trajectory_rows)
        print(f'[resume_context] path={trajectory_path} rows={len(trajectory_rows)} completed_samples={len(completed_context_samples)}', flush=True)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path.mkdir(parents=True, exist_ok=True)
    if args.render_support:
        support_output_path.mkdir(parents=True, exist_ok=True)
        support_comparison_path.mkdir(parents=True, exist_ok=True)
    transfer_output_path = Path(args.ttt_transfer_output_path) if args.ttt_transfer_output_path else None
    transfer_plan_output_path = Path(args.ttt_transfer_plan_output_path) if args.ttt_transfer_plan_output_path else None
    transfer_plan_rows = []
    if transfer_output_path is not None:
        transfer_output_path.mkdir(parents=True, exist_ok=True)
        transfer_plan_output_path.parent.mkdir(parents=True, exist_ok=True)
    for sample_index in sample_indices:
        sample_index = int(sample_index)
        resumed_sample = sample_index in completed_context_samples
        has_transfer_targets = sample_index in transfer_targets_by_source
        if resumed_sample and (not has_transfer_targets):
            print(f'[resume_context_skip] sample_index={sample_index}', flush=True)
            continue
        row = query_rows[sample_index]
        group_key = tuple((row.get(name.strip()) for name in args.stage2_group_keys.split(',') if name.strip()))
        cache_key = sample_cache_keys[sample_index]
        pred_name = _default_pred_name(sample_index, row)
        pred_path = output_path / pred_name
        if cache_key not in context_cache and resumed_sample:
            resumed_rows = resumed_trajectory_rows_by_sample[sample_index]
            adapted_context = torch.tensor(resumed_rows[-1]['context_flat'], device=pipe.device, dtype=context_dtype).reshape_as(initial_context_tensor)
            context_cache[cache_key] = (adapted_context, [], {}, resumed_rows)
            print(f"[resume_context_restore] sample_index={sample_index} inner_step={resumed_rows[-1]['inner_step']}", flush=True)
        if cache_key not in context_cache:
            support_indices = support_plan[cache_key]
            print(f'[adapt] group={group_key} cache_key={cache_key} support={support_indices}', flush=True)
            support_items = [support_dataset[index] for index in support_indices]
            target_context = _nearest_table_context(context_table, row.get('friction_mu'), device=pipe.device, dtype=context_dtype)
            sample_initial_context = shared_random_context if shared_random_context is not None else initial_context_tensor
            initial_context_seed = shared_random_context_seed
            if args.ttt_initial_context_random_uniform:
                template = initial_context_tensor
                if template is None:
                    template = pipe.physical_context_encoder.default_context.detach().to(device=pipe.device, dtype=context_dtype)
                initial_context_seed = int(args.seed) + sample_index * 1000003
                generator = torch.Generator(device=pipe.device)
                generator.manual_seed(initial_context_seed)
                sample_initial_context = torch.rand(template.shape, generator=generator, device=pipe.device, dtype=context_dtype)
            context_cache[cache_key] = _adapt_ttt_state(pipe, support_items, args, initial_context=sample_initial_context, target_context=target_context, trajectory_meta={'sample_index': sample_index, 'sample_id': row.get('sample_id'), 'episode_index': row.get('episode_index'), 'group_key': list(group_key), 'cache_key': list(cache_key), 'friction_mu': row.get('friction_mu'), 'pool_target_mu': row.get('pool_target_mu'), 'pool_index': row.get('pool_index'), 'support_displacement_m': row.get('support_displacement_m'), 'initial_context_seed': initial_context_seed, 'support_indices': support_indices, 'initial_context_source': None if initial_context_table is None else initial_context_table['path'], 'initial_context_table_index': initial_context_table_index, 'target_context_source': None if context_table is None else context_table['path']})
            trajectory_rows.extend(context_cache[cache_key][3])
            _write_jsonl(trajectory_path, trajectory_rows)
            torch.cuda.empty_cache()
        (adapted_context, inner_losses, adapt_metrics, cached_trajectory) = context_cache[cache_key]
        if args.adapt_only:
            result_rows.append({'sample_index': sample_index, 'sample_id': row.get('sample_id'), 'episode_index': row.get('episode_index'), 'friction_mu': row.get('friction_mu'), 'pool_target_mu': row.get('pool_target_mu'), 'pool_index': row.get('pool_index'), 'inner_losses': inner_losses, 'context_flat': [float(value) for value in adapted_context.float().cpu().reshape(-1).tolist()]})
            _write_jsonl(output_path.parent / 'results.jsonl', result_rows)
            torch.cuda.empty_cache()
            continue
        if args.render_support and cache_key not in rendered_support:
            for support_index in support_plan[cache_key]:
                support_row = support_rows[int(support_index)]
                support_pred_name = _default_pred_name(int(support_index), support_row)
                support_pred_path = support_output_path / support_pred_name
                if not (support_pred_path.exists() and args.skip_existing):
                    support_sample = query_dataset[int(support_index)]
                    support_sample = prepare_sample_for_rollout(support_sample, int(support_index), pipe, args)
                    support_sample['physical_context'] = _compose_known_physical_context(support_sample, adapted_context, args)
                    support_sample['output_path'] = str(support_pred_path)
                    print(f"[support_sample] group={group_key} support_index={support_index} episode={support_sample['episode_index']} output={support_pred_path}", flush=True)
                    _run_autoregressive(pipe=pipe, sample=support_sample, args=args)
                    torch.cuda.empty_cache()
                support_comparison = _write_two_way_comparison(row=support_row, sample_index=int(support_index), dataset_base_path=Path(args.dataset_base_path), pred_path=support_pred_path, output_dir=support_comparison_path, width=int(args.width), height=int(args.height), fps=int(args.fps), quality=int(args.quality), pred_label='SHOW+TTT')
                print(f'[support_comparison] {support_comparison}', flush=True)
            rendered_support.add(cache_key)
        if resumed_sample:
            print(f'[resume_context_skip_prediction] sample_index={sample_index}', flush=True)
        elif pred_path.exists() and args.skip_existing:
            print(f'[skip] existing prediction {pred_path}', flush=True)
        else:
            sample = query_dataset[sample_index]
            sample = prepare_sample_for_rollout(sample, sample_index, pipe, args)
            sample['physical_context'] = _compose_known_physical_context(sample, adapted_context, args)
            write_pred_path = pred_path
            if args.resume_context_trajectory:
                write_pred_path = pred_path.with_name(pred_path.stem + '.partial' + pred_path.suffix)
                write_pred_path.unlink(missing_ok=True)
            sample['output_path'] = str(write_pred_path)
            print(f"[sample] sample_index={sample_index} group={group_key} episode={sample['episode_index']} output={pred_path}", flush=True)
            _run_autoregressive(pipe=pipe, sample=sample, args=args)
            if write_pred_path != pred_path:
                os.replace(write_pred_path, pred_path)
            torch.cuda.empty_cache()
        if has_transfer_targets:
            source_mu = float(row['friction_mu'])
            source_transfer_dir = transfer_output_path / f'source{sample_index:04d}_mu{source_mu:.6f}'
            source_transfer_dir.mkdir(parents=True, exist_ok=True)
            target_indices = transfer_targets_by_source[sample_index]
            for target_index in target_indices:
                target_row = query_rows[int(target_index)]
                target_mu = float(target_row['friction_mu'])
                if abs(target_mu - source_mu) > 1e-08:
                    raise ValueError(f'Transfer target={target_index} has friction_mu={target_mu}, but source={sample_index} has friction_mu={source_mu}.')
                target_pred_path = source_transfer_dir / _default_pred_name(int(target_index), target_row)
                if target_pred_path.exists() and args.ttt_transfer_skip_existing:
                    print(f'[transfer_skip] {target_pred_path}', flush=True)
                    continue
                target_sample = query_dataset[int(target_index)]
                target_sample = prepare_sample_for_rollout(target_sample, int(target_index), pipe, args)
                target_sample['physical_context'] = _compose_known_physical_context(target_sample, adapted_context, args)
                temporary_target_path = target_pred_path.with_name(target_pred_path.stem + '.partial' + target_pred_path.suffix)
                temporary_target_path.unlink(missing_ok=True)
                target_sample['output_path'] = str(temporary_target_path)
                print(f'[transfer] source={sample_index} target={target_index} output={target_pred_path}', flush=True)
                _run_autoregressive(pipe=pipe, sample=target_sample, args=args)
                os.replace(temporary_target_path, target_pred_path)
                torch.cuda.empty_cache()
            transfer_plan_rows.append({'source_index': sample_index, 'source_sample_id': row.get('sample_id'), 'source_friction_mu': row.get('friction_mu'), 'source_inner_step': cached_trajectory[-1].get('inner_step'), 'target_indices': target_indices, 'target_sample_ids': [query_rows[index].get('sample_id') for index in target_indices]})
            temporary_plan_path = transfer_plan_output_path.with_name(transfer_plan_output_path.name + '.tmp')
            temporary_plan_path.write_text(json.dumps(transfer_plan_rows, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            os.replace(temporary_plan_path, transfer_plan_output_path)
        comparison_file = None
        if args.baseline_pred_dir:
            stage1_pred_path = Path(args.baseline_pred_dir) / pred_name
            if not stage1_pred_path.exists():
                print(f'[comparison_skip] missing stage1 baseline {stage1_pred_path}', flush=True)
            elif not pred_path.exists():
                print(f'[comparison_skip] missing stage2 prediction {pred_path}', flush=True)
            else:
                comparison_file = _write_three_way_comparison(row=row, sample_index=sample_index, dataset_base_path=Path(args.dataset_base_path), stage1_pred_path=stage1_pred_path, stage2_pred_path=pred_path, output_dir=comparison_path, width=int(args.width), height=int(args.height), fps=int(args.fps), quality=int(args.quality))
                print(f'[comparison] {comparison_file}', flush=True)
        result_rows.append({'sample_index': sample_index, 'sample_id': row.get('sample_id'), 'episode_index': row.get('episode_index'), 'group_key': list(group_key), 'friction_mu': row.get('friction_mu'), 'target_c': None if row.get('friction_mu') is None else float(row.get('friction_mu')) / 0.25, 'support_indices': support_plan[cache_key], 'ttt_adapt_scope': 'context', 'inner_losses': inner_losses, 'prediction_path': str(pred_path), 'comparison_path': None if comparison_file is None else str(comparison_file), 'context_norm': float(adapted_context.float().norm().cpu()), 'context_initial_mean': float(adapt_metrics.get('context_initial_mean', float('nan'))), 'context_initial_table_index': initial_context_table_index, 'context_adapted_mean': float(adapt_metrics.get('context_adapted_mean', float('nan'))), 'context_delta_mean': float(adapt_metrics.get('context_delta_mean', float('nan'))), 'context_final_target_l2': cached_trajectory[-1].get('target_context_l2') if cached_trajectory else None, 'known_context_keys': _known_context_keys(args)})
        _write_jsonl(output_path.parent / 'results.jsonl', result_rows)
    if trajectory_rows and context_table is not None:
        active_values = None
        if args.ttt_context_active_values:
            active_values = [float(value.strip()) for value in args.ttt_context_active_values.split(',') if value.strip()]
        pca_metadata_path = Path(args.dataset_metadata_path) if args.dataset_metadata_path else None
        if args.config:
            training_metadata = yaml.safe_load(Path(args.config).read_text()).get('dataset', {}).get('dataset_metadata_path')
            if training_metadata and Path(training_metadata).is_file():
                pca_metadata_path = Path(training_metadata)
        _write_context_pca_plot(pca_output_path, context_table, trajectory_rows, active_values=active_values, metadata_path=pca_metadata_path)
        print(f'[done] context_trajectory={trajectory_path}', flush=True)
        print(f'[done] context_pca={pca_output_path}', flush=True)
    print(f'[done] predictions={output_path}', flush=True)
    print(f'[done] comparisons={comparison_path}', flush=True)
if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    main()
