from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..io import write_json_atomic, write_jsonl_atomic
from ..task_state import load_task_state
from .core import evaluate_candidates, load_rules, summarize


def _resolve(path, root):
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def _state_value(specification, rules, root):
    if not isinstance(specification, dict):
        return specification
    state = load_task_state(_resolve(specification['state'], root))
    start = int(specification.get('start_frame', 0))
    stride = int(specification.get('frame_stride', 1))
    length = int(specification.get('length', max(1, (state.frame_count - start + stride - 1) // stride)))
    indices = np.clip(start + np.arange(length) * stride, 0, state.frame_count - 1)
    indices = indices[-int(rules.get('terminal_window', 5)):]
    object_index = int(rules.get('object_index', 0))
    kind = rules['outcome_kind']
    if kind == 'point':
        if state.centroids is None:
            return None
        points = np.asarray(state.centroids[indices, object_index], dtype=np.float64)
        points = points[np.all(np.isfinite(points), axis=1)]
        if not len(points):
            return None
        if not state.image_width or not state.image_height:
            raise ValueError('Normalized centroids require image_width and image_height.')
        point = np.median(points, axis=0)
        return [float(point[0] / state.image_width), float(point[1] / state.image_height)]
    if kind == 'scalar':
        if state.angles_rad is None:
            return None
        values = np.degrees(np.asarray(state.angles_rad[indices, object_index], dtype=np.float64))
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else None
    if state.light_score is None:
        raise ValueError('Lamp evaluation requires the tracked light_score array.')
    values = np.asarray(state.light_score[indices, object_index], dtype=np.float64)
    values = values[np.isfinite(values)]
    return bool(np.median(values) >= float(rules['light_threshold'])) if len(values) else None


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--task', choices=[
        'friction', 'gravity', 'mass_collision', 'light_switch',
        'mass_balance', 'mass_friction', 'multi_background',
    ])
    selection.add_argument('--config', type=Path)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rules = json.loads(args.config.read_text()) if args.config else load_rules(args.task)
    root = args.candidates.resolve().parent
    rows = []
    with args.candidates.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                for field in ('predicted_value', 'ground_truth_value'):
                    if field in row:
                        row[field] = _state_value(row[field], rules, root)
                rows.append(row)
    decisions = evaluate_candidates(rows, rules)
    summary = summarize(decisions, rules)
    write_jsonl_atomic(args.output_dir / 'decisions.jsonl', decisions)
    write_json_atomic(args.output_dir / 'summary.json', summary)
    write_json_atomic(args.output_dir / 'rules.json', rules)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
