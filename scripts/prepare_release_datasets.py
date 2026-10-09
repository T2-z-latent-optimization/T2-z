import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


CONTROL_FIELDS = (
    'action', 'observation.state', 'observation.eef_state',
    'raw.commanded_target.joint_rad', 'raw.commanded_target.arm_sent',
    'target_joint_action',
)
WINDOW_FIELDS = (
    'start_frame', 'end_frame', 'length', 'total_frames', 'frame_stride',
    'num_history_frames', 'native_frame_indices', 'valid_frames',
    'pad_short',
    'action_semantics',
)
STAT_FIELDS = {
    'min', 'max', 'mean', 'std', 'p01', 'p99', 'count',
    'quaternion_reference_wxyz', 'constant_channels', 'frame_stride',
}
PREFERRED = {
    'real_ball': ('precise', 0.5),
    'real_door': ('precise', 0.5),
    'real_stick': ('lift', 0.7),
}
FIXED = {'sim_pushbox_friction', 'sim_multi_background'}


def confined_file(root, relative):
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular local dataset file: {relative}')
    path.resolve().relative_to(root.resolve())
    return path


def clean_statistics(value):
    if not isinstance(value, dict):
        raise TypeError('Statistics must be an object')
    if 'min' in value and 'max' in value:
        return {key: item for key, item in value.items() if key in STAT_FIELDS}
    return {key: clean_statistics(item) for key, item in value.items()
            if isinstance(item, dict) and 'min' in item and 'max' in item}


def environment_value(task, row):
    if task == 'sim_multi_background':
        return row['physical_friction_mu']
    return row.get('friction_mu', row.get('environment_index'))


def episode_weights(task, rows, training):
    eligible = [bool(row.get('trainable_chunk', True)) for row in rows]
    if task == 'real_stick':
        eligible = [ok and not (row.get('initial_slide_override', False)
                    and row.get('sampling_kind') != 'lift')
                    for ok, row in zip(eligible, rows)]
    available = [i for i, ok in enumerate(eligible) if ok]
    if not available:
        raise ValueError('An exported episode has no usable windows')
    weights = [0.0] * len(rows)
    if task in PREFERRED:
        kind, probability = PREFERRED[task]
        preferred = [i for i in available if rows[i].get('sampling_kind') == kind]
        other = [i for i in available if rows[i].get('sampling_kind') != kind]
        if preferred and other:
            for i in preferred:
                weights[i] = probability / len(preferred)
            for i in other:
                weights[i] = (1 - probability) / len(other)
        else:
            for i in available:
                weights[i] = 1 / len(available)
    elif task == 'sim_mass_balance':
        count = int(training['grouped_context_actions_per_update'])
        required = max(int(rows[i].get('sampling_required_count', 0)) for i in available)
        preferred = [i for i in available if rows[i].get('sampling_required_pool', False)]
        if not 0 <= required <= count or (required and not preferred):
            raise ValueError('Invalid required-window allocation')
        probability = required / count
        for i in available:
            weights[i] = (1 - probability) / len(available)
        if preferred:
            for i in preferred:
                weights[i] += probability / len(preferred)
    else:
        for i in available:
            weights[i] = 1 / len(available)
    return weights


def load_inputs(root):
    datasets = {}
    configs = {}
    for path in sorted((root / 'datasets').glob('*/train.jsonl')):
        task = path.parent.name
        datasets[task] = {
            'rows': [json.loads(line) for line in path.read_text().splitlines() if line.strip()],
            'extra': {'action_stats.json': (path.parent / 'action_stats.json').read_text()},
        }
        config = root / 'configs' / 'ours' / f'{task}.yaml'
        configs[str(config.relative_to(root))] = {'data': yaml.safe_load(config.read_text())}
    return {'datasets': datasets, 'configs': configs}


def prepare_task(root, stage, task, data, config, seed):
    source = root / 'datasets' / task
    if source.is_symlink():
        raise ValueError('Dataset task directories must not be symlinks')
    if (source / 'sampling.json').exists():
        raise ValueError('Dataset already prepared; do not relabel an existing release')
    destination = stage / task
    destination.mkdir()
    training = config.get('training', config)
    rows = data['rows']
    values = list(dict.fromkeys(environment_value(task, row) for row in rows))
    if None in values:
        raise ValueError('Missing environment grouping')
    rng = random.Random(seed)
    rng.shuffle(values)
    environments = {value: i for i, value in enumerate(values)}
    episodes = defaultdict(list)
    for row in rows:
        videos = row['video'] if isinstance(row['video'], list) else [row['video']]
        episodes[(row['action'], tuple(videos))].append(row)
    identities = list(episodes)
    rng.shuffle(identities)
    output_rows = []
    for episode_index, identity in enumerate(identities):
        episode_rows = episodes[identity]
        group = {environment_value(task, row) for row in episode_rows}
        if len(group) != 1:
            raise ValueError('An episode belongs to multiple physical environments')
        environment_index = environments[next(iter(group))]
        relative = Path('episodes') / f'{episode_index:06d}'
        folder = destination / 'data' / relative
        folder.mkdir(parents=True)
        action_source = confined_file(source / 'data', identity[0])
        table = pq.read_table(action_source)
        columns = [key for key in CONTROL_FIELDS if key in table.column_names]
        if not columns:
            raise ValueError('No control/state columns found')
        anonymous = pa.Table.from_arrays([table[key] for key in columns], names=columns)
        pq.write_table(anonymous, folder / 'actions.parquet', compression='zstd')
        videos = []
        for view, video in enumerate(identity[1]):
            name = f'view_{view:02d}.mp4'
            os.link(confined_file(source / 'data', video), folder / name)
            videos.append((relative / name).as_posix())
        weights = episode_weights(task, episode_rows, training)
        windows = {}
        for row, weight in zip(episode_rows, weights):
            item = {key: row[key] for key in WINDOW_FIELDS if key in row}
            item.setdefault('frame_stride', int(training.get('frame_stride', 1)))
            item.setdefault('total_frames', table.num_rows)
            item.setdefault('num_history_frames', int(training.get('num_history_frames', 1)))
            item.setdefault('pad_short', bool(training.get('pad_short_chunks', False)))
            item.update(environment_index=environment_index, episode_index=episode_index,
                        video=videos, action=(relative / 'actions.parquet').as_posix(),
                        trainable_chunk=weight > 0, chunk_weight=weight)
            key = (item['start_frame'], item['end_frame'], item['frame_stride'], item['length'])
            if key in windows:
                windows[key]['chunk_weight'] += weight
                windows[key]['trainable_chunk'] |= weight > 0
            else:
                windows[key] = item
        if task in FIXED:
            selected = [item for item in windows.values() if item['trainable_chunk']]
            if len(selected) != 1 or selected[0]['start_frame'] != 65 or selected[0]['end_frame'] != 105:
                raise ValueError('Expected the fixed 65-105 window')
            template = selected[0]
            span = (int(template['length']) - 1) * int(template['frame_stride'])
            for start in range(max(0, int(template['total_frames']) - span)):
                key = (start, start + span, template['frame_stride'], template['length'])
                if key not in windows:
                    windows[key] = dict(template, start_frame=start, end_frame=start + span,
                                        trainable_chunk=False, chunk_weight=0.0)
        total_weight = math.fsum(item['chunk_weight'] for item in windows.values())
        if not math.isclose(total_weight, 1.0, abs_tol=1e-12):
            raise ValueError('Per-episode window probabilities do not sum to one')
        for chunk_index, (_, item) in enumerate(sorted(windows.items())):
            item['sample_id'] = f'e{episode_index:06d}_w{chunk_index:06d}'
            output_rows.append(item)
    with (destination / 'train.jsonl').open('w') as stream:
        for row in output_rows:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    statistics = clean_statistics(json.loads(data['extra']['action_stats.json']))
    (destination / 'action_stats.json').write_text(json.dumps(statistics, indent=2) + '\n')
    weighted = task in PREFERRED or task == 'sim_mass_balance'
    policy = {'chunk_selection_mode': 'weighted' if weighted else 'eligible',
              'eligible_field': 'trainable_chunk', 'weight_field': 'chunk_weight',
              'weight_scope': 'episode', 'unlisted_windows': 'ineligible'}
    (destination / 'sampling.json').write_text(json.dumps(policy, indent=2) + '\n')
    result = {'environments': len(values), 'episodes': len(episodes),
              'candidate_chunks': len(output_rows),
              'eligible_chunks': sum(row['trainable_chunk'] for row in output_rows),
              'chunk_selection_mode': policy['chunk_selection_mode']}
    print(task, json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--seed', type=int, default=173)
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / 'datasets'
    target.resolve().relative_to(root)
    inputs = json.loads(args.snapshot.read_text()) if args.snapshot else load_inputs(root)
    stage = Path(tempfile.mkdtemp(prefix='.preparing-', dir=target))
    manifest = {'split': 'train', 'environment_labels': 'anonymous_indices', 'tasks': {}}
    try:
        for index, (task, data) in enumerate(sorted(inputs['datasets'].items())):
            if Path(task).name != task:
                raise ValueError('Invalid task name')
            config = inputs['configs'][f'configs/ours/{task}.yaml']['data']
            manifest['tasks'][task] = prepare_task(root, stage, task, data, config, args.seed + index)
        for task in manifest['tasks']:
            destination = target / task
            previous = stage / f'.previous-{task}'
            os.replace(destination, previous)
            try:
                os.replace(stage / task, destination)
            except BaseException:
                os.replace(previous, destination)
                raise
            shutil.rmtree(previous)
        (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (target / 'README.md').write_text(
            '# Training datasets\n\n'
            'Only training episodes are included. Environment and episode identifiers are anonymous indices. '
            'Physical parameter values, action-level labels and source paths are not included. '
            'Recorded control/state values are preserved. Videos are unchanged; action files contain only control/state columns.\n\n'
            'Each task provides train.jsonl, action_stats.json, sampling.json and data/episodes/. '
            'Metadata paths are relative to the task data/ directory.\n\n'
            'The sampler first selects environments, then episodes, then a window within each episode. '
            'trainable_chunk is a boolean. chunk_weight is a non-negative relative weight, normalized within each episode. '
            'Eligible mode selects true rows uniformly; weighted mode uses chunk_weight; uniform mode ignores both fields. '
            'Unlisted windows are unavailable. Temporal indices, stride and padding remain explicit.\n\n'
            'Friction and diverse-background tasks enable only frames 65-105. '
            'Ball and door assign half the probability to each original window pool. '
            'Stick assigns 70 percent to lift windows and 30 percent to general windows; drift-only episodes use lift windows only. '
            'Soft uses only annotated stationary-start windows. Light uses the existing event-centered windows uniformly.\n\n'
            'Mass balance retains starts 0, 40 and 70, with stride 2 and 41 sampled frames. '
            'With eight samples per environment and four preferential late-window draws, its marginal probabilities are '
            '1/6, 5/12 and 5/12. Independent weighted draws reproduce these probabilities, not a strict per-update quota.\n'
        )
    finally:
        if stage.exists() and not any(stage.glob('.previous-*')):
            shutil.rmtree(stage)


if __name__ == '__main__':
    main()
