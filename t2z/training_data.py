import json
from pathlib import Path
import yaml

from t2z.config import flatten, resolve
from t2z.chunks import ChunkSelector
from t2z.groups import adapt_environment_indices


def prepare_baseline_metadata(config, directory):
    training = flatten(config)
    source = resolve(training['dataset_metadata_path'])
    with source.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    adapt_environment_indices(rows)
    selector = ChunkSelector.from_options(config.get('t2z_sampling', {}))
    rows = [row for row in rows if selector.accepts(row)]
    selector.validate_episodes(rows, 'friction_mu')
    if not rows:
        raise ValueError('No dataset-declared training chunks')
    physical_values = sorted({float(row['friction_mu']) for row in rows})
    indices = {value: index for index, value in enumerate(physical_values)}
    for row in rows:
        if 'episode_index' not in row:
            raise ValueError('Training metadata requires episode_index')
        if 'environment_index' in row:
            row['source_environment_index'] = row['environment_index']
        row['environment_index'] = indices[float(row['friction_mu'])]
        row['dataset_split'] = 'train'
        row.setdefault('sampling_kind', row.get('chunk_type', 'eligible'))
    directory = Path(directory)
    metadata = directory / 'training_metadata.jsonl'
    metadata.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = directory / 'environments.yaml'
    manifest.write_text(yaml.safe_dump({
        'selection': {'active_environment_ids': list(range(len(physical_values)))},
        'physical_environment_values': physical_values,
    }, sort_keys=False))
    training['dataset_metadata_path'] = str(metadata)
    for prefix in ('dinov2', 'ttt'):
        if prefix + '_environment_key' in training:
            training[prefix + '_environment_key'] = 'environment_index'
            training[prefix + '_action_key'] = 'episode_index'
            training[prefix + '_active_environment_manifest'] = str(manifest)
    config.clear()
    config['training'] = training


def describe_training_data(config, method):
    training = flatten(config)
    with resolve(training['dataset_metadata_path']).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    groups = {}
    selector = ChunkSelector.from_environment()
    key = 'friction_mu' if method == 'ours' else 'environment_index'
    for row in rows:
        if not selector.accepts(row):
            raise ValueError('Runtime training metadata contains an ineligible chunk')
        groups.setdefault(row[key], set()).add(row['episode_index'])
    if not groups:
        raise ValueError('Empty training metadata')
    stats_path = resolve(training['action_stat_path'])
    with stats_path.open() as stream:
        stats = json.load(stream)
    if not stats:
        raise ValueError('Empty action normalization statistics')
    base = resolve(training['dataset_base_path'])
    first = rows[0]
    video = first['video']
    paths = ([video] if isinstance(video, str) else list(video)) + [first['action']]
    for value in paths:
        path = Path(value)
        path = path if path.is_absolute() else base / path
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        'method': method,
        'chunk_selection_mode': selector.mode,
        'chunks': len(rows),
        'environments': len(groups),
        'minimum_episodes_per_environment': min(map(len, groups.values())),
        'first_episode_files_exist': True,
        'action_statistics_readable': True,
        'model_loaded': False,
    }
