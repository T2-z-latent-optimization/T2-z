import json
import os
from pathlib import Path
from t2z.config import flatten, resolve
from t2z.chunks import ChunkSelector
from t2z.groups import adapt_environment_indices


def select_environment_groups(config, root, temporary_directory):
    training = flatten(config)
    source = resolve(training['dataset_metadata_path'])
    options = config.get('t2z_sampling', {})
    selector = ChunkSelector.from_options(options)
    with source.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    adapt_environment_indices(rows)
    rows = [row for row in rows if selector.accepts(row)]
    selector.validate_episodes(rows, 'friction_mu')
    if not rows:
        raise ValueError('No dataset-declared trainable chunks')
    for row in rows:
        row['dataset_split'] = 'train'
    copies = int(options.get('latents_per_physical_environment', 1))
    if copies < 1:
        raise ValueError('latents_per_physical_environment must be positive')
    groups = sorted({float(row['friction_mu']) for row in rows})
    records = []
    if copies > 1:
        if any('latent_environment_index' in row for row in rows):
            if not all('latent_environment_index' in row and 'physical_environment_index' in row for row in rows):
                raise ValueError('Partially annotated latent aliases are not supported')
            pairs = {(int(row['latent_environment_index']), row['physical_environment_index']) for row in rows}
            for latent, physical in sorted(pairs):
                records.append({'latent_environment_index': latent, 'physical_environment_index': physical})
            rows = [dict(row, friction_mu=float(row['latent_environment_index']), environment_index=int(row['latent_environment_index'])) for row in rows]
        else:
            expanded = []
            for replica in range(copies):
                for physical_index, value in enumerate(groups):
                    latent_index = replica * len(groups) + physical_index
                    records.append({'latent_environment_index': latent_index, 'physical_environment_index': value, 'replica': replica})
                    for row in rows:
                        if float(row['friction_mu']) == value:
                            item = dict(row)
                            item.update(physical_environment_index=value, latent_environment_index=latent_index,
                                        environment_index=latent_index, friction_mu=float(latent_index), latent_replica=replica)
                            expanded.append(item)
            rows = expanded
    count = len({float(row['friction_mu']) for row in rows})
    training['grouped_context_curriculum_total_groups'] = count
    destination = Path(temporary_directory) / 'training_metadata.jsonl'
    destination.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    training['dataset_metadata_path'] = str(destination)
    if int(os.environ.get('RANK', '0')) == 0:
        output = resolve(training['output_path'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'latent_environment_mapping.json').write_text(json.dumps({'latents_per_physical_environment': copies, 'records': records}, indent=2))
    config.clear()
    config['training'] = training
