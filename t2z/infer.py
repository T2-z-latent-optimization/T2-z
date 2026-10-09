import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from t2z.config import flatten, load_config, resolve
from t2z.inference.selection import group_samples, parse_indices


def main():
    parser = argparse.ArgumentParser('Task-independent video prediction')
    parser.add_argument('--method', choices=('ours', 'standard', 'dino', 'ttt', 'lora'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--context-table')
    parser.add_argument('--metadata')
    parser.add_argument('--support-indices')
    parser.add_argument('--query-indices')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = load_config(args.config)
    checkpoint = resolve(args.checkpoint)
    if args.method == 'ours' and not args.context_table:
        parser.error('--context-table is required for Ours')
    context_table = resolve(args.context_table) if args.context_table else None
    metadata = resolve(args.metadata or spec['dataset_metadata_path'])
    supports = parse_indices(args.support_indices if args.support_indices is not None else spec.get('support_indices'))
    queries = parse_indices(args.query_indices if args.query_indices is not None else spec.get('query_indices'))
    if not queries:
        parser.error('Select query_indices in the config or with --query-indices')
    with metadata.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    environment_key = spec.get('environment_key', 'environment_index')
    groups = group_samples(rows, supports, queries, environment_key)
    output = resolve(args.output)
    runtime = output / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    cfg = flatten(load_config(resolve(spec['training_configs'][args.method])))
    cfg.update(spec['inference'])
    cfg.update(dataset_metadata_path=str(metadata), ckpt_path=str(checkpoint),
               output_path=str(output / 'raw'), sample_indices=','.join(map(str, queries)), max_samples=0,
               video_light_augmentation_enabled=False)
    if args.method == 'ours':
        support_path = runtime / 'support.jsonl'
        support_path.write_text(''.join(json.dumps(rows[index]) + '\n' for index in supports))
        cfg.update(spec['ours'])
        cfg.update(support_metadata_path=str(support_path), support_count=0,
                   stage2_group_keys=environment_key, ttt_initial_context_table_path=str(context_table),
                   ttt_initial_context_global=bool(spec['ours'].get('ttt_initial_context_global', False)))
    elif args.method in ('dino', 'ttt', 'lora'):
        cfg.update(support_indices=','.join(map(str, supports)), expected_supports_per_environment=0)
        if args.method == 'dino':
            cfg.update(dinov2_environment_key=environment_key, dinov2_checkpoint_path=str(checkpoint),
                       wan_checkpoint_output=str(runtime / 'world_model.safetensors'))
        elif args.method == 'ttt':
            cfg.update(ttt_environment_key=environment_key, ttt_checkpoint_path=str(checkpoint))
        else:
            cfg['support_environment_key'] = environment_key
    path = runtime / 'config.yaml'
    path.write_text(yaml.safe_dump({'inference': cfg}, sort_keys=False))
    provenance = {'method': args.method, 'task': spec['task'], 'checkpoint': str(checkpoint),
                  'context_table': str(context_table) if context_table else None,
                  'metadata_path': str(metadata), 'support_indices': supports, 'query_indices': queries,
                  'support_counts': {group['environment_id']: len(group['support_indices']) for group in groups}}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    sys.argv = [args.method, '--config', str(path)]
    if args.method == 'ours':
        sys.argv += ['--stage2_ckpt_path', str(checkpoint)]
    os.chdir(ROOT)
    importlib.import_module('t2z.inference.' + args.method).main()


if __name__ == '__main__':
    main()
