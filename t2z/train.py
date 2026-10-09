import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from t2z.config import flatten, load_config
from t2z.chunks import configure_chunk_selection
from t2z.metadata import select_environment_groups
from t2z.training_data import describe_training_data, prepare_baseline_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', choices=('ours', 'standard', 'dino', 'ttt'), default='ours')
    parser.add_argument('--check-data', action='store_true')
    args, extra = parser.parse_known_args()
    config = load_config(args.config)
    config.pop('t2z_task', None)
    sampling = config.get('t2z_sampling', {})
    configure_chunk_selection(sampling)
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix='t2z-training-') as temporary:
        if args.method == 'ours':
            if args.check_data:
                for section in config.values():
                    if isinstance(section, dict) and 'output_path' in section:
                        section['output_path'] = str(Path(temporary) / 'output')
            select_environment_groups(config, ROOT, temporary)
        else:
            prepare_baseline_metadata(config, temporary)
        if args.check_data:
            print(json.dumps(describe_training_data(config, args.method), sort_keys=True))
            return
        runner = importlib.import_module('t2z.training.' + args.method)
        path = Path(temporary) / 'config.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        sys.argv = [args.method, '--config', str(path), *extra]
        runner.main()


if __name__ == '__main__':
    main()
