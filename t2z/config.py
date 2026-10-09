import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def setup_environment():
    for key, value in {
        'T2Z_REPO_ROOT': ROOT,
        'T2Z_OUTPUT_ROOT': ROOT / 'outputs',
        'T2Z_LOCAL_CACHE': Path('/tmp/t2z'),
    }.items():
        os.environ.setdefault(key, str(value))


def expand(value):
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value


def load_config(path):
    setup_environment()
    return expand(yaml.safe_load(Path(path).read_text()))


def flatten(config):
    result = {}
    for key, value in config.items():
        if key.startswith('t2z_'):
            continue
        if isinstance(value, dict):
            result.update(value)
        else:
            result[key] = value
    return result


def resolve(path):
    path = Path(os.path.expandvars(str(path))).expanduser()
    return path if path.is_absolute() else ROOT / path
