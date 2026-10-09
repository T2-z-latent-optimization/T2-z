import argparse
import os
from pathlib import Path
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ROOT_ASSETS = {
    'datasets', 'data', 'outputs', 'results', 'models', 'checkpoints',
    'ckpt', 'logs', 'tmp', 'temp', 'cache', 'wandb', 'swanlog',
}
CACHE_DIRS = {
    '.git', '__pycache__', '.cache', '.pytest_cache', '.mypy_cache',
    '.ruff_cache', '.venv', 'venv', '.idea', '.vscode',
    '.ipynb_checkpoints', '.codex', '.ssh', 'node_modules',
}
EXCLUDED_SUFFIXES = {
    '.pyc', '.pyo', '.safetensors', '.pt', '.pth', '.ckpt', '.bin',
    '.mp4', '.avi', '.mov', '.mkv', '.parquet', '.arrow', '.zip',
    '.tar', '.gz', '.log', '.pem', '.key', '.swp', '.partial',
}
EXCLUDED_NAMES = {'AGENTS.md', 'CLAUDE.md', '.DS_Store', '.git', '.env'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.expanduser().resolve()
    files = []
    for directory, dirs, names in os.walk(ROOT):
        parent = Path(directory)
        dirs[:] = sorted(
            name for name in dirs
            if name not in CACHE_DIRS
            and not (parent == ROOT and name in ROOT_ASSETS)
            and not (parent / name).is_symlink()
        )
        for name in sorted(names):
            path = parent / name
            if path.is_symlink() or name in EXCLUDED_NAMES:
                continue
            if name.startswith('.env.') or name.endswith('~'):
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES or path.resolve() == target:
                continue
            files.append(path)
    relative = [path.relative_to(ROOT).as_posix() for path in files]
    required = ('wan_video_action/data/', 'wan_video_action/models/')
    counts = {
        prefix: sum(name.startswith(prefix) and name.endswith('.py') for name in relative)
        for prefix in required
    }
    if not files or not all(counts.values()):
        raise RuntimeError('Required source packages are missing from the archive inputs')
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.t2z-package-', dir=target.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path, name in zip(files, relative):
                info = zipfile.ZipInfo('T2-z/' + name, date_time=(2026, 1, 1, 0, 0, 0))
                info.create_system = 3
                mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                info.external_attr = (0o100000 | mode) << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    print(f'Archive: {target}')
    print(f'Files: {len(files)}')
    for prefix, count in counts.items():
        print(f'{prefix}: {count} Python source files')


if __name__ == '__main__':
    main()
