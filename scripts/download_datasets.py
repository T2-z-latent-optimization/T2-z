import argparse
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
from zipfile import ZipFile

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


def download_archive(directory, revision):
    for attempt in range(5):
        try:
            return hf_hub_download(
                repo_id='huagailuowen/T2-z-datasets',
                repo_type='dataset',
                filename='T2-z-datasets.zip',
                revision=revision,
                local_dir=directory,
            )
        except HfHubHTTPError as error:
            response = error.response
            if response is None or response.status_code != 429 or attempt == 4:
                raise
            reset = re.search(r'(?:^|;)\s*t=(\d+)', response.headers.get('RateLimit', ''))
            seconds = int(reset.group(1)) if reset else 300
            retry_after = response.headers.get('Retry-After', '')
            if retry_after.isdigit():
                seconds = max(seconds, int(retry_after))
            seconds = max(1, seconds) + 1
            print(f'Hugging Face rate limit; retrying in {seconds} seconds.', flush=True)
            time.sleep(seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'datasets')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    destination = args.destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        parser.error(f'Destination already exists; no files will be changed: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f'.{destination.name}.download'
    if staging.is_symlink():
        raise ValueError(f'Download staging directory must not be a symbolic link: {staging}')
    archive = download_archive(staging / 'download', args.revision)
    unpacked = staging / 'unpacked'
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if (not path.parts or path.parts[0] != 'datasets' or path.is_absolute()
                    or '..' in path.parts or '\\' in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f'Unsafe dataset archive member: {member.filename}')
        bundle.extractall(unpacked)
    dataset = unpacked / 'datasets'
    metadata = sorted(dataset.glob('*/train.jsonl'))
    if not metadata or not (dataset / 'manifest.json').is_file():
        raise FileNotFoundError('The downloaded repository does not contain the dataset layout.')
    for path in metadata:
        for name in ('sampling.json', 'action_stats.json', 'data'):
            if not (path.parent / name).exists():
                raise FileNotFoundError(path.parent / name)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Destination appeared during download: {destination}')
    dataset.rename(destination)
    shutil.rmtree(staging)
    print(f'Prepared {len(metadata)} task datasets at {destination}')


if __name__ == '__main__':
    main()
