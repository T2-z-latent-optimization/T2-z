#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../../.." && pwd)
base=${T2Z_SIM_BASE_VENV:-"$repo/../TTT4dynamics/.venv"}
if [[ ! -x "$base/bin/python" ]]; then
    printf 'Missing original simulator Python: %s\n' "$base/bin/python" >&2
    exit 1
fi
if [[ ! -x "$repo/.venv/bin/python" ]]; then
    uv venv --python "$base/bin/python" "$repo/.venv"
fi
base_packages=$("$base/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
overlay_packages=$("$repo/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$base_packages" > "$overlay_packages/t2z_sim_base.pth"
# Do not resolve/reinstall torch or mutate the original simulator environment.
uv pip install --python "$repo/.venv/bin/python" --default-index https://pypi.org/simple \
    --no-deps -r "$script_dir/requirements-model.txt"
