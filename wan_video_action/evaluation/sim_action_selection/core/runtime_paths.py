"""Process-local LIBERO configuration without machine-bound source YAMLs."""

from __future__ import annotations

import json
import os


def configure_libero_paths(sources, *, plus: bool):
    repository = sources.repos / ("LIBERO-plus" if plus else "LIBERO")
    package = repository / "libero" / "libero"
    directory = sources.work / "libero_runtime"
    directory.mkdir(parents=True, exist_ok=True)
    # JSON is a YAML subset, understood by the original LIBERO config reader.
    # Everything lives in this worker's owned scratch directory, not the
    # source collector's tracked config or a user's ~/.libero configuration.
    paths = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(repository / "libero" / "datasets"),
        "assets": str(package / "assets"),
    }
    (directory / "config.yaml").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    if plus:
        # The original Plus import chain historically selected its own tracked
        # hai-machine YAML. The optional override keeps offline use unchanged.
        os.environ["T2Z_LIBERO_PLUS_CONFIG_PATH"] = str(directory)
    else:
        os.environ.pop("T2Z_LIBERO_PLUS_CONFIG_PATH", None)
    return directory
