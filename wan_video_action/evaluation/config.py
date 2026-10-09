"""Evaluation config loading without coupling to training configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Evaluation config must be a mapping: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload
