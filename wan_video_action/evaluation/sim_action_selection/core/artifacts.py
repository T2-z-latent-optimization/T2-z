from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_video(path, views, fps):
    import imageio.v2 as imageio

    path = Path(path)
    with imageio.get_writer(
        path, fps=float(fps), codec="libx264", macro_block_size=None,
        ffmpeg_params=["-crf", "18"],
    ) as writer:
        for frame in views:
            writer.append_data(np.concatenate(list(frame), axis=1).astype(np.uint8))


def save_state(path, state):
    payload = {"fps": state.fps, "image_width": state.image_width, "image_height": state.image_height}
    for name in ("centroids", "visible", "angles_rad", "light_on", "light_score"):
        value = getattr(state, name)
        if value is not None:
            payload[name] = value
    for name, value in (state.events or {}).items():
        payload["event_" + name] = value
    np.savez_compressed(path, **payload)
