import json
import os
import random
from typing import Optional

import imageio
import numpy as np
import torch


def resolve_path(base_path: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_path, path)


def resolve_model_path(model_path: str):
    model_path = os.path.expanduser(str(model_path).strip())
    if not model_path.endswith(".safetensors.index.json"):
        return model_path

    with open(model_path, "r") as f:
        weight_map = json.load(f).get("weight_map", {})
    return [
        os.path.join(os.path.dirname(model_path), shard_name)
        for shard_name in sorted(set(weight_map.values()))
    ]


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    max_value = float(np.max(array))
    min_value = float(np.min(array))
    if max_value <= 1.0 and min_value >= -1.0:
        array = (array + 1.0) * 127.5
    elif max_value <= 1.0 and min_value >= 0.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def save_video(video: torch.Tensor, output_path: str, fps: int, quality: int):
    video = video.detach().to(dtype=torch.float32).cpu()
    num_views = int(video.shape[0])
    num_frames = int(video.shape[2])

    frames = []
    for frame_idx in range(num_frames):
        view_frames = [
            _normalize_to_uint8(video[view_idx, :, frame_idx].permute(1, 2, 0).numpy())
            for view_idx in range(num_views)
        ]
        frame = view_frames[0] if len(view_frames) == 1 else np.concatenate(view_frames, axis=1)
        frames.append(frame)

    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=quality) as writer:
        for frame in frames:
            writer.append_data(frame)


def align_num_frames(
    num_frames: int,
    time_division_factor: int = 4,
    time_division_remainder: int = 1,
    mode: str = "floor",
) -> int:
    num_frames = int(num_frames)
    factor = int(time_division_factor)
    remainder = int(time_division_remainder)
    if factor <= 0:
        raise ValueError(f"time_division_factor must be positive, got {factor}")
    if mode not in ("floor", "ceil"):
        raise ValueError(f"mode must be 'floor' or 'ceil', got {mode}")
    if num_frames % factor == remainder % factor:
        return num_frames
    if mode == "ceil":
        return num_frames + ((remainder - num_frames) % factor)
    aligned = num_frames - ((num_frames - remainder) % factor)
    return max(1, aligned) if num_frames > 0 else num_frames


def resolve_num_frames(
    available_frames: int,
    requested_frames: Optional[int],
    time_division_factor: int = 4,
    time_division_remainder: int = 1,
    align: bool = True,
) -> int:
    available_frames = int(available_frames)
    if requested_frames is None:
        return available_frames

    requested_frames = int(requested_frames)
    if available_frames >= requested_frames:
        return requested_frames

    num_frames = max(0, available_frames)
    if align:
        return align_num_frames(
            num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            mode="floor",
        )
    return num_frames


def set_global_seed(seed=42):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
