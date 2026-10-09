"""Explicit recorded command sources. Never substitute measured robot states."""
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq


@lru_cache(maxsize=128)
def read_recorded_targets(parquet_path, action_type):
    if action_type == "joint_target":
        field = "raw.commanded_target.joint_rad"
        sent = "raw.commanded_target.arm_sent"
        data = pq.read_table(parquet_path, columns=[field, sent]).to_pydict()
        if not np.asarray(data[sent], dtype=bool).all():
            raise ValueError(f"Unsent joint target in {parquet_path}; no measured-action fallback")
        values = np.asarray(data[field], dtype=np.float32)
        width = 7
    elif action_type == "joint_target_export":


        field = "target_joint_action"
        exported = np.asarray(
            pq.read_table(parquet_path, columns=[field]).to_pydict()[field],
            dtype=np.float32,
        )
        if exported.ndim != 2 or exported.shape[1] != 8:
            raise ValueError(f"Expected target_joint_action [T,8]: {parquet_path}")
        values = exported[:, :7]
        width = 7
    elif action_type == "eef_target":

        values = np.asarray(
            pq.read_table(parquet_path, columns=["action"]).to_pydict()["action"],
            dtype=np.float32,
        )
        width = 8
    else:
        raise ValueError(f"Not a recorded target representation: {action_type}")
    if values.ndim != 2 or values.shape[1] != width or not len(values):
        raise ValueError(f"Invalid {action_type} shape {values.shape}: {parquet_path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite recorded targets: {parquet_path}")
    values.setflags(write=False)
    return values


def load_target_chunk(loader, parquet_path, start, end, num_frames):
    import torch

    values = read_recorded_targets(str(parquet_path), loader.action_type)
    last = min(int(end), len(values) - 1)
    if start < 0 or start > last:
        raise ValueError(f"Empty target chunk [{start}, {end}] in {parquet_path}")

    indices = np.minimum(start + np.arange(num_frames) * loader.frame_stride, last)
    low, high = loader._get_min_max()
    span = high - low
    normalized = np.clip(2.0 * (values[indices] - low) / np.maximum(span, 1e-8) - 1.0, -1, 1)
    normalized[:, span < 1e-8] = 0.0
    width = loader.output_dim or values.shape[1]
    if width < values.shape[1]:
        raise ValueError("Target channels cannot be silently dropped")
    packed = np.zeros((num_frames, width), dtype=np.float32)
    packed[:, :values.shape[1]] = normalized
    return torch.from_numpy(packed).unsqueeze(0)
