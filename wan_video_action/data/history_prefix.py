"""Causal history windows and continuous recorded target EEF inputs.

For anchor t, history length H and stride s, observed frames run from
t-(H-1)*s through t. Future frames follow at stride s. Unavailable history
is padded with the first episode frame for both video and action.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np


def canonicalize_target_eef(values, reference):
    """Normalize quaternion magnitude/sign without changing physical rotation.

The first quaternion uses a single training-derived reference hemisphere.
Later signs are chosen causally against the preceding quaternion, never from
the current chunk's median or an observed/future robot state.
"""
    result = np.array(values, dtype=np.float32, copy=True)
    if result.ndim != 2 or result.shape[1] != 8 or not len(result):
        raise ValueError(f"Expected recorded target EEF [T,8], got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite recorded EEF target")
    quaternion = result[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Zero-norm target quaternion")
    quaternion /= norms
    ref = np.asarray(reference, dtype=np.float32)
    if ref.shape != (4,) or not np.isfinite(ref).all() or np.linalg.norm(ref) < 1e-8:
        raise ValueError("Invalid training quaternion reference")
    if np.dot(quaternion[0], ref) < 0:
        quaternion[0] *= -1
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0:
            quaternion[index] *= -1
    return result


@lru_cache(maxsize=128)
def _canonical_episode(path, reference):
    from .recorded_targets import read_recorded_targets

    result = canonicalize_target_eef(read_recorded_targets(path, "eef_target"), reference)
    result.setflags(write=False)
    return result


class CanonicalTargetEEF:
    """Recorded target EEF, normalized with training-only statistics."""

    def __init__(self, args, stats):
        self.base = Path(args.dataset_base_path)
        self.frames = int(args.num_frames)
        self.stride = int(args.frame_stride)
        self.width = int(args.action_dim)
        self.low = np.asarray(stats["min"], dtype=np.float32)
        self.high = np.asarray(stats["max"], dtype=np.float32)
        self.reference = tuple(stats["quaternion_reference_wxyz"])
        if self.width < 8 or self.low.shape != (8,) or self.high.shape != (8,):
            raise ValueError("Target EEF requires eight native channels and matching statistics")

    def __call__(self, payload):
        import torch

        path = Path(payload["data"])
        if not path.is_absolute():
            path = self.base / path
        values = _canonical_episode(str(path), self.reference)
        start = int(payload["start_frame"])
        last = min(int(payload["end_frame"]), len(values) - 1)
        if start < 0 or start > last:
            raise ValueError(f"Invalid target window [{start},{last}] in {path}")
        indices = np.minimum(start + self.stride * np.arange(self.frames), last)
        span = self.high - self.low
        normalized = np.clip(
            2 * (values[indices] - self.low) / np.maximum(span, 1e-8) - 1, -1, 1,
        )
        normalized[:, span < 1e-8] = 0
        packed = np.zeros((self.frames, self.width), dtype=np.float32)
        packed[:, :8] = normalized
        return torch.from_numpy(packed).unsqueeze(0)


def _frame_zero_payload(payload):
    if isinstance(payload, (list, tuple)):
        return [_frame_zero_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {**payload, "start_frame": 0, "end_frame": 0}
    return {"data": payload, "start_frame": 0, "end_frame": 0}


class HistoryPrefixDataset:
    """Wrap the existing video decoder/resize path; only add missing history.

The manifest's nonnegative start is the first real strided frame. Decode with
the original operator, prepend episode frame zero for unavailable history,
then keep the configured number of frames without negative parquet slicing.
"""

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.data = dataset.data
        self.frames = int(args.num_frames)
        self.history = int(args.num_history_frames)
        self.stride = int(args.frame_stride)
        if self.stride < 1 or not 1 <= self.history < self.frames:
            raise ValueError("History windows require positive stride and both observed and future frames")
        for row in self.data:
            anchor = int(row["history_anchor_frame"])
            logical_start = anchor - (self.history - 1) * self.stride
            padding = max(0, (-logical_start + self.stride - 1) // self.stride)
            if int(row["history_padding_frames"]) != padding:
                raise ValueError("History padding disagrees with anchor")
            if int(row["start_frame"]) != logical_start + padding * self.stride:
                raise ValueError("First real frame is not aligned with history anchor")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        import torch

        row = self.data[int(index) % len(self.data)]
        sample = self.dataset[index]
        padding = int(row["history_padding_frames"])
        if padding:
            if int(row["start_frame"]) == 0:
                first_video = sample["video"][:, :, :1]
                first_action = sample["action"][:, :1]
            else:
                first_video = self.dataset.main_data_operator(
                    _frame_zero_payload(row["video"]),
                )[:, :, :1]
                first_action = self.dataset.special_operator_map["action"](
                    _frame_zero_payload(row["action"]),
                )[:, :1]
            sample["video"] = torch.cat(
                [first_video.repeat(1, 1, padding, 1, 1),
                 sample["video"][:, :, :self.frames - padding]], dim=2,
            )
            sample["action"] = torch.cat(
                [first_action.repeat(1, padding, 1),
                 sample["action"][:, :self.frames - padding]], dim=1,
            )
        if sample["video"].shape[2] != self.frames or sample["action"].shape[1] != self.frames:
            raise ValueError(f"History video/action must have {self.frames} aligned frames")
        return sample
