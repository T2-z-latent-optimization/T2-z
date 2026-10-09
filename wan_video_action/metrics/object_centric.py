"""Primary mask IoU and centroid trajectory metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from wan_video_action.evaluation.io import read_mask_array
from .pairs import VideoPair


def _select_object(array: np.ndarray, object_index: int) -> np.ndarray:
    if array.ndim == 3:
        return array.astype(bool)
    if array.ndim == 4:
        if object_index < 0 or object_index >= array.shape[1]:
            raise IndexError(
                f"object_index={object_index} outside mask shape {array.shape}."
            )
        return array[:, object_index].astype(bool)
    raise ValueError(
        f"Mask array must have shape (T,H,W) or (T,N,H,W), got {array.shape}."
    )


def _resize_masks(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    if cv2 is not None:
        return np.stack([
            cv2.resize(
                frame.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for frame in masks
        ])
    from PIL import Image

    return np.stack([
        np.asarray(
            Image.fromarray(frame.astype(np.uint8)).resize(
                (width, height),
                resample=Image.Resampling.NEAREST,
            )
        ).astype(bool)
        for frame in masks
    ])


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return np.array([float(np.mean(x)), float(np.mean(y))], dtype=np.float64)


def evaluate_mask_pair(
    gt_mask_path: str,
    pred_mask_path: str,
    *,
    mask_key: str = "masks",
    **options,
) -> dict[str, Any]:
    record = VideoPair(gt_mask_path=gt_mask_path, pred_mask_path=pred_mask_path, **options)
    if not record.gt_mask_path or not record.pred_mask_path:
        raise ValueError(
            f"Object-centric record {record.sample_id!r} requires both mask paths."
        )
    gt = _select_object(
        read_mask_array(record.gt_mask_path, key=mask_key),
        record.object_index,
    )
    pred = _select_object(
        read_mask_array(record.pred_mask_path, key=mask_key),
        record.object_index,
    )
    gt = gt[record.gt_start_frame::record.gt_frame_stride]
    pred = pred[record.pred_start_frame::record.pred_frame_stride]
    count = min(len(gt), len(pred))
    if record.num_frames is not None:
        count = min(count, record.num_frames)
    gt = gt[:count]
    pred = pred[:count]
    if count == 0:
        raise ValueError("GT and prediction masks have no overlapping frames.")
    if pred.shape[1:] != gt.shape[1:]:
        pred = _resize_masks(pred, gt.shape[1], gt.shape[2])

    diagonal = float(np.hypot(gt.shape[1], gt.shape[2]))
    ious: list[float] = []
    centroid_errors: list[float] = []
    missing = 0
    valid_indices: list[int] = []

    for index, (gt_mask, pred_mask) in enumerate(zip(gt, pred)):
        gt_center = _centroid(gt_mask)
        if gt_center is None:
            continue
        valid_indices.append(index)
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        ious.append(float(intersection / union) if union else 0.0)
        pred_center = _centroid(pred_mask)
        if pred_center is None:
            missing += 1
            centroid_errors.append(1.0)
        else:
            centroid_errors.append(
                float(np.linalg.norm(gt_center - pred_center) / diagonal)
            )

    if not valid_indices:
        raise ValueError("No GT-visible frames are available for object metrics.")

    return {
        "sample_id": record.sample_id,
        "environment_id": record.environment_id,
        "method": record.method,
        "split": record.split,
        "support_size": record.support_size,
        "seed": record.seed,
        "valid_visible_frames": len(valid_indices),
        "excluded_gt_invisible_frames": count - len(valid_indices),
        "mean_iou": float(np.mean(ious)),
        "final_iou": float(ious[-1]),
        "centroid_ade": float(np.mean(centroid_errors)),
        "centroid_fde": float(centroid_errors[-1]),
        "missing_mask_rate": float(missing / len(valid_indices)),
    }
