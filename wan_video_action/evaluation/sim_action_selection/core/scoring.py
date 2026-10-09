"""Model prediction accuracy on the SELECTED, genuinely executed rollout."""
from __future__ import annotations

import numpy as np


def prediction_accuracy(predicted, observed, rules):
    index = int(rules.get("object_index", 0))
    kind = rules["outcome_kind"]
    count = min(predicted.frame_count, observed.frame_count)
    if kind == "point":
        left = predicted.centroids[:count, index].astype(np.float64)
        right = observed.centroids[:count, index].astype(np.float64)
        left /= [predicted.image_width, predicted.image_height]
        right /= [observed.image_width, observed.image_height]
        valid = np.all(np.isfinite(left) & np.isfinite(right), axis=1)
        if predicted.visible is not None and observed.visible is not None:
            valid &= predicted.visible[:count, index] & observed.visible[:count, index]
        error = np.linalg.norm(left[valid] - right[valid], axis=1)
        metric = "mean_normalized_image_centroid_error_visible_frames"
    elif kind == "scalar":
        left = np.degrees(predicted.angles_rad[:count, index])
        right = np.degrees(observed.angles_rad[:count, index])
        valid = np.isfinite(left) & np.isfinite(right)
        error = np.abs((left[valid] - right[valid] + 90.0) % 180.0 - 90.0)
        metric = "mean_principal_axis_angle_error_degrees"
    else:
        left = predicted.light_score[:count, index]
        right = observed.light_score[:count, index]
        valid = np.isfinite(left) & np.isfinite(right)
        error = np.abs(left[valid] - right[valid])
        metric = "mean_lamp_roi_score_error"
    # Conditioning/history frame is copied from the observation and must not
    # artificially improve model accuracy. Compute future-only error again.
    valid_indices = np.flatnonzero(valid)
    error = error[valid_indices > 0]
    return {
        "metric": metric, "value": float(error.mean()) if len(error) else None,
        "compared_future_frames": len(error), "requested_future_frames": max(count - 1, 0),
        "scope": "RGB tracked selected rollout; no privileged physical poses",
    }
