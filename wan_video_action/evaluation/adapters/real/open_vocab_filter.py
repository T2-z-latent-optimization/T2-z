"""Silver-terminal proposal filtering, independent of world-model training."""
from __future__ import annotations

import cv2
import numpy as np


def filter_proposals(image, detections, config):
    """Use image appearance/geometry only; never consume reference coordinates."""
    height, width = image.shape[:2]
    if [width, height] != config["native_size"]:
        raise ValueError(f"Uncalibrated image size: {width}x{height}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    proposals = []
    for detection in detections:
        row = dict(detection)
        x0, y0, x1, y1 = map(float, row["box"])
        w, h = x1 - x0, y1 - y0
        left, top = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        right, bottom = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        row["boundary_clipped"] = x0 < 0 or y0 < 0 or x1 > width or y1 > height
        size_ok = (config["min_box_side"] <= w <= config["max_box_side"] and
                   config["min_box_side"] <= h <= config["max_box_side"])
        reasons = [] if size_ok else ["implausible_terminal_size"]
        if right <= left or bottom <= top:
            row.update(size_ok=False, color_ok=False, accepted=False,
                       rejection_reasons=reasons + ["empty_crop"], features={})
            proposals.append(row)
            continue
        patch = hsv[top:bottom, left:right]
        saturation = patch[:, :, 1]
        green = ((patch[:, :, 0] >= config["green_hue_min"]) &
                 (patch[:, :, 0] <= config["green_hue_max"]) &
                 (saturation >= config["saturation_threshold"]) &
                 (patch[:, :, 2] >= config["green_min_value"]))
        green_fraction = float(green.mean())
        saturated_fraction = float((saturation >= config["saturation_threshold"]).mean())
        if green_fraction > config["max_green_fraction"]:
            reasons.append("green_gripper_appearance")
        if saturated_fraction > config["max_saturated_fraction"]:
            reasons.append("non_silver_saturated_appearance")
        color_ok = (green_fraction <= config["max_green_fraction"] and
                    saturated_fraction <= config["max_saturated_fraction"])

        pad = config["background_ring_px"]
        a, b = max(0, left-pad), max(0, top-pad)
        c, d = min(width, right+pad), min(height, bottom+pad)
        ring = np.ones((d-b, c-a), dtype=bool)
        ring[top-b:bottom-b, left-a:right-a] = False
        ring &= hsv[b:d, a:c, 1] < config["saturation_threshold"]
        background = lightness[b:d, a:c][ring]
        background_l = float(np.percentile(background, 75)) if background.size else None
        darker_fraction = None if background_l is None else float(
            (lightness[top:bottom, left:right].astype(float) <
             background_l-config["background_lightness_difference"]).mean())
        row.update(size_ok=size_ok, color_ok=color_ok, accepted=size_ok and color_ok,
                   rejection_reasons=reasons,
                   features=dict(width_px=w, height_px=h, green_fraction=green_fraction,
                                 saturated_fraction=saturated_fraction,
                                 local_background_lightness=background_l,
                                 darker_than_table_fraction=darker_fraction))
        proposals.append(row)
    return sorted(proposals, key=lambda row: row["score"], reverse=True)
