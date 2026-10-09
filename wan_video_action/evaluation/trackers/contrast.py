from __future__ import annotations
from dataclasses import dataclass
from ..tracking_types import TrackedMasks
import numpy as np

@dataclass(frozen=True)
class ContrastTrackerConfig:
    initial_x: float
    initial_y: float
    roi_x_min: int
    roi_x_max: int
    roi_y_min: int
    roi_y_max: int
    search_x_radius: int
    search_y_backward: int
    search_y_forward: int
    missing_search_y_forward: int
    outer_ring_margin: int
    minimum_initial_candidate_y: float
    minimum_contrast: float
    initial_contrast_ratio: float
    maximum_tracking_jump_px: float
    interpolation_grace_frames: int
    offscreen_min_y: float
    offscreen_velocity_min_y: float
    offscreen_velocity_threshold: float
    rectangle_half_sizes: tuple[tuple[int, int], ...]
    contrast_score_weight: float
    continuity_score_divisor: float
    lateral_score_divisor: float
    position_score_divisor: float
    movement_threshold: float
    recent_history_frames: int

def _integral_image(frame: np.ndarray) -> np.ndarray:
    return np.pad(frame.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)))

def _rectangle_mean(integral: np.ndarray, x: np.ndarray, y: np.ndarray, half_width: int, half_height: int) -> np.ndarray:
    (x0, x1) = (x - half_width, x + half_width + 1)
    (y0, y1) = (y - half_height, y + half_height + 1)
    area = float((2 * half_width + 1) * (2 * half_height + 1))
    return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / area

def _local_contrast(integral: np.ndarray, x: np.ndarray, y: np.ndarray, half_width: int, half_height: int, ring_margin: int) -> np.ndarray:
    inner = _rectangle_mean(integral, x, y, half_width, half_height)
    outer_half_width = half_width + ring_margin
    outer_half_height = half_height + ring_margin
    outer_area = float((2 * outer_half_width + 1) * (2 * outer_half_height + 1))
    inner_area = float((2 * half_width + 1) * (2 * half_height + 1))
    outer_sum = _rectangle_mean(integral, x, y, outer_half_width, outer_half_height) * outer_area
    ring = (outer_sum - inner * inner_area) / (outer_area - inner_area)
    return np.linalg.norm(inner - ring, axis=-1)

def _candidate_bounds(previous: np.ndarray, missing_count: int, height: int, width: int, config: ContrastTrackerConfig) -> tuple[np.ndarray, np.ndarray]:
    maximum_half_width = max((value[0] for value in config.rectangle_half_sizes))
    maximum_half_height = max((value[1] for value in config.rectangle_half_sizes))
    margin = config.outer_ring_margin
    x_min = max(config.roi_x_min, maximum_half_width + margin, int(round(previous[0])) - config.search_x_radius)
    x_max = min(config.roi_x_max, width - maximum_half_width - margin - 1, int(round(previous[0])) + config.search_x_radius)
    forward = config.missing_search_y_forward if missing_count else config.search_y_forward
    y_min = max(config.roi_y_min, maximum_half_height + margin, int(round(previous[1])) - config.search_y_backward)
    y_max = min(config.roi_y_max, height - maximum_half_height - margin - 1, int(round(previous[1])) + forward)
    if x_min > x_max or y_min > y_max:
        return (np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=int))
    return np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))

def _best_candidate(frame: np.ndarray, previous: np.ndarray, moved: bool, missing_count: int, config: ContrastTrackerConfig) -> tuple[float, np.ndarray, int, int] | None:
    (height, width) = frame.shape[:2]
    (x, y) = _candidate_bounds(previous, missing_count, height, width, config)
    if x.size == 0:
        return None
    integral = _integral_image(frame)
    distance = np.hypot(x - previous[0], y - previous[1])
    best = None
    for (half_width, half_height) in config.rectangle_half_sizes:
        contrast = _local_contrast(integral, x, y, half_width, half_height, config.outer_ring_margin)
        score = config.contrast_score_weight * contrast - distance / config.continuity_score_divisor - np.abs(x - config.initial_x) / config.lateral_score_divisor + y / config.position_score_divisor
        if not moved:
            score = np.where(y >= config.minimum_initial_candidate_y, score, -np.inf)
        location = np.unravel_index(int(np.argmax(score)), score.shape)
        candidate = (float(contrast[location]), np.array([float(x[location]), float(y[location])]), half_width, half_height, float(score[location]))
        if best is None or candidate[-1] > best[-1]:
            best = candidate
    if best is None:
        return None
    return (best[0], best[1], best[2], best[3])

def track_contrast_object(frames: np.ndarray, config: ContrastTrackerConfig) -> TrackedMasks:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f'Expected RGB frames [T,H,W,3], got {frames.shape}.')
    if len(frames) == 0:
        raise ValueError('Cannot track an empty video.')
    values = frames.astype(np.float64) / 255.0
    (height, width) = values.shape[1:3]
    initial_x = np.array([[int(round(config.initial_x))]])
    initial_y = np.array([[int(round(config.initial_y))]])
    initial_integral = _integral_image(values[0])
    initial_contrast = max((float(_local_contrast(initial_integral, initial_x, initial_y, half_width, half_height, config.outer_ring_margin)[0, 0]) for (half_width, half_height) in config.rectangle_half_sizes))
    contrast_threshold = max(config.minimum_contrast, initial_contrast * config.initial_contrast_ratio)
    masks = np.zeros((len(frames), height, width), dtype=bool)
    visible = np.zeros(len(frames), dtype=bool)
    offscreen = np.zeros(len(frames), dtype=bool)
    centers = np.full((len(frames), 2), np.nan, dtype=np.float64)
    previous = np.array([config.initial_x, config.initial_y], dtype=np.float64)
    centers[0] = previous
    visible[0] = True
    masks[0, int(round(previous[1])), int(round(previous[0]))] = True
    recent_centers = [previous.copy()]
    moved = False
    exited = False
    missing_count = 0
    for frame_index in range(1, len(frames)):
        candidate = None if exited else _best_candidate(values[frame_index], previous, moved, missing_count, config)
        if candidate is not None:
            (contrast, center, half_width, half_height) = candidate
            jump = float(np.linalg.norm(center - previous))
            if contrast >= contrast_threshold and jump <= config.maximum_tracking_jump_px:
                (center_x, center_y) = np.rint(center).astype(int)
                masks[frame_index, max(0, center_y - half_height):min(height, center_y + half_height + 1), max(0, center_x - half_width):min(width, center_x + half_width + 1)] = True
                visible[frame_index] = True
                centers[frame_index] = center
                moved = moved or float(np.linalg.norm(center - np.array([config.initial_x, config.initial_y]))) > config.movement_threshold
                previous = center
                recent_centers.append(center.copy())
                recent_centers = recent_centers[-config.recent_history_frames:]
                missing_count = 0
                continue
        missing_count += 1
        downward_velocity = 0.0
        if len(recent_centers) >= 2:
            downward_velocity = float(recent_centers[-1][1] - recent_centers[0][1]) / float(len(recent_centers) - 1)
        if moved and (previous[1] >= config.offscreen_min_y or (previous[1] >= config.offscreen_velocity_min_y and downward_velocity > config.offscreen_velocity_threshold)):
            exited = True
        if exited:
            offscreen[frame_index] = True
            centers[frame_index] = previous
        elif missing_count <= config.interpolation_grace_frames:
            centers[frame_index] = previous
        if np.isfinite(centers[frame_index]).all():
            held_x = int(np.clip(round(previous[0]), 0, width - 1))
            held_y = int(np.clip(round(previous[1]), 0, height - 1))
            masks[frame_index, held_y, held_x] = True
    offscreen_indices = np.flatnonzero(offscreen)
    first_offscreen = int(offscreen_indices[0]) if len(offscreen_indices) else None
    return TrackedMasks(masks=masks, visible=visible, offscreen=offscreen, centers=centers, first_offscreen_frame=first_offscreen)
