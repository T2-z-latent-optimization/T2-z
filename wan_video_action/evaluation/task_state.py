"""Frozen task-state artifacts used by task-specific evaluators."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TaskState:
    centroids: np.ndarray | None = None
    visible: np.ndarray | None = None
    angles_rad: np.ndarray | None = None
    light_on: np.ndarray | None = None
    light_score: np.ndarray | None = None
    image_height: int | None = None
    image_width: int | None = None
    fps: float = 1.0
    events: dict[str, np.ndarray] | None = None

    @property
    def frame_count(self) -> int:
        arrays = (
            self.centroids,
            self.visible,
            self.angles_rad,
            self.light_on,
            self.light_score,
        )
        counts = [len(value) for value in arrays if value is not None]
        if self.events:
            counts.extend(len(value) for value in self.events.values())
        if not counts:
            raise ValueError("Task state contains no temporal arrays.")
        if len(set(counts)) != 1:
            raise ValueError(f"Task-state arrays have inconsistent lengths: {counts}.")
        return counts[0]

    def sliced(self, start: int, count: int | None = None) -> "TaskState":
        stop = None if count is None else start + count

        def take(value: np.ndarray | None) -> np.ndarray | None:
            return None if value is None else value[start:stop]

        return replace(
            self,
            centroids=take(self.centroids),
            visible=take(self.visible),
            angles_rad=take(self.angles_rad),
            light_on=take(self.light_on),
            light_score=take(self.light_score),
            events=(
                {key: value[start:stop] for key, value in self.events.items()}
                if self.events else None
            ),
        )


def _as_tracks(value: np.ndarray, *, trailing: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if trailing is not None and array.shape[-1] != trailing:
        raise ValueError(f"Expected final dimension {trailing}, got {array.shape}.")
    if array.ndim == 2 and trailing is not None:
        array = array[:, None, :]
    elif array.ndim == 1 and trailing is None:
        array = array[:, None]
    if array.ndim != (3 if trailing is not None else 2):
        raise ValueError(f"Invalid task-state track shape: {array.shape}.")
    return array


def states_from_masks(masks: np.ndarray, *, fps: float = 1.0) -> TaskState:
    """Extract object centroids and unoriented principal-axis angles from masks."""
    array = np.asarray(masks).astype(bool)
    if array.ndim == 3:
        array = array[:, None]
    if array.ndim != 4:
        raise ValueError(f"Masks must be [T,H,W] or [T,N,H,W], got {array.shape}.")
    frames, objects, height, width = array.shape
    centroids = np.full((frames, objects, 2), np.nan, dtype=np.float64)
    angles = np.full((frames, objects), np.nan, dtype=np.float64)
    visible = np.zeros((frames, objects), dtype=bool)
    for frame_index in range(frames):
        for object_index in range(objects):
            ys, xs = np.nonzero(array[frame_index, object_index])
            if len(xs) == 0:
                continue
            visible[frame_index, object_index] = True
            center = np.array([xs.mean(), ys.mean()], dtype=np.float64)
            centroids[frame_index, object_index] = center
            if len(xs) >= 2:
                points = np.column_stack((xs, ys)).astype(np.float64) - center
                covariance = points.T @ points / float(len(points))
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                axis = eigenvectors[:, int(np.argmax(eigenvalues))]
                angles[frame_index, object_index] = np.arctan2(axis[1], axis[0])
    return TaskState(
        centroids=centroids,
        visible=visible,
        angles_rad=angles,
        image_height=height,
        image_width=width,
        fps=float(fps),
    )


def load_task_state(path: str | Path, *, mask_key: str = "masks") -> TaskState:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        keys = set(payload.files)
        if "centroids" not in keys and mask_key in keys:
            fps = float(np.asarray(payload["fps"]).reshape(-1)[0]) if "fps" in keys else 1.0
            state = states_from_masks(payload[mask_key], fps=fps)
        else:
            centroids = (
                _as_tracks(payload["centroids"], trailing=2).astype(np.float64)
                if "centroids" in keys else None
            )
            visible = (
                _as_tracks(payload["visible"]).astype(bool)
                if "visible" in keys else (
                    np.all(np.isfinite(centroids), axis=-1)
                    if centroids is not None else None
                )
            )
            angles = (
                _as_tracks(payload["angles_rad"]).astype(np.float64)
                if "angles_rad" in keys else None
            )
            light_on = (
                _as_tracks(payload["light_on"]).astype(bool)
                if "light_on" in keys else None
            )
            light_score = (
                _as_tracks(payload["light_score"]).astype(np.float64)
                if "light_score" in keys else None
            )
            scalar = lambda key, default: (
                np.asarray(payload[key]).reshape(-1)[0] if key in keys else default
            )
            state = TaskState(
                centroids=centroids,
                visible=visible,
                angles_rad=angles,
                light_on=light_on,
                light_score=light_score,
                image_height=int(scalar("image_height", 0)) or None,
                image_width=int(scalar("image_width", 0)) or None,
                fps=float(scalar("fps", 1.0)),
            )
        events = {
            key.removeprefix("event_"): np.asarray(payload[key]).astype(bool)
            for key in keys if key.startswith("event_")
        }
    return replace(state, events=events or None)


def save_task_state(path: str | Path, state: TaskState) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "fps": np.asarray(state.fps, dtype=np.float64),
    }
    for name in ("centroids", "visible", "angles_rad", "light_on", "light_score"):
        value = getattr(state, name)
        if value is not None:
            payload[name] = np.asarray(value)
    if state.image_height is not None:
        payload["image_height"] = np.asarray(state.image_height, dtype=np.int64)
    if state.image_width is not None:
        payload["image_width"] = np.asarray(state.image_width, dtype=np.int64)
    if state.events:
        payload.update({f"event_{key}": value for key, value in state.events.items()})
    np.savez_compressed(destination, **payload)
