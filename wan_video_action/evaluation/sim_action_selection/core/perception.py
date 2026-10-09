from __future__ import annotations

from dataclasses import replace

import numpy as np

from ...adapters.simulation import extract_sim_task_state
from ...task_state import TaskState, states_from_masks
from ...tracking import track_frames


def extract_state(task, views, fps, config):
    """Both prediction and execution use RGB only, never privileged poses."""
    views = np.asarray(views, dtype=np.uint8)
    if views.ndim != 5:
        raise ValueError("Expected [time, views, height, width, RGB]")
    main = np.ascontiguousarray(views[:, 0])
    if task in {"friction", "multi_background"}:
        tracked = track_frames(main, config)
        state = states_from_masks(tracked.masks, fps=fps)
        centers = np.asarray(tracked.centers, dtype=np.float64)
        if centers.ndim == 2:
            centers = centers[:, None, :]
        visible = np.asarray(tracked.visible)
        if visible.ndim == 1:
            visible = visible[:, None]
        return replace(
            state, centroids=centers, visible=visible,
            events={"offscreen": np.asarray(tracked.offscreen)},
        )
    return extract_sim_task_state(task, main, fps=fps, main_view_width=main.shape[2], **config)


def terminal_value(state: TaskState, rules):
    index = int(rules.get("object_index", 0))
    count = int(rules.get("terminal_window", 5))
    kind = rules["outcome_kind"]
    if kind == "point":
        values = np.asarray(state.centroids[-count:, index], dtype=np.float64)
        values = values[np.all(np.isfinite(values), axis=1)]
        if not len(values):
            return None
        return (np.median(values, axis=0) / [state.image_width, state.image_height]).tolist()
    if kind == "scalar":
        values = np.degrees(state.angles_rad[-count:, index])
    else:
        values = state.light_score[-count:, index]
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    result = float(np.median(values))
    return result if kind == "scalar" else bool(result >= rules["light_threshold"])
