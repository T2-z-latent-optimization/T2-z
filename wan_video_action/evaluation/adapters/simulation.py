"""RGB task-state extractors for the current simulation benchmarks.

All extractors operate on the left/main camera in the horizontally concatenated
Wan rollout. Coordinates remain in that camera's image coordinate system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ..task_state import TaskState


DEFAULT_MAIN_VIEW_WIDTH = 224
DEFAULT_LIGHT_ROI = (98, 108, 151, 166)
DEFAULT_YELLOW_THRESHOLD = 0.35
DEFAULT_MAX_TRACKING_JUMP_PX = 64.0


def crop_main_view(frames: np.ndarray, width: int = DEFAULT_MAIN_VIEW_WIDTH) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB video [T,H,W,C], got {array.shape}.")
    if width <= 0 or width > array.shape[2]:
        raise ValueError(f"Invalid main-view width {width} for video width {array.shape[2]}.")
    return array[:, :, :width, :3]


def _colour_mask(frame: np.ndarray, colour: str) -> np.ndarray:
    rgb = frame.astype(np.float64)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if colour == "blue":
        return (
            (blue > 90)
            & (blue > 1.22 * red)
            & (blue > 1.10 * green)
            & ((blue - red) > 28)
        )
    if colour == "red":
        return (
            (red > 80)
            & (red > 1.32 * green)
            & (red > 1.35 * blue)
            & ((red - green) > 25)
        )
    if colour == "bar_brown":
        return (
            (red > 55)
            & (red < 180)
            & (green > 20)
            & (green < 125)
            & (blue < 75)
            & (red > 1.20 * green)
            & (green > 1.15 * blue)
        )
    raise ValueError(f"Unknown colour mask {colour!r}.")


def _components(
    mask: np.ndarray, *, min_area: int, max_area: int
) -> list[dict[str, np.ndarray | float | int | tuple[int, int, int, int]]]:
    labels, count = ndimage.label(mask)
    output = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        area = len(xs)
        if not min_area <= area <= max_area:
            continue
        output.append(
            {
                "area": area,
                "center": np.asarray([xs.mean(), ys.mean()], dtype=np.float64),
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                "xs": xs,
                "ys": ys,
            }
        )
    return output


def _nearest_exit_side(
    center: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    edge_margin: int,
) -> tuple[str, bool] | None:
    x0, y0, x1, y1 = bbox
    candidates = [
        (float(center[0]), "left", x0 <= 0),
        (float(width - 1 - center[0]), "right", x1 >= width - 1),
        (float(center[1]), "top", y0 <= 0),
        (float(height - 1 - center[1]), "bottom", y1 >= height - 1),
    ]
    distance, side, touches = min(candidates, key=lambda item: item[0])
    return (side, touches) if touches or distance <= edge_margin else None


def _track_colour(
    frames: np.ndarray,
    colour: str,
    *,
    min_area: int,
    max_area: int,
    edge_margin: int,
    max_tracking_jump_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    frame_count, height, width = frames.shape[:3]
    centroids = np.full((frame_count, 2), np.nan, dtype=np.float64)
    visible = np.zeros(frame_count, dtype=bool)
    selected: list[dict | None] = []
    previous: np.ndarray | None = None

    for frame in frames:
        components = _components(
            _colour_mask(frame, colour), min_area=min_area, max_area=max_area
        )
        if not components:
            selected.append(None)
            continue
        if previous is None:
            choice = max(components, key=lambda item: int(item["area"]))
        else:
            choice = min(
                components,
                key=lambda item: float(np.linalg.norm(item["center"] - previous)),
            )
            if float(np.linalg.norm(choice["center"] - previous)) > max_tracking_jump_px:


                selected.append(None)
                continue
        previous = np.asarray(choice["center"], dtype=np.float64)
        selected.append(choice)
        index = len(selected) - 1
        centroids[index] = previous
        visible[index] = True

    events = {
        "offscreen": np.zeros(frame_count, dtype=bool),
        "exit_left": np.zeros(frame_count, dtype=bool),
        "exit_right": np.zeros(frame_count, dtype=bool),
        "exit_top": np.zeros(frame_count, dtype=bool),
        "exit_bottom": np.zeros(frame_count, dtype=bool),
    }
    visible_indices = np.flatnonzero(visible)
    if len(visible_indices):
        last = int(visible_indices[-1])
        choice = selected[last]
        exit_info = None if choice is None else _nearest_exit_side(
            np.asarray(choice["center"]),
            choice["bbox"],
            width=width,
            height=height,
            edge_margin=edge_margin,
        )
        if exit_info is not None and last < frame_count - 1:
            side, _ = exit_info
            start = last + 1



            centroids[start:] = np.asarray(choice["center"], dtype=np.float64)
            events["offscreen"][start:] = True
            events[f"exit_{side}"][start:] = True
    return centroids, visible, events


def extract_coloured_object_state(
    frames: np.ndarray,
    colours: Iterable[str],
    *,
    fps: float = 1.0,
    main_view_width: int = DEFAULT_MAIN_VIEW_WIDTH,
    min_area: int = 8,
    max_area: int = 3000,
    edge_margin: int = 16,
    max_tracking_jump_px: float = DEFAULT_MAX_TRACKING_JUMP_PX,
) -> TaskState:
    main = crop_main_view(frames, main_view_width)
    tracks = [
        _track_colour(
            main,
            colour,
            min_area=min_area,
            max_area=max_area,
            edge_margin=edge_margin,
            max_tracking_jump_px=max_tracking_jump_px,
        )
        for colour in colours
    ]
    centroids = np.stack([track[0] for track in tracks], axis=1)
    visible = np.stack([track[1] for track in tracks], axis=1)
    event_names = tracks[0][2].keys()
    events = {
        name: np.stack([track[2][name] for track in tracks], axis=1)
        for name in event_names
    }
    return TaskState(
        centroids=centroids,
        visible=visible,
        image_height=main.shape[1],
        image_width=main.shape[2],
        fps=float(fps),
        events=events,
    )


def extract_mass_balance_state(
    frames: np.ndarray,
    *,
    fps: float = 1.0,
    main_view_width: int = DEFAULT_MAIN_VIEW_WIDTH,
    min_area: int = 20,
    max_area: int = 3000,
) -> TaskState:
    main = crop_main_view(frames, main_view_width)
    count, height, width = main.shape[:3]
    centroids = np.full((count, 1, 2), np.nan, dtype=np.float64)
    angles = np.full((count, 1), np.nan, dtype=np.float64)
    visible = np.zeros((count, 1), dtype=bool)
    for index, frame in enumerate(main):
        components = _components(
            _colour_mask(frame, "bar_brown"), min_area=min_area, max_area=max_area
        )
        if not components:
            continue
        component = max(components, key=lambda item: int(item["area"]))
        points = np.column_stack((component["xs"], component["ys"])).astype(np.float64)
        center = points.mean(axis=0)
        covariance = np.cov(points - center, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if axis[0] < 0:
            axis = -axis
        centroids[index, 0] = center
        angles[index, 0] = np.arctan2(axis[1], axis[0])
        visible[index, 0] = True
    return TaskState(
        centroids=centroids,
        visible=visible,
        angles_rad=angles,
        image_height=height,
        image_width=width,
        fps=float(fps),
    )


def extract_lightswitch_state(
    frames: np.ndarray,
    *,
    fps: float = 1.0,
    main_view_width: int = DEFAULT_MAIN_VIEW_WIDTH,
    roi: tuple[int, int, int, int] = DEFAULT_LIGHT_ROI,
    yellow_threshold: float = DEFAULT_YELLOW_THRESHOLD,
) -> TaskState:
    main = crop_main_view(frames, main_view_width)
    x0, y0, x1, y1 = roi
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid LightSwitch ROI {roi}.")
    if x1 > main.shape[2] or y1 > main.shape[1]:
        raise ValueError(f"LightSwitch ROI {roi} lies outside {main.shape[2]}x{main.shape[1]} view.")
    rgb = main[:, y0:y1, x0:x1].astype(np.float64) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    yellow = (
        (red > 0.45)
        & (green > 0.38)
        & (blue < 0.38)
        & (red > 1.5 * blue)
        & (green > 1.4 * blue)
    )
    score = np.mean(yellow, axis=(1, 2))
    return TaskState(
        light_on=(score >= yellow_threshold)[:, None],
        light_score=score[:, None],
        image_height=main.shape[1],
        image_width=main.shape[2],
        fps=float(fps),
    )


def extract_sim_task_state(
    task: str,
    frames: np.ndarray,
    *,
    fps: float = 1.0,
    main_view_width: int = DEFAULT_MAIN_VIEW_WIDTH,
    min_area: int = 8,
    max_area: int = 3000,
    edge_margin: int = 16,
    max_tracking_jump_px: float = DEFAULT_MAX_TRACKING_JUMP_PX,
    light_roi: tuple[int, int, int, int] = DEFAULT_LIGHT_ROI,
    yellow_threshold: float = DEFAULT_YELLOW_THRESHOLD,
) -> TaskState:
    name = task.strip().lower().replace("-", "_")
    if name == "gravity":
        return extract_coloured_object_state(
            frames,
            ("blue",),
            fps=fps,
            main_view_width=main_view_width,
            min_area=min_area,
            max_area=max_area,
            edge_margin=edge_margin,
            max_tracking_jump_px=max_tracking_jump_px,
        )
    if name in {"mass_collision", "collision", "mass_friction", "joint_mass_friction"}:

        return extract_coloured_object_state(
            frames,
            ("red", "blue"),
            fps=fps,
            main_view_width=main_view_width,
            min_area=min_area,
            max_area=max_area,
            edge_margin=edge_margin,
            max_tracking_jump_px=max_tracking_jump_px,
        )
    if name in {"mass_balance", "balance"}:
        return extract_mass_balance_state(
            frames,
            fps=fps,
            main_view_width=main_view_width,
            min_area=max(20, min_area),
            max_area=max_area,
        )
    if name in {"lightswitch", "light_switch"}:
        return extract_lightswitch_state(
            frames,
            fps=fps,
            main_view_width=main_view_width,
            roi=light_roi,
            yellow_threshold=yellow_threshold,
        )
    raise ValueError(f"No sim_rgb_v1 extractor for task {task!r}.")


def render_task_state_audit(
    frames: np.ndarray,
    state: TaskState,
    output_path: str | Path,
    *,
    task: str,
    main_view_width: int = DEFAULT_MAIN_VIEW_WIDTH,
    light_roi: tuple[int, int, int, int] = DEFAULT_LIGHT_ROI,
) -> None:
    main = crop_main_view(frames, main_view_width)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        destination,
        format="ffmpeg",
        fps=max(float(state.fps), 1.0),
        codec="libx264",
        quality=8,
    )
    try:
        for index, frame in enumerate(main):
            image = Image.fromarray(frame).convert("RGB")
            draw = ImageDraw.Draw(image)
            if state.centroids is not None:
                for object_index, center in enumerate(state.centroids[index]):
                    if not np.all(np.isfinite(center)):
                        continue
                    x, y = (float(center[0]), float(center[1]))
                    offscreen = bool(
                        state.events
                        and "offscreen" in state.events
                        and state.events["offscreen"][index, object_index]
                    )
                    colour = "red" if offscreen else ("cyan" if object_index == 0 else "yellow")
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=colour, width=2)
                    draw.text((x + 6, y - 10), f"obj{object_index}", fill=colour)
            if state.angles_rad is not None and np.isfinite(state.angles_rad[index, 0]):
                center = state.centroids[index, 0]
                angle = float(state.angles_rad[index, 0])
                axis = np.asarray([np.cos(angle), np.sin(angle)])
                start, stop = center - 60 * axis, center + 60 * axis
                draw.line((tuple(start), tuple(stop)), fill="magenta", width=3)
                draw.text((4, 4), f"tilt={np.degrees(angle):+.2f} deg", fill="magenta")
            if state.light_score is not None:
                draw.rectangle(light_roi, outline="yellow", width=2)
                draw.text(
                    (4, 4),
                    f"yellow={state.light_score[index, 0]:.3f} on={bool(state.light_on[index, 0])}",
                    fill="yellow",
                )
            writer.append_data(np.asarray(image))
    finally:
        writer.close()
