"""I/O helpers for evaluation artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:                                                          
    cv2 = None


def read_video_frames(
    path: str | Path,
    start_frame: int = 0,
    num_frames: int | None = None,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    if cv2 is not None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        try:
            while num_frames is None or len(frames) < num_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    else:
        import imageio.v2 as imageio

        last_error: Exception | None = None
        for attempt in range(3):
            frames.clear()
            reader = None
            try:
                reader = imageio.get_reader(str(path), "ffmpeg")
                stop = None if num_frames is None else start_frame + num_frames
                for index, frame in enumerate(reader):
                    if index < start_frame:
                        continue
                    if stop is not None and index >= stop:
                        break
                    frames.append(np.asarray(frame))
                last_error = None
                break
            except (OSError, RuntimeError, ValueError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(2**attempt)
            finally:
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass
        if last_error is not None:
            raise OSError(
                f"Unable to decode video after 3 attempts: {path}"
            ) from last_error
    if not frames:
        raise ValueError(f"No frames read from {path} at offset {start_frame}.")
    return np.stack(frames)


def read_mask_array(path: str | Path, key: str = "masks") -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            if key not in payload:
                raise KeyError(f"Mask archive {path} does not contain key {key!r}.")
            array = payload[key]
        finally:
            payload.close()
        return array
    return payload


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_jsonl_atomic(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
