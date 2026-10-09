from pathlib import Path
from tempfile import TemporaryDirectory

import imageio
import numpy as np


def roundtrip_prediction_video(frames, fps, quality):
    frames = np.asarray(frames)
    if frames.ndim != 5 or frames.shape[-1] != 3 or min(frames.shape[:4]) < 1:
        raise ValueError("Expected nonempty [time, views, height, width, RGB] frames")
    if frames.dtype != np.uint8:
        raise TypeError("Prediction video encoding requires uint8 RGB")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Prediction video FPS must be positive and finite")
    count, views, height, width, _ = frames.shape
    with TemporaryDirectory(prefix="t2z_prediction_video_") as directory:
        path = Path(directory) / "prediction.mp4"
        with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=quality) as writer:
            for frame in frames:
                writer.append_data(np.concatenate(frame, axis=1))
        reader = imageio.get_reader(str(path))
        try:
            packed = np.stack([np.asarray(frame, dtype=np.uint8) for frame in reader])
        finally:
            reader.close()
        if packed.shape != (count, height, width * views, 3):
            raise ValueError("Prediction video encoding changed frame count or image dimensions")
        return np.stack([packed[:, :, view * width:(view + 1) * width] for view in range(views)], axis=1)
