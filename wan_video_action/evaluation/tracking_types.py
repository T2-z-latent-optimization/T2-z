from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrackedMasks:
    masks: np.ndarray
    visible: np.ndarray
    offscreen: np.ndarray
    centers: np.ndarray
    first_offscreen_frame: int | None
