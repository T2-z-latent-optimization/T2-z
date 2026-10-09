from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoPair:
    gt_video_path: str = ''
    pred_video_path: str = ''
    gt_mask_path: str | None = None
    pred_mask_path: str | None = None
    gt_state_path: str | None = None
    pred_state_path: str | None = None
    gt_start_frame: int = 0
    pred_start_frame: int = 0
    num_frames: int | None = None
    gt_frame_stride: int = 1
    pred_frame_stride: int = 1
    gt_state_start_frame: int | None = None
    pred_state_start_frame: int | None = None
    object_index: int = 0
    task: str | None = None
    sample_id: str = ''
    environment_id: str = ''
    method: str = ''
    split: str = ''
    support_size: int = 0
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
