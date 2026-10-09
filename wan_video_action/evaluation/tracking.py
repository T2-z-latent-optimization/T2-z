from pathlib import Path

from .config import load_evaluation_config
from .tracking_types import TrackedMasks


def track_frames(frames, config) -> TrackedMasks:
    if isinstance(config, (str, Path)):
        config = load_evaluation_config(config)
    method = config['method']
    parameters = dict(config['parameters'])
    if method == 'color':
        from .trackers.color import ColorTrackerConfig, track_color_object
        return track_color_object(frames, ColorTrackerConfig(**parameters))
    if method == 'contrast':
        from .trackers.contrast import ContrastTrackerConfig, track_contrast_object
        return track_contrast_object(frames, ContrastTrackerConfig(**parameters))
    if method == 'appearance':
        from .trackers.appearance import AppearanceTrackerConfig, track_appearance_object
        return track_appearance_object(frames, AppearanceTrackerConfig(**parameters))
    raise ValueError(f'Unknown tracking method: {method!r}')
