from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import imageio
import numpy as np
import torch

from ..core.model import WanPredictor
from ..core.methods import OnlineMethod
from ..core.prediction_video import roundtrip_prediction_video
from wan_video_action.utils import save_video


def decode_views(path, view_count, width):
    reader = imageio.get_reader(str(path))
    try:
        packed = np.stack([np.asarray(frame, dtype=np.uint8) for frame in reader])
    finally:
        reader.close()
    return np.stack([packed[:, :, view * width:(view + 1) * width] for view in range(view_count)], axis=1)


def make_predictor(roundtrip=False):
    class Pipe:
        def __call__(self, **kwargs):
            return torch.zeros((2, 3, 5, 32, 48), dtype=torch.bfloat16)

    model = WanPredictor.__new__(WanPredictor)
    model.torch = torch
    model.pipe = Pipe()
    model.method = "ours"
    model.online_method = OnlineMethod.__new__(OnlineMethod)
    model.online_method.method = model.method
    model.context = torch.zeros((1, 32), dtype=torch.float32)
    model.args = SimpleNamespace(
        num_frames=5, height=32, width=48, cfg_scale=1.0,
        num_inference_steps=1, fps=20, quality=6,
    )
    if roundtrip:
        model.args.prediction_video_roundtrip = True
    model.video_tensor = lambda values: (
        torch.from_numpy(values).permute(1, 4, 0, 2, 3).float().mul(2.0 / 255.0).sub(1.0).to(torch.bfloat16)
    )
    model.action_tensor = lambda values: torch.zeros((1, 5, 14), dtype=torch.bfloat16)
    return model


class PredictionVideoTests(unittest.TestCase):
    def test_roundtrip_matches_existing_float_video_export(self):
        video = torch.linspace(-1, 1, 2 * 3 * 5 * 32 * 48).reshape(2, 3, 5, 32, 48)
        frames = np.clip((video.permute(2, 0, 3, 4, 1).numpy() + 1.0) * 127.5, 0, 255).astype(np.uint8)
        actual = roundtrip_prediction_video(frames, fps=20, quality=6)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.mp4"
            save_video(video, str(path), fps=20, quality=6)
            expected = decode_views(path, 2, 48)
        self.assertTrue(np.array_equal(actual, expected))

    def test_default_predict_does_not_encode_or_change_current_frame(self):
        model = make_predictor()
        views = np.full((1, 2, 32, 48, 3), 128, dtype=np.uint8)
        with patch("wan_video_action.evaluation.sim_action_selection.core.prediction_video.roundtrip_prediction_video") as encode:
            actual = model.predict({"views": views, "actions": np.zeros((5, 14)), "seed": 17})
        encode.assert_not_called()
        self.assertTrue(np.array_equal(actual[0], views[0]))

    def test_predict_roundtrip_preserves_current_and_matches_exported_future(self):
        model = make_predictor(roundtrip=True)
        views = np.full((1, 2, 32, 48, 3), 128, dtype=np.uint8)
        actual = model.predict({"views": views, "actions": np.zeros((5, 14)), "seed": 17})
        expected_video = model.pipe()
        expected_video[:, :, 0] = model.video_tensor(views)[:, :, 0]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.mp4"
            save_video(expected_video, str(path), fps=20, quality=6)
            expected = decode_views(path, 2, 48)
        self.assertTrue(np.array_equal(actual[0], views[0]))
        self.assertTrue(np.array_equal(actual[1:], expected[1:]))

    def test_roundtrip_does_not_mix_views(self):
        frames = np.zeros((5, 2, 32, 48, 3), dtype=np.uint8)
        frames[:, 0] = [220, 20, 30]
        frames[:, 1] = [20, 30, 220]
        result = roundtrip_prediction_video(frames, fps=20, quality=6)
        self.assertEqual(result.shape, frames.shape)
        self.assertGreater(float(result[:, 0, :, :, 0].mean()), float(result[:, 0, :, :, 2].mean()))
        self.assertGreater(float(result[:, 1, :, :, 2].mean()), float(result[:, 1, :, :, 0].mean()))


if __name__ == "__main__":
    unittest.main()
