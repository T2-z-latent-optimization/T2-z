from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wan_video_action.evaluation.sim_action_selection.core.model import WanPredictor


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_online_video_normalization_matches_dataset(dtype):
    rgb = np.resize(np.arange(256, dtype=np.uint8), (3, 2, 16, 16, 3))
    original = rgb.copy()
    predictor = WanPredictor.__new__(WanPredictor)
    predictor.torch = torch
    predictor.pipe = SimpleNamespace(device="cpu", torch_dtype=dtype)
    predictor.args = SimpleNamespace(height=16, width=16)
    expected = torch.from_numpy(rgb.astype(np.float32)).permute(1, 4, 0, 2, 3)
    expected = (expected * (2.0 / 255.0) - 1.0).to(dtype=dtype)
    actual = predictor.video_tensor(rgb)
    assert torch.equal(actual, expected)
    assert np.array_equal(rgb, original)
    assert actual.shape == (2, 3, 3, 16, 16)
