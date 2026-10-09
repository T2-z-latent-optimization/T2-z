"""Real T2-z backend interfaces; heavyweight checkpoint loading is mocked.

These tests require the dedicated model environment. They do not claim to
validate a trained checkpoint's predictions or accuracy.
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("diffsynth", reason="Run with T2-z/.venv/bin/python for model backend tests")

from wan_video_action.data import LoadCobotAction
from .. import TASKS
from ..core.model import WanPredictor
from ..core.types import Chunk
from .fixtures import config_for


@pytest.mark.parametrize("action_type", ["eef_delta", "action_joint"])
def test_real_action_normalizer_and_channel_packing(action_type):
    stats = {"min": [-0.5] * 7, "max": [0.5] * 7, "p01": [-0.25] * 7, "p99": [0.25] * 7}
    predictor = WanPredictor.__new__(WanPredictor)
    predictor.torch = torch
    predictor.pipe = SimpleNamespace(device="cpu", torch_dtype=torch.float32)
    predictor.action_loader = LoadCobotAction(action_type=action_type, stat={action_type: stats}, output_dim=14)
    raw = np.asarray([[-0.25] * 7, [0] * 7, [0.25] * 7, [5] * 7], dtype=np.float32)
    normalized = predictor.action_tensor(raw).numpy()[0]
    assert normalized.shape == (4, 14)
    active = slice(0, 7) if action_type == "action_joint" else slice(7, 14)
    inactive = slice(7, 14) if action_type == "action_joint" else slice(0, 7)
    np.testing.assert_allclose(normalized[:, active], np.asarray([[-1] * 7, [0] * 7, [1] * 7, [1] * 7]), atol=1e-6)
    assert np.all(normalized[:, inactive] == 0)
    with pytest.raises(ValueError, match="width"):
        predictor.action_tensor(np.zeros((4, 6)))
    with pytest.raises(ValueError, match="finite"):
        predictor.action_tensor(np.full((4, 7), np.nan))


@pytest.mark.parametrize("task", TASKS)
def test_original_inference_configuration_and_loader(task, tmp_path):
    config = config_for(task)
    model_dir = tmp_path / "base_model"
    model_dir.mkdir()
    base, checkpoint = tmp_path / "base.safetensors", tmp_path / "evaluation.safetensors"
    base.touch()
    checkpoint.touch()
    stats = tmp_path / "action_stats.json"
    entry = {"min": [-1] * 7, "max": [1] * 7}
    stats.write_text(json.dumps({"eef_delta": entry, "action_joint": entry}))
    configuration = {
        "method": "ours", "checkpoint": str(checkpoint),
        "inference_config": config["inference_config"],
        "overrides": {"model_paths": str(model_dir), "ckpt_path": str(base), "action_stat_path": str(stats)},
    }
    fake_pipe = SimpleNamespace(device="cpu", torch_dtype=torch.float32)
    with patch("t2z.inference.standard.build_pipeline", return_value=fake_pipe) as build, patch("t2z.inference.ours._freeze_pipe") as freeze, patch("wan_video_action.pipelines.wan_video_action.load_checkpoint_weights") as load:
        predictor = WanPredictor(configuration)
    assert int(predictor.args.num_frames) == Chunk(**config["chunk"]).length
    assert int(predictor.args.num_history_frames) == 1
    assert predictor.args.stage2_known_context_keys == ""
    assert predictor.args.video_light_augmentation_enabled is False
    assert predictor.action_loader.output_dim == 14
    assert predictor.action_loader.use_joint == (task == "light_switch")
    build.assert_called_once()
    load.assert_called_once_with(fake_pipe, str(checkpoint))
    freeze.assert_called_once_with(fake_pipe)


def test_backend_imports_without_changing_existing_modules():
    from t2z.inference import ours, standard
    from wan_video_action.pipelines.wan_video_action import build_wan_video_action_pipeline

    assert callable(ours._adapt_ttt_state)
    assert callable(standard.build_pipeline)
    assert callable(build_wan_video_action_pipeline)
