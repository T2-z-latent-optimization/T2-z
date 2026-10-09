from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch
import os
import json

import numpy as np
import pytest
import torch

from .. import TASKS
from ..core.model import WanPredictor
from ..core.methods import OnlineMethod
from ..core.artifacts import jsonable
from ..core.perception import terminal_value
from ..core.rpc import read_message, write_message
from ..core.runner import Runner
from ..core.selection import load_rules, select_action
from ..core.simulator import NativeSimulator
from ..core.types import Candidate, Chunk
from ...config import load_evaluation_config
from .fixtures import config_for, resources_for


@pytest.mark.parametrize("task", TASKS)
def test_chunk_contract(task):
    chunk = Chunk(**config_for(task)["chunk"])
    assert chunk.length == (61 if task in {"gravity", "mass_collision", "mass_friction"} else 33 if task == "light_switch" else 41)
    assert chunk.indices[0] == chunk.start and chunk.indices[-1] == chunk.end


@pytest.mark.parametrize("arguments", [(-1, 10, 1), (2, 1, 1), (0, 4, 0), (0, 5, 2)])
def test_invalid_chunk(arguments):
    with pytest.raises(ValueError):
        Chunk(*arguments)


def test_rpc_roundtrip():
    incoming, outgoing = os.pipe()
    try:
        with os.fdopen(outgoing, "wb", buffering=0) as writer:
            write_message(writer, {"array": np.arange(12).reshape(3, 4), "message": "ok"})
        with os.fdopen(incoming, "rb", buffering=0) as reader:
            result = read_message(reader)
        np.testing.assert_array_equal(result["array"], np.arange(12).reshape(3, 4))
    finally:
        for descriptor in (incoming, outgoing):
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize("task", TASKS)
def test_selection_rejects_oracle_and_duplicates(task):
    rules = load_rules(task)
    target = rules["targets"][0]
    with pytest.raises(ValueError, match="Ground-truth"):
        select_action([{"action_index": 0, "ground_truth_value": 1}], target, rules)
    with pytest.raises(ValueError, match="Duplicate"):
        select_action([{"action_index": 0}, {"action_index": 0}], target, rules)


class CountingSimulator(NativeSimulator):
    def __init__(self, start=2, end=6, stride=2, last=6):
        self.chunk = Chunk(start, end, stride)
        self.frame_index = 0
        self.config = {"native_last_frame": last}
        self.fps, self.case, self.bank = 20, {}, []
        self.sources = SimpleNamespace(provenance={})
        self.renders = 0

    def initialize(self):
        pass

    def current(self):
        self.renders += 1
        return {"views": np.full((2, 8, 8, 3), self.renders, dtype=np.uint8)}

    def step_native(self):
        return {"model_action": np.array([self.frame_index, 0]), "native_action": np.array([self.frame_index, 1])}

    def build_candidates(self):
        self.bank = [Candidate(0, 0, {}, np.stack([self.chunk.indices, np.zeros(self.chunk.length)], axis=1))]

    def fingerprint(self):
        return str(self.frame_index)


def test_boundary_and_native_export_are_distinct():
    simulator = CountingSimulator()
    paused = simulator.prepare()
    result = simulator.execute(0)
    np.testing.assert_array_equal(paused["views"], result["views"][0])
    assert simulator.renders == 5  # One boundary render, four later observations.
    assert result["native_control_iterations"] == 5
    np.testing.assert_array_equal(result["native_control_frame_indices"], np.arange(2, 7))
    np.testing.assert_array_equal(result["recorded_frame_indices"], np.arange(2, 7))
    assert np.all(result["native_actions"][:, 1] == 1)
    assert np.all(result["actions"][:, 1] == 0)
    assert result["conditioning_max_abs_error"] == 0
    with pytest.raises(RuntimeError):
        simulator.execute(0)


def test_padding_does_not_step_simulator():
    simulator = CountingSimulator(start=0, end=4, stride=1, last=2)
    simulator.prepare()
    result = simulator.execute(0)
    assert result["native_control_iterations"] == 3
    assert result["padded"].tolist() == [False, False, False, True, True]
    assert result["source_frame_indices"].tolist() == [0, 1, 2, 2, 2]
    np.testing.assert_array_equal(result["views"][2], result["views"][4])


def minimal_predictor():
    predictor = WanPredictor.__new__(WanPredictor)
    predictor.torch = torch
    predictor.args = SimpleNamespace(height=8, width=8, num_frames=5, cfg_scale=1, num_inference_steps=1)
    predictor.pipe = SimpleNamespace(device="cpu", torch_dtype=torch.float32)
    predictor.method, predictor.context = "standard", None
    predictor.online_method = OnlineMethod.__new__(OnlineMethod)
    predictor.online_method.method = predictor.method
    return predictor


def test_video_tensor_exact_layout_and_range():
    predictor = minimal_predictor()
    video = np.zeros((1, 2, 8, 8, 3), dtype=np.uint8)
    video[:, 1] = 255
    tensor = predictor.video_tensor(video)
    assert tensor.shape == (2, 3, 1, 8, 8)
    assert torch.all(tensor[0] == -1) and torch.all(tensor[1] == 1)
    with pytest.raises(ValueError):
        predictor.video_tensor(video.astype(np.float32))
    with pytest.raises(ValueError):
        predictor.video_tensor(np.zeros((1, 2, 16, 8, 3), dtype=np.uint8))


def test_prediction_decoder_and_input_whitelist():
    predictor = minimal_predictor()

    class Pipe:
        device, torch_dtype = "cpu", torch.float32

        def __call__(self, **kwargs):
            assert kwargs["input_video"].shape[2] == 1
            assert kwargs["num_frames"] == 5
            return torch.zeros((2, 3, 5, 8, 8))

    predictor.pipe = Pipe()
    predictor.action_tensor = lambda actions: torch.as_tensor(actions)[None]
    current = np.full((1, 2, 8, 8, 3), 19, dtype=np.uint8)
    request = {"views": current, "actions": np.zeros((5, 7)), "seed": 1}
    decoded = predictor.predict(request)
    assert decoded.shape == (5, 2, 8, 8, 3)
    np.testing.assert_array_equal(decoded[0], current[0])
    assert np.all(decoded[1:] == 127)
    with pytest.raises(ValueError, match="only"):
        predictor.predict({**request, "friction_mu": 0.1})
    with pytest.raises(ValueError, match="current frame"):
        predictor.predict({**request, "views": np.repeat(current, 2, axis=0)})


@pytest.mark.parametrize("task", TASKS)
def test_tracking_terminal_semantics(task):
    rules = load_rules(task)
    state = SimpleNamespace(image_width=224, image_height=224)
    kind = rules["outcome_kind"]
    if kind == "point":
        state.centroids = np.full((5, 1, 2), 112.0)
        assert terminal_value(state, rules) == [0.5, 0.5]
    elif kind == "scalar":
        state.angles_rad = np.full((5, 1), np.pi / 6)
        assert terminal_value(state, rules) == pytest.approx(30)
    else:
        state.light_score = np.full((5, 1), float(rules["light_threshold"]) + 0.01)
        assert terminal_value(state, rules) is True


def test_light_support_seed_stays_tied_to_annotation(tmp_path):
    runner = Runner("light_switch", config_for("light_switch"), resources_for("light_switch"), tmp_path)
    runner.resources.update(method="ours", support_actions=[0])
    seeds = []

    class FakeSimulator:
        def call(self, method, argument=None):
            if method == "execute":
                return {"views": np.zeros((33, 2, 8, 8, 3), dtype=np.uint8), "actions": np.zeros((33, 7)), "native_frame_indices": np.arange(33), "source_frame_indices": np.arange(33), "conditioning": "fixture"}

        def shutdown(self):
            pass

    def simulator(case, path):
        seeds.append(case["seed"])
        return FakeSimulator()

    runner.simulator = simulator
    runner._supports({"seed": 20260725}, tmp_path)
    assert seeds == [20260725]


def test_no_selection_still_writes_result(tmp_path):
    runner = Runner("gravity", config_for("gravity"), resources_for("gravity"), tmp_path)
    simulator = CountingSimulator(start=0, end=4, stride=1)

    class Client:
        def call(self, method, argument=None):
            return getattr(simulator, method)() if argument is None else getattr(simulator, method)(argument)

        def shutdown(self):
            pass

    class Model:
        def call(self, method, argument):
            return {} if method == "freeze_context" else np.repeat(argument["views"], 5, axis=0)

    runner.simulator = lambda case, path: Client()
    runner.model = Model()
    with patch.object(runner.perception, "outcome", return_value=(SimpleNamespace(), None)), patch("wan_video_action.evaluation.sim_action_selection.core.runner.save_state"), patch("wan_video_action.evaluation.sim_action_selection.core.runner.select_action", return_value=(None, {})):
        result = runner.run_decision({}, 0, runner.rules["targets"][0]["name"])
    assert result["reason"] == "no_selected_action"
    saved = json.loads((tmp_path / ("decision_00000_" + runner.rules["targets"][0]["name"]) / "result.json").read_text())
    assert saved == jsonable(result)
