from types import SimpleNamespace

import numpy as np
import pytest

from ..core.capture import encode_observations
from ..core.decision_capture import capture_decision
from ..core.simulator import NativeSimulator
from ..core.types import Candidate, Chunk


class PrefixSimulator(NativeSimulator):
    def __init__(self, enabled):
        self.config = {"capture_history": enabled}
        self.chunk = Chunk(2, 4)
        self.frame_index = 0
        self.case = {}
        self.fps = 20
        self.sources = SimpleNamespace(provenance={})

    def initialize(self):
        self.steps = 0
        self.observed = []

    def step_native(self):
        self.steps += 1

    def current(self):
        self.observed.append(self.frame_index)
        return {"views": np.full((2, 16, 24, 3), self.frame_index, dtype=np.uint8)}

    def build_candidates(self):
        self.bank = [Candidate(0, 0, {}, np.zeros((3, 7), dtype=np.float32))]

    def fingerprint(self):
        return str(self.steps)


def test_native_prefix_has_no_future_or_extra_steps():
    simulator = PrefixSimulator(True)
    result = simulator.prepare()
    assert simulator.steps == 2
    assert simulator.observed == [0, 1, 2]
    assert np.array_equal(result["capture_frame_indices"], [0, 1, 2])
    assert np.array_equal(result["capture_history_views"][-1], result["views"])


def test_default_native_capture_does_not_collect_history():
    simulator = PrefixSimulator(False)
    result = simulator.prepare()
    assert simulator.observed == [2]
    assert "capture_history_views" not in result


def observation():
    history = np.random.default_rng(7).integers(0, 256, (3, 2, 16, 24, 3), dtype=np.uint8)
    return {"views": history[-1], "capture_history_views": history, "capture_frame_indices": np.arange(3), "frame_index": 2}


def profile():
    return {"mode": "jpeg_h264", "jpeg_quality": 98, "crf": 18, "gop": 2, "use_history": True}


def test_history_capture_uses_correct_decision_frame():
    paused = observation()
    expected, _ = encode_observations(paused["capture_history_views"], profile())
    actual, audit = capture_decision(paused, profile())
    assert np.array_equal(actual, expected[-1:])
    assert actual.shape[0] == 1
    assert audit["query_future_used"] is False
    assert audit["native_history_frame_count"] == 3


def test_history_capture_rejects_future_indices():
    paused = observation()
    paused["capture_frame_indices"] = [0, 1, 3]
    with pytest.raises(ValueError, match="past"):
        capture_decision(paused, profile())


def test_history_capture_rejects_changed_current_frame():
    paused = observation()
    paused["views"] = np.zeros_like(paused["views"])
    with pytest.raises(ValueError, match="exact current"):
        capture_decision(paused, profile())
