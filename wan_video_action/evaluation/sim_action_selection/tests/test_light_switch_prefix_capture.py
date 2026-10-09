from types import SimpleNamespace

import numpy as np
import pytest

from wan_video_action.evaluation.sim_action_selection.core.types import Chunk
from wan_video_action.evaluation.sim_action_selection.tasks.light_switch.simulator import (
    LightSwitchSimulator, PrefixReached, PrefixSink,
)


def recorded_frame(index):
    return {
        "observation.images.image": np.full((8, 8, 3), index, dtype=np.uint8),
        "observation.images.wrist_image": np.full((8, 8, 3), index + 20, dtype=np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
        "action": np.zeros(7, dtype=np.float32),
    }


class PlannedLightPrefix(LightSwitchSimulator):
    """Replay planner and recorder callbacks without native engines."""

    def __init__(self, *, enabled, start=5):
        self.config = {"capture_history": enabled}
        self.case = {}
        self.chunk = Chunk(start, start + 32, 1, "post_step")
        self.frame_index = 0
        self.fps = 30
        self.sources = SimpleNamespace(provenance={})
        self.recorded_rows = []
        self.press_lead_steps = 23
        self.decision_button = "blue"
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(press_history=[]))

    def initialize(self):
        self._capture_prefix_views = []
        sink = PrefixSink(self)

        def follow_path(result):
            for i in range(len(result["position"])):
                frame = recorded_frame(i)
                self.recorded_rows.append(i)
                sink.add_frame(frame)
                frame["observation.images.image"][:] = 255

        def move_to_pose_with_screw(pose):
            result = {"position": np.zeros((self.chunk.start + self.press_lead_steps + 1, 7))}
            follow_path(result)

        try:
            move_to_pose_with_screw(SimpleNamespace(p=np.array([0.0, 0.0, 0.05])))
        except PrefixReached:
            pass
        self.frame_index = self.chunk.start

    def build_candidates(self):
        self.bank = [SimpleNamespace(index=0, order=0,
                                     model_actions=np.zeros((self.chunk.length, 7), dtype=np.float32))]

    def fingerprint(self):
        return "paused-original-boundary"

    def step_native(self):
        raise AssertionError("History capture must not execute any extra control")


@pytest.mark.parametrize("start", [0, 5, 25])
def test_light_history_is_real_contiguous_past_and_current(start):
    simulator = PlannedLightPrefix(enabled=True, start=start)
    paused = simulator.prepare()
    assert simulator.recorded_rows == list(range(start + 1))
    np.testing.assert_array_equal(paused["capture_frame_indices"], np.arange(start + 1))
    np.testing.assert_array_equal(paused["capture_history_views"][:, 0, 0, 0, 0], np.arange(start + 1))
    np.testing.assert_array_equal(paused["capture_history_views"][-1], paused["views"])


def test_light_raw_default_does_not_store_history():
    simulator = PlannedLightPrefix(enabled=False)
    paused = simulator.prepare()
    assert simulator._capture_prefix_views == []
    assert "capture_history_views" not in paused


def test_missing_prefix_is_not_silently_padded():
    simulator = PlannedLightPrefix(enabled=True)
    simulator.captured_prefix_views = lambda: ()
    with pytest.raises(ValueError, match="initialization prefix"):
        simulator.prepare()
