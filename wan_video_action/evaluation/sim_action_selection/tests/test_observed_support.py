from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ..core.selection import load_rules, merge_observed_supports, select_action


def test_support_observation_replaces_only_its_own_action():
    rules = load_rules("gravity")
    predictions = [
        {"action_index": 0, "action_order": 0, "selection_value": [0.8, 0.65], "selection_source": "model_prediction"},
        {"action_index": 1, "action_order": 1, "selection_value": [0.6, 0.62], "selection_source": "model_prediction"},
    ]
    merged = merge_observed_supports(predictions, [{"action_index": 0, "selection_value": [0.54, 0.56]}], rules)
    selected, _ = select_action(merged, rules["targets"][0], rules)
    assert selected["action_index"] == 0
    assert selected["selection_source"] == "observed_support"
    assert merged[1] == predictions[1]
    assert predictions[0]["selection_value"] == [0.8, 0.65]


def test_repeated_support_outcomes_use_offline_aggregation():
    rules = load_rules("mass_balance")
    predictions = [{"action_index": 1, "selection_value": None}]
    merged = merge_observed_supports(predictions, [
        {"action_index": 1, "selection_value": -2.0},
        {"action_index": 1, "selection_value": 4.0},
    ], rules)
    assert merged[0]["selection_value"] == 1.0


def test_support_must_belong_to_candidate_bank():
    with pytest.raises(ValueError, match="candidate bank"):
        merge_observed_supports([{"action_index": 0}], [{"action_index": 4, "selection_value": 0}], load_rules("mass_balance"))


@pytest.mark.parametrize("enabled,expected_action,expected_calls", [(True, 0, [1]), (False, 1, [0, 1])])
def test_runner_switch_controls_support_reuse(tmp_path, enabled, expected_action, expected_calls):
    from ..core.runner import Runner

    views = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames = np.arange(5)
    candidates = [
        {"index": index, "order": index, "parameters": {}, "conditioning": "exact_open_loop",
         "model_actions": np.full((5, 7), index, dtype=np.float32)}
        for index in range(2)
    ]

    class Simulator:
        def call(self, method, argument=None):
            if method == "prepare":
                return {"views": views, "native_frame_indices": frames, "fps": 20, "fingerprint": "paused"}
            if method == "candidates":
                return candidates
            if method == "fingerprint":
                return "paused"
            if method == "execute":
                actions = candidates[argument]["model_actions"]
                return {"views": np.repeat(views[None], 5, axis=0), "actions": actions,
                        "planned_actions": actions, "native_actions": actions,
                        "native_control_frame_indices": frames, "recorded_actions": actions,
                        "recorded_frame_indices": frames, "native_frame_indices": frames,
                        "source_frame_indices": frames, "padded": np.zeros(5, dtype=bool),
                        "native_control_iterations": 5, "conditioning": "exact_open_loop",
                        "conditioning_max_abs_error": 0}
            raise ValueError(method)

        def shutdown(self):
            pass

    class Model:
        def __init__(self):
            self.calls = []

        def call(self, method, request):
            if method == "freeze_context":
                return {"observed_supports": [{"action_index": 0, "views": np.full((5, 2, 8, 8, 3), 99, dtype=np.uint8)}]}
            index = int(request["actions"][0, 0])
            self.calls.append(index)
            video = np.full((5, 2, 8, 8, 3), 10 + index, dtype=np.uint8)
            video[0] = request["views"][0]
            return video

    def outcome(video, *args):
        marker = int(video[-1, 0, 0, 0, 0])
        value = {99: [0.54, 0.56], 10: [0.8, 0.65], 11: [0.6, 0.62]}.get(marker, [0.54, 0.56])
        return SimpleNamespace(), value

    resources = {"method": "ours", "support_actions": [], "support_indices": [],
                 "tracking": {}, "seed": 1, "save_predictions": False,
                 "use_observed_support_outcomes": enabled}
    runner = Runner("gravity", {}, resources, tmp_path)
    runner.simulator = lambda *args: Simulator()
    runner.model = Model()
    with patch.object(runner.perception, "outcome", side_effect=outcome), \
         patch("wan_video_action.evaluation.sim_action_selection.core.runner.save_state"), \
         patch("wan_video_action.evaluation.sim_action_selection.core.runner.save_video"), \
         patch("wan_video_action.evaluation.sim_action_selection.core.runner.prediction_accuracy", return_value={}):
        result = runner.run_decision({}, 0, "near")
    assert result["selected_action_index"] == expected_action
    assert runner.model.calls == expected_calls
    assert result["selected_outcome_source"] == ("observed_support" if enabled else "model_prediction")
