import pytest

from wan_video_action.evaluation.sim_action_selection.core.selection import load_rules, select_action


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("target_index", [0, 1])
def test_equal_light_predictions_preserve_original_label_tiebreak(value, target_index):
    rules = load_rules("light_switch")
    candidates = [
        {"action_index": index, "action_order": float(index), "selection_value": value}
        for index in range(2)
    ]
    selected, _ = select_action(candidates, rules["targets"][target_index], rules)
    assert selected["action_index"] == 1


def test_light_label_tiebreak_does_not_override_outcome_distance():
    rules = load_rules("light_switch")
    candidates = [
        {"action_index": 0, "action_order": 0.0, "selection_value": True},
        {"action_index": 1, "action_order": 1.0, "selection_value": False},
    ]
    selected, _ = select_action(candidates, rules["targets"][0], rules)
    assert selected["action_index"] == 0
