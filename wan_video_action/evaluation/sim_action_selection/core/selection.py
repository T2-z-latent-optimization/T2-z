"""Prediction-only facade over the existing offline selection strategies."""
from __future__ import annotations

from ...action_selection.core import _aggregate, _reaches, _select, _valid, load_rules


def merge_observed_supports(predictions, observations, rules):
    candidates = {int(item["action_index"]) for item in predictions}
    values = {}
    for observation in observations:
        index = int(observation["action_index"])
        if index not in candidates:
            raise ValueError(f"Observed support action {index} is not in the candidate bank")
        values.setdefault(index, []).append(observation["selection_value"])
    merged = []
    for prediction in predictions:
        item = dict(prediction)
        index = int(item["action_index"])
        if index in values:
            value = _aggregate(values[index], rules["outcome_kind"])
            if not _valid(value, rules["outcome_kind"]):
                raise ValueError(f"Observed support action {index} has no valid tracked outcome")
            item.update(selection_value=value, selection_source="observed_support")
        merged.append(item)
    return merged


def select_action(predictions, target, rules):
    if len({item["action_index"] for item in predictions}) != len(predictions):
        raise ValueError("Duplicate action candidate IDs")
    if any("ground_truth_value" in item for item in predictions):
        raise ValueError("Ground-truth outcomes must never enter online selection")
    selected, details = _select(predictions, target, rules)
    return selected, details


def executed_score(value, target):
    return {
        "target_reached": bool(_reaches(value, target)),
        "outcome_available": value is not None,
        "offline_policy": target["score"],
        "offline_policy_score": int(_reaches(value, target)) if target["score"] == "target_reach" and value is not None else None,
        "oracle_required": target["score"] != "target_reach",
    }
