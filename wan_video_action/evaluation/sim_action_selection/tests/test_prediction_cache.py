import numpy as np
import pytest

from ..core.prediction_cache import CasePredictionCache


class Worker:
    def __init__(self):
        self.calls = []

    def call(self, method, argument):
        self.calls.append((method, argument))
        if method == "freeze_context":
            return {"context": [[0.5]], "observed_supports": [{"views": np.zeros((1, 2, 2, 3), dtype=np.uint8)}]}
        return np.ones((2, 1, 2, 2, 3), dtype=np.uint8)


def freeze(uid="first", value=0):
    return {"decision_uid": uid, "supports": [{"actions": np.full((2, 7), value, dtype=np.float32)}],
            "support_indices": [], "collect_observed_supports": True}


def prediction(seed=0):
    return {"views": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
            "actions": np.zeros((2, 7), dtype=np.float32), "seed": seed}


def test_targets_share_one_adaptation_and_preserve_observed_supports():
    worker = Worker()
    cache = CasePredictionCache(worker)
    first = cache.call("freeze_context", freeze())
    first.pop("observed_supports")
    second = cache.call("freeze_context", freeze("second"))
    assert len(worker.calls) == 1
    assert second["context_reused_across_targets"]
    assert second["observed_supports"][0]["views"].shape == (1, 2, 2, 3)


def test_changed_support_does_not_silently_reuse_a_context():
    cache = CasePredictionCache(Worker())
    cache.call("freeze_context", freeze())
    with pytest.raises(RuntimeError, match="Support inputs changed"):
        cache.call("freeze_context", freeze("second", value=1))


def test_identical_predictions_are_reused_without_mutable_aliases():
    worker = Worker()
    cache = CasePredictionCache(worker)
    cache.call("freeze_context", freeze())
    first = cache.call("predict", prediction())
    first[:] = 0
    second = cache.call("predict", prediction())
    assert np.all(second == 1)
    assert [method for method, _ in worker.calls].count("predict") == 1


def test_changed_seed_or_actions_trigger_an_independent_prediction():
    worker = Worker()
    cache = CasePredictionCache(worker)
    cache.call("freeze_context", freeze())
    cache.call("predict", prediction())
    cache.call("predict", prediction(seed=1))
    request = prediction()
    request["actions"][0, 0] = 1
    cache.call("predict", request)
    assert [method for method, _ in worker.calls].count("predict") == 3


def test_new_case_has_an_independent_context():
    worker = Worker()
    CasePredictionCache(worker).call("freeze_context", freeze())
    CasePredictionCache(worker).call("freeze_context", freeze(value=1))
    assert len(worker.calls) == 2
