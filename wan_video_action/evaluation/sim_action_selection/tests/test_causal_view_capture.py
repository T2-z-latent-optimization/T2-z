import hashlib

import numpy as np
import pytest

from ..core.decision_capture import capture_decision


def paused():
    history = np.random.default_rng(51).integers(0, 256, (3, 2, 32, 32, 3), dtype=np.uint8)
    return {"views": history[-1], "frame_index": 2, "capture_history_views": history,
            "capture_frame_indices": np.arange(3)}


def profile():
    return {"mode": "jpeg_h264", "jpeg_quality": 95, "crf": 18, "gop": 2,
            "use_history": True, "current_repetitions": 1}


def test_per_view_capture_equals_independent_causal_encoding():
    state = paused()
    settings = profile()
    settings["view_profiles"] = [{"current_repetitions": 3}, {"crf": 19, "current_repetitions": 2}]
    actual, audit = capture_decision(state, settings)
    expected = []
    for camera, overrides in enumerate(settings["view_profiles"]):
        individual = {**state, "views": state["views"][camera:camera + 1],
                      "capture_history_views": state["capture_history_views"][:, camera:camera + 1]}
        captured, _ = capture_decision(individual, {**profile(), **overrides})
        expected.append(captured)
    np.testing.assert_array_equal(actual, np.concatenate(expected, axis=1))
    assert actual.shape == (1, 2, 32, 32, 3)
    assert audit["query_future_used"] is False
    assert audit["model_condition_frame_count"] == 1
    assert audit["repeated_current_padding_per_view"] == [2, 1]
    assert audit["native_history_frame_count_per_view"] == [3, 3]
    assert audit["source_rgb_sha256"] == hashlib.sha256(state["views"].tobytes()).hexdigest()


def test_empty_view_overrides_preserve_original_capture():
    state = paused()
    original, _ = capture_decision(state, profile())
    independent, _ = capture_decision(state, {**profile(), "view_profiles": [{}, {}]})
    np.testing.assert_array_equal(independent, original)


def test_raw_per_view_capture_preserves_native_rgb():
    state = paused()
    captured, audit = capture_decision(state, {"mode": "raw", "view_profiles": [{}, {}]})
    np.testing.assert_array_equal(captured, state["views"][None])
    assert audit["query_future_used"] is False


@pytest.mark.parametrize("overrides", [[], [{}], [{}, {}, {}], [{}, 3]])
def test_invalid_camera_profiles_rejected(overrides):
    with pytest.raises(ValueError):
        capture_decision(paused(), {**profile(), "view_profiles": overrides})
