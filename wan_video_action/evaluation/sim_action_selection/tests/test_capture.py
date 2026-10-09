import numpy as np
import pytest

from ..core.capture import encode_observations


def images(length=1):
    return np.random.default_rng(5).integers(0, 256, (length, 2, 16, 24, 3), dtype=np.uint8)


def test_raw_preserves_pixels():
    original = images()
    encoded, audit = encode_observations(original, current_only=True)
    assert np.array_equal(encoded, original)
    assert audit["source_rgb_sha256"] == audit["model_rgb_sha256"]
    assert audit["mae_uint8"] == 0


def test_current_capture_is_deterministic_and_preserves_source():
    original = images()
    saved = original.copy()
    profile = {"mode": "jpeg_h264", "jpeg_quality": 95, "crf": 18, "gop": 2}
    first, audit = encode_observations(original, profile, current_only=True)
    second, _ = encode_observations(original, profile, current_only=True)
    assert np.array_equal(first, second)
    assert np.array_equal(original, saved)
    assert first.shape == original.shape
    assert audit["query_future_used"] is False
    assert audit["mae_uint8"] > 0


def test_capture_rejects_query_future():
    with pytest.raises(ValueError, match="future"):
        encode_observations(images(2), {"mode": "jpeg_h264", "jpeg_quality": 95, "crf": 18, "gop": 2}, current_only=True)


def test_past_support_sequence_preserves_all_frames():
    original = images(3)
    encoded, _ = encode_observations(original, {"mode": "jpeg_h264", "jpeg_quality": 95, "crf": 18, "gop": 2})
    assert encoded.shape == original.shape


@pytest.mark.parametrize("profile", [
    {"mode": "unknown"},
    {"mode": "jpeg_h264"},
    {"mode": "jpeg_h264", "jpeg_quality": 0, "crf": 18, "gop": 2},
    {"mode": "jpeg_h264", "jpeg_quality": 95, "crf": 52, "gop": 2},
    {"mode": "jpeg_h264", "jpeg_quality": 95, "crf": 18, "gop": 0},
    {"mode": "jpeg_h264", "jpeg_quality": 95, "crf": 18, "gop": 2, "current_repetitions": 0},
])
def test_invalid_capture_profile(profile):
    with pytest.raises(ValueError):
        encode_observations(images(), profile, current_only=True)
