from __future__ import annotations

import hashlib

import numpy as np

from .capture import encode_observations


def select_model_views(views, indices=None):
    array = np.asarray(views)
    if array.ndim != 5:
        raise ValueError("Model observations must have time and camera axes")
    if indices is None:
        return array
    selection = np.asarray(indices)
    if selection.ndim != 1 or not len(selection) or selection.dtype.kind not in "iu":
        raise ValueError("Model camera indices must be a nonempty integer sequence")
    if len(np.unique(selection)) != len(selection):
        raise ValueError("Model camera indices must not repeat")
    if np.any(selection < 0) or np.any(selection >= array.shape[1]):
        raise ValueError("Model camera index is outside the native observations")
    return array[:, selection].copy()


def capture_decision(paused, profile=None, fps=20):
    current = np.asarray(paused["views"])[None]
    profile = dict(profile or {"mode": "raw"})
    view_profiles = profile.pop("view_profiles", None)
    if view_profiles is not None:
        if not isinstance(view_profiles, list) or len(view_profiles) != current.shape[1]:
            raise ValueError("Decision capture profiles must match the native camera count")
        captures, audits = [], []
        for camera, overrides in enumerate(view_profiles):
            if not isinstance(overrides, dict):
                raise ValueError("Each camera capture profile must be a mapping")
            camera_paused = dict(paused)
            camera_paused["views"] = current[0, camera:camera + 1]
            if "capture_history_views" in paused:
                camera_paused["capture_history_views"] = np.asarray(paused["capture_history_views"])[:, camera:camera + 1]
            captured, audit = capture_decision(camera_paused, {**profile, **overrides}, fps)
            captures.append(captured)
            audits.append(audit)
        captured = np.concatenate(captures, axis=1)
        return captured, {
            "profile": {**profile, "view_profiles": view_profiles},
            "source_rgb_sha256": hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest(),
            "model_rgb_sha256": hashlib.sha256(np.ascontiguousarray(captured).tobytes()).hexdigest(),
            "mae_uint8": float(np.abs(current.astype(np.float32) - captured.astype(np.float32)).mean()),
            "current_frame_only": all(item["current_frame_only"] for item in audits),
            "native_decision_frame": int(paused.get("frame_index", 0)),
            "native_history_frame_count_per_view": [item.get("native_history_frame_count", 1) for item in audits],
            "repeated_current_padding_per_view": [item["profile"].get("current_repetitions", 1) - 1 for item in audits],
            "model_condition_frame_count": 1,
            "query_future_used": False,
            "view_audits": audits,
            "scope": "Independent causal capture per camera; only native past/current RGB and copies of the current RGB are encoded.",
        }
    if not profile.get("use_history", False) or profile.get("mode", "raw") == "raw":
        return encode_observations(current, profile, fps, current_only=True)
    if "capture_history_views" not in paused or "capture_frame_indices" not in paused:
        raise ValueError("History capture requires actual past native observations")
    history = np.asarray(paused["capture_history_views"])
    indices = np.asarray(paused["capture_frame_indices"])
    boundary = int(paused["frame_index"])
    if history.dtype != np.uint8 or history.ndim != 5 or history.shape[1:] != current.shape[1:]:
        raise ValueError("Native history and current RGB shapes must agree")
    if len(history) != boundary + 1 or not np.array_equal(indices, np.arange(boundary + 1)):
        raise ValueError("Capture history must contain only the contiguous past through the decision frame")
    if not np.array_equal(history[-1:], current):
        raise ValueError("Native history must end with the exact current observation")
    repetitions = profile.get("current_repetitions", 1)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("Current-frame repetitions must be a positive integer")
    source = np.concatenate([history, np.repeat(current, repetitions - 1, axis=0)])
    encoded, audit = encode_observations(source, profile, fps)
    captured = encoded[boundary:boundary + 1].copy()
    audit.update({
        "source_rgb_sha256": hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest(),
        "model_rgb_sha256": hashlib.sha256(np.ascontiguousarray(captured).tobytes()).hexdigest(),
        "history_rgb_sha256": hashlib.sha256(np.ascontiguousarray(history).tobytes()).hexdigest(),
        "mae_uint8": float(np.abs(current.astype(np.float32) - captured.astype(np.float32)).mean()),
        "native_history_frame_count": len(history),
        "native_decision_frame": boundary,
        "repeated_current_padding": repetitions - 1,
        "model_condition_frame_count": 1,
        "query_future_used": False,
        "scope": "Encoding uses actual native past/current frames; the model receives only the encoded current frame. Whole-recording bit-exact parity is not claimed.",
    })
    return captured, audit


def capture_prefix(paused, executed, profile=None, fps=20, frame_count=3):
    frames = np.asarray(executed["recorded_views"])[:frame_count]
    indices = np.asarray(executed["recorded_frame_indices"])[:frame_count]
    boundary = int(paused.get("frame_index", 0))
    current = np.asarray(paused["views"])[None]
    if frames.dtype != np.uint8 or frames.ndim != 5 or frames.shape[1:] != current.shape[1:]:
        raise ValueError("Candidate prefix and decision RGB shapes must agree")
    if len(frames) != frame_count or not np.array_equal(indices, boundary + np.arange(frame_count)):
        raise ValueError("Candidate prefix must contain contiguous actual native observations")
    profile = dict(profile or {"mode": "jpeg_h264", "jpeg_quality": 95, "crf": 18, "gop": 2})
    view_profiles = profile.pop("view_profiles", None)
    if view_profiles is None:
        encoded, audit = encode_observations(frames, profile, fps)
    else:
        if not isinstance(view_profiles, list) or len(view_profiles) != frames.shape[1]:
            raise ValueError("Candidate prefix profiles must match the native camera count")
        encoded_views, view_audits = [], []
        for camera, overrides in enumerate(view_profiles):
            if not isinstance(overrides, dict):
                raise ValueError("Each camera capture profile must be a mapping")
            encoded_view, view_audit = encode_observations(frames[:, camera:camera + 1], {**profile, **overrides}, fps)
            encoded_views.append(encoded_view)
            view_audits.append(view_audit)
        encoded = np.concatenate(encoded_views, axis=1)
        audit = {"profile": {**profile, "view_profiles": view_profiles}, "view_audits": view_audits}
    captured = encoded[:1].copy()
    audit.update({
        "source_rgb_sha256": hashlib.sha256(np.ascontiguousarray(frames[:1]).tobytes()).hexdigest(),
        "model_rgb_sha256": hashlib.sha256(np.ascontiguousarray(captured).tobytes()).hexdigest(),
        "prefix_rgb_sha256": hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest(),
        "mae_uint8": float(np.abs(frames[:1].astype(np.float32) - captured.astype(np.float32)).mean()),
        "native_current_rgb_exact": bool(np.array_equal(frames[:1], current)),
        "native_current_mae_uint8": float(np.abs(frames[:1].astype(np.float32) - current.astype(np.float32)).mean()),
        "native_decision_frame": boundary,
        "native_prefix_frame_count": frame_count,
        "native_prefix_frame_indices": indices.tolist(),
        "model_condition_frame_count": 1,
        "current_frame_only": False,
        "query_future_used": frame_count > 1,
        "query_future_scope": "candidate_specific_preimpact_encoding_prefix",
        "scope": "Actual candidate-specific native prefix is encoded together; only its encoded first frame conditions the model.",
    })
    return captured, audit


def capture_execution(paused, executed, profile=None, fps=20):
    profile = dict(profile or {"mode": "raw"})
    if profile.get("mode", "raw") == "raw":
        return np.asarray(executed["views"]).copy(), {
            "profile": profile, "query_future_used_for_model": False,
            "actual_future_encoded_after_selection": False,
        }
    recorded_indices = np.asarray(executed["recorded_frame_indices"])
    grid = np.asarray(executed["native_frame_indices"])
    recorded = executed.get("recorded_views")
    if recorded is None:
        if not np.array_equal(recorded_indices, grid):
            raise ValueError("Encoded execution requires original-rate native observations")
        recorded = executed["views"]
    recorded = np.asarray(recorded)
    if len(recorded) != len(recorded_indices) or recorded.dtype != np.uint8:
        raise ValueError("Recorded native observations and frame indices do not match")
    if not len(recorded_indices) or not np.array_equal(recorded_indices, np.arange(recorded_indices[0], recorded_indices[-1] + 1)):
        raise ValueError("Native execution capture must be contiguous at the original frame rate")
    frames, indices = recorded, recorded_indices
    if profile.get("use_history", False):
        history = np.asarray(paused["capture_history_views"])
        history_indices = np.asarray(paused["capture_frame_indices"])
        past = history_indices < recorded_indices[0]
        frames = np.concatenate([history[past], recorded])
        indices = np.concatenate([history_indices[past], recorded_indices])
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("Past and executed camera observations must form a contiguous capture")
    encoded, audit = encode_observations(frames, profile, fps)
    positions = np.minimum(np.searchsorted(indices, grid), len(indices) - 1)
    audit.update({
        "query_future_used_for_model": False,
        "actual_future_encoded_after_selection": True,
        "native_capture_frame_count": len(frames),
        "native_frame_indices": grid.tolist(),
        "source_frame_indices": indices[positions].tolist(),
        "scope": "Actual selected-action observations encoded after selection; not bit-exact whole-recording archive encoding.",
    })
    return encoded[positions].copy(), audit
