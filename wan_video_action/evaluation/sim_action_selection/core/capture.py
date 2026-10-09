from __future__ import annotations

import hashlib
import io

import numpy as np


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _encode_view(frames, profile, fps):
    import av
    from PIL import Image

    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("h264", rate=int(fps), options={"crf": str(profile["crf"]), "g": str(profile["gop"])})
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for index, image in enumerate(frames):
            jpeg = io.BytesIO()
            Image.fromarray(image).save(jpeg, format="JPEG", quality=profile["jpeg_quality"])
            jpeg.seek(0)
            with Image.open(jpeg) as decoded:
                frame = av.VideoFrame.from_image(decoded.convert("RGB"))
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    buffer.seek(0)
    with av.open(buffer, "r") as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def encode_observations(views, profile=None, fps=20, current_only=False):
    array = np.asarray(views)
    if array.ndim != 5 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("Observations must be uint8 [time, views, height, width, RGB]")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError("Observation time and camera dimensions must be nonempty")
    profile = dict(profile or {"mode": "raw"})
    mode = profile.get("mode", "raw")
    if mode == "raw":
        encoded = array
    elif mode == "jpeg_h264":
        required = {"jpeg_quality", "crf", "gop"}
        if required - profile.keys():
            raise ValueError(f"Capture profile is missing {sorted(required - profile.keys())}")
        quality, crf, gop = profile["jpeg_quality"], profile["crf"], profile["gop"]
        repetitions = profile.get("current_repetitions", 1)
        if not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be an integer in [1, 100]")
        if not isinstance(crf, int) or not 0 <= crf <= 51 or not isinstance(gop, int) or gop < 1:
            raise ValueError("Invalid H264 CRF or GOP")
        if not isinstance(repetitions, int) or repetitions < 1:
            raise ValueError("Current-frame repetitions must be a positive integer")
        if current_only and len(array) != 1:
            raise ValueError("Current-only capture cannot encode query future observations")
        source = np.repeat(array, repetitions, axis=0) if current_only else array
        encoded = np.stack([_encode_view(source[:, camera], profile, fps) for camera in range(array.shape[1])], axis=1)
        if current_only:
            encoded = encoded[:1]
        if encoded.shape != array.shape:
            raise ValueError("Capture encoding changed the observation shape")
    else:
        raise ValueError(f"Unknown capture mode: {mode}")
    audit = {
        "profile": profile,
        "source_rgb_sha256": _digest(array),
        "model_rgb_sha256": _digest(encoded),
        "mae_uint8": float(np.abs(array.astype(np.float32) - encoded.astype(np.float32)).mean()),
        "current_frame_only": current_only,
        "query_future_used": False,
        "scope": "Current-frame encoding approximates recorded capture, not bit-exact whole-video encoding.",
    }
    return encoded, audit
