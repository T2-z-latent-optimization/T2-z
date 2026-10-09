"""Task-specific, split-blind RGB measurement proposals.

These extractors never receive an environment ID, action level, or outcome label.
A detection is not an accuracy certificate. Formal metrics require independent
manual audit, and uncertain/absent tracks remain missing rather than forward-filled.
Inputs must be individual native dataset-view frames, not comparison grids.
"""

from __future__ import annotations

import cv2
import numpy as np

SIZES = {"ball": (560, 248), "door": (640, 480),
         "stick": (640, 480), "soft": (512, 256)}


def components(mask, minimum=10):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    result = []
    for idx in range(1, count):
        x, y, w, h, area = map(int, stats[idx])
        if area >= minimum:
            result.append({"center": centers[idx].astype(float),
                           "bbox": (x, y, w, h), "area": area})
    return result


def point(value):
    return None if value is None else [float(v) for v in value]


class BallTracker:
    def __init__(self, first, **kwargs):
        self.previous = None

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = (((h >= 155) | (h <= 8)) & (s >= 65) & (v >= 60))
        mask[:60] = False
        mask[225:] = False
        candidates = [
            c for c in components(mask, 25)
            if 6 <= c["bbox"][2] <= 80 and 4 <= c["bbox"][3] <= 65
            and c["area"] <= 2200
        ]
        if self.previous is not None:
            candidates = [c for c in candidates
                          if np.linalg.norm(c["center"] - self.previous) <= 100]
        if not candidates:
            return {"center": None, "confidence": 0.0, "reason": "missing_red_ball"}
        candidates.sort(key=lambda c: c["area"], reverse=True)
        best = candidates[0]
        ambiguous = len(candidates) > 1 and candidates[1]["area"] > best["area"] * .55
        self.previous = best["center"]
        x, y, w, h = best["bbox"]
        return {"center": point(self.previous), "bbox": list(best["bbox"]),
                "confidence": 0.4 if ambiguous else min(1.0, best["area"] / 100.0),
                "ambiguous": ambiguous, "touches_right_edge": x + w >= frame.shape[1] - 2,
                "reason": "ambiguous_red_components" if ambiguous else "detected"}


class DoorTracker:
    def __init__(self, first, **kwargs):
        gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 3), 0)
        gradient = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))


        score = np.mean(gradient[145:365, 230:355], axis=0)
        self.closed_reference_x = float(230 + np.argmax(score))

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b, g, r = [a.astype(float) for a in cv2.split(frame)]
        mask = ((hsv[:, :, 0] <= 38) & (hsv[:, :, 1] >= 45)
                & (hsv[:, :, 2] >= 30) & (r > 1.10 * g) & (g > 1.03 * b))
        xs = []
        for y in range(155, 355, 3):
            indices = np.flatnonzero(mask[y, :540])
            if len(indices) >= 80:
                xs.append(float(indices[-1]))
        if len(xs) < 40:
            return {"center": None, "confidence": 0.0, "reason": "missing_door_edge"}
        x = float(np.median(xs))
        gap = max(0.0, x - self.closed_reference_x)
        return {"center": [x, 255.0], "door_edge_x": x,
                "closed_reference_x": self.closed_reference_x,
                "closure_gap_px": gap, "closed_proposal": gap <= 8.0,
                "confidence": min(1.0, len(xs) / 60.0),
                "reason": "edge_proposal_requires_seam_audit"}


class StickTracker:
    def __init__(self, first, **kwargs):
        pass

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue = ((hsv[:, :, 0] >= 92) & (hsv[:, :, 0] <= 125)
                & (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 30)).astype(np.uint8)
        blue[:150] = 0
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((7, 17), np.uint8))
        candidates = [c for c in components(blue, 80)
                      if c["bbox"][2] >= 10 and c["bbox"][3] >= 12]
        candidates.sort(key=lambda c: c["area"], reverse=True)
        if len(candidates) < 2:
            return {"center": None, "confidence": 0.0, "reason": "missing_endpoint"}
        ends = sorted(candidates[:2], key=lambda c: c["center"][0])
        left, right = (c["center"] for c in ends)
        span = float(np.linalg.norm(right - left))
        if span < 180:
            return {"center": None, "confidence": 0.0, "reason": "implausible_endpoint_pair"}
        yellow = ((hsv[:, :, 0] >= 17) & (hsv[:, :, 0] <= 40)
                  & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 55))
        yellow[:170] = False
        supports = sorted(components(yellow, 100), key=lambda c: c["area"], reverse=True)
        support = supports[0]["center"] if supports else None
        fraction = None if support is None else float(
            np.dot(support - left, right - left) / (span * span)
        )
        clipped = any(c["bbox"][0] <= 1 or
                      c["bbox"][0] + c["bbox"][2] >= frame.shape[1] - 1 for c in ends)
        return {"center": point((left + right) / 2), "left": point(left),
                "right": point(right), "support_center": point(support),
                "support_fraction": fraction, "span_px": span,
                "angle_deg": float(np.degrees(np.arctan2(
                    right[1] - left[1], right[0] - left[0]))),
                "confidence": .5 if clipped else 1.0,
                "reason": "clipped_endpoint" if clipped else "detected"}


class SoftTracker:
    """Rigid terminal-block tracking, with explicitly auditable initial boxes."""

    def __init__(self, first, seed_bbox=None, seed_approved=False, **kwargs):
        self.reference = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        self.approved = bool(seed_approved)
        self.polygon = None
        self.features = None
        self.initial = True
        if seed_bbox is None:
            seed_bbox = self.propose_seed(first)
        if seed_bbox is not None:
            x, y, w, h = map(float, seed_bbox)
            self.polygon = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                                    dtype=np.float32)
            self.features = self.find_features(self.reference)

    @staticmethod
    def propose_seed(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((13, 13), np.uint8))
        spots = (contrast > 25) & (gray < 135) & (hsv[:, :, 1] < 65)
        spots[:70] = False
        dots = [c for c in components(spots, 3)
                if c["area"] <= 90 and c["bbox"][2] <= 14 and c["bbox"][3] <= 14]
        candidates = []
        for i, a in enumerate(dots):
            for b in dots[i+1:]:
                separation = float(np.linalg.norm(a["center"] - b["center"]))
                if not 6 <= separation <= 24:
                    continue
                center = (a["center"] + b["center"]) / 2
                x, y = center
                x0, y0 = max(0, int(x)-17), max(0, int(y)-12)
                x1, y1 = min(frame.shape[1], x0+34), min(frame.shape[0], y0+38)
                patch = hsv[y0:y1, x0:x1]
                if patch.size and float(np.mean(patch[:, :, 1] > 70)) < .08:
                    candidates.append((a["area"] + b["area"], (x0, y0, x1-x0, y1-y0)))
        return max(candidates, key=lambda c: c[0])[1] if candidates else None

    def find_features(self, gray):
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.round(self.polygon).astype(np.int32), 255)
        return cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=.03,
                                      minDistance=3, mask=mask, blockSize=3)

    def update(self, frame):
        if self.polygon is None or self.features is None or len(self.features) < 4:
            return {"center": None, "confidence": 0.0, "seed_approved": self.approved,
                    "reason": "manual_terminal_block_seed_required"}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        confidence = 1.0
        if not self.initial:
            current, status, _ = cv2.calcOpticalFlowPyrLK(
                self.reference, gray, self.features, None,
                winSize=(31, 31), maxLevel=3)
            if current is None:
                return {"center": None, "confidence": 0.0, "reason": "flow_failed"}
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.reference, current, None, winSize=(31, 31), maxLevel=3)
            if back is None:
                return {"center": None, "confidence": 0.0, "reason": "backward_flow_failed"}
            good = ((status[:, 0] != 0) & (back_status[:, 0] != 0)
                    & (np.linalg.norm(back[:, 0] - self.features[:, 0], axis=1) <= 1.5))
            if int(good.sum()) < 4:
                return {"center": None, "confidence": 0.0, "reason": "insufficient_consistent_features"}
            affine, inliers = cv2.estimateAffinePartial2D(
                self.features[good, 0], current[good, 0],
                method=cv2.RANSAC, ransacReprojThreshold=2.0)
            if affine is None:
                return {"center": None, "confidence": 0.0, "reason": "rigid_fit_failed"}
            scale = float(np.linalg.norm(affine[0, :2]))
            confidence = float(np.mean(inliers))
            if not .88 <= scale <= 1.12 or confidence < .65:
                return {"center": None, "confidence": confidence, "reason": "unstable_rigid_fit"}
            self.polygon = cv2.transform(self.polygon[None], affine)[0]
            self.reference = gray
            self.features = self.find_features(gray)
        self.initial = False
        center = self.polygon.mean(axis=0)
        return {"center": point(center), "polygon": self.polygon.tolist(),
                "confidence": confidence, "seed_approved": self.approved,
                "reason": "tracked" if self.approved else "unapproved_seed_proposal"}


TRACKERS = {"ball": BallTracker, "door": DoorTracker,
            "stick": StickTracker, "soft": SoftTracker}


def make_tracker(task, frame, **kwargs):
    expected = SIZES[task]
    if (frame.shape[1], frame.shape[0]) != expected:
        raise ValueError(
            f"{task}: expected native dataset view {expected}, got "
            f"{(frame.shape[1], frame.shape[0])}; invert resize/letterbox first."
        )
    return TRACKERS[task](frame, **kwargs)
