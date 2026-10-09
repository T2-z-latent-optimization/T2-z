"""Calibrated real-video tracking without outcome labels."""
from __future__ import annotations

import cv2
import numpy as np

from .tracking import BallTracker, DoorTracker, StickTracker, SIZES


def highpass(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray - cv2.GaussianBlur(gray, (0, 0), 3.0)


class TerminalBlockTracker:
    """Framewise object-template localization; never integrates an optical-flow box.

    Templates are reviewed TRAIN-image crops. Their centers represent the projected
    visible metal-body center, not a rope point or physical 3-D center of mass.
    Detector confidence is not an accuracy certificate.
    """

    def __init__(self, first, templates, settings, **kwargs):
        self.templates = templates
        self.settings = settings
        self.previous = None
        self.preferred = None
        self.index = 0

    def search(self, frame, full):
        height, width = frame.shape[:2]
        if full or self.previous is None:
            x0, y0, x1, y1 = 0, self.settings.get("search_min_y", 0), width, height
            indices = range(len(self.templates))
        else:
            x, y = self.previous
            radius = self.settings["local_search_radius_px"]
            x0, y0 = max(0, int(x-radius)), max(self.settings.get("search_min_y", 0), int(y-radius))
            x1, y1 = min(width, int(x+radius)), min(height, int(y+radius))
            indices = self.preferred or range(len(self.templates))
        region = highpass(frame)[y0:y1, x0:x1]
        saturated = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1] > 70
        sat_integral = cv2.integral(saturated.astype(np.uint8))
        proposals = []
        for index in indices:
            template = self.templates[index]
            h, w = template["image"].shape
            if h > region.shape[0] or w > region.shape[1]:
                continue
            response = cv2.matchTemplate(region, template["image"], cv2.TM_CCOEFF_NORMED)
            response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
            for _ in range(2):
                _, score, _, loc = cv2.minMaxLoc(response)
                x, y = loc
                cx, cy = x+x0+(w-1)/2, y+y0+(h-1)/2
                left, top = x+x0, y+y0
                right, bottom = left+w, top+h
                saturated_fraction = float(sat_integral[bottom,right]-sat_integral[top,right]-
                    sat_integral[bottom,left]+sat_integral[top,left])/(w*h)
                if saturated_fraction <= self.settings.get("max_saturated_fraction", 1.):
                    proposals.append((float(score), cx, cy, index, left, top, w, h))
                exclusion = self.settings["candidate_merge_radius_px"]
                response[max(0,y-exclusion):y+exclusion+1,
                         max(0,x-exclusion):x+exclusion+1] = -1.
        proposals.sort(reverse=True)
        return proposals

    def update(self, frame):
        full = self.previous is None or self.index % self.settings["global_search_every"] == 0
        proposals = self.search(frame, full)
        if not full and (not proposals or proposals[0][0] < self.settings["min_score"]):
            proposals = self.search(frame, True)
        self.index += 1
        if not proposals or proposals[0][0] < self.settings["min_score"]:
            return {"center": None, "confidence": 0., "reason": "no_terminal_template_match",
                    "seed_approved": False}
        best = proposals[0]
        score, x, y, index, bx, by, w, h = best
        alternative = next((p for p in proposals[1:]
                            if np.hypot(p[1]-x, p[2]-y) >
                            self.settings["candidate_merge_radius_px"]), None)
        margin = score-alternative[0] if alternative else 1.
        if margin < self.settings["min_margin"]:
            return {"center": None, "confidence": score, "ambiguous": True,
                    "reason": "ambiguous_terminal_match", "match_margin": margin,
                    "candidate_center": [x,y], "seed_approved": False}
        self.previous = np.array([x,y], dtype=float)

        sources = []
        for proposal in proposals:
            source = self.templates[proposal[3]]["source"]
            if source not in sources:
                sources.append(source)
            if len(sources) == 3:
                break
        self.preferred = [i for i,t in enumerate(self.templates) if t["source"] in sources]
        clipped = bx <= 1 or by <= 1 or bx+w >= frame.shape[1]-1 or by+h >= frame.shape[0]-1
        polygon = [[bx,by],[bx+w-1,by],[bx+w-1,by+h-1],[bx,by+h-1]]
        return {"center": [x,y], "polygon": polygon, "bbox": [bx,by,w,h],
                "confidence": score, "match_margin": margin,
                "template_source": self.templates[index]["source"],
                "touches_image_edge": clipped, "seed_approved": False,
                "reason": "boundary_requires_review" if clipped else "terminal_template_proposal"}


class AnchoredTerminalTracker(TerminalBlockTracker):
    """Short-horizon rigid tracking with independently re-detected anchors."""

    def __init__(self, first, templates, settings, **kwargs):
        super().__init__(first, templates, settings)
        self.gray = None
        self.polygon = None
        self.features = None
        self.anchor = None
        self.anchor_size = None
        self.anchor_age = 0
        self.misses = 0
        self.frames = 0

    def corners(self, gray, polygon):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((5,5), np.uint8))
        return cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=.02,
                                      minDistance=2, mask=mask, blockSize=3)

    @staticmethod
    def correlation(a, b):
        aa, bb = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        aa -= aa.mean()
        bb -= bb.mean()
        denominator = np.linalg.norm(aa)*np.linalg.norm(bb)
        return float(np.dot(aa,bb)/denominator) if denominator > 1.e-8 else 0.

    def rigid_candidate(self, frame, gray):
        if self.gray is None or self.features is None or len(self.features) < 4:
            return None, "insufficient_object_features"
        current,status,_ = cv2.calcOpticalFlowPyrLK(
            self.gray,gray,self.features,None,winSize=(21,21),maxLevel=3)
        if current is None or status is None:
            return None,"forward_flow_failed"
        backward,back_status,_ = cv2.calcOpticalFlowPyrLK(
            gray,self.gray,current,None,winSize=(21,21),maxLevel=3)
        if backward is None or back_status is None:
            return None,"backward_flow_failed"
        error=np.linalg.norm(backward[:,0]-self.features[:,0],axis=1)
        good=(status[:,0]!=0)&(back_status[:,0]!=0)&(error<=self.settings["flow_fb_error_px"])
        if int(good.sum())<4:
            return None,"inconsistent_object_features"
        affine,inliers=cv2.estimateAffinePartial2D(
            self.features[good,0],current[good,0],method=cv2.RANSAC,
            ransacReprojThreshold=1.8)
        if affine is None or inliers is None:
            return None,"rigid_fit_failed"
        inlier_fraction=float(inliers.mean())
        scale=float(np.linalg.norm(affine[0,:2]))
        rotation=abs(float(np.degrees(np.arctan2(affine[1,0],affine[0,0]))))
        polygon=cv2.transform(self.polygon[None],affine)[0]
        center=polygon.mean(axis=0)
        displacement=float(np.linalg.norm(center-self.polygon.mean(axis=0)))
        if (inlier_fraction<.75 or not .9<=scale<=1.1 or rotation>20 or
                displacement>self.settings["max_frame_displacement_px"]):
            return None,"implausible_rigid_motion"
        width,height=self.anchor_size
        destination=np.array([[0,0],[width-1,0],[width-1,height-1]],np.float32)
        warp=cv2.getAffineTransform(polygon[:3].astype(np.float32),destination)
        appearance=cv2.warpAffine(frame,warp,(width,height))
        appearance_score=self.correlation(highpass(appearance),self.anchor)
        if appearance_score<self.settings["min_anchor_correlation"]:
            return None,"anchor_appearance_disagrees"
        xmin,ymin=polygon.min(axis=0)
        xmax,ymax=polygon.max(axis=0)
        clipped=bool(xmin<1 or ymin<1 or xmax>frame.shape[1]-2 or ymax>frame.shape[0]-2)
        return dict(center=center.tolist(),polygon=polygon.tolist(),
                    confidence=inlier_fraction,appearance_score=appearance_score,
                    flow_fb_median_px=float(np.median(error[good])),
                    displacement_px=displacement,touches_image_edge=clipped,
                    seed_approved=False,measurement_valid=not clipped,
                    reason="rigid_flow_boundary" if clipped else "anchored_rigid_flow"),None

    def anchor_detection(self, frame, gray, detection):
        self.polygon=np.array(detection["polygon"],np.float32)
        x,y,w,h=detection["bbox"]
        self.anchor=highpass(frame[y:y+h,x:x+w])
        self.anchor_size=(w,h)
        self.gray=gray
        self.features=self.corners(gray,self.polygon)
        self.anchor_age=0
        self.misses=0
        detection["measurement_valid"]=not detection.get("touches_image_edge",False)
        detection["reason"]="terminal_template_anchor"
        return detection

    def accept_flow(self, gray, candidate):
        self.polygon=np.array(candidate["polygon"],np.float32)
        self.gray=gray
        self.features=self.corners(gray,self.polygon)
        self.previous=np.array(candidate["center"])
        self.anchor_age+=1
        self.misses=0
        candidate["anchor_age"]=self.anchor_age
        return candidate

    def update(self, frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.frames+=1
        candidate,flow_failure=self.rigid_candidate(frame,gray)
        must_detect=(candidate is None or self.frames % self.settings["detect_every"]==0
                     or self.anchor_age>=self.settings["max_anchor_age"])
        if must_detect:
            if candidate is None:
                self.previous=None
            else:
                self.previous=np.array(candidate["center"])
            detection=super().update(frame)
            if detection.get("center") is not None:
                agreement=(candidate is None or
                    np.linalg.norm(np.array(detection["center"])-np.array(candidate["center"]))
                    <=self.settings["detector_flow_agreement_px"])
                if agreement or detection["confidence"]>=self.settings["strong_reacquire_score"]:
                    return self.anchor_detection(frame,gray,detection)
        if candidate is not None and self.anchor_age<self.settings["max_anchor_age"]:
            return self.accept_flow(gray,candidate)
        self.misses+=1
        if self.misses>=2:
            self.gray=None
            self.features=None
            self.polygon=None
            self.previous=None
        return {"center":None,"confidence":0.,"measurement_valid":False,
                "seed_approved":False,"reason":flow_failure or "independent_anchor_required"}


class CalibratedDoorTracker(DoorTracker):
    def __init__(self, first, settings, **kwargs):
        super().__init__(first)
        self.settings = settings

    def update(self, frame):
        result = super().update(frame)
        if result.get("center") is None:
            return result
        raw_gap = result["door_edge_x"] - self.closed_reference_x
        excess = raw_gap - self.settings["closed_panel_offset_px"]
        tolerance = self.settings["closed_tolerance_px"]
        if -tolerance <= excess <= tolerance:
            label = "closed"
        elif excess >= self.settings["open_excess_px"]:
            label = "open"
        else:
            label = "uncertain"
        result.update(raw_edge_seam_gap_px=raw_gap,
                      calibrated_closure_excess_px=excess,
                      closed_outer_reference_x=self.closed_reference_x+
                      self.settings["closed_panel_offset_px"],
                      closure_state=label,
                      closed_proposal=label=="closed",
                      reason="calibrated_edge_"+label)
        return result


def build_templates(config, dataset_root):
    from pathlib import Path
    templates = []
    for source in config["templates"]:
        env, ep, f = source["environment"], source["episode_index"], source["frame"]
        path = (Path(dataset_root) / (env+"_lerobot") / "videos" /
                f"chunk-{ep//1000:03d}" / "observation.images.image" /
                f"episode_{ep:06d}.mp4")
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot decode template {path} frame {f}")
        x,y,w,h = source["bbox"]
        crop = frame[y:y+h,x:x+w]
        if crop.shape[:2] != (h,w):
            raise ValueError(f"Template box outside image: {source}")
        for mirrored in ([False,True] if config.get("mirror_templates",False) else [False]):
            view=cv2.flip(crop,1) if mirrored else crop
            for scale in config["scales"]:
                resized = cv2.resize(view, (round(w*scale), round(h*scale)))
                rh,rw = resized.shape[:2]
                for angle in config["angles_deg"]:
                    matrix = cv2.getRotationMatrix2D(((rw-1)/2,(rh-1)/2), angle, 1.)
                    rotated = cv2.warpAffine(resized, matrix, (rw,rh),
                                             borderMode=cv2.BORDER_REFLECT_101)
                    templates.append({"source": f"{env}/{ep}/{f}",
                                      "image": highpass(rotated)})
    return templates


def make_calibrated_tracker(task, frame, calibration, templates):
    if (frame.shape[1],frame.shape[0]) != SIZES[task]:
        raise ValueError("Invert model letterbox/resize into native dataset pixels first")
    if task == "soft":
        cls = (AnchoredTerminalTracker if calibration["soft"].get("tracker_version")=="v3"
               else TerminalBlockTracker)
        return cls(frame, templates, calibration["soft"])
    if task == "door":
        return CalibratedDoorTracker(frame, calibration["door"])
    return {"ball": BallTracker, "stick": StickTracker}[task](frame)
