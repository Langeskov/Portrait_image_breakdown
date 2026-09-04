"""
景深分析 - DepthOfFieldEstimator
"""
from __future__ import annotations
import cv2
import numpy as np
from reverse_engineering.data_types import EstimatedValue, DepthOfFieldResult
from core.pose_detector import PoseResult, LandmarkIndex as LI

def _lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _region_sharp(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask.astype(np.uint8))
    roi = masked[mask > 0]
    return _lap_var(roi) if len(roi) > 100 else 0.0

def _make_mask(h, w, region, bbox=None):
    mask = np.zeros((h, w), dtype=np.uint8)
    if region == "subject" and bbox:
        x1, y1, x2, y2 = bbox
        pad = int((x2-x1)*0.1)
        mask[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)] = 255
    elif region == "background": mask[:int(h*0.33), :] = 255
    elif region == "foreground": mask[int(h*0.66):, :] = 255
    elif region == "periphery": mask[:, :int(w*0.15)] = 255; mask[:, int(w*0.85):] = 255
    return mask

def analyze_depth_of_field(image, pose=None, bbox=None):
    h, w = image.shape[:2]
    if bbox is None and pose is not None:
        pts = [(lm.world_x*w, lm.world_y*h) for lm in pose.landmarks[:17] if lm.visibility > 0.4]
        if pts:
            pts = np.array(pts)
            bbox = (int(pts[:,0].min()), int(pts[:,1].min()), int(pts[:,0].max()), int(pts[:,1].max()))
    subj_sharp = _region_sharp(image, _make_mask(h, w, "subject", bbox)) if bbox else _lap_var(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    bg_sharp = _region_sharp(image, _make_mask(h, w, "background"))
    fg_sharp = _region_sharp(image, _make_mask(h, w, "foreground"))
    peri_sharp = _region_sharp(image, _make_mask(h, w, "periphery"))
    ref = max(subj_sharp, 1.0)
    bg_blur = 1.0 - min(bg_sharp / ref, 1.0)
    fg_blur = 1.0 - min(fg_sharp / ref, 1.0)
    peri_blur = 1.0 - min(peri_sharp / ref, 1.0)
    bg_blur_final = max(bg_blur, peri_blur)
    avg = (bg_blur + fg_blur + peri_blur) / 3
    if avg > 0.5: dt, dc, ap, ac = "shallow", min(avg*1.3, 0.9), "f/1.4-f/2.8", 0.5
    elif avg > 0.25: dt, dc, ap, ac = "medium", 0.6, "f/2.8-f/5.6", 0.4
    else: dt, dc, ap, ac = "deep", min((1-avg)*1.3, 0.85), "f/5.6-f/11", 0.35
    basis = [f"bg_blur={bg_blur_final:.2f}", f"fg_blur={fg_blur:.2f}"]
    return DepthOfFieldResult(
        dof_type=EstimatedValue(dt, confidence=dc, basis=basis),
        subject_sharp=subj_sharp > 50, foreground_blur=round(fg_blur, 3),
        background_blur=round(bg_blur_final, 3),
        aperture_range=EstimatedValue(ap, unit="f-stop", confidence=ac, basis=basis + ["aperture cannot be uniquely determined"]))
