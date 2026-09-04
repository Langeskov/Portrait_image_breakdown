"""
运动模糊分析 - MotionBlurAnalyzer
"""
from __future__ import annotations
import cv2
import numpy as np
import math
from reverse_engineering.data_types import EstimatedValue, MotionBlurResult

def _blur_mag(gray):
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return max(0.0, 1.0 - var / 800.0)

def _blur_dir(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    strong = mag > np.percentile(mag, 85)
    if strong.sum() < 100: return None
    mean_a = float(np.mean(angle[strong]))
    return (math.cos(mean_a + math.pi/2), math.sin(mean_a + math.pi/2))

def analyze_motion_blur(image, bbox=None):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    full_mag = _blur_mag(gray)
    subj_mag = full_mag
    if bbox:
        x1, y1, x2, y2 = bbox
        roi = gray[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if roi.size > 100: subj_mag = _blur_mag(roi)
    blur_dir = _blur_dir(gray)
    if full_mag < 0.15:
        bt, bc, se, sr, sc = "none", 0.8, "1/250+", ("1/250","1/8000"), 0.3
    elif subj_mag > 0.4 and full_mag > 0.3:
        if blur_dir and abs(blur_dir[0]) > abs(blur_dir[1]):
            bt, bc, se, sr, sc = "panning", 0.6, "1/30-1/80", ("1/30","1/80"), 0.55
        else:
            bt, bc, se, sr, sc = "long_exposure", 0.5, "1/4-2s", ("1/4","2"), 0.4
    elif subj_mag > 0.35:
        bt, bc, se, sr, sc = "subject", 0.55, "1/60-1/125", ("1/60","1/125"), 0.45
    elif full_mag > 0.3:
        bt, bc, se, sr, sc = "camera", 0.5, "1/30-1/60", ("1/30","1/60"), 0.4
    else:
        bt, bc, se, sr, sc = "none", 0.5, "1/125+", ("1/125","1/8000"), 0.3
    return MotionBlurResult(
        blur_type=EstimatedValue(bt, confidence=bc, basis=[f"full={full_mag:.2f}", f"subj={subj_mag:.2f}"]),
        blur_direction=blur_dir, blur_magnitude=round(full_mag, 3),
        shutter_range=EstimatedValue(se, unit="s", range_min=sr[0], range_max=sr[1],
                                      confidence=sc, basis=[f"type={bt}", "shutter cannot be uniquely determined"]))
