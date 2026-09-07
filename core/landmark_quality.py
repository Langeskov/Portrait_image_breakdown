"""Pose landmark quality and semantic anchor utilities for field assistance."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LandmarkQuality:
    visible_count: int
    mean_confidence: float
    lower_body_confidence: float
    face_confidence: float
    anchor_confidence: float
    pose_state: str


def assess_landmarks(landmarks, threshold: float = 0.4) -> LandmarkQuality:
    conf = np.array([float(getattr(lm, "visibility", 0.0)) for lm in landmarks[:17]], dtype=float)
    visible = conf > threshold
    lower = conf[11:17] if len(conf) >= 17 else np.array([], dtype=float)
    face = conf[0:5] if len(conf) >= 5 else np.array([], dtype=float)
    visible_count = int(visible.sum())
    mean = float(conf[visible].mean()) if visible.any() else 0.0
    lower_conf = float(lower.mean()) if lower.size else 0.0
    face_conf = float(face.mean()) if face.size else 0.0
    anchor_conf = max(lower_conf, face_conf * 0.7, mean * 0.5)

    if lower_conf >= 0.45 and visible_count >= 10:
        pose_state = "grounded"
    elif lower_conf >= 0.35 and visible_count >= 8:
        pose_state = "lower-body-partial"
    else:
        pose_state = "upper-body"
    return LandmarkQuality(visible_count, mean, lower_conf, face_conf, anchor_conf, pose_state)


def semantic_anchor_pixels(landmarks, image_w: int, image_h: int, quality: LandmarkQuality | None = None) -> tuple[float, float]:
    """Choose a stable body anchor: hips first, torso second, head last."""
    groups = ((11, 12), (5, 6), (0,))
    for group in groups:
        pts = [(float(landmarks[i].x), float(landmarks[i].y))
               for i in group if i < len(landmarks) and float(landmarks[i].visibility) >= 0.4]
        if pts:
            return tuple(np.mean(np.asarray(pts), axis=0).tolist())
    return image_w * 0.5, image_h * 0.5


def face_direction_degrees(landmarks) -> float | None:
    """Estimate a coarse left/right face direction from ears relative to nose."""
    if len(landmarks) < 5:
        return None
    nose, le, re = landmarks[0], landmarks[3], landmarks[4]
    if min(float(nose.visibility), float(le.visibility), float(re.visibility)) < 0.45:
        return None
    eye_mid = np.array([(float(le.x) + float(re.x)) * .5, (float(le.y) + float(re.y)) * .5])
    dx = float(nose.x) - eye_mid[0]
    dy = float(nose.y) - eye_mid[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, max(abs(dy), 1e-6)))
