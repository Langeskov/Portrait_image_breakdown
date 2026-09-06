"""Camera pose estimation from the shared bounded 2D-pose fitting model."""
from __future__ import annotations

import math

import cv2
import numpy as np

from reverse_engineering.data_types import EstimatedValue, CameraPoseResult
from reverse_engineering.geometry import PoseSolver
from core.pose_detector import PoseResult


def estimate_image_roll(image) -> tuple[float, float, int]:
    """Estimate image roll from long near-horizontal scene lines.

    Returns ``(roll_degrees, confidence, line_count)``.  The estimator is
    deliberately conservative: it only uses long Hough segments and reports
    confidence from the amount and consistency of usable evidence.
    """
    if image is None:
        return 0.0, 0.0, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if getattr(image, "ndim", 0) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    h, w = gray.shape[:2]
    min_len = max(40.0, w * 0.45)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800.0, threshold=max(30, int(w * 0.08)), minLineLength=int(min_len), maxLineGap=max(10, int(w * 0.03)))
    if lines is None:
        return 0.0, 0.0, 0

    angles = []
    for raw in lines[:, 0]:
        x1, y1, x2, y2 = map(float, raw)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        # Scene lines used for roll should be reasonably close to horizontal.
        if abs(angle) <= 20.0:
            angles.append((angle, length))

    if not angles:
        return 0.0, 0.0, 0

    values = np.array([a for a, _ in angles], dtype=float)
    weights = np.array([l for _, l in angles], dtype=float)
    roll = float(np.average(values, weights=weights))
    spread = float(np.sqrt(np.average((values - roll) ** 2, weights=weights)))
    count_conf = min(1.0, len(angles) / 6.0)
    consistency = math.exp(-spread / 5.0)
    confidence = float(np.clip(count_conf * consistency, 0.0, 1.0))
    return roll, confidence, len(angles)


def estimate_camera_pose(
    pose: PoseResult,
    perspective_vanishing_points=None,
    representative_focal_mm: float = 50.0,
    image=None,
    subject_bbox=None,
) -> CameraPoseResult:
    """Return the same best-fit camera solution used by the 3D workspace.

    Vanishing-point and image-roll experiments are intentionally not used as
    independent overrides here. The camera state must come from the shared
    reprojection objective so Camera reconstruction and Projection Preview stay
    numerically consistent.
    """
    del perspective_vanishing_points, representative_focal_mm, image
    h, w = pose.image_height, pose.image_width
    kp = [[lm.x, lm.y, lm.visibility] for lm in pose.landmarks[:17]]
    candidates = PoseSolver.fit_camera_to_pose(
        kp, w, h,
        subject_bbox=subject_bbox,
        num_candidates=8,
    )
    if candidates:
        best = max(candidates, key=lambda c: c.score)
        return CameraPoseResult(
            camera_height=EstimatedValue(round(best.height, 2), unit="m", range_min=0.25, range_max=2.2, confidence=min(0.8, best.score), basis=["shared 2D pose reprojection fit", "pose-driven 3D proxy"]),
            camera_distance=EstimatedValue(round(best.distance, 2), unit="m", range_min=round(max(0.5, best.distance * 0.75), 2), range_max=round(best.distance * 1.35, 2), confidence=min(0.75, best.score), basis=["shared reprojection fit", "focal/distance ambiguity retained"]),
            camera_pitch=EstimatedValue(round(best.extrinsics.pitch, 1), unit="deg", range_min=-30, range_max=30, confidence=min(0.75, best.score), basis=["shared reprojection fit"]),
            camera_yaw=EstimatedValue(round(best.extrinsics.yaw, 1), unit="deg", range_min=-30, range_max=30, confidence=min(0.7, best.score), basis=["shared reprojection fit"]),
            camera_roll=EstimatedValue(round(best.extrinsics.roll, 1), unit="deg", range_min=-12, range_max=12, confidence=min(0.65, best.score), basis=["shared reprojection fit", "roll is weakly constrained by 2D pose"]),
        )

    return CameraPoseResult(
        camera_height=EstimatedValue(1.0, unit="m", range_min=0.5, range_max=2.2, confidence=0.1, basis=["fallback"]),
        camera_distance=EstimatedValue(4.0, unit="m", range_min=2.0, range_max=8.0, confidence=0.1, basis=["fallback"]),
        camera_pitch=EstimatedValue(0.0, unit="deg", range_min=-30, range_max=30, confidence=0.1, basis=["fallback"]),
        camera_yaw=EstimatedValue(0.0, unit="deg", range_min=-30, range_max=30, confidence=0.1, basis=["fallback"]),
        camera_roll=EstimatedValue(0.0, unit="deg", range_min=-12, range_max=12, confidence=0.1, basis=["fallback"]),
    )
