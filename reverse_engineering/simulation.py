"""Automatic camera reconstruction from 2D pose evidence.

The primary solver is a bounded nonlinear reprojection fit. OpenCV supplies the
pinhole projection model while SciPy supplies robust bounded least-squares.
A deterministic geometry fallback is retained only as a runtime safety net if
the numerical solver cannot produce a solution; it is explicitly marked in the
candidate losses so it cannot be mistaken for a fitted optimum.
"""
from __future__ import annotations

import numpy as np

from reverse_engineering.geometry import (
    CameraIntrinsics,
    CameraExtrinsics,
    PoseCandidate,
    PoseSolver,
    REF_PERSON_HEIGHT,
)


def _subject_position_loss(observed: tuple[float, float]) -> float:
    x, y = observed
    return max(0.0, abs(x - 0.5) - 0.5) ** 2 + max(0.0, abs(y - 0.5) - 0.5) ** 2


def _dedupe_candidates(candidates, max_candidates):
    out = []
    for c in sorted(candidates, key=lambda x: (
        float(x.losses.get("mean_reprojection_px", 1e9)), -float(x.score)
    )):
        if any(
            abs(c.focal_equiv_35mm - u.focal_equiv_35mm) < 4.0
            and abs(c.distance - u.distance) < 0.2
            and abs(c.extrinsics.yaw - u.extrinsics.yaw) < 2.0
            for u in out
        ):
            continue
        out.append(c)
        if len(out) >= max(1, max_candidates):
            break
    return out


def _fallback_candidate(image_w, image_h, pose_keypoints, focal_mm=50.0):
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
        return None
    valid = kp[:17, 2] > 0.35
    if int(valid.sum()) < 5:
        return None
    ys = kp[:17, 1][valid]
    body_h = max(float(np.ptp(ys)), 20.0)
    intr = CameraIntrinsics.from_focal_mm(focal_mm, image_w, image_h)
    distance = max(1.0, REF_PERSON_HEIGHT * intr.fy / body_h)
    center_y = float(np.mean(ys) / max(image_h, 1))
    height = float(np.clip(REF_PERSON_HEIGHT * (0.8 - 0.45 * center_y), 0.5, 1.8))
    return PoseCandidate(
        intrinsics=intr,
        extrinsics=CameraExtrinsics(
            rvec=np.zeros(3),
            tvec=np.array([0.0, 0.0, distance]),
            position=np.array([0.0, height, -distance]),
            pitch=0.0, yaw=0.0, roll=0.0,
        ),
        distance=float(distance),
        height=float(height),
        focal_equiv_35mm=float(focal_mm),
        score=0.18,
        losses={
            "mean_reprojection_px": float(body_h),
            "median_reprojection_px": float(body_h),
            "bbox_iou": 0.0,
            "fallback": 1.0,
            "fallback_reason": "numerical camera fit returned no solution",
        },
    )


def optimize_parameters(
    image_w, image_h, subject_scale, subject_position, perspective_strength,
    pose_keypoints=None, num_candidates=5, subject_bbox=None,
):
    """Fit Distance/Focal/Height/Yaw/Pitch/Roll from 2D pose + BBox evidence."""
    del subject_scale, perspective_strength
    if pose_keypoints is None:
        return []
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
        return []

    try:
        candidates = PoseSolver.fit_camera_to_pose(
            kp, image_w, image_h, subject_bbox=subject_bbox,
            focal_seeds=(24, 35, 50, 70, 85, 105, 135, 200),
            num_candidates=max(8, num_candidates),
        )
    except Exception as exc:
        fallback = _fallback_candidate(image_w, image_h, kp)
        if fallback is None:
            return []
        fallback.losses["fit_exception"] = f"{type(exc).__name__}: {exc}"
        return [fallback]

    if not candidates:
        fallback = _fallback_candidate(image_w, image_h, kp)
        return [fallback] if fallback is not None else []

    position_penalty = min(_subject_position_loss(subject_position), 0.05)
    for c in candidates:
        c.losses["subject_position_prior"] = round(position_penalty, 5)
        c.score = float(np.clip(c.score - position_penalty, 0.05, 0.99))
    return _dedupe_candidates(candidates, num_candidates)
