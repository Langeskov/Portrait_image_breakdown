"""Automatic camera reconstruction from 2D pose and scene evidence."""
from __future__ import annotations

import numpy as np

from reverse_engineering.geometry import CameraIntrinsics, CameraExtrinsics, PoseCandidate, PoseSolver, REF_PERSON_HEIGHT
from reverse_engineering.rotation_solver import fuse_pose_and_scene, estimate_rotation_candidates
from reverse_engineering.scene_geometry import SceneGeometryEvidence


def _subject_position_loss(observed: tuple[float, float]) -> float:
    x, y = observed
    return max(0.0, abs(x - 0.5) - 0.5) ** 2 + max(0.0, abs(y - 0.5) - 0.5) ** 2


def _dedupe_candidates(candidates, max_candidates):
    out = []
    for c in sorted(candidates, key=lambda x: (-float(x.score), float(x.losses.get("mean_reprojection_px", 1e9)))):
        if any(
            abs(c.focal_equiv_35mm - u.focal_equiv_35mm) < 5.0
            and abs(c.distance - u.distance) < 0.25
            and abs(c.extrinsics.yaw - u.extrinsics.yaw) < 4.0
            and abs(c.extrinsics.pitch - u.extrinsics.pitch) < 4.0
            and abs(c.extrinsics.roll - u.extrinsics.roll) < 2.0
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
    body_h = max(float(np.ptp(kp[:17, 1][valid])), 20.0)
    intr = CameraIntrinsics.from_focal_mm(focal_mm, image_w, image_h)
    distance = max(1.0, REF_PERSON_HEIGHT * intr.fy / body_h)
    center_y = float(np.mean(kp[:17, 1][valid]) / max(image_h, 1))
    height = float(np.clip(REF_PERSON_HEIGHT * (0.8 - 0.45 * center_y), 0.5, 1.8))
    return PoseCandidate(
        intrinsics=intr,
        extrinsics=CameraExtrinsics(
            rvec=np.zeros(3), tvec=np.array([0.0, 0.0, distance]),
            position=np.array([0.0, height, -distance]), pitch=0.0, yaw=0.0, roll=0.0,
        ),
        distance=float(distance), height=float(height), focal_equiv_35mm=float(focal_mm), score=0.18,
        losses={
            "mean_reprojection_px": float(body_h), "median_reprojection_px": float(body_h),
            "bbox_iou": 0.0, "fallback": 1.0,
            "fallback_reason": "numerical camera fit returned no solution",
        },
    )


def optimize_parameters(
    image_w,
    image_h,
    subject_scale,
    subject_position,
    perspective_strength,
    pose_keypoints=None,
    num_candidates=5,
    subject_bbox=None,
    scene_evidence: SceneGeometryEvidence | None = None,
):
    """Build ranked camera solutions from pose framing plus scene orientation.

    The pose solver establishes plausible framing/distance. When Manhattan
    scene evidence is available, a second solver establishes rotation and a
    scene-supported focal family; the two are then fused and re-scored.
    """
    del subject_scale, perspective_strength
    if pose_keypoints is None:
        return []
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
        return []

    try:
        pose_candidates = PoseSolver.fit_camera_to_pose(
            kp, image_w, image_h,
            subject_bbox=subject_bbox,
            focal_seeds=(28, 35, 50, 70, 85, 105, 135, 200),
            num_candidates=max(8, num_candidates),
        )
    except Exception as exc:
        fallback = _fallback_candidate(image_w, image_h, kp)
        if fallback is not None:
            fallback.losses["fit_exception"] = f"{type(exc).__name__}: {exc}"
            return [fallback]
        return []

    if not pose_candidates:
        fallback = _fallback_candidate(image_w, image_h, kp)
        return [fallback] if fallback is not None else []

    position_penalty = min(_subject_position_loss(subject_position), 0.05)
    for c in pose_candidates:
        c.losses["subject_position_prior"] = round(position_penalty, 5)
        c.score = float(np.clip(c.score - position_penalty, 0.05, 0.99))

    if scene_evidence is not None:
        rotation_candidates = estimate_rotation_candidates(scene_evidence, image_w, image_h, max_candidates=8)
        fused = fuse_pose_and_scene(
            pose_candidates,
            rotation_candidates,
            image_w,
            image_h,
            kp,
            subject_bbox=subject_bbox,
            max_candidates=max(8, num_candidates),
        )
        if fused:
            return _dedupe_candidates(fused, num_candidates)

    return _dedupe_candidates(pose_candidates, num_candidates)
