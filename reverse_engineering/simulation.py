"""Automatic camera reconstruction from 2D pose and scene evidence."""
from __future__ import annotations

import math
import numpy as np

from reverse_engineering.geometry import (
    CameraIntrinsics,
    CameraModel,
    PoseCandidate,
    PoseSolver,
    REF_PERSON_HEIGHT,
    _camera_pose_from_params,
    pose_driven_person_points,
)
from reverse_engineering.intrinsics import IntrinsicsEvidence
from reverse_engineering.calibration import CalibrationProfile, resolve_profile
from reverse_engineering.rotation_solver import fuse_pose_and_scene, estimate_rotation_candidates
from reverse_engineering.scene_geometry import SceneGeometryEvidence
from reverse_engineering.scene_constraints import (
    DepthConstraintEvidence,
    candidate_depth_score,
    candidate_feasibility_score,
    estimate_camera_feasibility,
)


def _intrinsics_from_profile(focal_mm, image_w, image_h, profile: CalibrationProfile | None = None):
    """Build pixel intrinsics using a selected calibration profile."""
    profile = profile or resolve_profile("Generic")
    sensor_width = float(profile.sensor_width_mm) if profile.sensor_width_mm else 36.0
    sensor_height = float(profile.sensor_height_mm) if profile.sensor_height_mm else None
    intr = CameraIntrinsics.from_focal_mm(focal_mm, image_w, image_h, sensor_width_mm=sensor_width, sensor_height_mm=sensor_height)
    calibrated = profile.for_image(image_w, image_h)
    if calibrated.principal_point_x is not None:
        intr.cx = float(calibrated.principal_point_x)
    if calibrated.principal_point_y is not None:
        intr.cy = float(calibrated.principal_point_y)
    intr.fy /= max(float(calibrated.pixel_aspect_ratio), 1e-6)
    intr.sensor_width_mm = sensor_width
    intr.sensor_height_mm = sensor_height if sensor_height is not None else intr.sensor_height_mm
    return intr


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


def _fallback_candidate(image_w, image_h, pose_keypoints, focal_mm=50.0, calibration_profile=None):
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
        return None
    valid = kp[:17, 2] > 0.35
    if int(valid.sum()) < 5:
        return None
    body_h = max(float(np.ptp(kp[:17, 1][valid])), 20.0)
    intr = _intrinsics_from_profile(focal_mm, image_w, image_h, calibration_profile)
    distance = max(1.0, REF_PERSON_HEIGHT * intr.fy / body_h)
    center_y = float(np.mean(kp[:17, 1][valid]) / max(image_h, 1))
    height = float(np.clip(REF_PERSON_HEIGHT * (0.8 - 0.45 * center_y), 0.5, 1.8))
    position, extrinsics = _camera_pose_from_params(distance, height, 0.0, 0.0, 0.0)
    return PoseCandidate(
        intrinsics=intr, extrinsics=extrinsics, distance=float(distance), height=float(height),
        focal_equiv_35mm=float(focal_mm), score=0.18,
        losses={"mean_reprojection_px": float(body_h), "median_reprojection_px": float(body_h), "bbox_iou": 0.0, "fallback": 1.0, "fallback_reason": "numerical camera fit returned no solution"},
    )


def _camera_candidate_visibility(candidate, pose_keypoints, image_w, image_h, subject_bbox=None):
    kp = np.asarray(pose_keypoints, dtype=float)
    proxy = pose_driven_person_points(kp, image_w, image_h)
    projected = CameraModel(candidate.intrinsics, candidate.extrinsics).project_points(proxy)
    valid = np.isfinite(projected).all(axis=1) & (kp[:, 2] > 0.35)
    if int(valid.sum()) < 5:
        return 0.0, 0.0
    p = projected[valid]
    inside = ((p[:, 0] >= 0.0) & (p[:, 0] <= image_w) & (p[:, 1] >= 0.0) & (p[:, 1] <= image_h))
    point_inside = float(np.mean(inside))
    if subject_bbox is None:
        return point_inside, point_inside
    x0, y0, x1, y1 = subject_bbox
    pb = np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
    iw = max(0.0, min(pb[2], float(x1)) - max(pb[0], float(x0)))
    ih = max(0.0, min(pb[3], float(y1)) - max(pb[1], float(y0)))
    inter = iw * ih
    pred_area = max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1])
    observed_area = max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
    union = pred_area + observed_area - inter
    bbox_iou = inter / union if union > 1e-9 else 0.0
    return point_inside, bbox_iou


def _scene_has_level_reference(scene_evidence: SceneGeometryEvidence | None):
    if scene_evidence is None or scene_evidence.vertical_cluster is None:
        return False
    clusters = scene_evidence.clusters
    if scene_evidence.vertical_cluster >= len(clusters):
        return False
    vertical_lines = [scene_evidence.lines[i] for i in clusters[scene_evidence.vertical_cluster]]
    if len(vertical_lines) < 3:
        return False
    deviations = [abs(((line.angle_deg - 90.0 + 90.0) % 180.0) - 90.0) for line in vertical_lines]
    return float(np.median(deviations)) < 2.5 and float(np.percentile(deviations, 90)) < 5.0


def _apply_level_reference(candidate, scene_evidence):
    if not _scene_has_level_reference(scene_evidence):
        return candidate
    pitch = float(candidate.extrinsics.pitch)
    if abs(pitch) <= 8.0:
        return candidate
    clamped = math.copysign(8.0, pitch)
    _, extrinsics = _camera_pose_from_params(candidate.distance, candidate.height, candidate.extrinsics.yaw, clamped, candidate.extrinsics.roll)
    candidate.extrinsics = extrinsics
    candidate.losses["pitch_level_reference"] = True
    candidate.losses["original_pitch_deg"] = round(pitch, 3)
    return candidate


def _apply_depth_constraint(candidate, pose_keypoints, image_w, image_h, depth_evidence):
    score = candidate_depth_score(candidate, pose_keypoints, image_w, image_h, depth_evidence)
    if score is None:
        return
    candidate.losses["depth_order_score"] = round(float(score), 4)
    candidate.losses["depth_constraint_confidence"] = round(float(depth_evidence.confidence), 4)
    candidate.score = float(np.clip(candidate.score + 0.12 * depth_evidence.confidence * (score - 0.5), 0.01, 0.99))


def _apply_feasibility_constraint(candidate, pose_keypoints, image_w, image_h, depth_evidence):
    feasibility = estimate_camera_feasibility(
        pose_keypoints, image_w, image_h, candidate.focal_equiv_35mm, depth_evidence,
    )
    score = candidate_feasibility_score(candidate, feasibility)
    if score is None:
        return feasibility
    candidate.losses["feasibility_score"] = round(float(score), 4)
    candidate.losses["feasibility_confidence"] = round(float(feasibility.confidence), 4)
    candidate.score = float(np.clip(candidate.score + 0.08 * feasibility.confidence * (score - 0.5), 0.01, 0.99))
    return feasibility


def optimize_parameters(
    image_w, image_h, subject_scale, subject_position, perspective_strength,
    pose_keypoints=None, num_candidates=5, subject_bbox=None,
    scene_evidence=None, intrinsics_evidence=None, calibration_profile=None,
    depth_evidence=None,
):
    """Build ranked camera solutions from pose, scene, EXIF, calibration and depth evidence."""
    del subject_scale, perspective_strength
    if pose_keypoints is None:
        return []
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
        return []
    profile = calibration_profile or resolve_profile("Generic")
    focal_seeds = [28, 35, 50, 70, 85, 105, 135, 200]
    if profile.default_focal_length_mm:
        pf = float(profile.default_focal_length_mm)
        focal_seeds = sorted(set(focal_seeds + [max(20.0, min(220.0, pf + d)) for d in (-8.0, -4.0, 0.0, 4.0, 8.0)]))
    if intrinsics_evidence is not None and intrinsics_evidence.has_focal_prior:
        focal = intrinsics_evidence.preferred_focal_mm()
        if focal is not None:
            focal_seeds = sorted({round(float(np.clip(focal + delta, 20.0, 220.0)), 2) for delta in (-8.0, -4.0, 0.0, 4.0, 8.0)} | {float(v) for v in focal_seeds})

    pose_candidates = []
    fit_exception = None
    try:
        factory = lambda focal: _intrinsics_from_profile(focal, image_w, image_h, profile)
        pose_candidates = PoseSolver.fit_camera_to_pose(kp, image_w, image_h, subject_bbox=subject_bbox, focal_seeds=tuple(focal_seeds), num_candidates=max(8, num_candidates), intrinsics_factory=factory)
    except Exception as exc:
        fit_exception = f"{type(exc).__name__}: {exc}"

    if not pose_candidates:
        fallback_focal = intrinsics_evidence.preferred_focal_mm() if intrinsics_evidence else None
        fallback = _fallback_candidate(image_w, image_h, kp, fallback_focal or profile.default_focal_length_mm or 50.0, profile)
        if fallback is None:
            return []
        if fit_exception:
            fallback.losses["fit_exception"] = fit_exception
        pose_candidates = [fallback]

    position_penalty = min(_subject_position_loss(subject_position), 0.05)
    scored_pose = []
    for c in pose_candidates:
        c = _apply_level_reference(c, scene_evidence)
        c.losses["subject_position_prior"] = round(position_penalty, 5)
        c.losses["calibration_profile"] = profile.name
        if intrinsics_evidence is not None and intrinsics_evidence.preferred_focal_mm() is not None:
            hint = intrinsics_evidence.preferred_focal_mm()
            focal_delta = abs(float(c.focal_equiv_35mm) - float(hint))
            exif_score = float(np.exp(-focal_delta / max(8.0, 0.10 * float(hint))))
            c.losses["exif_focal_mm"] = round(float(hint), 3)
            c.losses["exif_focal_score"] = round(exif_score, 4)
            c.score = float(np.clip(c.score * (0.82 + 0.18 * exif_score) - position_penalty, 0.05, 0.99))
        else:
            c.score = float(np.clip(c.score - position_penalty, 0.05, 0.99))
        point_inside, bbox_iou = _camera_candidate_visibility(c, kp, image_w, image_h, subject_bbox)
        c.losses["in_frame_fraction"] = round(point_inside, 4)
        c.losses["visibility_bbox_iou"] = round(bbox_iou, 4)
        _apply_depth_constraint(c, kp, image_w, image_h, depth_evidence)
        _apply_feasibility_constraint(c, kp, image_w, image_h, depth_evidence)
        if point_inside < 0.55:
            c.losses["visibility_penalty"] = round(0.20 * (0.55 - point_inside), 4)
            c.score = float(np.clip(c.score - 0.20 * (0.55 - point_inside), 0.01, 0.99))
        else:
            c.score = float(np.clip(c.score + 0.10 * (point_inside - 0.5) + 0.06 * (bbox_iou - 0.5), 0.01, 0.99))
        scored_pose.append(c)

    pose_candidates = scored_pose or pose_candidates

    if scene_evidence is not None:
        rotation_candidates = estimate_rotation_candidates(scene_evidence, image_w, image_h, max_candidates=8)
        fused = fuse_pose_and_scene(pose_candidates, rotation_candidates, image_w, image_h, kp, subject_bbox=subject_bbox, max_candidates=max(8, num_candidates))
        if fused:
            fused_scored = []
            for c in fused:
                c = _apply_level_reference(c, scene_evidence)
                point_inside, bbox_iou = _camera_candidate_visibility(c, kp, image_w, image_h, subject_bbox)
                c.losses["in_frame_fraction"] = round(point_inside, 4)
                c.losses["visibility_bbox_iou"] = round(bbox_iou, 4)
                c.losses["calibration_profile"] = profile.name
                _apply_depth_constraint(c, kp, image_w, image_h, depth_evidence)
                _apply_feasibility_constraint(c, kp, image_w, image_h, depth_evidence)
                if point_inside < 0.55:
                    c.losses["visibility_penalty"] = round(0.20 * (0.55 - point_inside), 4)
                    c.score = float(np.clip(c.score - 0.20 * (0.55 - point_inside), 0.01, 0.99))
                else:
                    c.score = float(np.clip(c.score + 0.10 * (point_inside - 0.5) + 0.06 * (bbox_iou - 0.5), 0.01, 0.99))
                fused_scored.append(c)
            ranked = _dedupe_candidates(fused_scored, num_candidates)
            if len(ranked) >= 2:
                return ranked
            # Do not allow a single fused solution to erase the alternative
            # focal/distance family found from the pose fit.
            return _dedupe_candidates(ranked + pose_candidates, num_candidates)

    return _dedupe_candidates(pose_candidates, num_candidates)
