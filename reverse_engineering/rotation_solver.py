"""Camera rotation recovery and scene/pose candidate fusion."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from reverse_engineering.geometry import CameraIntrinsics, CameraModel, CameraExtrinsics, PoseCandidate, pose_driven_person_points
from reverse_engineering.scene_geometry import SceneGeometryEvidence, VanishingPoint


@dataclass(frozen=True)
class RotationCandidate:
    focal_length_mm: float
    extrinsics: CameraExtrinsics
    scene_score: float
    orthogonality_error: float
    horizon_error_deg: float
    vanishing_point_support: float
    evidence: tuple[str, ...]


def _normalize_angle(angle: float) -> float:
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _vp_ray(vp: VanishingPoint, intrinsics: CameraIntrinsics) -> np.ndarray:
    ray = np.linalg.inv(intrinsics.to_matrix()) @ np.array([vp.x, vp.y, 1.0], dtype=np.float64)
    return ray / max(np.linalg.norm(ray), 1e-12)


def _focal_from_orthogonal_vps(a: VanishingPoint, b: VanishingPoint, width: int, height: int, sensor_w: float = 36.0) -> float | None:
    sensor_h = sensor_w * height / max(width, 1)
    ax, ay = a.x - width * 0.5, a.y - height * 0.5
    bx, by = b.x - width * 0.5, b.y - height * 0.5
    term = ax * bx * (sensor_w / width) ** 2 + ay * by * (sensor_h / height) ** 2
    if term >= -1e-6:
        return None
    focal_sq = -term
    if focal_sq <= 0:
        return None
    focal = math.sqrt(focal_sq)
    return float(focal) if 18.0 <= focal <= 220.0 else None


def _rotation_from_vps(vps: tuple[VanishingPoint, VanishingPoint, VanishingPoint], intrinsics: CameraIntrinsics):
    best = None
    for hx, hz in ((vps[0], vps[1]), (vps[1], vps[0])):
        rx0, rz0, ry0 = _vp_ray(hx, intrinsics), _vp_ray(hz, intrinsics), _vp_ray(vps[2], intrinsics)
        if rz0[2] < 0:
            rz0 = -rz0
        if ry0[1] > 0:
            ry0 = -ry0
        raw_err = abs(float(np.dot(rx0, ry0))) + abs(float(np.dot(rx0, rz0))) + abs(float(np.dot(ry0, rz0)))

        # Polar decomposition converts the noisy VP triad into the nearest
        # proper rotation while preserving the measured directions.
        M = np.column_stack([rx0, ry0, rz0])
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1.0
            R = U @ Vt

        camera_to_world = R.T
        forward = camera_to_world @ np.array([0.0, 0.0, 1.0])
        forward /= max(np.linalg.norm(forward), 1e-12)
        yaw = math.degrees(math.atan2(-forward[0], forward[2]))
        pitch = math.degrees(math.asin(float(np.clip(-forward[1], -1.0, 1.0))))

        world_up = np.array([0.0, 1.0, 0.0])
        right0 = np.cross(world_up, forward)
        if np.linalg.norm(right0) < 1e-8:
            continue
        right0 /= np.linalg.norm(right0)
        up0 = np.cross(forward, right0)
        up0 /= max(np.linalg.norm(up0), 1e-12)
        camera_up = camera_to_world @ np.array([0.0, -1.0, 0.0])
        camera_up /= max(np.linalg.norm(camera_up), 1e-12)
        roll = math.degrees(math.atan2(
            float(np.dot(np.cross(up0, camera_up), forward)),
            float(np.dot(up0, camera_up)),
        ))
        candidate = (R, (pitch, yaw, roll), raw_err)
        if best is None or raw_err < best[2]:
            best = candidate
    return best


def estimate_rotation_candidates(evidence: SceneGeometryEvidence, image_w: int, image_h: int, max_candidates: int = 8) -> list[RotationCandidate]:
    if not evidence.has_three_directions:
        return []
    vertical = next((vp for vp in evidence.vanishing_points if vp.cluster == evidence.vertical_cluster), None)
    horizontal = [vp for vp in evidence.vanishing_points if vp.cluster in evidence.horizontal_clusters]
    if vertical is None or len(horizontal) < 2:
        return []

    focal_values: set[float] = set()
    for pair in ((horizontal[0], horizontal[1]), (horizontal[0], vertical), (horizontal[1], vertical)):
        focal = _focal_from_orthogonal_vps(pair[0], pair[1], image_w, image_h)
        if focal is not None:
            # Keep a local neighborhood so noisy VPs do not force one exact f.
            for delta in (-8.0, -4.0, 0.0, 4.0, 8.0):
                focal_values.add(round(float(np.clip(focal + delta, 20.0, 200.0)), 2))
    focal_values.update((28.0, 35.0, 50.0, 70.0, 85.0, 105.0, 135.0))

    results: list[RotationCandidate] = []
    for focal in sorted(focal_values):
        intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h, 36.0, 36.0 * image_h / max(image_w, 1))
        rotation = _rotation_from_vps((horizontal[0], horizontal[1], vertical), intr)
        if rotation is None:
            continue
        R, (pitch, yaw, roll), orth_err = rotation
        horizon = evidence.horizon_angle_deg
        horizon_error = abs(_normalize_angle(roll - horizon)) if horizon is not None else 0.0
        support = float(np.mean([horizontal[0].confidence, horizontal[1].confidence, vertical.confidence]))
        orth_score = math.exp(-orth_err / 0.18)
        horizon_score = math.exp(-horizon_error / 6.0) if horizon is not None else 0.45

        # Penalize focal hypotheses that explain the VP triad poorly.  A scene
        # supported focal gets a substantially higher score than a pose-only
        # arbitrary long-lens hypothesis.
        scene_score = float(np.clip(
            0.58 * orth_score + 0.22 * support + 0.20 * horizon_score,
            0.01, 0.99,
        ))
        rvec, _ = cv2.Rodrigues(R)
        results.append(RotationCandidate(
            focal_length_mm=float(focal),
            extrinsics=CameraExtrinsics(rvec.reshape(3), np.zeros(3), np.zeros(3), pitch, yaw, roll),
            scene_score=scene_score,
            orthogonality_error=float(orth_err),
            horizon_error_deg=float(horizon_error),
            vanishing_point_support=support,
            evidence=(
                f"{len(evidence.lines)} scene lines",
                f"VP support {vertical.support}/{horizontal[0].support}/{horizontal[1].support}",
                f"orthogonality error {orth_err:.3f}",
                f"horizon roll {horizon:.1f}°" if horizon is not None else "horizon unavailable",
            ),
        ))

    results.sort(key=lambda c: (-c.scene_score, c.orthogonality_error, c.focal_length_mm))
    unique: list[RotationCandidate] = []
    for candidate in results:
        if any(
            abs(candidate.focal_length_mm - u.focal_length_mm) < 5.0
            and abs(candidate.extrinsics.yaw - u.extrinsics.yaw) < 4.0
            and abs(candidate.extrinsics.pitch - u.extrinsics.pitch) < 4.0
            and abs(candidate.extrinsics.roll - u.extrinsics.roll) < 2.0
            for u in unique
        ):
            continue
        unique.append(candidate)
        if len(unique) >= max(1, max_candidates):
            break
    return unique


def _camera_from_candidate(candidate: PoseCandidate, rotation: RotationCandidate, image_w: int, image_h: int, subject_bbox=None, pose_keypoints=None):
    focal = float(rotation.focal_length_mm)
    distance = float(candidate.distance) * focal / max(float(candidate.focal_equiv_35mm), 1e-6)
    height = float(candidate.height)
    position = np.array([
        0.0,
        height,
        -distance,
    ], dtype=float)
    R = cv2.Rodrigues(rotation.extrinsics.rvec)[0]
    tvec = -R @ position
    intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h, 36.0, 36.0 * image_h / max(image_w, 1))
    ext = CameraExtrinsics(
        rotation.extrinsics.rvec.copy(), tvec, position,
        rotation.extrinsics.pitch, rotation.extrinsics.yaw, rotation.extrinsics.roll,
    )
    camera = CameraModel(intr, ext)
    if pose_keypoints is None:
        return PoseCandidate(intr, ext, distance, height, focal, candidate.score, dict(candidate.losses))

    obj = pose_driven_person_points(pose_keypoints, image_w, image_h)
    valid = np.isfinite(obj).all(axis=1)
    points = np.asarray(pose_keypoints, dtype=float)
    valid &= points[:, 2] > 0.35
    projected = camera.project_points(obj)
    valid &= np.isfinite(projected).all(axis=1)
    if int(valid.sum()) < 5:
        return None
    errors = np.linalg.norm(projected[valid] - points[valid, :2], axis=1)
    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))

    iou = 0.0
    if subject_bbox is not None:
        pb = np.array([np.min(projected[valid, 0]), np.min(projected[valid, 1]), np.max(projected[valid, 0]), np.max(projected[valid, 1])])
        bx0, by0, bx1, by1 = map(float, subject_bbox)
        iw = max(0.0, min(pb[2], bx1) - max(pb[0], bx0))
        ih = max(0.0, min(pb[3], by1) - max(pb[1], by0))
        inter = iw * ih
        ap = max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1])
        ao = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        iou = inter / max(ap + ao - inter, 1e-9)

    pose_score = math.exp(-mean_error / max(12.0, 0.012 * math.hypot(image_w, image_h)))
    combined = float(np.clip(0.52 * pose_score + 0.28 * iou + 0.20 * rotation.scene_score, 0.01, 0.99))
    losses = dict(candidate.losses)
    losses.update({
        "mean_reprojection_px": round(mean_error, 3),
        "median_reprojection_px": round(median_error, 3),
        "bbox_iou": round(iou, 4),
        "scene_score": round(rotation.scene_score, 4),
        "scene_focal_mm": round(focal, 3),
        "scene_yaw": round(rotation.extrinsics.yaw, 3),
        "scene_pitch": round(rotation.extrinsics.pitch, 3),
        "scene_roll": round(rotation.extrinsics.roll, 3),
    })
    return PoseCandidate(intr, ext, distance, height, focal, combined, losses)


def fuse_pose_and_scene(
    pose_candidates: list[PoseCandidate],
    rotation_candidates: list[RotationCandidate],
    image_w: int,
    image_h: int,
    pose_keypoints: np.ndarray,
    subject_bbox=None,
    max_candidates: int = 5,
) -> list[PoseCandidate]:
    if not pose_candidates or not rotation_candidates:
        return []
    fused: list[PoseCandidate] = []
    for rotation in rotation_candidates:
        nearest = sorted(pose_candidates, key=lambda p: abs(p.focal_equiv_35mm - rotation.focal_length_mm))[:3]
        for pose_candidate in nearest:
            candidate = _camera_from_candidate(
                pose_candidate, rotation, image_w, image_h,
                subject_bbox=subject_bbox, pose_keypoints=pose_keypoints,
            )
            if candidate is not None:
                fused.append(candidate)
    fused.sort(key=lambda c: (-c.score, c.losses.get("mean_reprojection_px", 1e9)))
    unique: list[PoseCandidate] = []
    for candidate in fused:
        if any(
            abs(candidate.focal_equiv_35mm - u.focal_equiv_35mm) < 5
            and abs(candidate.extrinsics.yaw - u.extrinsics.yaw) < 4
            and abs(candidate.extrinsics.pitch - u.extrinsics.pitch) < 4
            and abs(candidate.extrinsics.roll - u.extrinsics.roll) < 2
            for u in unique
        ):
            continue
        unique.append(candidate)
        if len(unique) >= max(1, max_candidates):
            break
    return unique
