"""Camera rotation recovery and scene/pose candidate fusion.

Rotation contracts:
- ``RotationCandidate.focal_length_mm`` is a 35mm-equivalent focal prior.
- ``orientation_source='manhattan'`` contains scene yaw/pitch/roll evidence.
- ``orientation_source='roll_only'`` contains only scene roll; pose yaw/pitch,
  focal length and distance remain authoritative during fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from reverse_engineering.geometry import (
    CameraIntrinsics,
    CameraModel,
    CameraExtrinsics,
    PoseCandidate,
    pose_driven_person_points,
    _camera_pose_from_params,
)
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
    orientation_source: str = "manhattan"

    @property
    def focal_equiv_35mm(self) -> float:
        return float(self.focal_length_mm)


def _normalize_angle(angle: float) -> float:
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _axis_angle_distance(angle: float, axis: float) -> float:
    return abs(_normalize_angle(angle - axis))


def _signed_roll_from_line(angle: float, horizontal_family: bool) -> float:
    """Map a scene-line orientation to the equivalent camera roll."""
    return _normalize_angle(angle if horizontal_family else angle - 90.0)


def _circular_mean_line_roll(values: list[float], weights: list[float]) -> float:
    """Weighted mean for line orientation, whose period is 180 degrees."""
    if not values:
        return 0.0
    theta = np.radians(2.0 * np.asarray(values, dtype=float))
    w = np.asarray(weights, dtype=float)
    z = np.sum(w * np.exp(1j * theta))
    if abs(z) <= 1e-12:
        return float(np.average(values, weights=w))
    return _normalize_angle(0.5 * math.degrees(math.atan2(float(z.imag), float(z.real))))


def _estimate_line_roll(evidence: SceneGeometryEvidence) -> tuple[float | None, float, int]:
    """Estimate image roll from tiered line evidence.

    A broad orientation search identifies the dominant scene family. The final
    angle is then recovered from the signed original segment angles rather than
    the centre of the tolerance window, avoiding systematic quantization toward
    zero. Two orthogonal families remain the strongest case; one family is
    accepted only with strong length, spatial coverage and coherence evidence.
    """
    raw_lines = getattr(evidence, "lines", ())
    width = max(float(evidence.width), 1.0)
    height = max(float(evidence.height), 1.0)
    min_length = max(60.0, 0.08 * min(width, height))
    usable: list[tuple[float, float, float, float]] = []
    for line in raw_lines:
        if not hasattr(line, "angle_deg") or not hasattr(line, "length"):
            continue
        length = float(getattr(line, "length", 0.0))
        if not np.isfinite(length) or length < min_length:
            continue
        angle = _normalize_angle(float(getattr(line, "angle_deg", 0.0)))
        if not np.isfinite(angle):
            continue
        mx = (float(getattr(line, "x1", 0.0)) + float(getattr(line, "x2", 0.0))) * 0.5
        my = (float(getattr(line, "y1", 0.0)) + float(getattr(line, "y2", 0.0))) * 0.5
        usable.append((angle, length, mx, my))

    count = len(usable)
    if count < 4:
        return None, 0.0, count

    tolerance = 9.0
    grid = np.arange(-45.0, 45.0001, 0.5)
    weights = np.sqrt(np.asarray([item[1] for item in usable], dtype=float))
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-9:
        return None, 0.0, count

    scored: list[tuple[float, float, float, float]] = []
    for roll in grid:
        h_mask = np.array([
            _axis_angle_distance(angle, roll) <= tolerance
            for angle, _, _, _ in usable
        ])
        v_mask = np.array([
            _axis_angle_distance(angle, _normalize_angle(roll + 90.0)) <= tolerance
            for angle, _, _, _ in usable
        ])
        h_support = float(np.sum(weights[h_mask]))
        v_support = float(np.sum(weights[v_mask]))
        support = h_support + v_support
        balance = min(h_support, v_support) / max(max(h_support, v_support), 1e-9)
        score = support * (0.72 + 0.28 * balance)
        scored.append((score, float(roll), h_support, v_support))

    best_score, best_roll_seed, h_support, v_support = max(scored, key=lambda x: x[0])
    ordered = sorted(scored, key=lambda x: x[0], reverse=True)
    second_score = next(
        (item[0] for item in ordered[1:] if abs(item[1] - best_roll_seed) >= 6.0),
        0.0,
    )
    separation = max(0.0, best_score - second_score)

    dominant_is_horizontal = h_support >= v_support
    dominant_axis = best_roll_seed if dominant_is_horizontal else _normalize_angle(best_roll_seed + 90.0)
    dominant_mask = np.array([
        _axis_angle_distance(angle, dominant_axis) <= tolerance
        for angle, _, _, _ in usable
    ])
    dominant_values = [
        _signed_roll_from_line(item[0], dominant_is_horizontal)
        for item, keep in zip(usable, dominant_mask) if keep
    ]
    dominant_weights = [
        weights[i]
        for i, keep in enumerate(dominant_mask) if keep
    ]

    if dominant_values:
        final_roll = _circular_mean_line_roll(dominant_values, dominant_weights)
        residuals = np.asarray([
            _axis_angle_distance(value, final_roll) for value in dominant_values
        ], dtype=float)
        spread = float(np.sqrt(np.average(residuals ** 2, weights=np.asarray(dominant_weights))))
        mids_x = np.asarray([item[2] for item, keep in zip(usable, dominant_mask) if keep], dtype=float)
        mids_y = np.asarray([item[3] for item, keep in zip(usable, dominant_mask) if keep], dtype=float)
        x_coverage = float(np.ptp(mids_x) / width) if len(mids_x) > 1 else 0.0
        y_coverage = float(np.ptp(mids_y) / height) if len(mids_y) > 1 else 0.0
        max_length = max(item[1] for item, keep in zip(usable, dominant_mask) if keep)
    else:
        final_roll, spread, x_coverage, y_coverage, max_length = 0.0, 99.0, 0.0, 0.0, 0.0

    dominant_support = h_support if dominant_is_horizontal else v_support
    family_fraction = dominant_support / max(total_weight, 1e-9)
    coverage = x_coverage if dominant_is_horizontal else y_coverage
    count_conf = min(1.0, count / 10.0)
    coverage_conf = float(np.clip(coverage / 0.45, 0.0, 1.0))
    length_conf = float(np.clip(max_length / max(0.30 * math.hypot(width, height), 1.0), 0.0, 1.0))
    coherence_conf = math.exp(-max(0.0, spread - 3.0) / 5.0)
    separation_conf = min(1.0, separation / max(best_score * 0.30, 1e-9))

    two_family_conf = float(np.clip(
        0.25 * count_conf
        + 0.30 * min(1.0, (h_support + v_support) / max(total_weight * 0.35, 1e-9))
        + 0.20 * (min(h_support, v_support) / max(max(h_support, v_support), 1e-9))
        + 0.10 * separation_conf
        + 0.15 * coherence_conf,
        0.0,
        1.0,
    ))
    single_family_conf = float(np.clip(
        0.22 * count_conf
        + 0.28 * family_fraction
        + 0.20 * coverage_conf
        + 0.15 * length_conf
        + 0.15 * coherence_conf,
        0.0,
        1.0,
    ))

    strong_two_family = (
        h_support > total_weight * 0.08
        and v_support > total_weight * 0.08
        and two_family_conf >= 0.50
    )
    strong_single_family = (
        dominant_support > total_weight * 0.22
        and family_fraction >= 0.55
        and spread <= 6.0
        and coverage >= 0.30
        and max_length >= 0.25 * math.hypot(width, height)
        and abs(final_roll) <= 25.0
        and single_family_conf >= 0.55
    )

    if not (strong_two_family or strong_single_family):
        return None, float(np.clip(max(two_family_conf, single_family_conf) * 0.65, 0.0, 1.0)), count

    confidence = two_family_conf if strong_two_family else single_family_conf
    return float(final_roll), float(np.clip(confidence, 0.0, 1.0)), count


def _vp_ray(vp: VanishingPoint, intrinsics: CameraIntrinsics) -> np.ndarray:
    ray = np.array([
        (vp.x - intrinsics.cx) / max(intrinsics.fx, 1e-9),
        -(vp.y - intrinsics.cy) / max(intrinsics.fy, 1e-9),
        1.0,
    ], dtype=np.float64)
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


def _angles_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    camera_to_world = np.asarray(R, dtype=np.float64).T
    forward = camera_to_world @ np.array([0.0, 0.0, 1.0])
    forward /= max(np.linalg.norm(forward), 1e-12)
    yaw = math.degrees(math.atan2(-forward[0], forward[2]))
    pitch = math.degrees(math.asin(float(np.clip(-forward[1], -1.0, 1.0))))
    world_up = np.array([0.0, 1.0, 0.0])
    right0 = np.cross(world_up, forward)
    if np.linalg.norm(right0) < 1e-8:
        return pitch, yaw, 0.0
    right0 /= np.linalg.norm(right0)
    up0 = np.cross(forward, right0)
    up0 /= max(np.linalg.norm(up0), 1e-12)
    camera_up = camera_to_world @ np.array([0.0, 1.0, 0.0])
    camera_up /= max(np.linalg.norm(camera_up), 1e-12)
    roll = math.degrees(math.atan2(
        float(np.dot(np.cross(up0, camera_up), forward)),
        float(np.dot(up0, camera_up)),
    ))
    return pitch, yaw, roll


def _rotation_from_vps(vps: tuple[VanishingPoint, VanishingPoint, VanishingPoint], intrinsics: CameraIntrinsics, preferred_horizon_roll: float | None = None):
    best = None
    for hx, hz in ((vps[0], vps[1]), (vps[1], vps[0])):
        rx0, rz0, ry0 = _vp_ray(hx, intrinsics), _vp_ray(hz, intrinsics), _vp_ray(vps[2], intrinsics)
        if rz0[2] < 0:
            rz0 = -rz0
        if ry0[1] < 0:
            ry0 = -ry0
        for sx in (-1.0, 1.0):
            rx = rx0 * sx
            M = np.column_stack([rx, ry0, rz0])
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1.0
                R = U @ Vt
            pitch, yaw, roll = _angles_from_rotation(R)
            raw_err = abs(float(np.dot(rx, ry0))) + abs(float(np.dot(rx, rz0))) + abs(float(np.dot(ry0, rz0)))
            horizon_err = abs(_normalize_angle(roll - preferred_horizon_roll)) if preferred_horizon_roll is not None else 0.0
            depth_vp_radius = math.hypot(hz.x - intrinsics.cx, hz.y - intrinsics.cy) / max(math.hypot(intrinsics.width, intrinsics.height), 1.0)
            preference = 0.0025 * abs(yaw) + 0.01 * max(0.0, abs(pitch) - 60.0) + 0.006 * max(0.0, abs(roll) - 30.0) + 0.15 * depth_vp_radius + 0.25 * horizon_err / 90.0
            quality = raw_err + preference
            candidate = (quality, R, (pitch, yaw, roll), raw_err, horizon_err)
            if best is None or quality < best[0]:
                best = candidate
    return None if best is None else (best[1], best[2], best[3])


def _force_roll(R: np.ndarray, pitch: float, yaw: float, roll: float) -> np.ndarray:
    _, extrinsics = _camera_pose_from_params(1.0, 0.0, yaw, pitch, roll)
    return cv2.Rodrigues(extrinsics.rvec)[0]


def _roll_only_candidate(roll: float, confidence: float, image_w: int, image_h: int, focal: float = 50.0) -> RotationCandidate:
    intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h, 36.0, 36.0 * image_h / max(image_w, 1))
    _, extrinsics = _camera_pose_from_params(1.0, 0.0, 0.0, 0.0, roll)
    return RotationCandidate(
        focal_length_mm=float(focal),
        extrinsics=extrinsics,
        scene_score=float(np.clip(confidence, 0.01, 0.99)),
        orthogonality_error=0.0,
        horizon_error_deg=0.0,
        vanishing_point_support=0.0,
        evidence=("line-roll scene evidence", f"scene roll {roll:.1f}° ({confidence:.0%})"),
        orientation_source="roll_only",
    )


def estimate_rotation_candidates(evidence: SceneGeometryEvidence, image_w: int, image_h: int, max_candidates: int = 8) -> list[RotationCandidate]:
    """Return scene rotation candidates, including roll-only candidates."""
    line_roll, line_roll_confidence, usable_line_count = _estimate_line_roll(evidence)
    if not evidence.has_three_directions:
        return [_roll_only_candidate(line_roll, line_roll_confidence, image_w, image_h)] if line_roll is not None else []

    vertical = next((vp for vp in evidence.vanishing_points if vp.cluster == evidence.vertical_cluster), None)
    horizontal = [vp for vp in evidence.vanishing_points if vp.cluster in evidence.horizontal_clusters]
    if vertical is None or len(horizontal) < 2:
        return [_roll_only_candidate(line_roll, line_roll_confidence, image_w, image_h)] if line_roll is not None else []

    preferred_roll = line_roll if line_roll is not None and line_roll_confidence >= 0.50 else 0.0
    pair_focals: list[float] = []
    for pair in ((horizontal[0], horizontal[1]), (horizontal[0], vertical), (horizontal[1], vertical)):
        f = _focal_from_orthogonal_vps(pair[0], pair[1], image_w, image_h)
        if f is not None:
            pair_focals.append(f)

    focal_values: set[float] = set()
    if pair_focals:
        median_f = float(np.median(pair_focals))
        for base in pair_focals + [median_f]:
            for delta in (-8.0, -4.0, 0.0, 4.0, 8.0):
                focal_values.add(round(float(np.clip(base + delta, 20.0, 200.0)), 2))
    focal_values.update((28.0, 35.0, 50.0, 70.0, 85.0, 105.0, 135.0))

    results: list[RotationCandidate] = []
    for focal in sorted(focal_values):
        intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h, 36.0, 36.0 * image_h / max(image_w, 1))
        rotation = _rotation_from_vps((horizontal[0], horizontal[1], vertical), intr, preferred_roll)
        if rotation is None:
            continue
        R, (pitch, yaw, raw_roll), orth_err = rotation
        trusted_roll = (
            line_roll
            if line_roll is not None and line_roll_confidence >= 0.50
            else (float(evidence.horizon_angle_deg) if usable_line_count == 0 and evidence.horizon_angle_deg is not None else 0.0)
        )
        if abs(raw_roll - trusted_roll) > 1e-6:
            R = _force_roll(R, pitch, yaw, trusted_roll)
            pitch, yaw, _ = _angles_from_rotation(R)
        horizon = evidence.horizon_angle_deg
        horizon_error = abs(_normalize_angle(trusted_roll - horizon)) if horizon is not None else 0.0
        support = float(np.mean([horizontal[0].confidence, horizontal[1].confidence, vertical.confidence]))
        orth_score = math.exp(-orth_err / 0.18)
        horizon_score = math.exp(-horizon_error / 6.0) if horizon is not None else 0.45
        focal_consistency = (
            math.exp(-float(np.median([abs(focal - pf) for pf in pair_focals])) / max(5.0, 0.15 * float(np.median(pair_focals))))
            if pair_focals else 0.2
        )
        roll_score = line_roll_confidence if line_roll is not None else 0.35
        scene_score = float(np.clip(
            .42 * orth_score + .16 * support + .17 * horizon_score + .10 * focal_consistency + .15 * roll_score,
            0.01,
            0.99,
        ))
        rvec, _ = cv2.Rodrigues(R)
        results.append(RotationCandidate(
            focal_length_mm=float(focal),
            extrinsics=CameraExtrinsics(rvec.reshape(3), np.zeros(3), np.zeros(3), pitch, yaw, trusted_roll),
            scene_score=scene_score,
            orthogonality_error=float(orth_err),
            horizon_error_deg=float(horizon_error),
            vanishing_point_support=support,
            evidence=(
                f"{len(evidence.lines)} scene lines",
                f"VP support {vertical.support}/{horizontal[0].support}/{horizontal[1].support}",
                f"orthogonality error {orth_err:.3f}",
                f"focal consistency {focal_consistency:.2f}" if pair_focals else "focal from generic candidate family",
                f"scene roll {trusted_roll:.1f}° ({line_roll_confidence:.0%}, {usable_line_count} lines)" if line_roll is not None else f"scene roll forced to neutral 0° ({usable_line_count} lines)",
            ),
            orientation_source="manhattan",
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
    roll_only = rotation.orientation_source == "roll_only"
    focal = float(candidate.focal_equiv_35mm) if roll_only else float(rotation.focal_length_mm)
    distance = float(candidate.distance) * focal / max(float(candidate.focal_equiv_35mm), 1e-6)
    height = float(candidate.height)
    yaw = float(candidate.extrinsics.yaw) if roll_only else float(rotation.extrinsics.yaw)
    pitch = float(candidate.extrinsics.pitch) if roll_only else float(rotation.extrinsics.pitch)
    roll = float(rotation.extrinsics.roll)
    if roll_only:
        _, ext_seed = _camera_pose_from_params(distance, height, yaw, pitch, roll)
        R = cv2.Rodrigues(ext_seed.rvec)[0]
    else:
        R = cv2.Rodrigues(rotation.extrinsics.rvec)[0]
    position = np.array([
        math.sin(math.radians(yaw)) * distance,
        height,
        -math.cos(math.radians(yaw)) * distance,
    ], dtype=float)
    tvec = -R @ position
    intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h, 36.0, 36.0 * image_h / max(image_w, 1))
    rvec, _ = cv2.Rodrigues(R)
    ext = CameraExtrinsics(rvec.reshape(3), tvec, position, pitch, yaw, roll)
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
    mean_error, median_error = float(np.mean(errors)), float(np.median(errors))
    iou = 0.0
    if subject_bbox is not None:
        pb = np.array([
            np.min(projected[valid, 0]), np.min(projected[valid, 1]),
            np.max(projected[valid, 0]), np.max(projected[valid, 1]),
        ])
        bx0, by0, bx1, by1 = map(float, subject_bbox)
        iw = max(0.0, min(pb[2], bx1) - max(pb[0], bx0))
        ih = max(0.0, min(pb[3], by1) - max(pb[1], by0))
        inter = iw * ih
        ap = max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1])
        ao = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        iou = inter / max(ap + ao - inter, 1e-9)

    pose_score = math.exp(-mean_error / max(12.0, 0.012 * math.hypot(image_w, image_h)))
    combined = float(np.clip(0.48 * pose_score + 0.26 * iou + 0.26 * rotation.scene_score, 0.01, 0.99))
    losses = dict(candidate.losses)
    losses.update({
        "mean_reprojection_px": round(mean_error, 3),
        "median_reprojection_px": round(median_error, 3),
        "bbox_iou": round(iou, 4),
        "scene_score": round(rotation.scene_score, 4),
        "scene_focal_mm": round(focal, 3),
        "scene_yaw": round(yaw, 3),
        "scene_pitch": round(pitch, 3),
        "scene_roll": round(roll, 3),
        "scene_orientation_source": rotation.orientation_source,
    })
    return PoseCandidate(intr, ext, distance, height, focal, combined, losses)


def fuse_pose_and_scene(pose_candidates: list[PoseCandidate], rotation_candidates: list[RotationCandidate], image_w: int, image_h: int, pose_keypoints: np.ndarray, subject_bbox=None, max_candidates: int = 5) -> list[PoseCandidate]:
    """Fuse scene evidence while preserving pose-authoritative fields for roll-only evidence."""
    if not pose_candidates or not rotation_candidates:
        return []
    fused: list[PoseCandidate] = []
    for rotation in rotation_candidates:
        nearest = sorted(pose_candidates, key=lambda p: abs(p.focal_equiv_35mm - rotation.focal_length_mm))[:3]
        for pose_candidate in nearest:
            candidate = _camera_from_candidate(pose_candidate, rotation, image_w, image_h, subject_bbox=subject_bbox, pose_keypoints=pose_keypoints)
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