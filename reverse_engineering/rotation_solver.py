"""Camera rotation recovery from scene geometry.

Human pose constrains framing, but scene geometry constrains camera orientation.
This module uses vanishing-point orthogonality and the existing pinhole camera
model to estimate Yaw/Pitch/Roll without inventing a custom renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from reverse_engineering.data_types import EstimatedValue
from reverse_engineering.geometry import CameraIntrinsics, CameraExtrinsics
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
    p = np.array([vp.x, vp.y, 1.0], dtype=np.float64)
    ray = np.linalg.inv(intrinsics.to_matrix()) @ p
    return ray / max(np.linalg.norm(ray), 1e-12)


def _focal_from_orthogonal_vps(a: VanishingPoint, b: VanishingPoint, width: int, height: int, sensor_w: float = 36.0, sensor_h: float | None = None) -> float | None:
    sensor_h = float(sensor_h if sensor_h is not None else sensor_w * height / max(width, 1))
    ax = a.x - width * 0.5
    ay = a.y - height * 0.5
    bx = b.x - width * 0.5
    by = b.y - height * 0.5
    term = ax * bx * (sensor_w / width) ** 2 + ay * by * (sensor_h / height) ** 2
    if term >= -1e-6:
        return None
    f2 = -term
    if f2 <= 0:
        return None
    f = math.sqrt(f2)
    return float(f) if 15.0 <= f <= 300.0 else None


def _rotation_from_vps(vps: tuple[VanishingPoint, VanishingPoint, VanishingPoint], intrinsics: CameraIntrinsics) -> tuple[np.ndarray, tuple[float, float, float], float] | None:
    # The first two are the two horizontal Manhattan directions; the third is
    # the world-up direction. Try both horizontal assignments so yaw is not
    # arbitrarily rotated by 90 degrees.
    best = None
    for h0, h1 in ((vps[0], vps[1]), (vps[1], vps[0])):
        rx = _vp_ray(h0, intrinsics)
        rz = _vp_ray(h1, intrinsics)
        ry = _vp_ray(vps[2], intrinsics)

        # Vanishing points are direction-only: choose signs so +Z points in
        # front of the camera and +Y means world-up (camera Y is down).
        if rz[2] < 0:
            rz = -rz
        if ry[1] > 0:
            ry = -ry
        rx = rx - np.dot(rx, ry) * ry - np.dot(rx, rz) * rz
        if np.linalg.norm(rx) < 1e-8:
            continue
        rx /= np.linalg.norm(rx)

        # Re-orthogonalize the triad with a polar decomposition.
        R0 = np.column_stack([rx, ry, rz])
        U, _, Vt = np.linalg.svd(R0)
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

        orth_err = abs(float(np.dot(rx, ry))) + abs(float(np.dot(rx, rz))) + abs(float(np.dot(ry, rz)))
        candidate = (R, (pitch, yaw, roll), orth_err)
        if best is None or orth_err < best[2]:
            best = candidate
    return best


def estimate_rotation_candidates(evidence: SceneGeometryEvidence, image_w: int, image_h: int, max_candidates: int = 5) -> list[RotationCandidate]:
    vps = list(evidence.vanishing_points)
    if len(vps) < 3 or not evidence.has_three_directions:
        return []
    vertical = next((vp for vp in vps if vp.cluster == evidence.vertical_cluster), None)
    horizontal = [vp for vp in vps if vp.cluster in evidence.horizontal_clusters]
    if vertical is None or len(horizontal) < 2:
        return []

    focal_values: list[float] = []
    horizontal_pair = (horizontal[0], horizontal[1])
    for a, b in (
        horizontal_pair,
        (horizontal[0], vertical),
        (horizontal[1], vertical),
    ):
        f = _focal_from_orthogonal_vps(a, b, image_w, image_h)
        if f is not None:
            focal_values.append(f)
    focal_values.extend([28.0, 35.0, 50.0, 70.0, 85.0, 105.0, 135.0])
    focal_values = sorted({round(float(np.clip(f, 20.0, 200.0)), 2) for f in focal_values})

    results: list[RotationCandidate] = []
    for focal in focal_values:
        intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h, 36.0, 36.0 * image_h / max(image_w, 1))
        rotation_result = _rotation_from_vps((horizontal[0], horizontal[1], vertical), intr)
        if rotation_result is None:
            continue
        R, (pitch, yaw, roll), orth_err = rotation_result
        roll_horizon = evidence.horizon_angle_deg if evidence.horizon_angle_deg is not None else 0.0
        horizon_error = abs(_normalize_angle(roll - roll_horizon)) if evidence.horizon_angle_deg is not None else 0.0
        vp_support = float(np.mean([horizontal[0].confidence, horizontal[1].confidence, vertical.confidence]))
        orth_score = math.exp(-orth_err / 0.25)
        horizon_score = math.exp(-horizon_error / 8.0) if evidence.horizon_angle_deg is not None else 0.5
        score = float(np.clip(0.55 * orth_score + 0.25 * vp_support + 0.20 * horizon_score, 0.01, 0.99))
        rvec, _ = cv2.Rodrigues(R)
        extrinsics = CameraExtrinsics(
            rvec=rvec.reshape(3),
            tvec=np.zeros(3, dtype=float),
            position=np.zeros(3, dtype=float),
            pitch=float(pitch), yaw=float(yaw), roll=float(roll),
        )
        results.append(RotationCandidate(
            focal_length_mm=float(focal),
            extrinsics=extrinsics,
            scene_score=score,
            orthogonality_error=float(orth_err),
            horizon_error_deg=float(horizon_error),
            vanishing_point_support=vp_support,
            evidence=(
                f"{len(evidence.lines)} line segments",
                f"VP support {vertical.support}+{horizontal[0].support}+{horizontal[1].support}",
                "Manhattan orthogonality",
                "horizon orientation" if evidence.horizon_angle_deg is not None else "no horizon angle",
            ),
        ))

    results.sort(key=lambda c: -c.scene_score)
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
