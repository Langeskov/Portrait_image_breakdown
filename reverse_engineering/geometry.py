from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

REF_PERSON_HEIGHT = 1.70


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    sensor_width_mm: float = 36.0
    sensor_height_mm: float = 24.0

    @classmethod
    def from_focal_mm(
        cls,
        focal_length_mm: float,
        width: int,
        height: int,
        sensor_width_mm: float = 36.0,
        sensor_height_mm: Optional[float] = None,
    ) -> "CameraIntrinsics":
        if sensor_height_mm is None:
            sensor_height_mm = sensor_width_mm * height / max(width, 1)
        fx = float(focal_length_mm) * width / max(sensor_width_mm, 1e-9)
        fy = float(focal_length_mm) * height / max(sensor_height_mm, 1e-9)
        return cls(fx, fy, width * 0.5, height * 0.5, width, height, sensor_width_mm, sensor_height_mm)

    def to_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class CameraExtrinsics:
    rvec: np.ndarray
    tvec: np.ndarray
    position: np.ndarray
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class ProjectionResult:
    projected_points: np.ndarray
    reproj_error: float
    in_front: bool


@dataclass
class PoseCandidate:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    distance: float
    height: float
    focal_equiv_35mm: float
    score: float
    losses: dict


def _pose_arrays(pose_keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
    points = kp[:17, :2].copy()
    conf = np.clip(kp[:17, 2], 0.0, 1.0)
    conf[~np.isfinite(points).all(axis=1)] = 0.0
    return points, conf


def canonical_person_points(height: float = REF_PERSON_HEIGHT) -> np.ndarray:
    """Metric fallback person template centered around the subject origin."""
    h = float(height)
    y = {0: .95, 1: .975, 2: .975, 3: .94, 4: .94, 5: .79, 6: .79,
         7: .61, 8: .61, 9: .47, 10: .47, 11: .52, 12: .52,
         13: .31, 14: .31, 15: .05, 16: .05}
    x = {0: 0.0, 1: -.035, 2: .035, 3: -.085, 4: .085, 5: -.215, 6: .215,
         7: -.31, 8: .31, 9: -.38, 10: .38, 11: -.16, 12: .16,
         13: -.15, 14: .15, 15: -.15, 16: .15}
    z = {0: .08, 1: .08, 2: .08, 3: .08, 4: .08, 7: -.04, 8: -.04,
         9: -.04, 10: -.04, 13: -.04, 14: -.04, 15: -.04, 16: -.04}
    return np.array([[x[i], (y[i] - .5) * h, z.get(i, 0.0)] for i in range(17)], dtype=np.float64)


def pose_driven_person_points(pose_keypoints: np.ndarray, image_w: int, image_h: int, ref_height: float = REF_PERSON_HEIGHT) -> np.ndarray:
    """Create a metric proxy whose 2D silhouette follows the observed pose."""
    points, conf = _pose_arrays(pose_keypoints)
    valid = conf > .30
    if int(valid.sum()) < 5:
        return canonical_person_points(ref_height)
    observed = points[valid]
    x0, y0 = observed.min(axis=0)
    x1, y1 = observed.max(axis=0)
    span_y = max(float(y1 - y0), 20.0)
    scale = float(ref_height) / span_y
    cx, cy = (float(x0) + float(x1)) * .5, (float(y0) + float(y1)) * .5
    world = np.full((17, 3), np.nan, dtype=np.float64)
    world[valid, 0] = (points[valid, 0] - cx) * scale
    world[valid, 1] = (cy - points[valid, 1]) * scale
    world[valid, 2] = 0.0
    world[[0, 1, 2, 3, 4], 2] = .08
    world[[7, 8, 9, 10, 13, 14, 15, 16], 2] = -.04
    return world


def _camera_pose_from_params(distance: float, height: float, yaw_deg: float, pitch_deg: float, roll_deg: float, target: Optional[np.ndarray] = None) -> tuple[np.ndarray, CameraExtrinsics]:
    target = np.asarray(target if target is not None else [0.0, 0.0, 0.0], dtype=float)
    yaw, pitch, roll = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    position = target + np.array([math.sin(yaw) * distance, height, -math.cos(yaw) * distance], dtype=float)
    forward = np.array([-math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch)], dtype=float)
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-9)
    c, s = math.cos(roll), math.sin(roll)
    rolled_right = right * c + up * s
    rolled_up = -right * s + up * c
    rotation = np.vstack([rolled_right, -rolled_up, forward])
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = -rotation @ position
    return position, CameraExtrinsics(rvec.reshape(3), tvec.reshape(3), position, float(pitch_deg), float(yaw_deg), float(roll_deg))


class CameraModel:
    def __init__(self, intrinsics: CameraIntrinsics, extrinsics: Optional[CameraExtrinsics] = None):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        points = np.asarray(points_3d, dtype=float).reshape(-1, 3)
        out = np.full((len(points), 2), np.nan, dtype=float)
        if self.extrinsics is None:
            return out
        valid = np.isfinite(points).all(axis=1)
        if valid.any():
            projected, _ = cv2.projectPoints(points[valid], self.extrinsics.rvec, self.extrinsics.tvec, self.intrinsics.to_matrix(), None)
            out[valid] = projected.reshape(-1, 2)
        return out

    def project_point(self, point_3d: np.ndarray) -> tuple[float, float]:
        p = self.project_points(np.asarray(point_3d).reshape(1, 3))[0]
        return float(p[0]), float(p[1])

    def unproject_point(self, px: float, py: float, depth: float = 1.0) -> np.ndarray:
        ray = np.linalg.inv(self.intrinsics.to_matrix()) @ np.array([px, py, 1.0], dtype=float)
        if abs(ray[2]) <= 1e-9:
            return np.array([np.nan, np.nan, np.nan])
        return ray * (float(depth) / ray[2])


class PoseSolver:
    """Bounded camera fitting using a shared metric pose proxy."""

    @staticmethod
    def fit_camera_to_pose(
        pose_keypoints: np.ndarray,
        image_w: int,
        image_h: int,
        subject_bbox: Optional[tuple[int, int, int, int]] = None,
        ref_height: float = REF_PERSON_HEIGHT,
        focal_seeds: tuple[float, ...] = (28, 35, 50, 70, 85, 105, 135),
        num_candidates: int = 5,
    ) -> list[PoseCandidate]:
        points, conf = _pose_arrays(pose_keypoints)
        valid = (conf > .35) & np.isfinite(points).all(axis=1)
        obj = pose_driven_person_points(pose_keypoints, image_w, image_h, ref_height)
        valid &= np.isfinite(obj).all(axis=1)
        if int(valid.sum()) < 6:
            return []

        obs = points[valid]
        weights = np.sqrt(np.clip(conf[valid], .25, 1.0))
        diag = max(math.hypot(image_w, image_h), 1.0)
        bbox = tuple(map(float, subject_bbox)) if subject_bbox is not None else None
        yspan = max(float(np.ptp(obs[:, 1])), 20.0)

        reference_focal_mm = float(np.median(focal_seeds)) if focal_seeds else 50.0
        reference_intrinsics = CameraIntrinsics.from_focal_mm(reference_focal_mm, image_w, image_h)
        base_distance = max(0.8, ref_height * reference_intrinsics.fy / yspan)
        base_height = float(np.clip(
            ref_height * (0.35 + 0.45 * (1.0 - np.clip(np.mean(obs[:, 1]) / image_h, 0.0, 1.0))),
            .45,
            1.9,
        ))

        lo = np.array([max(.8, base_distance * .35), 20.0, .40, -35.0, -35.0, -12.0], dtype=float)
        hi = np.array([min(30.0, max(6.0, base_distance * 2.8)), 220.0, 2.10, 35.0, 35.0, 12.0], dtype=float)
        if hi[0] <= lo[0]:
            hi[0] = lo[0] + .5

        def evaluate(p):
            distance, focal, height, yaw, pitch, roll = p
            intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h)
            position, extrinsics = _camera_pose_from_params(distance, height, yaw, pitch, roll)
            projected = CameraModel(intr, extrinsics).project_points(obj)
            return projected, intr, extrinsics, position

        def residual(p):
            projected, _, _, _ = evaluate(p)
            pred = projected[valid]
            point_r = ((pred - obs) / 4.0) * weights[:, None]
            if not np.isfinite(point_r).all():
                return np.full(point_r.size + 9, 100.0)
            extra = []
            if bbox is not None:
                bx0, by0, bx1, by1 = bbox
                pb = np.array([np.nanmin(projected[:, 0]), np.nanmin(projected[:, 1]), np.nanmax(projected[:, 0]), np.nanmax(projected[:, 1])])
                observed_w = max(bx1 - bx0, 1.0)
                observed_h = max(by1 - by0, 1.0)
                predicted_w = max(pb[2] - pb[0], 1.0)
                predicted_h = max(pb[3] - pb[1], 1.0)
                extra.extend([(pb[0] - bx0) / 6.0, (pb[1] - by0) / 6.0, math.log(predicted_w / observed_w) * 6.0, math.log(predicted_h / observed_h) * 6.0])
            extra.extend([
                (p[0] / max(base_distance, 1.0) - 1.0) * .08,
                ((p[2] - base_height) / .8) * .08,
                (p[3] / 35.0) * .04,
                (p[4] / 35.0) * .04,
                (p[5] / 12.0) * .02,
            ])
            return np.concatenate([point_r.reshape(-1), np.asarray(extra, dtype=float)])

        results: list[PoseCandidate] = []
        for focal in focal_seeds:
            for yaw0 in (-10.0, 0.0, 10.0):
                for pitch0 in (-10.0, 0.0, 10.0):
                    for roll0 in (-4.0, 0.0, 4.0):
                        initial_distance = base_distance * focal / max(reference_focal_mm, 1e-6)
                        x0 = np.array([initial_distance, focal, base_height, yaw0, pitch0, roll0], dtype=float)
                        x0 = np.clip(x0, lo + 1e-4, hi - 1e-4)
                        try:
                            sol = least_squares(
                                residual,
                                x0,
                                bounds=(lo, hi),
                                loss="soft_l1",
                                f_scale=5.0,
                                max_nfev=450,
                                xtol=1e-7,
                                ftol=1e-7,
                                gtol=1e-7,
                            )
                        except (ValueError, RuntimeError, FloatingPointError):
                            continue
                        projected, intr, extrinsics, position = evaluate(sol.x)
                        errors = np.linalg.norm(projected[valid] - obs, axis=1)
                        if not np.isfinite(errors).all():
                            continue
                        mean_error = float(np.mean(errors))
                        median_error = float(np.median(errors))
                        if mean_error > 45.0:
                            continue
                        pb = np.array([np.nanmin(projected[:, 0]), np.nanmin(projected[:, 1]), np.nanmax(projected[:, 0]), np.nanmax(projected[:, 1])])
                        iou = 0.0
                        center_delta = diag
                        if bbox is not None:
                            bx0, by0, bx1, by1 = bbox
                            iw = max(0.0, min(pb[2], bx1) - max(pb[0], bx0))
                            ih = max(0.0, min(pb[3], by1) - max(pb[1], by0))
                            inter = iw * ih
                            area_p = max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1])
                            area_o = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
                            iou = inter / max(area_p + area_o - inter, 1e-9)
                            center_delta = math.hypot((pb[0] + pb[2]) * .5 - (bx0 + bx1) * .5, (pb[1] + pb[3]) * .5 - (by0 + by1) * .5)
                        reproj_score = math.exp(-mean_error / max(12.0, .015 * diag))
                        center_score = math.exp(-center_delta / max(30.0, .03 * diag))
                        prior = math.exp(-0.5 * ((sol.x[0] / max(base_distance, 1.0) - 1.0) / 2.0) ** 2)
                        size_score = math.exp(-abs(math.log(max(pb[2] - pb[0], 1.0) / max((bbox[2] - bbox[0]) if bbox else pb[2] - pb[0], 1.0)))) if bbox else 1.0
                        score = float(np.clip(.55 * reproj_score + .25 * iou + .15 * center_score + .05 * prior * size_score, 0.0, 1.0))
                        results.append(PoseCandidate(intr, extrinsics, float(sol.x[0]), float(sol.x[2]), float(sol.x[1]), score, {
                            "mean_reprojection_px": round(mean_error, 3),
                            "median_reprojection_px": round(median_error, 3),
                            "bbox_iou": round(iou, 4),
                            "center_delta_px": round(center_delta, 3),
                            "optimization_cost": round(float(sol.cost), 6),
                            "distance_prior_m": round(base_distance, 3),
                        }))

        results.sort(key=lambda c: (-c.score, c.losses.get("mean_reprojection_px", 1e9)))
        unique: list[PoseCandidate] = []
        for candidate in results:
            if any(
                abs(candidate.focal_equiv_35mm - u.focal_equiv_35mm) < 4.0
                and abs(candidate.distance - u.distance) < .20
                and abs(candidate.extrinsics.yaw - u.extrinsics.yaw) < 2.0
                and abs(candidate.extrinsics.pitch - u.extrinsics.pitch) < 2.0
                and abs(candidate.extrinsics.roll - u.extrinsics.roll) < 1.5
                for u in unique
            ):
                continue
            unique.append(candidate)
            if len(unique) >= max(1, num_candidates):
                break
        return unique

    @staticmethod
    def solve_from_body_geometry(pose_keypoints, image_w, image_h, **kwargs):
        return PoseSolver.fit_camera_to_pose(pose_keypoints, image_w, image_h, **kwargs)
