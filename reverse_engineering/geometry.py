"""Camera geometry and bounded 2D-pose camera fitting."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

REF_PERSON_HEIGHT = 1.70
REF_SHOULDER_WIDTH = 0.45


@dataclass(frozen=True)
class ImagePoint:
    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True)
class WorldPoint:
    x: float
    y: float
    z: float
    confidence: float = 0.0


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0
    sensor_width_mm: float = 36.0
    sensor_height_mm: float = 24.0

    @property
    def fov_x(self) -> float:
        return 0.0 if self.fx <= 0 or self.width <= 0 else 2.0 * math.degrees(math.atan((self.width * 0.5) / self.fx))

    @property
    def fov_y(self) -> float:
        return 0.0 if self.fy <= 0 or self.height <= 0 else 2.0 * math.degrees(math.atan((self.height * 0.5) / self.fy))

    @property
    def focal_length_mm(self) -> float:
        return 0.0 if self.width <= 0 or self.sensor_width_mm <= 0 else self.fx * self.sensor_width_mm / self.width

    def to_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @classmethod
    def from_focal_mm(cls, focal_mm: float, width: int, height: int, sensor_w: float = 36.0, sensor_h: float = 24.0) -> "CameraIntrinsics":
        return cls(
            float(focal_mm) * width / max(sensor_w, 1e-6),
            float(focal_mm) * height / max(sensor_h, 1e-6),
            width * 0.5,
            height * 0.5,
            int(width),
            int(height),
            sensor_w,
            sensor_h,
        )


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
    """Create a metric proxy with the observed 2D pose as its silhouette."""
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
    # Small, fixed depth offsets prevent a perfectly planar PnP configuration
    # without inventing arbitrary per-joint depth.
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
    R = np.vstack([rolled_right, -rolled_up, forward])
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ position
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
        return ray * (float(depth) / ray[2]) if abs(ray[2]) > 1e-9 else np.array([np.nan, np.nan, np.nan])


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
        xspan = max(float(np.ptp(obs[:, 0])), 20.0)
        base_distance = max(1.0, 50.0 * image_h / yspan * ref_height / 24.0)
        base_height = float(np.clip(ref_height * (0.35 + 0.45 * (1.0 - np.clip((np.mean(obs[:, 1]) / image_h), 0.0, 1.0))), .45, 1.9))

        # [distance, focal, height, yaw, pitch, roll]
        lo = np.array([max(.8, base_distance * .20), 28.0, .40, -20.0, -20.0, -8.0], dtype=float)
        hi = np.array([min(18.0, max(6.0, base_distance * 3.0)), 180.0, 2.10, 20.0, 20.0, 8.0], dtype=float)
        if hi[0] <= lo[0]:
            hi[0] = lo[0] + .5

        target_center = np.array([0.0, 0.0, 0.0], dtype=float)

        def evaluate(p):
            distance, focal, height, yaw, pitch, roll = p
            intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h)
            position, extrinsics = _camera_pose_from_params(distance, height, yaw, pitch, roll, target_center)
            camera = CameraModel(intr, extrinsics)
            projected = camera.project_points(obj)
            return projected, intr, extrinsics, position

        def residual(p):
            distance, _, height, yaw, pitch, roll = p
            projected, _, _, _ = evaluate(p)
            pred = projected[valid]
            point_r = ((pred - obs) / 4.0) * weights[:, None]
            if not np.isfinite(point_r).all():
                return np.full(point_r.size + 4, 100.0)

            extra = []
            if bbox is not None:
                bx0, by0, bx1, by1 = bbox
                pb = np.array([np.nanmin(projected[:, 0]), np.nanmin(projected[:, 1]), np.nanmax(projected[:, 0]), np.nanmax(projected[:, 1])])
                observed_w, observed_h = max(bx1 - bx0, 1.0), max(by1 - by0, 1.0)
                predicted_w, predicted_h = max(pb[2] - pb[0], 1.0), max(pb[3] - pb[1], 1.0)
                extra.extend([
                    (pb[0] - bx0) / 6.0,
                    (pb[1] - by0) / 6.0,
                    (math.log(predicted_w / observed_w)) * 6.0,
                    (math.log(predicted_h / observed_h)) * 6.0,
                ])
            # Weak regularization prevents meaningless boundary solutions while
            # keeping pose orientation free to move when image evidence supports it.
            extra.extend([
                (distance / max(base_distance, 1.0) - 1.0) * .15,
                ((height - base_height) / .8) * .10,
                (yaw / 20.0) * .06,
                (pitch / 20.0) * .06,
                (roll / 8.0) * .03,
            ])
            return np.concatenate([point_r.reshape(-1), np.asarray(extra, dtype=float)])

        results: list[PoseCandidate] = []
        yaw_seeds = (-6.0, 0.0, 6.0)
        pitch_seeds = (-6.0, 0.0, 6.0)
        roll_seeds = (-3.0, 0.0, 3.0)
        for focal in focal_seeds:
            for yaw0 in yaw_seeds:
                for pitch0 in pitch_seeds:
                    x0 = np.array([base_distance * focal / 50.0, focal, base_height, yaw0, pitch0, 0.0], dtype=float)
                    x0 = np.clip(x0, lo + 1e-4, hi - 1e-4)
                    try:
                        sol = least_squares(
                            residual, x0, bounds=(lo, hi),
                            loss="soft_l1", f_scale=5.0,
                            max_nfev=350, xtol=1e-7, ftol=1e-7, gtol=1e-7,
                        )
                    except (ValueError, RuntimeError, FloatingPointError):
                        continue
                    projected, intr, extrinsics, position = evaluate(sol.x)
                    errors = np.linalg.norm(projected[valid] - obs, axis=1)
                    if not np.isfinite(errors).all():
                        continue
                    mean_error = float(np.mean(errors)); median_error = float(np.median(errors))
                    if mean_error > 60.0:
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
                        union = area_p + area_o - inter
                        iou = inter / union if union > 1e-9 else 0.0
                        center_delta = math.hypot((pb[0] + pb[2]) * .5 - (bx0 + bx1) * .5, (pb[1] + pb[3]) * .5 - (by0 + by1) * .5)

                    reproj_score = math.exp(-mean_error / max(12.0, 0.015 * diag))
                    size_score = math.exp(-abs(math.log(max(pb[2]-pb[0],1) / max((bbox[2]-bbox[0]) if bbox else pb[2]-pb[0],1)))) if bbox else 1.0
                    center_score = math.exp(-center_delta / max(30.0, 0.03 * diag))
                    prior = math.exp(-0.5 * ((sol.x[0] / max(base_distance,1.0) - 1.0) / 1.75) ** 2)
                    score = float(np.clip(.50 * reproj_score + .25 * iou + .15 * center_score + .10 * prior * size_score, 0.0, 1.0))
                    results.append(PoseCandidate(
                        intrinsics=intr,
                        extrinsics=extrinsics,
                        distance=float(sol.x[0]),
                        height=float(sol.x[2]),
                        focal_equiv_35mm=float(sol.x[1]),
                        score=score,
                        losses={
                            "mean_reprojection_px": round(mean_error, 3),
                            "median_reprojection_px": round(median_error, 3),
                            "bbox_iou": round(iou, 4),
                            "center_delta_px": round(center_delta, 3),
                            "optimization_cost": round(float(sol.cost), 6),
                        },
                    ))

        # The bounded parameterization is useful as a prior, but its orientation
        # model can become a poor local minimum for synthetic or strongly tilted
        # poses. OpenCV's PnP solver provides a direct reprojection solution for
        # the same metric proxy. Use it to recover a solution family whenever the
        # bounded search did not produce enough usable candidates.
        if len(results) < max(1, num_candidates):
            pnp_obj = obj[valid].astype(np.float32)
            pnp_obs = obs.astype(np.float32)
            for focal in focal_seeds:
                if len(results) >= max(1, num_candidates):
                    break
                intr = CameraIntrinsics.from_focal_mm(focal, image_w, image_h)
                try:
                    ok, rvec, tvec = cv2.solvePnP(
                        pnp_obj, pnp_obs, intr.to_matrix(), None,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                except (cv2.error, ValueError, FloatingPointError):
                    continue
                if not ok:
                    continue
                rvec = np.asarray(rvec, dtype=float).reshape(3)
                tvec = np.asarray(tvec, dtype=float).reshape(3)
                projected, _ = cv2.projectPoints(pnp_obj, rvec, tvec, intr.to_matrix(), None)
                projected = projected.reshape(-1, 2)
                errors = np.linalg.norm(projected - obs, axis=1)
                if not np.isfinite(errors).all():
                    continue
                mean_error = float(np.mean(errors))
                median_error = float(np.median(errors))
                if mean_error > 60.0:
                    continue

                R, _ = cv2.Rodrigues(rvec)
                position = (-R.T @ tvec).reshape(3)
                distance = float(np.linalg.norm(position))
                height = float(abs(position[1]))
                extrinsics = CameraExtrinsics(
                    rvec=rvec,
                    tvec=tvec,
                    position=position,
                )
                pb = np.array([
                    np.min(projected[:, 0]), np.min(projected[:, 1]),
                    np.max(projected[:, 0]), np.max(projected[:, 1]),
                ])
                iou = 0.0
                center_delta = diag
                if bbox is not None:
                    bx0, by0, bx1, by1 = bbox
                    iw = max(0.0, min(pb[2], bx1) - max(pb[0], bx0))
                    ih = max(0.0, min(pb[3], by1) - max(pb[1], by0))
                    inter = iw * ih
                    area_p = max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1])
                    area_o = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
                    union = area_p + area_o - inter
                    iou = inter / union if union > 1e-9 else 0.0
                    center_delta = math.hypot(
                        (pb[0] + pb[2]) * .5 - (bx0 + bx1) * .5,
                        (pb[1] + pb[3]) * .5 - (by0 + by1) * .5,
                    )
                reproj_score = math.exp(-mean_error / max(12.0, 0.015 * diag))
                center_score = math.exp(-center_delta / max(30.0, 0.03 * diag))
                score = float(np.clip(.65 * reproj_score + .25 * iou + .10 * center_score, 0.0, 1.0))
                results.append(PoseCandidate(
                    intrinsics=intr,
                    extrinsics=extrinsics,
                    distance=distance,
                    height=height,
                    focal_equiv_35mm=float(focal),
                    score=score,
                    losses={
                        "mean_reprojection_px": round(mean_error, 3),
                        "median_reprojection_px": round(median_error, 3),
                        "bbox_iou": round(iou, 4),
                        "center_delta_px": round(center_delta, 3),
                        "solver": "solvePnP",
                    },
                ))

        results.sort(key=lambda c: (-c.score, c.losses.get("mean_reprojection_px", 1e9)))
        unique: list[PoseCandidate] = []
        for candidate in results:
            if any(
                abs(candidate.focal_equiv_35mm - u.focal_equiv_35mm) < 4.0
                and abs(candidate.distance - u.distance) < .20
                and abs(candidate.extrinsics.yaw - u.extrinsics.yaw) < 2.0
                and abs(candidate.extrinsics.pitch - u.extrinsics.pitch) < 2.0
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

    @staticmethod
    def solve_from_vanishing_points(vps, image_w, image_h, intrinsics=None):
        if len(vps) < 2:
            return None
        if intrinsics is None:
            focal = 0.9 * max(image_w, image_h)
            intrinsics = CameraIntrinsics(focal, focal, image_w * .5, image_h * .5, image_w, image_h)
        inv = np.linalg.inv(intrinsics.to_matrix())
        dirs = []
        for vx, vy in vps[:3]:
            if np.isfinite(vx) and np.isfinite(vy):
                d = inv @ np.array([vx * image_w, vy * image_h, 1.0])
                n = np.linalg.norm(d)
                if n > 1e-8:
                    dirs.append(d / n)
        if len(dirs) < 2 or abs(float(np.dot(dirs[0], dirs[1]))) > .45:
            return None
        r1 = dirs[0]
        r2 = dirs[1] - np.dot(dirs[1], r1) * r1
        n = np.linalg.norm(r2)
        if n < 1e-8:
            return None
        r2 /= n
        r3 = np.cross(r1, r2)
        R = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        rvec, _ = cv2.Rodrigues(R)
        return CameraExtrinsics(rvec.reshape(3), np.zeros(3), np.array([np.nan, np.nan, np.nan]))
