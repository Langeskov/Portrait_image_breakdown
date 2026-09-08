"""Image-space refinement of a reconstructed camera candidate.

The solver still treats focal length/distance as coupled. This pass only makes
small bounded corrections using the observed pose and subject anchor so the 3D
proxy is actually aimed at the photographed subject instead of merely producing
an internally consistent camera.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from reverse_engineering.geometry import CameraIntrinsics, CameraModel, PoseCandidate, _camera_pose_from_params, pose_driven_person_points


def subject_anchor(pose_keypoints: np.ndarray) -> np.ndarray:
    """Return a stable image-space anchor, preferring hips/torso over the head."""
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17:
        return np.array([0.5, 0.5], dtype=float)
    groups = ((11, 12), (5, 6), (13, 14), (0,))
    for group in groups:
        pts = []
        for i in group:
            if i < len(kp) and kp[i, 2] > 0.45 and np.isfinite(kp[i, :2]).all():
                pts.append(kp[i, :2])
        if pts:
            return np.mean(pts, axis=0)
    valid = kp[:17, 2] > 0.35
    if valid.any():
        return np.mean(kp[:17, :2][valid], axis=0)
    return np.array([0.5, 0.5], dtype=float)


def _anchor_world(pose_keypoints: np.ndarray, image_w: int, image_h: int, proxy: np.ndarray) -> np.ndarray:
    """Choose the corresponding proxy landmark for the image-space anchor."""
    kp = np.asarray(pose_keypoints, dtype=float)
    a = subject_anchor(kp)
    valid = np.isfinite(proxy).all(axis=1) & (kp[:len(proxy), 2] > 0.30)
    if not valid.any():
        return np.array([0.0, 0.0, 0.0], dtype=float)
    norm = np.column_stack([kp[:len(proxy), 0] / max(image_w, 1), kp[:len(proxy), 1] / max(image_h, 1)])
    target = np.array([a[0] / max(image_w, 1), a[1] / max(image_h, 1)])
    idxs = np.flatnonzero(valid)
    idx = idxs[int(np.argmin(np.linalg.norm(norm[valid] - target, axis=1)))]
    return proxy[idx]


def refine_camera_candidate(
    candidate: PoseCandidate,
    pose_keypoints: np.ndarray,
    image_w: int,
    image_h: int,
    subject_bbox: Optional[tuple[int, int, int, int]] = None,
    max_delta: tuple[float, float, float] = (9.0, 9.0, 4.0),
    allow_roll_refinement: bool = False,
) -> PoseCandidate:
    """Apply a small image-space correction to yaw/pitch and, optionally, roll.

    Roll is not identifiable from human pose alone: a person may lean while the
    camera remains level. Therefore roll refinement is disabled by default and
    is only allowed when an independent scene-roll measurement has been
    supplied by the rotation solver.
    """
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17:
        return candidate
    proxy = pose_driven_person_points(kp, image_w, image_h)
    valid = (kp[:17, 2] > 0.35) & np.isfinite(proxy).all(axis=1)
    if int(valid.sum()) < 6:
        return candidate

    obs = kp[:17, :2]
    weights = np.clip(kp[:17, 2], 0.35, 1.0)
    anchor_world = _anchor_world(kp, image_w, image_h, proxy)
    anchor_obs = subject_anchor(kp)

    base = np.array([candidate.extrinsics.yaw, candidate.extrinsics.pitch, candidate.extrinsics.roll], dtype=float)
    roll_offsets = np.linspace(-max_delta[2], max_delta[2], 5) if allow_roll_refinement else np.array([0.0])
    best = base.copy(); best_cost = float("inf")

    for dy in np.linspace(-max_delta[0], max_delta[0], 7):
        for dp in np.linspace(-max_delta[1], max_delta[1], 7):
            for dr in roll_offsets:
                yaw, pitch, roll = base + np.array([dy, dp, dr])
                _, extr = _camera_pose_from_params(candidate.distance, candidate.height, yaw, pitch, roll)
                intr = candidate.intrinsics
                projected = CameraModel(intr, extr).project_points(proxy)
                if not np.isfinite(projected[valid]).all():
                    continue
                residual = np.linalg.norm(projected[valid] - obs[valid], axis=1)
                robust = np.minimum(residual, 80.0)
                cost = float(np.average(robust, weights=weights[valid]))
                anchor_px = CameraModel(intr, extr).project_point(anchor_world)
                if np.isfinite(anchor_px).all():
                    cost += 0.60 * math.hypot(anchor_px[0] - anchor_obs[0], anchor_px[1] - anchor_obs[1])
                if subject_bbox is not None:
                    x0, y0, x1, y1 = map(float, subject_bbox)
                    pb = np.array([np.nanmin(projected[:, 0]), np.nanmin(projected[:, 1]), np.nanmax(projected[:, 0]), np.nanmax(projected[:, 1])])
                    dx = max(x0 - pb[2], 0.0) + max(pb[0] - x1, 0.0)
                    dyb = max(y0 - pb[3], 0.0) + max(pb[1] - y1, 0.0)
                    cost += 0.25 * (dx + dyb)
                if cost < best_cost:
                    best_cost, best = cost, np.array([yaw, pitch, roll], dtype=float)

    if np.linalg.norm(best - base) < 1e-6:
        candidate.losses["image_refinement"] = "no_change"
        candidate.losses["image_refinement_cost_px"] = round(best_cost, 3)
        candidate.losses["image_refinement_roll_refined"] = bool(allow_roll_refinement)
        return candidate

    _, extr = _camera_pose_from_params(candidate.distance, candidate.height, *best)
    candidate.extrinsics = extr
    candidate.losses["image_refinement"] = "bounded_pose_refinement"
    candidate.losses["image_refinement_cost_px"] = round(best_cost, 3)
    candidate.losses["image_refinement_delta_deg"] = [round(float(v), 2) for v in (best - base)]
    candidate.losses["image_refinement_roll_refined"] = bool(allow_roll_refinement)
    return candidate
