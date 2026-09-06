from __future__ import annotations

import math

import cv2
import numpy as np

from reverse_engineering.geometry import CameraIntrinsics, _camera_pose_from_params
from reverse_engineering.rotation_solver import estimate_rotation_candidates
from reverse_engineering.scene_geometry import SceneGeometryEvidence, VanishingPoint


def _vp_for_axis(rotation: np.ndarray, intrinsics: CameraIntrinsics, axis: np.ndarray) -> tuple[float, float]:
    direction = rotation @ axis
    pixel = intrinsics.to_matrix() @ direction
    assert abs(pixel[2]) > 1e-9
    return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])


def test_rotation_solver_recovers_known_camera_pose():
    width, height = 2400, 1600
    focal = 50.0
    intr = CameraIntrinsics.from_focal_mm(focal, width, height, 36.0, 24.0)
    _, extrinsics = _camera_pose_from_params(5.0, 1.4, 12.0, -5.0, 6.0)
    rotation = cv2.Rodrigues(extrinsics.rvec)[0]

    x_vp = _vp_for_axis(rotation, intr, np.array([1.0, 0.0, 0.0]))
    z_vp = _vp_for_axis(rotation, intr, np.array([0.0, 0.0, 1.0]))
    y_vp = _vp_for_axis(rotation, intr, np.array([0.0, 1.0, 0.0]))
    vps = (
        VanishingPoint(*x_vp, cluster=0, support=12, confidence=.92, mean_line_residual_px=1.0),
        VanishingPoint(*z_vp, cluster=1, support=14, confidence=.94, mean_line_residual_px=1.0),
        VanishingPoint(*y_vp, cluster=2, support=10, confidence=.90, mean_line_residual_px=1.0),
    )
    evidence = SceneGeometryEvidence(
        width=width, height=height,
        lines=tuple(), clusters=((0, 1), (2, 3), (4, 5)),
        vanishing_points=vps,
        vertical_cluster=2,
        horizontal_clusters=(0, 1),
        horizon_angle_deg=6.0,
        confidence=.9,
    )

    candidates = estimate_rotation_candidates(evidence, width, height, max_candidates=8)
    assert candidates
    best = max(candidates, key=lambda c: c.scene_score)
    assert abs(best.focal_length_mm - focal) < 2.0
    assert abs(best.extrinsics.yaw - 12.0) < 1.5
    assert abs(best.extrinsics.pitch + 5.0) < 1.5
    assert abs(best.extrinsics.roll - 6.0) < 1.5
    assert best.scene_score > .7
