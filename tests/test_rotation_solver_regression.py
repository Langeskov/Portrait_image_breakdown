"""Regression tests for scene-rotation score contracts."""
from __future__ import annotations

import numpy as np

from reverse_engineering.rotation_solver import RotationCandidate, _camera_from_candidate, _estimate_line_roll
from reverse_engineering.geometry import CameraIntrinsics, CameraModel, PoseCandidate, _camera_pose_from_params, pose_driven_person_points
from reverse_engineering.scene_geometry import LineSegment, SceneGeometryEvidence


def _line(angle: float, length: float = 260.0, index: int = 0, x_step: float = 9.0) -> LineSegment:
    x1 = 40.0 + index * x_step
    y1 = 80.0 + index * 4.0
    rad = np.radians(angle)
    x2 = x1 + length * np.cos(rad)
    y2 = y1 + length * np.sin(rad)
    return LineSegment(x1, y1, float(x2), float(y2), length, angle)


def _evidence(lines, width=1200, height=800):
    return SceneGeometryEvidence(
        width=width,
        height=height,
        lines=tuple(lines),
        clusters=tuple(),
        vanishing_points=tuple(),
        vertical_cluster=None,
        horizontal_clusters=tuple(),
        horizon_angle_deg=None,
        confidence=0.9,
    )


def test_line_roll_runner_up_score_is_scalar_not_candidate_tuple():
    lines = [
        _line(0.0, index=0), _line(0.5, index=1), _line(-0.5, index=2),
        _line(89.5, index=3), _line(-89.0, index=4), _line(90.0, index=5),
        _line(1.0, index=6), _line(89.0, index=7),
    ]
    roll, confidence, count = _estimate_line_roll(_evidence(lines))
    assert count == len(lines)
    assert roll is not None and np.isfinite(roll)
    assert 0.0 <= confidence <= 1.0


def test_single_strong_scene_family_recovers_modest_camera_tilt():
    lines = [
        _line(7.0 + delta, length=520.0, index=i, x_step=125.0)
        for i, delta in enumerate((-0.8, -0.3, 0.0, 0.2, 0.5, 0.8))
    ]
    lines += [_line(38.0, length=180.0, index=i, x_step=35.0) for i in range(4)]
    roll, confidence, count = _estimate_line_roll(_evidence(lines))
    assert count == len(lines)
    assert roll is not None
    assert abs(roll - 7.0) < 1.5
    assert confidence >= 0.55


def test_diagonal_only_geometry_is_not_promoted_to_camera_roll():
    lines = [_line(angle, length=220.0, index=i, x_step=7.0) for i, angle in enumerate((32.0, 37.0, 42.0, 47.0, 52.0) * 8)]
    roll, confidence, count = _estimate_line_roll(_evidence(lines))
    assert count == len(lines)
    assert roll is None
    assert 0.0 <= confidence <= 1.0


def test_roll_only_rotation_preserves_pose_yaw_pitch_and_focal():
    intr = CameraIntrinsics.from_focal_mm(70.0, 1000, 800)
    _, pose_ext = _camera_pose_from_params(4.0, 1.35, 11.0, -6.0, 0.0)
    pose = PoseCandidate(intr, pose_ext, 4.0, 1.35, 70.0, 0.8, {})
    _, roll_ext = _camera_pose_from_params(1.0, 0.0, 0.0, 0.0, 7.0)
    rotation = RotationCandidate(
        focal_length_mm=50.0,
        extrinsics=roll_ext,
        scene_score=0.75,
        orthogonality_error=0.0,
        horizon_error_deg=0.0,
        vanishing_point_support=0.0,
        evidence=("roll-only fixture",),
        orientation_source="roll_only",
    )
    proxy = pose_driven_person_points(np.zeros((17, 3), dtype=float), 1000, 800)
    # Build a self-consistent observed pose from the pose camera so fusion is
    # tested for contract preservation rather than for arbitrary reprojection.
    k = np.c_[CameraModel(intr, pose_ext).project_points(proxy), np.full(17, 0.9)]
    fused = _camera_from_candidate(pose, rotation, 1000, 800, pose_keypoints=k)
    assert fused is not None
    assert abs(fused.extrinsics.yaw - pose_ext.yaw) < 1e-6
    assert abs(fused.extrinsics.pitch - pose_ext.pitch) < 1e-6
    assert abs(fused.extrinsics.roll - 7.0) < 1e-6
    assert abs(fused.focal_equiv_35mm - pose.focal_equiv_35mm) < 1e-6
