"""Regression tests for scene-rotation and pose-scoring contracts."""
from __future__ import annotations

import numpy as np

from reverse_engineering.rotation_solver import RotationCandidate, _camera_from_candidate, _estimate_line_roll
from reverse_engineering.geometry import CameraIntrinsics, CameraModel, PoseCandidate, _camera_pose_from_params, PoseSolver, pose_driven_person_points
from reverse_engineering.camera_pose import estimate_camera_pose_candidates
from reverse_engineering.scene_geometry import LineSegment, SceneGeometryEvidence
from core.pose_detector import PoseLandmark, PoseResult


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


def _pose_for_candidate_family(width=1600, height=1065):
    bbox = (450, 90, 1150, 990)
    points = [
        (800, 150), (770, 145), (830, 145), (745, 165), (855, 165),
        (650, 300), (950, 300), (610, 450), (990, 450), (575, 600),
        (1025, 600), (700, 590), (900, 590), (720, 735), (880, 735),
        (715, 930), (885, 930),
    ]
    landmarks = [PoseLandmark(i, float(x), float(y), 0.0, 0.95, x / width, y / height, 0.0) for i, (x, y) in enumerate(points)]
    return PoseResult(landmarks, width, height, 0.95, bbox)


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
    k = np.c_[CameraModel(intr, pose_ext).project_points(proxy), np.full(17, 0.9)]
    fused = _camera_from_candidate(pose, rotation, 1000, 800, pose_keypoints=k)
    assert fused is not None
    assert abs(fused.extrinsics.yaw - pose_ext.yaw) < 1e-6
    assert abs(fused.extrinsics.pitch - pose_ext.pitch) < 1e-6
    assert abs(fused.extrinsics.roll - 7.0) < 1e-6
    assert abs(fused.focal_equiv_35mm - pose.focal_equiv_35mm) < 1e-6


def test_pose_solver_accepts_scalar_bbox_dimensions():
    """A normal XYXY bbox must produce scalar width/height terms without TypeError."""
    _, ext = _camera_pose_from_params(4.0, 1.3, 0.0, 0.0, 0.0)
    intr = CameraIntrinsics.from_focal_mm(70.0, 1000, 800)
    proxy = pose_driven_person_points(np.zeros((17, 3), dtype=float), 1000, 800)
    observed = CameraModel(intr, ext).project_points(proxy)
    keypoints = np.c_[observed, np.full(17, 0.9)]
    x0, y0 = np.nanmin(observed, axis=0)
    x1, y1 = np.nanmax(observed, axis=0)
    candidates = PoseSolver.fit_camera_to_pose(
        keypoints,
        1000,
        800,
        subject_bbox=(int(x0), int(y0), int(x1), int(y1)),
        focal_seeds=(70.0,),
        num_candidates=1,
    )
    assert candidates
    assert np.isfinite(candidates[0].score)


def test_candidate_family_replaces_pathological_optimizer_singleton():
    pose = _pose_for_candidate_family()
    candidates = estimate_camera_pose_candidates(pose, subject_bbox=pose.bbox, num_candidates=6)
    assert len(candidates) >= 4
    assert all(24.0 <= c.focal_equiv_35mm <= 150.0 for c in candidates)
    assert all(0.70 <= c.height <= 2.10 for c in candidates)
    assert all(1.20 <= c.distance <= 15.0 for c in candidates)
    assert all(c.losses.get("candidate_source") in {"analytic_pose_family", "numerical_pose_fit"} for c in candidates)
    assert not any(abs(c.height - 0.40) < 1e-6 for c in candidates)


def test_candidate_family_has_explicit_focal_distance_ambiguity():
    pose = _pose_for_candidate_family()
    candidates = estimate_camera_pose_candidates(pose, subject_bbox=pose.bbox, num_candidates=6)
    focal_values = [round(c.focal_equiv_35mm, 1) for c in candidates]
    distances = [round(c.distance, 2) for c in candidates]
    assert len(set(focal_values)) >= 4
    assert len(set(distances)) >= 4
    assert all(c.losses.get("focal_distance_ambiguity") is True or c.losses.get("candidate_source") == "numerical_pose_fit" for c in candidates)
