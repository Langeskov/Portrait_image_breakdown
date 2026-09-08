"""Regression tests for image/pose coordinate contracts."""
from __future__ import annotations

import numpy as np

from core.pose_detector import PoseLandmark, PoseResult
from reverse_engineering.engine_v2 import _normalize_pose_input


def _pose(width=2048, height=1364):
    landmarks = [
        PoseLandmark(
            index=i,
            x=200.0 + i * 20.0,
            y=100.0 + i * 40.0,
            z=0.0,
            visibility=0.95,
            world_x=(200.0 + i * 20.0) / width,
            world_y=(100.0 + i * 40.0) / height,
            world_z=0.0,
        )
        for i in range(17)
    ]
    return PoseResult(
        landmarks=landmarks,
        image_width=width,
        image_height=height,
        detection_confidence=0.9,
        bbox=(160, 80, 980, 1430),
    )


def test_reverse_engineering_boundary_rescales_pose_and_bbox():
    pose = _pose()
    normalized, bbox, changed = _normalize_pose_input(pose, 1600, 1065, pose.bbox)
    assert changed is True
    assert normalized.image_width == 1600
    assert normalized.image_height == 1065
    assert np.isclose(normalized.landmarks[0].x, 200.0 * (1600 / 2048))
    assert np.isclose(normalized.landmarks[0].y, 100.0 * (1065 / 1364))
    assert bbox == (125, 62, 766, 1118)


def test_reverse_engineering_boundary_preserves_matching_coordinates():
    pose = _pose(1600, 1065)
    normalized, bbox, changed = _normalize_pose_input(pose, 1600, 1065, None)
    assert changed is False
    assert normalized is pose
    assert bbox == pose.bbox
