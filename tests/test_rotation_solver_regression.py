"""Regression tests for scene-rotation score contracts."""
from __future__ import annotations

import numpy as np

from reverse_engineering.rotation_solver import _estimate_line_roll
from reverse_engineering.scene_geometry import LineSegment, SceneGeometryEvidence


def _line(angle: float, length: float = 260.0, index: int = 0) -> LineSegment:
    x1 = 40.0 + index * 9.0
    y1 = 80.0 + index * 4.0
    rad = np.radians(angle)
    x2 = x1 + length * np.cos(rad)
    y2 = y1 + length * np.sin(rad)
    return LineSegment(x1, y1, float(x2), float(y2), length, angle)


def test_line_roll_runner_up_score_is_scalar_not_candidate_tuple():
    """The runner-up score must be numeric before score separation arithmetic."""
    lines = [
        _line(0.0, index=0),
        _line(0.5, index=1),
        _line(-0.5, index=2),
        _line(89.5, index=3),
        _line(-89.0, index=4),
        _line(90.0, index=5),
        _line(1.0, index=6),
        _line(89.0, index=7),
    ]
    evidence = SceneGeometryEvidence(
        width=1200,
        height=800,
        lines=tuple(lines),
        clusters=tuple(),
        vanishing_points=tuple(),
        vertical_cluster=None,
        horizontal_clusters=tuple(),
        horizon_angle_deg=None,
        confidence=0.9,
    )

    roll, confidence, count = _estimate_line_roll(evidence)

    assert count == len(lines)
    assert roll is not None
    assert np.isfinite(roll)
    assert 0.0 <= confidence <= 1.0
