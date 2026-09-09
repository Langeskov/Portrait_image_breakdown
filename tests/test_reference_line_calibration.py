from __future__ import annotations

import math

from reverse_engineering.reference_line_calibration import (
    ReferenceLineConstraint,
    ReferenceLineEvidence,
    angular_error_deg,
    line_angle_deg,
    normalize_line_angle_deg,
)
from reverse_engineering.scene_anchors import SceneAnchor, AnchorKind


def test_horizontal_line_reports_zero_angle_and_zero_correction():
    evidence = ReferenceLineEvidence((10.0, 100.0), (210.0, 100.0), ReferenceLineConstraint.HORIZONTAL)
    assert evidence.observed_angle_deg == 0.0
    assert evidence.correction_deg == 0.0
    assert evidence.supports_roll
    assert evidence.confidence > 0.8


def test_sloped_line_produces_roll_correction_without_changing_line_length():
    angle = 8.0
    length = 400.0
    p1 = (100.0, 100.0)
    p2 = (p1[0] + length * math.cos(math.radians(angle)), p1[1] + length * math.sin(math.radians(angle)))
    evidence = ReferenceLineEvidence(p1, p2, ReferenceLineConstraint.HORIZONTAL)
    assert abs(evidence.observed_angle_deg - angle) < 1e-6
    assert abs(evidence.correction_deg + angle) < 1e-6
    assert abs(evidence.length_px - length) < 1e-6


def test_vertical_constraint_uses_ninety_degree_target():
    evidence = ReferenceLineEvidence((100.0, 200.0), (100.0, 20.0), ReferenceLineConstraint.VERTICAL)
    assert abs(abs(evidence.observed_angle_deg) - 90.0) < 1e-6
    assert abs(evidence.correction_deg) < 1e-6


def test_free_constraint_does_not_claim_roll_support():
    evidence = ReferenceLineEvidence((0.0, 0.0), (100.0, 50.0), ReferenceLineConstraint.FREE)
    assert evidence.target_angle_deg is None
    assert evidence.correction_deg is None
    assert not evidence.supports_roll


def test_angle_helpers_normalize_unoriented_lines():
    assert normalize_line_angle_deg(180.0) == 0.0
    assert normalize_line_angle_deg(-180.0) == 0.0
    assert angular_error_deg(89.0, -90.0) == 1.0
    assert abs(line_angle_deg((0.0, 0.0), (0.0, -10.0))) == 90.0


def test_anchor_serialization_preserves_reference_constraint():
    anchor = SceneAnchor(
        "rail", "Rail", AnchorKind.PLANE,
        image_points=((10.0, 20.0), (100.0, 21.0)),
        reference_line_constraint="horizontal",
    )
    restored = SceneAnchor.from_dict(anchor.to_dict())
    assert restored.reference_line_constraint == "horizontal"
    assert restored.image_points == anchor.image_points
