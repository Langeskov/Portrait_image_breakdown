from __future__ import annotations

from reverse_engineering.reference_camera import estimate_reference_camera_hypothesis
from reverse_engineering.reference_line_calibration import ReferenceLineConstraint, ReferenceLineEvidence
from reverse_engineering.reference_reconstruction import ReferenceComposition
from reverse_engineering.scene import SceneModel
from reverse_engineering.scene_anchors import AnchorKind, SceneAnchor


def _composition(center=(900.0, 600.0), scale=0.15):
    return ReferenceComposition(
        1800, 1200,
        anchors=(),
        subject_bbox=(700.0, 300.0, 1100.0, 900.0),
        subject_center=center,
        subject_scale=scale,
    )


def test_horizontal_reference_line_adds_roll_support():
    scene = SceneModel()
    reference = _composition(scale=0.16)
    current = _composition(scale=0.12)
    anchor = SceneAnchor("rail", "Rail", AnchorKind.PLANE)
    evidence = ReferenceLineEvidence((100.0, 100.0), (500.0, 140.0), ReferenceLineConstraint.HORIZONTAL)

    base = estimate_reference_camera_hypothesis(scene, reference, current, selected_anchor=anchor)
    constrained = estimate_reference_camera_hypothesis(
        scene, reference, current, selected_anchor=anchor, line_evidence=evidence
    )

    assert constrained.roll_correction_deg is not None
    assert constrained.line_constraint == "horizontal"
    assert constrained.line_observed_angle_deg is not None
    assert constrained.confidence >= base.confidence
    assert abs(constrained.roll_correction_deg + constrained.line_observed_angle_deg) < 1e-6


def test_free_reference_line_does_not_change_roll_hypothesis():
    scene = SceneModel()
    reference = _composition(scale=0.16)
    current = _composition(scale=0.12)
    anchor = SceneAnchor("rail", "Rail", AnchorKind.PLANE)
    evidence = ReferenceLineEvidence((100.0, 100.0), (500.0, 140.0), ReferenceLineConstraint.FREE)

    base = estimate_reference_camera_hypothesis(scene, reference, current, selected_anchor=anchor)
    constrained = estimate_reference_camera_hypothesis(
        scene, reference, current, selected_anchor=anchor, line_evidence=evidence
    )

    assert constrained.roll_correction_deg is None
    assert constrained.confidence == base.confidence
