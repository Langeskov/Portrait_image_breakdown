"""Regression tests for photographer-ready verbal cues."""

from core.action_classifier import ActionCategory, ActionResult
from core.camera_analyzer import CameraAngle, CameraResult, ShotType
from core.composition import CompositionResult, CompositionType
from core.orientation import FacingDirection, OrientationResult, TiltDirection
from core.photographer_cues import generate_photographer_cues, speakable_summary


def _action(**features):
    base = {
        "knee_angle_avg": 160.0,
        "knee_angle_diff": 2.0,
        "stance_width": 0.05,
        "hands_above_shoulders": 0.0,
        "ankle_y_diff": 0.02,
    }
    base.update(features)
    return ActionResult(
        ActionCategory.STANDING,
        0.85,
        "standard",
        {"left_elbow": 172.0, "right_elbow": 171.0},
        base,
    )


def _orientation(facing=FacingDirection.FRONT):
    return OrientationResult(
        facing=facing,
        tilt=TiltDirection.UPRIGHT,
        facing_angle=0.0,
        tilt_angle=0.0,
        confidence=0.9,
        detail="fixture",
    )


def _camera(ratio=0.4):
    return CameraResult(
        shot_type=ShotType.MEDIUM,
        camera_angle=CameraAngle.EYE_LEVEL,
        subject_ratio=ratio,
        subject_center_offset=(0.0, 0.0),
        dutch_angle_deg=0.0,
        detail="fixture",
    )


def _composition(x=0.5):
    return CompositionResult(
        primary_type=CompositionType.CENTER,
        subject_position=(x, 0.5),
        thirds_alignment=0.8,
        symmetry_score=0.5,
        headroom=0.15,
        lead_space="balanced",
        visual_weight=(x, 0.5),
        balance_score=0.8,
        suggestions=[],
        detail="fixture",
    )


def test_cues_are_speakable_and_preserve_reasoning():
    cues = generate_photographer_cues(_action(), _orientation(), _camera(), _composition())
    assert cues
    assert cues[0].cue.endswith(("。", "。"))
    assert cues[0].reason
    assert "重心" in speakable_summary(cues)


def test_composition_changes_direction_of_spoken_cue():
    left = generate_photographer_cues(_action(), _orientation(), _camera(), _composition(x=0.2))
    right = generate_photographer_cues(_action(), _orientation(), _camera(), _composition(x=0.8))
    assert any("右边" in c.cue for c in left)
    assert any("左边" in c.cue for c in right)


def test_tight_frame_limits_movement():
    cues = generate_photographer_cues(_action(), _orientation(), _camera(ratio=0.8), _composition())
    assert any(c.category == "camera" for c in cues)


def test_cue_engine_does_not_require_action_label():
    a1 = _action()
    a2 = _action(); a2.category = ActionCategory.SITTING
    c1 = [c.cue for c in generate_photographer_cues(a1, _orientation(), _camera(), _composition())]
    c2 = [c.cue for c in generate_photographer_cues(a2, _orientation(), _camera(), _composition())]
    assert c1 == c2
