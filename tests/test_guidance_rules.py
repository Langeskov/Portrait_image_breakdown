"""Targeted unit tests for guidance rule fixes.

Tests two specific changes:
1. Sitting/squatting should NOT trigger stagger (action category guard)
2. Elbow angle >= 165 triggers outward, < 165 does not
"""
import pytest
from core.action_classifier import ActionCategory, ActionResult
from core.camera_analyzer import CameraAngle, CameraResult, ShotType
from core.composition import CompositionResult, CompositionType
from core.guidance import Direction, GuidanceState, generate_guidance
from core.orientation import FacingDirection, OrientationResult, TiltDirection


def _action(**overrides):
    defaults = dict(
        category=ActionCategory.STANDING,
        confidence=0.9,
        sub_description="fixture",
        joint_angles={"left_elbow": 170.0, "right_elbow": 170.0,
                      "left_knee": 160.0, "right_knee": 160.0},
        features={"knee_angle_avg": 160.0, "knee_angle_diff": 0.0,
                  "stance_width": 0.1, "ankle_y_diff": 0.0,
                  "shoulder_y": 0.4, "hip_y": 0.55,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
    )
    defaults.update(overrides)
    features = defaults.pop("features")
    joint_angles = defaults.pop("joint_angles")
    return ActionResult(
        category=defaults["category"],
        confidence=defaults["confidence"],
        sub_description=defaults["sub_description"],
        joint_angles=joint_angles,
        features=features,
    )


def _orientation(facing=FacingDirection.LEFT):
    return OrientationResult(facing, TiltDirection.UPRIGHT, 0.0, 0.0, 0.9, "fixture")


def _camera(ratio=0.4):
    return CameraResult(ShotType.MEDIUM, CameraAngle.EYE_LEVEL, ratio,
                        (0.0, 0.0), 0.0, "fixture")


def _composition(x=0.5):
    return CompositionResult(CompositionType.CENTER, (x, 0.5), 0.8, 0.5,
                             0.15, "balanced", (x, 0.5), 0.8, [], "fixture")


# ── Stagger guard: sitting ──

def test_sitting_with_bent_knees_no_stagger():
    """SITTING + knee_avg < 125 must NOT produce stagger."""
    action = _action(
        category=ActionCategory.SITTING,
        features={"knee_angle_avg": 95.0, "knee_angle_diff": 5.0,
                  "stance_width": 0.05, "ankle_y_diff": 0.0,
                  "shoulder_y": 0.4, "hip_y": 0.55,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
    )
    result = generate_guidance(action, _orientation(), _camera(), _composition())
    stagger = [a for a in result.all_actions if a.direction == Direction.STAGGER]
    assert stagger == [], f"SITTING should not stagger, got: {[a.cue for a in stagger]}"


def test_squatting_with_bent_knees_no_stagger():
    """SQUATTING + knee_avg < 125 must NOT produce stagger."""
    action = _action(
        category=ActionCategory.SQUATTING,
        features={"knee_angle_avg": 80.0, "knee_angle_diff": 3.0,
                  "stance_width": 0.1, "ankle_y_diff": 0.0,
                  "shoulder_y": 0.5, "hip_y": 0.65,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
    )
    result = generate_guidance(action, _orientation(), _camera(), _composition())
    stagger = [a for a in result.all_actions if a.direction == Direction.STAGGER]
    assert stagger == [], f"SQUATTING should not stagger, got: {[a.cue for a in stagger]}"


def test_standing_with_bent_knees_can_stagger():
    """STANDING + knee_avg < 125 + low knee_diff → stagger should fire."""
    action = _action(
        category=ActionCategory.STANDING,
        features={"knee_angle_avg": 110.0, "knee_angle_diff": 5.0,
                  "stance_width": 0.1, "ankle_y_diff": 0.0,
                  "shoulder_y": 0.4, "hip_y": 0.55,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
    )
    result = generate_guidance(action, _orientation(), _camera(), _composition())
    stagger = [a for a in result.all_actions if a.direction == Direction.STAGGER]
    assert len(stagger) >= 1, "STANDING with bent knees should be eligible for stagger"


# ── Elbow threshold ──

def test_elbow_161_no_outward():
    """Elbow angle 161° should NOT trigger outward (below threshold)."""
    action = _action(
        features={"knee_angle_avg": 160.0, "knee_angle_diff": 25.0,
                  "stance_width": 0.15, "ankle_y_diff": 0.05,
                  "shoulder_y": 0.4, "hip_y": 0.55,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
        joint_angles={"left_elbow": 161.0, "right_elbow": 161.0,
                      "left_knee": 160.0, "right_knee": 160.0},
    )
    result = generate_guidance(action, _orientation(), _camera(), _composition())
    outward = [a for a in result.all_actions if a.direction == Direction.OUTWARD]
    assert outward == [], f"161° should not trigger outward, got: {[a.cue for a in outward]}"


def test_elbow_166_triggers_outward():
    """Elbow angle 166° should trigger outward (at threshold)."""
    action = _action(
        features={"knee_angle_avg": 160.0, "knee_angle_diff": 25.0,
                  "stance_width": 0.15, "ankle_y_diff": 0.05,
                  "shoulder_y": 0.4, "hip_y": 0.55,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
        joint_angles={"left_elbow": 166.0, "right_elbow": 166.0,
                      "left_knee": 160.0, "right_knee": 160.0},
    )
    result = generate_guidance(action, _orientation(), _camera(), _composition())
    outward = [a for a in result.all_actions if a.direction == Direction.OUTWARD]
    assert len(outward) >= 1, "166° should trigger outward"


def test_elbow_177_triggers_outward():
    """Elbow angle 177° should trigger outward (well above threshold)."""
    action = _action(
        features={"knee_angle_avg": 160.0, "knee_angle_diff": 25.0,
                  "stance_width": 0.15, "ankle_y_diff": 0.05,
                  "shoulder_y": 0.4, "hip_y": 0.55,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
        joint_angles={"left_elbow": 177.0, "right_elbow": 177.0,
                      "left_knee": 160.0, "right_knee": 160.0},
    )
    result = generate_guidance(action, _orientation(), _camera(), _composition())
    outward = [a for a in result.all_actions if a.direction == Direction.OUTWARD]
    assert len(outward) >= 1, "177° should trigger outward"


def test_elbow_asymmetric_no_outward():
    """If only one elbow is high, outward should NOT trigger (both required)."""
    action = _action(
        features={"knee_angle_avg": 160.0, "knee_angle_diff": 25.0,
                  "stance_width": 0.15, "ankle_y_diff": 0.05,
                  "shoulder_y": 0.4, "hip_y": 0.55,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
        joint_angles={"left_elbow": 170.0, "right_elbow": 140.0,
                      "left_knee": 160.0, "right_knee": 160.0},
    )
    result = generate_guidance(action, _orientation(), _camera(), _composition())
    outward = [a for a in result.all_actions if a.direction == Direction.OUTWARD]
    assert outward == [], "Asymmetric elbows should not trigger outward"


# ── Sanity: existing behavior preserved ──

def test_hold_when_already_asymmetric():
    """knee_diff > 20 with no other issues → HOLD."""
    action = _action(
        features={"knee_angle_avg": 160.0, "knee_angle_diff": 25.0,
                  "stance_width": 0.15, "ankle_y_diff": 0.05,
                  "shoulder_y": 0.4, "hip_y": 0.55,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
        joint_angles={"left_elbow": 120.0, "right_elbow": 130.0,
                      "left_knee": 140.0, "right_knee": 165.0},
    )
    result = generate_guidance(action, _orientation(), _camera(), _composition())
    assert result.state == GuidanceState.HOLD


def test_composition_subject_right_when_left_of_center():
    """px < 0.38 → SUBJECT_RIGHT cue."""
    action = _action(
        features={"knee_angle_avg": 160.0, "knee_angle_diff": 25.0,
                  "stance_width": 0.15, "ankle_y_diff": 0.05,
                  "shoulder_y": 0.4, "hip_y": 0.55,
                  "hands_above_shoulders": 0.0, "wrist_y_avg": 0.6},
        joint_angles={"left_elbow": 120.0, "right_elbow": 130.0,
                      "left_knee": 140.0, "right_knee": 165.0},
    )
    result = generate_guidance(action, _orientation(), _camera(), _composition(x=0.30))
    if result.primary and result.primary.direction in (Direction.SUBJECT_RIGHT,):
        pass  # composition cue is primary
    else:
        comp = [a for a in result.all_actions if a.direction == Direction.SUBJECT_RIGHT]
        assert len(comp) >= 1, "Subject at x=0.30 should get SUBJECT_RIGHT cue"
