"""Regression tests for goal-oriented pose guidance."""

from core.action_classifier import ActionCategory, ActionResult
from core.camera_analyzer import CameraAngle, CameraResult, ShotType
from core.composition import CompositionResult, CompositionType
from core.orientation import FacingDirection, OrientationResult, TiltDirection
from core.suggestion import generate_suggestions, _generate_pose_guidance


def _action(**features):
    base = {
        "knee_angle_avg": 160.0,
        "knee_angle_diff": 2.0,
        "stance_width": 0.05,
        "shoulder_y": 0.42,
        "hip_y": 0.43,
        "hands_above_shoulders": 0.0,
        "wrist_y_avg": 0.56,
        "ankle_y_diff": 0.02,
    }
    base.update(features)
    angles = {
        "left_knee": 160.0,
        "right_knee": 160.0,
        "left_elbow": 172.0,
        "right_elbow": 171.0,
        "left_hip": 150.0,
        "right_hip": 150.0,
    }
    return ActionResult(ActionCategory.STANDING, 0.85, "standard", angles, base)


def _orientation(facing=FacingDirection.FRONT, tilt=TiltDirection.UPRIGHT):
    return OrientationResult(
        facing=facing,
        tilt=tilt,
        facing_angle=0.0,
        tilt_angle=0.0,
        confidence=0.9,
        detail="fixture",
    )


def _camera():
    return CameraResult(
        shot_type=ShotType.MEDIUM,
        camera_angle=CameraAngle.EYE_LEVEL,
        subject_ratio=0.4,
        subject_center_offset=(0.0, 0.0),
        dutch_angle_deg=0.0,
        detail="fixture",
    )


def _composition(x=0.5, y=0.5):
    return CompositionResult(
        primary_type=CompositionType.CENTER,
        subject_position=(x, y),
        thirds_alignment=0.8,
        symmetry_score=0.5,
        headroom=0.15,
        lead_space="balanced",
        visual_weight=(x, y),
        balance_score=0.8,
        suggestions=[],
        detail="fixture",
    )


def test_static_pose_gets_geometry_based_coaching():
    guidance = _generate_pose_guidance(_action(), _orientation(), _composition())
    titles = [s.title for s in guidance]
    assert "先释放对称站姿" in titles
    assert "把手臂从躯干上分开" in titles


def test_guidance_does_not_depend_on_current_action_label():
    a1 = _action()
    a2 = _action()
    a2.category = ActionCategory.SITTING
    g1 = [s.title for s in _generate_pose_guidance(a1, _orientation(), _composition())]
    g2 = [s.title for s in _generate_pose_guidance(a2, _orientation(), _composition())]
    assert g1 == g2


def test_goal_direction_follows_composition():
    result = generate_suggestions(
        _action(), _orientation(), _camera(), _composition(x=0.75)
    )
    assert any("朝画面中央打开" in s.title for s in result.suggestions)
    assert "让动作朝负空间展开" in result.creative_direction


def test_sitting_like_geometry_gets_leg_separation_guidance():
    action = _action(knee_angle_avg=98.0, knee_angle_diff=6.0)
    guidance = _generate_pose_guidance(action, _orientation(), _composition())
    assert any(s.title == "让双腿产生前后层次" for s in guidance)
