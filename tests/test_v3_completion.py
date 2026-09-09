import numpy as np

from reverse_engineering.plane_constraints import (
    PlaneConstraint,
    PlaneRelation,
    apply_position_constraint,
    evaluate_constraint,
    signed_distance_to_plane,
)
from reverse_engineering.reference_pose_generation import generate_composition_aware_pose_target
from reverse_engineering.reference_reconstruction import ReferenceComposition
from reverse_engineering.reconstruction_session import scene_from_dict, scene_to_dict
from reverse_engineering.scene import SceneModel
from reverse_engineering.scene_anchors import AnchorKind, SceneAnchor
from reverse_engineering.temporal import TemporalCameraState, TemporalFrameState, smooth_sequence


def test_session_roundtrip_preserves_camera_anchor_and_subjects():
    scene = SceneModel()
    scene.camera.yaw = 17.5
    scene.camera.roll = -2.0
    scene.subject.center_x = 0.8
    scene.anchors.append(SceneAnchor("wall", "Wall", AnchorKind.PLANE, (0, 2, 1), (0, 0, 1), (5, 2)))
    scene.plane_constraints = [{"constraint_id": "c1", "plane_anchor_id": "ground", "target_id": "primary_subject", "relation": "on_plane", "offset_m": 0.0}]
    payload = scene_to_dict(scene, image_path="photo.jpg", image_shape=(4000, 3000))
    restored, constraints = scene_from_dict(payload)
    assert restored.camera.yaw == 17.5
    assert restored.camera.roll == -2.0
    assert restored.subject.center_x == 0.8
    assert len(restored.anchors) == 2
    assert constraints[0]["constraint_id"] == "c1"


def test_plane_constraint_projects_point_and_reports_residual():
    plane = SceneAnchor("ground", "Ground", AnchorKind.PLANE, (0, 0, 0), (0, 1, 0), (4, 4))
    constraint = PlaneConstraint("c1", "ground", relation=PlaneRelation.ON_PLANE)
    point = (1.0, 1.2, 2.0)
    assert signed_distance_to_plane(point, plane) == 1.2
    corrected = apply_position_constraint(point, constraint, [plane])
    assert corrected == (1.0, 0.0, 2.0)
    result = evaluate_constraint(point, constraint, [plane])
    assert result.residual_m == 1.2
    assert not result.satisfied


def test_angular_plane_constraint_requires_target_direction():
    plane = SceneAnchor("wall", "Wall", AnchorKind.PLANE, (0, 0, 0), (0, 0, 1), (4, 4))
    constraint = PlaneConstraint("c2", "wall", relation=PlaneRelation.PARALLEL)
    result = evaluate_constraint((0, 0, 0), constraint, [plane], direction=None)
    assert not result.satisfied
    assert result.residual_m == float("inf")
    assert "requires a target direction" in result.message


def test_composition_aware_target_uses_reference_center_and_scale():
    reference = ReferenceComposition(1000, 1000, (), (400, 200, 600, 800), (500, 500), 0.12)
    current = ReferenceComposition(1000, 1000, (), (300, 200, 500, 700), (400, 450), 0.08)
    landmarks = [type("LM", (), {"x": 400.0, "y": 200.0, "visibility": 1.0})(), type("LM", (), {"x": 600.0, "y": 800.0, "visibility": 1.0})()]
    result = generate_composition_aware_pose_target(landmarks, reference, current)
    assert result.success
    assert result.target is not None
    assert result.target.subject_center == (500, 500)
    assert result.target.visible_count == 2


def test_temporal_smoothing_reduces_camera_jitter_and_holds_keypoints():
    camera_a = TemporalCameraState(4.0, 1.5, 10.0, 2.0, 0.0, 50.0)
    camera_b = TemporalCameraState(4.2, 1.6, 14.0, 2.8, 0.5, 50.0)
    frames = [
        TemporalFrameState(0.0, np.zeros((3, 2)), camera_a, 0.8),
        TemporalFrameState(0.1, np.ones((3, 2)) * 2.0, camera_b, 0.9),
        TemporalFrameState(0.2, None, camera_b, 0.0),
    ]
    states = smooth_sequence(frames, alpha=0.5)
    assert 4.0 < states[1].camera.distance < 4.2
    assert states[2].keypoints is not None
    assert states[2].keypoints.mean() > 0.0
