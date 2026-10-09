import numpy as np

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
