from __future__ import annotations

import numpy as np

from reverse_engineering.scene import SceneModel
from reverse_engineering.multi_person_layout import MultiPersonLayoutResult, PersonLayout


def test_scene_model_subject_is_self_healing_after_subjects_reset():
    scene = SceneModel()
    scene.subjects = []
    subject = scene.subject
    assert len(scene.subjects) == 1
    assert subject.person_index == 0
    assert np.isfinite(subject.height)


def test_from_reverse_result_preserves_primary_subject_height_when_rebuilding_layout():
    class Scalar:
        def __init__(self, value):
            self.value = value

    class Focal:
        def __init__(self, value):
            self.equivalent_35mm = Scalar(value)

    class CameraPose:
        camera_distance = Scalar(4.0)
        camera_height = Scalar(1.5)
        camera_yaw = Scalar(0.0)
        camera_pitch = Scalar(0.0)
        camera_roll = Scalar(0.0)

    points = tuple((x, y, 0.95) for x, y in (
        (0.45, 0.20), (0.50, 0.20), (0.40, 0.30), (0.60, 0.30),
        (0.38, 0.45), (0.62, 0.45), (0.40, 0.55), (0.60, 0.55),
        (0.42, 0.70), (0.58, 0.70), (0.44, 0.82), (0.56, 0.82),
        (0.43, 0.88), (0.57, 0.88), (0.43, 0.96), (0.57, 0.96),
        (0.50, 0.12),
    ))
    people = (
        PersonLayout(0, (0.35, 0.10, 0.65, 0.98), (0.50, 0.54), 0.264, 0.45, -0.1, 0.7, True, points),
        PersonLayout(1, (0.10, 0.15, 0.30, 0.75), (0.20, 0.45), 0.12, 0.80, 0.6, 0.7, True, points),
    )
    result = type("Result", (), {})()
    result.image_size = (1000, 800)
    result.subject_keypoints = None
    result.candidate_solutions = []
    result._sim_candidates = []
    result.camera_pose = CameraPose()
    result.focal_length = Focal(50.0)
    result.multi_person_layout = MultiPersonLayoutResult(
        people=people,
        depth_backend="fixture",
        depth_confidence=0.7,
        independent_depth=True,
    )

    scene = SceneModel()
    scene.subject.height = 1.82
    rebuilt = SceneModel.from_reverse_result(result)
    assert len(rebuilt.subjects) == 2
    assert all(np.isclose(subject.height, 1.82) for subject in rebuilt.subjects)
    assert rebuilt.subject.person_index == 0
