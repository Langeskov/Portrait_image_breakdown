from reverse_engineering.reference_camera import estimate_reference_camera_hypothesis
from reverse_engineering.reference_reconstruction import ReferenceComposition
from reverse_engineering.scene import SceneModel


def _composition(width, height, cx, cy, scale):
    return ReferenceComposition(
        width=width,
        height=height,
        anchors=(),
        subject_bbox=(cx - 100.0, cy - 200.0, cx + 100.0, cy + 200.0),
        subject_center=(cx, cy),
        subject_scale=scale,
    )


def test_reference_hypothesis_distance_uses_inverse_square_area_relation():
    scene = SceneModel()
    scene.camera.distance = 6.0
    current = _composition(1000, 1000, 500, 500, 0.08)
    reference = _composition(1000, 1000, 500, 500, 0.02)

    result = estimate_reference_camera_hypothesis(scene, reference, current)

    assert result.success
    assert result.reference_distance_m == 12.0
    assert result.distance_delta_m == 6.0
    assert result.reframe_yaw_deg == 0.0
    assert result.reframe_pitch_deg == 0.0


def test_reference_hypothesis_center_offset_becomes_reaim_delta():
    scene = SceneModel()
    current = _composition(1000, 1000, 500, 500, 0.08)
    reference = _composition(1000, 1000, 600, 400, 0.08)

    result = estimate_reference_camera_hypothesis(scene, reference, current)

    assert result.success
    assert result.center_dx == 0.1
    assert result.center_dy == -0.1
    assert result.reframe_yaw_deg > 0
    assert result.reframe_pitch_deg > 0


def test_reference_hypothesis_does_not_require_or_mutate_scene_camera():
    scene = SceneModel()
    before = (scene.camera.distance, scene.camera.yaw, scene.camera.pitch, scene.camera.focal_length_mm)
    current = _composition(1000, 1000, 500, 500, 0.08)
    reference = _composition(1000, 1000, 500, 500, 0.06)

    result = estimate_reference_camera_hypothesis(scene, reference, current)

    assert result.success
    after = (scene.camera.distance, scene.camera.yaw, scene.camera.pitch, scene.camera.focal_length_mm)
    assert before == after
