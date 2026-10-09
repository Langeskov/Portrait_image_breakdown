from core.reference_pose_generation import generate_composition_aware_pose_target
from core.reference_reconstruction import ReferenceComposition


def test_composition_aware_target_uses_reference_center_and_scale():
    reference = ReferenceComposition(1000, 1000, (), (400, 200, 600, 800), (500, 500), 0.12)
    current = ReferenceComposition(1000, 1000, (), (300, 200, 500, 700), (400, 450), 0.08)
    landmarks = [type("LM", (), {"x": 400.0, "y": 200.0, "visibility": 1.0})(), type("LM", (), {"x": 600.0, "y": 800.0, "visibility": 1.0})()]
    result = generate_composition_aware_pose_target(landmarks, reference, current)
    assert result.success
    assert result.target is not None
    assert result.target.subject_center == (500, 500)
    assert result.target.visible_count == 2
