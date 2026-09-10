from core.model_config import DEFAULT_POSE_MODEL, pose_model_choices


def test_settings_exposes_all_builtin_pose_models():
    keys = [spec.key for spec in pose_model_choices()]
    assert keys == ["n", "s", "m", "l", "x"]
    assert DEFAULT_POSE_MODEL == "m"


def test_default_model_has_expected_label():
    spec = next(spec for spec in pose_model_choices() if spec.key == DEFAULT_POSE_MODEL)
    assert "YOLO26m" in spec.label
