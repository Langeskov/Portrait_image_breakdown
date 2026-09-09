from pathlib import Path

from core.model_config import (
    DEFAULT_POSE_MODEL,
    MODEL_DIR,
    get_pose_model,
    pose_model_choices,
    resolve_pose_model_path,
)


def test_default_pose_model_is_yolo26x_in_model_directory():
    model = get_pose_model()
    assert model.key == DEFAULT_POSE_MODEL == "x"
    assert resolve_pose_model_path() == MODEL_DIR / "yolo26x-pose.pt"


def test_builtin_pose_model_keys_resolve_to_local_checkpoints():
    names = {spec.key: spec.filename for spec in pose_model_choices()}
    assert names == {
        "n": "yolo26n-pose.pt",
        "s": "yolo26s-pose.pt",
        "m": "yolo26m-pose.pt",
        "l": "yolo26l-pose.pt",
        "x": "yolo26x-pose.pt",
    }
    assert resolve_pose_model_path("m") == MODEL_DIR / "yolo26m-pose.pt"
    assert resolve_pose_model_path("yolo26l-pose.pt") == MODEL_DIR / "yolo26l-pose.pt"


def test_bare_custom_checkpoint_names_are_kept_under_model_directory():
    assert resolve_pose_model_path("custom-pose.pt") == MODEL_DIR / "custom-pose.pt"
    assert Path(resolve_pose_model_path("custom-pose.pt")).parent == MODEL_DIR


def test_explicit_paths_remain_explicit():
    assert resolve_pose_model_path("custom/checkpoint.pt") == Path("custom/checkpoint.pt")
