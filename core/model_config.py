"""Centralized local model configuration for Portrait Image Breakdown."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"


@dataclass(frozen=True)
class PoseModelSpec:
    key: str
    filename: str
    label: str


POSE_MODELS: tuple[PoseModelSpec, ...] = (
    PoseModelSpec("n", "yolo26n-pose.pt", "YOLO26n · fastest"),
    PoseModelSpec("s", "yolo26s-pose.pt", "YOLO26s · balanced"),
    PoseModelSpec("m", "yolo26m-pose.pt", "YOLO26m · higher accuracy"),
    PoseModelSpec("l", "yolo26l-pose.pt", "YOLO26l · high accuracy"),
    PoseModelSpec("x", "yolo26x-pose.pt", "YOLO26x · highest accuracy"),
)

DEFAULT_POSE_MODEL = "x"


def get_pose_model(key_or_path: str | None = None) -> PoseModelSpec | str:
    """Resolve a built-in model key, filename, or arbitrary checkpoint path."""
    value = str(key_or_path or os.getenv("PIB_POSE_MODEL", DEFAULT_POSE_MODEL)).strip()
    for spec in POSE_MODELS:
        if value.lower() in {spec.key, spec.filename.lower()}:
            return spec
    path = Path(value).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return str(path)
    return str(MODEL_DIR / path.name)


def resolve_pose_model_path(key_or_path: str | None = None) -> Path:
    """Return the local checkpoint path expected by the application."""
    resolved = get_pose_model(key_or_path)
    if isinstance(resolved, PoseModelSpec):
        return MODEL_DIR / resolved.filename
    return Path(resolved).expanduser()


def pose_model_choices() -> tuple[PoseModelSpec, ...]:
    return POSE_MODELS


def pose_model_label(key_or_path: str | None = None) -> str:
    resolved = get_pose_model(key_or_path)
    if isinstance(resolved, PoseModelSpec):
        return resolved.label
    return Path(str(resolved)).name


def validate_pose_model(key_or_path: str | None = None) -> Path:
    """Validate a configured checkpoint and provide an actionable error."""
    path = resolve_pose_model_path(key_or_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Pose model not found: {path}\n\n"
            f"Place the YOLO26 pose checkpoint in {MODEL_DIR} or choose another local checkpoint."
        )
    return path
