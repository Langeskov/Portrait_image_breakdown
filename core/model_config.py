"""Centralized local model configuration for Portrait Image Breakdown."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from urllib.request import urlopen


def runtime_root_dir() -> Path:
    """Return the application root in source, PyInstaller, or Nuitka builds."""
    compiled = globals().get("__compiled__")
    containing_dir = getattr(compiled, "containing_dir", None)
    if containing_dir:
        return Path(containing_dir)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT_DIR = runtime_root_dir()
MODEL_DIR = ROOT_DIR / "model"
ASSET_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


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

DEFAULT_POSE_MODEL = "m"


def is_packaged_application() -> bool:
    """Whether the current process is a compiled/standalone distribution."""
    return bool(globals().get("__compiled__")) or bool(getattr(sys, "frozen", False))


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


def _download_builtin_pose_model(spec: PoseModelSpec, target: Path) -> Path:
    """Download an official built-in checkpoint directly into ``model/``."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".download")
    url = f"{ASSET_BASE_URL}/{spec.filename}"
    try:
        with urlopen(url, timeout=30) as response, temp.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temp.stat().st_size <= 0:
            raise OSError("Downloaded model file is empty")
        temp.replace(target)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def ensure_pose_model(key_or_path: str | None = None, *, allow_download: bool | None = None) -> Path:
    """Return a local checkpoint, downloading only when explicitly allowed in source builds."""
    resolved = get_pose_model(key_or_path)
    path = resolve_pose_model_path(key_or_path)
    if path.is_file():
        return path

    if allow_download is None:
        allow_download = not is_packaged_application()

    if not allow_download:
        raise FileNotFoundError(
            f"Pose model not found in the installed application: {path}\n\n"
            f"Reinstall the application with the model files included."
        )

    if isinstance(resolved, PoseModelSpec):
        try:
            return _download_builtin_pose_model(resolved, path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Pose model is missing and automatic download failed: {path}\n\n"
                f"Download {resolved.filename} manually into {MODEL_DIR}.\n"
                f"Network error: {exc}"
            ) from exc
    raise FileNotFoundError(
        f"Pose model not found: {path}\n\n"
        f"Place the checkpoint in {MODEL_DIR} or provide an existing local checkpoint path."
    )


def validate_pose_model(key_or_path: str | None = None) -> Path:
    """Validate a configured checkpoint without triggering a download."""
    path = resolve_pose_model_path(key_or_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Pose model not found: {path}\n\n"
            f"Place the checkpoint in {MODEL_DIR} or choose another local checkpoint."
        )
    return path
