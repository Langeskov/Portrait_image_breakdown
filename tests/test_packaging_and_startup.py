from pathlib import Path

import pytest

from core import model_config


ROOT = Path(__file__).resolve().parents[1]


def test_pyinstaller_packaging_files_exist():
    assert (ROOT / "packaging" / "build_pyinstaller_windows.ps1").exists()
    spec = ROOT / "packaging" / "portrait_image_breakdown.spec"
    source = spec.read_text(encoding="utf-8")
    assert 'icon=str(ROOT / "assets" / "app.ico")' in source
    assert '"assets" / "app.ico"' in source
    assert '"model"' in source


def test_application_uses_ico_icon():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert 'ROOT / "assets" / "app.ico"' in source
    assert "nuitka-project" not in source


def test_application_loads_pose_model_without_top_level_ultralytics_import():
    source = (ROOT / "gui" / "application_window.py").read_text(encoding="utf-8")
    assert "from core.pose_detector import PoseDetector" not in source.split("class PoseModelLoadWorker", 1)[0]
    assert "QTimer.singleShot(50, self.start_model_loading)" in source
    assert "pose_model_selection_v2" in source
    assert "DEFAULT_POSE_MODEL" in source


def test_packaged_missing_model_never_attempts_download(monkeypatch):
    monkeypatch.setattr(model_config, "is_packaged_application", lambda: True)
    monkeypatch.setattr(
        model_config,
        "resolve_pose_model_path",
        lambda key=None: ROOT / "model" / "definitely-missing-release-model.pt",
    )
    with pytest.raises(FileNotFoundError, match="installed application"):
        model_config.ensure_pose_model("m")
