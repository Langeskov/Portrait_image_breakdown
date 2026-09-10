from pathlib import Path

import pytest

from core import model_config


ROOT = Path(__file__).resolve().parents[1]


def test_pyinstaller_release_files_exist():
    spec = ROOT / "packaging" / "portrait_image_breakdown.spec"
    script = ROOT / "packaging" / "build_pyinstaller_windows.ps1"
    icon = ROOT / "assets" / "app.ico"

    assert spec.exists()
    assert script.exists()
    assert icon.exists()


def test_pyinstaller_spec_includes_models_and_ico():
    source = (ROOT / "packaging" / "portrait_image_breakdown.spec").read_text(encoding="utf-8")
    assert 'collect_submodules("ultralytics")' in source
    assert '"assets" / "app.ico"' in source
    assert '(ROOT / "model").glob("*.pt")' in source
    assert "console=False" in source
    assert "COLLECT(" in source


def test_application_uses_ico_runtime_icon():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert 'ROOT / "assets" / "app.ico"' in source
    assert 'ROOT / "assets" / "icon.png"' not in source


def test_application_loads_pose_model_without_top_level_ultralytics_import():
    source = (ROOT / "gui" / "application_window.py").read_text(encoding="utf-8")
    assert "from core.pose_detector import PoseDetector" not in source.split("class PoseModelLoadWorker", 1)[0]
    assert "class PoseModelLoadWorker(QThread):" in source
    assert "QTimer.singleShot(0, self.start_model_loading)" in source
    assert "正在后台加载姿态模型" in source


def test_packaged_missing_model_never_attempts_download(monkeypatch):
    monkeypatch.setattr(model_config, "is_packaged_application", lambda: True)
    monkeypatch.setattr(
        model_config,
        "resolve_pose_model_path",
        lambda key=None: ROOT / "model" / "definitely-missing-release-model.pt",
    )
    with pytest.raises(FileNotFoundError, match="installed application"):
        model_config.ensure_pose_model("m")
