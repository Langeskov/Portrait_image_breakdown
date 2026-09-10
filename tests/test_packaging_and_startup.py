from pathlib import Path

import pytest

from core import model_config


ROOT = Path(__file__).resolve().parents[1]


def test_nuitka_project_options_include_release_data_dirs():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "# nuitka-project: --mode=standalone" in source
    assert "# nuitka-project: --enable-plugin=pyside6" in source
    assert "--include-data-dir={MAIN_DIRECTORY}/model=model" in source
    assert "--include-data-dir={MAIN_DIRECTORY}/assets=assets" in source


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
