"""Application-owned MainWindow facade with explicit service injection."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings, QThread, QTimer, Signal

from core.model_config import DEFAULT_POSE_MODEL, get_pose_model, pose_model_label
from gui.main_window import MainWindow as _BaseMainWindow, _resize_for_analysis, AnalysisWorker, AnalysisBundle
from gui.cache import AnalysisCache


class PoseModelLoadWorker(QThread):
    """Load Ultralytics and the selected checkpoint outside the Qt GUI thread."""

    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, model: str):
        super().__init__()
        self._model = model

    def run(self) -> None:
        try:
            # Keep the heavyweight Ultralytics import and YOLO construction off
            # the GUI thread, including the first-time package initialization.
            from core.pose_detector import PoseDetector
            detector = PoseDetector(model=self._model)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if self.isInterruptionRequested():
            detector.close()
            return
        self.loaded.emit(detector)


class ApplicationMainWindow(_BaseMainWindow):
    """MainWindow variant that receives engine/cache/profile/model services explicitly."""

    _SETTINGS_ORG = "PortraitImageBreakdown"
    _SETTINGS_APP = "PortraitImageBreakdown"
    _POSE_MODEL_SETTING = "pose_model"

    def __init__(self, services, parent=None):
        settings = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        configured = settings.value(self._POSE_MODEL_SETTING, DEFAULT_POSE_MODEL)
        resolved = get_pose_model(str(configured) if configured is not None else DEFAULT_POSE_MODEL)
        pose_model = resolved.key if hasattr(resolved, "key") else str(resolved)

        # The base constructor builds all widgets but delegates detector
        # construction to _initialize_pose_detector(), which this facade
        # overrides so the heavy import/model load can happen asynchronously.
        super().__init__(pose_model=pose_model)
        if parent is not None:
            self.setParent(parent)
        self._services = services
        self._calibration_profile = "Generic"
        self._current_path: Optional[str] = None
        self._engine_factory = services.engine_factory
        self._image_cache_key = services.image_cache_key
        self._result_cache = AnalysisCache(capacity=8)
        self._pose_model = pose_model
        self._model_loader: Optional[PoseModelLoadWorker] = None
        self._model_ready = False
        self._pending_image_path: Optional[str] = None

        # The zero-delay timer runs after QApplication enters its event loop,
        # so the first paint of the window is not blocked by model startup.
        QTimer.singleShot(0, self.start_model_loading)
        QTimer.singleShot(0, lambda: self._set_model_controls_enabled(False))

    def _initialize_pose_detector(self, pose_model: str | None = None):
        """Do not import Ultralytics in the GUI thread; load it in PoseModelLoadWorker."""
        return None

    def _set_model_controls_enabled(self, enabled: bool) -> None:
        group = getattr(self, "_pose_model_group", None)
        if group is not None:
            group.setEnabled(enabled)

    @property
    def model_ready(self) -> bool:
        return self._model_ready and self._det is not None

    @property
    def current_image(self):
        return self._img

    @property
    def pose_model(self) -> str:
        """Current configured pose model key or explicit checkpoint path."""
        return self._pose_model

    @property
    def pose_model_name(self) -> str:
        if self._det is not None:
            return self._det.model_name
        return pose_model_label(self._pose_model)

    def start_model_loading(self) -> None:
        """Start the initial model load exactly once, without blocking the GUI."""
        if self._det is not None or self._model_loader is not None:
            return
        self._model_ready = False
        self._set_progress(0, f"正在后台加载姿态模型：{pose_model_label(self._pose_model)}")
        worker = PoseModelLoadWorker(self._pose_model)
        self._model_loader = worker
        worker.loaded.connect(self._on_model_loaded)
        worker.failed.connect(self._on_model_load_failed)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._on_model_loader_finished)
        worker.start()

    def _on_model_loader_finished(self) -> None:
        # The active worker reference is cleared by the GUI thread after the
        # result/error signal has been delivered.
        self._model_loader = None

    def _on_model_loaded(self, detector) -> None:
        self._det = detector
        self._model_ready = True
        self._set_model_controls_enabled(True)
        label = pose_model_label(self._pose_model)
        self._finish_progress(f"姿态模型已就绪：{label}")

        pending = self._pending_image_path
        self._pending_image_path = None
        if pending:
            self.load_image(pending)

    def _on_model_load_failed(self, message: str) -> None:
        self._model_ready = False
        self._set_model_controls_enabled(True)
        self._progress.setVisible(False)
        QMessageBox.critical(
            self,
            "Pose model",
            "无法加载姿态模型。\n\n"
            f"{message}\n\n"
            "发布版不会在启动时自动下载模型，请确认安装目录中的 model\\*.pt 文件完整。",
        )

    def _stop_model_loader(self) -> None:
        worker = self._model_loader
        if worker is None or not worker.isRunning():
            return
        worker.requestInterruption()
        if not worker.wait(3000):
            worker.terminate()
            worker.wait(1000)
        self._model_loader = None

    def set_current_image_path(self, path: str):
        self._current_path = str(path)

    def set_pose_model(self, model: str) -> None:
        """Switch the detector model and re-analyze the current image."""
        if not self.model_ready:
            raise RuntimeError("Pose model is still loading")
        model = str(model or DEFAULT_POSE_MODEL).strip()
        resolved = get_pose_model(model)
        normalized = resolved.key if hasattr(resolved, "key") else str(resolved)
        if normalized == self._pose_model:
            return

        self._cancel_worker()
        old_detector = self._det
        try:
            from core.pose_detector import PoseDetector
            new_detector = PoseDetector(model=normalized)
        except Exception:
            # Keep the current detector intact when the new checkpoint cannot load.
            raise

        self._det = new_detector
        self._pose_model = normalized
        QSettings(self._SETTINGS_ORG, self._SETTINGS_APP).setValue(
            self._POSE_MODEL_SETTING, normalized
        )
        self._result_cache.clear()
        self._bundle = AnalysisBundle()
        self._eng = None
        try:
            old_detector.close()
        except Exception:
            pass

        if self._current_path:
            self.load_image(self._current_path)
        else:
            self._st.showMessage(f"Pose model: {self.pose_model_name}")

    def set_calibration_profile(self, profile: str):
        self._calibration_profile = str(profile or "Generic")
        self._services = self._services.create(calibration_profile=self._calibration_profile)
        self._engine_factory = self._services.engine_factory
        self._eng = None

    def load_image(self, path: str):
        self._la(path)

    def _cancel_worker(self):
        worker = self._wk
        if worker is None or not worker.isRunning():
            return
        worker.requestInterruption()
        if not worker.wait(3000):
            worker.terminate()
            worker.wait(1000)

    def _la(self, path: str):
        if not self.model_ready:
            self._pending_image_path = str(path)
            self._st.showMessage("正在后台加载姿态模型，模型就绪后将自动开始分析。")
            return

        from core.image_io import load_image, frame_orientation
        img = load_image(path)
        if img is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Cannot read image")
            return
        self.set_current_image_path(path)
        self._img = img
        self._bundle = AnalysisBundle()
        self._w2.set_image(img)
        cache_key = self._image_cache_key(img)
        if cache_key in self._result_cache:
            self._bundle = self._result_cache[cache_key]
            self._apply_bundle(self._bundle)
            self._finish_progress(f"Loaded from cache | {frame_orientation(img)} | {path.split('/')[-1]}")
            return
        self._set_progress(0, f"Preparing analysis | {frame_orientation(img)} | {path.split('/')[-1]} | {self.pose_model_name}")
        self._cancel_worker()
        analysis_img = _resize_for_analysis(img, max_side=1600)
        if self._re_enabled and self._eng is None:
            self._eng = self._engine_factory(enable_simulation=False)
        self._wk = AnalysisWorker(self._det, self._eng, img, analysis_img, enable_re=self._re_enabled)
        self._wk.progress.connect(self._set_progress)
        self._wk.pose_ready.connect(self._on_pose_ready)
        self._wk.core_ready.connect(self._on_core_ready)
        self._wk.reverse_ready.connect(self._on_reverse_ready)
        self._wk.error.connect(self._err)
        self._wk.start()

    def closeEvent(self, event):
        self._cancel_worker()
        self._stop_model_loader()
        try:
            if self._det is not None:
                self._det.close()
        finally:
            super().closeEvent(event)
