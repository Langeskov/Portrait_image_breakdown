"""Application-owned MainWindow facade with explicit service injection."""
from __future__ import annotations
from typing import Optional
from gui.main_window import MainWindow as _BaseMainWindow, _resize_for_analysis, AnalysisWorker, AnalysisBundle
from gui.cache import AnalysisCache


class ApplicationMainWindow(_BaseMainWindow):
    """MainWindow variant that receives engine/cache/profile services explicitly."""

    def __init__(self, services, parent=None):
        # MainWindow is a top-level QMainWindow and its constructor intentionally
        # owns its own application wiring; it does not accept a parent argument.
        super().__init__()
        if parent is not None:
            self.setParent(parent)
        self._services = services
        self._calibration_profile = "Generic"
        self._current_path: Optional[str] = None
        self._engine_factory = services.engine_factory
        self._image_cache_key = services.image_cache_key
        self._result_cache = AnalysisCache(capacity=8)

    @property
    def current_image(self):
        return self._img

    def set_current_image_path(self, path: str):
        self._current_path = str(path)

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
        self._set_progress(0, f"Preparing analysis | {frame_orientation(img)} | {path.split('/')[-1]}")
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
        try:
            self._det.close()
        finally:
            super().closeEvent(event)
