"""MainWindow - Two-phase analysis architecture (Fast + Full RE)"""
from __future__ import annotations

import hashlib
import os
import platform
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QKeySequence, QColor, QPalette, QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QToolBar, QFileDialog, QLabel, QStatusBar, QMessageBox, QCheckBox,
    QComboBox, QApplication, QTabBar, QStackedWidget, QScrollArea,
    QProgressBar, QPushButton, QGroupBox,
)

THEME = dict(
    window="#F5F6F8", panel="#FFFFFF", surface="#FAFBFC",
    border="#D9DDE3", text="#1F2937", text2="#6B7280",
    accent="#2563EB", success="#16A34A", warning="#D97706", danger="#DC2626",
    toolbar="#FFFFFF",
)


def apply_light_theme(app):
    p = app.palette()
    for role, color in [
        (QPalette.Window, THEME["window"]),
        (QPalette.WindowText, THEME["text"]),
        (QPalette.Base, THEME["panel"]),
        (QPalette.Text, THEME["text"]),
        (QPalette.Button, THEME["toolbar"]),
        (QPalette.ButtonText, THEME["text"]),
        (QPalette.Highlight, THEME["accent"]),
        (QPalette.HighlightedText, "#FFFFFF"),
    ]:
        p.setColor(role, QColor(color))
    app.setPalette(p)
    app.setStyleSheet(f"""
        QToolBar {{ background: {THEME['toolbar']}; border-bottom: 1px solid {THEME['border']}; padding: 2px; }}
        QToolBar QToolButton {{ padding: 4px 8px; }}
        QStatusBar {{ background: {THEME['surface']}; border-top: 1px solid {THEME['border']}; color: {THEME['text2']}; }}
        QProgressBar {{ min-width: 180px; max-width: 260px; min-height: 14px; border: 1px solid {THEME['border']}; border-radius: 3px; background: {THEME['panel']}; text-align: center; color: {THEME['text']}; }}
        QProgressBar::chunk {{ background: {THEME['accent']}; border-radius: 2px; }}
        QTabBar::tab {{ background: {THEME['surface']}; border: 1px solid {THEME['border']}; padding: 6px 16px; margin-right: 2px; color: {THEME['text']}; }}
        QTabBar::tab:selected {{ background: {THEME['panel']}; border-bottom: 2px solid {THEME['accent']}; }}
    """)


@dataclass
class AnalysisBundle:
    """Accumulates all analysis results for a single image."""
    pose: Optional[object] = None
    orientation: Optional[object] = None
    action: Optional[object] = None
    camera: Optional[object] = None
    composition: Optional[object] = None
    suggestions: Optional[object] = None
    reverse_result: Optional[object] = None


def _image_hash(image: np.ndarray) -> str:
    small = cv2.resize(image, (64, 64))
    return hashlib.md5(small.tobytes()).hexdigest()


def _resize_for_analysis(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _safe_repr(value, max_len: int = 1200) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    return text[:max_len] + ("…" if len(text) > max_len else "")


def _diagnostic_log_path() -> Path:
    candidates = []
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "PortraitImageBreakdown")
    else:
        state_home = os.environ.get("XDG_STATE_HOME")
        if state_home:
            candidates.append(Path(state_home) / "PortraitImageBreakdown")
        candidates.append(Path.home() / ".local" / "state" / "PortraitImageBreakdown")
    candidates.append(Path.cwd() / "PortraitImageBreakdown_logs")
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return directory / "analysis_errors.log"
        except Exception:
            continue
    return Path.cwd() / "analysis_errors.log"


def _write_diagnostic_log(report: str) -> Path:
    path = _diagnostic_log_path()
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + "=" * 88 + "\n" + report + "\n")
        return path
    except Exception:
        return Path("analysis_errors.log")


class AnalysisWorker(QThread):
    pose_ready = Signal(object)
    core_ready = Signal(object)
    reverse_ready = Signal(object)
    progress = Signal(int, str)
    error = Signal(str)

    def __init__(self, det, eng, image: np.ndarray, analysis_image: np.ndarray, enable_re: bool = True):
        super().__init__()
        self._det = det; self._eng = eng; self._image = image; self._analysis_image = analysis_image
        self._enable_re = enable_re; self._bundle = AnalysisBundle(); self._stage = "initialization"

    def _stage_begin(self, stage: str, progress: Optional[tuple[int, str]] = None):
        self._stage = stage
        if progress is not None: self.progress.emit(progress[0], progress[1])

    def _context(self) -> str:
        image = self._analysis_image; pose = self._bundle.pose
        lines = [f"stage: {self._stage}", f"python: {platform.python_version()}", f"platform: {platform.platform()}", f"numpy: {np.__version__}", f"opencv: {cv2.__version__}", f"worker image shape: {getattr(image, 'shape', None)}", f"worker image dtype: {getattr(image, 'dtype', None)}", f"worker image type: {type(image).__name__}", f"full image shape: {getattr(self._image, 'shape', None)}", f"engine type: {type(self._eng).__module__}.{type(self._eng).__name__ if self._eng is not None else None}", f"reverse enabled: {self._enable_re}"]
        if pose is not None:
            landmarks = getattr(pose, "landmarks", None)
            lines.extend([f"pose type: {type(pose).__module__}.{type(pose).__name__}", f"pose bbox type: {type(getattr(pose, 'bbox', None)).__name__}", f"pose bbox: {_safe_repr(getattr(pose, 'bbox', None))}", f"pose landmarks type: {type(landmarks).__name__}", f"pose landmarks count: {len(landmarks) if hasattr(landmarks, '__len__') else 'n/a'}"])
            if landmarks:
                lines.append(f"pose landmark[0]: {_safe_repr(landmarks[0])}")
                lines.append(f"pose landmark[0] type: {type(landmarks[0]).__module__}.{type(landmarks[0]).__name__}")
        return "\n".join(lines)

    def _emit_exception(self, exc: BaseException):
        tb = traceback.format_exc()
        report = "\n".join([f"Portrait Image Breakdown diagnostic @ {datetime.now().isoformat(timespec='seconds')}", self._context(), "", f"exception type: {type(exc).__module__}.{type(exc).__name__}", f"exception message: {exc}", "", "FULL TRACEBACK:", tb.rstrip()])
        log_path = _write_diagnostic_log(report)
        self.error.emit(f"Analysis failed at stage: {self._stage}\n\n{type(exc).__name__}: {exc}\n\nFull traceback was saved to:\n{log_path}\n\n----- FULL TRACEBACK -----\n{tb.rstrip()}")

    def run(self):
        try:
            self._stage_begin("1/8 PoseDetector.detect", (5, "[1/8] Detecting subject…"))
            pose = self._det.detect(self._analysis_image)
            if pose is None:
                self._stage = "1/8 PoseDetector.detect (no person detected)"; self.error.emit("No person detected in image"); return
            self._bundle.pose = pose; self.pose_ready.emit(pose)
            self._stage_begin("2/8 2D analyzer imports", (20, "[2/8] Loading 2D analyzers…"))
            from core.orientation import analyze_orientation
            from core.action_classifier import classify_action
            from core.camera_analyzer import analyze_camera
            from core.composition import analyze_composition
            from core.suggestion import generate_suggestions
            self._stage_begin("3/8 analyze_orientation", (24, "[3/8] Analyzing orientation…")); orientation = analyze_orientation(pose)
            self._stage_begin("4/8 classify_action", (28, "[4/8] Classifying action…")); action = classify_action(pose)
            self._stage_begin("5/8 analyze_camera", (32, "[5/8] Analyzing camera…")); camera = analyze_camera(pose, self._analysis_image)
            self._stage_begin("6/8 analyze_composition", (36, "[6/8] Analyzing composition…")); composition = analyze_composition(self._analysis_image, pose)
            self._stage_begin("7/8 generate_suggestions", (40, "[7/8] Generating suggestions…")); suggestions = generate_suggestions(action, orientation, camera, composition)
            self._bundle.orientation = orientation; self._bundle.action = action; self._bundle.camera = camera; self._bundle.composition = composition; self._bundle.suggestions = suggestions
            self.core_ready.emit(self._bundle); self.progress.emit(45, "2D analysis complete · reconstructing camera…")
            if self._enable_re and self._eng is not None:
                self._stage_begin("8/8 reverse_engineering image preparation", (55, "[8/8] Preparing 3D reconstruction input…"))
                re_image = _resize_for_analysis(self._analysis_image, max_side=1600)
                self._stage_begin("8/8 ReverseEngineeringEngine.analyze", (60, "[8/8] Calculating 3D camera geometry…")); re_result = self._eng.analyze(re_image, pose, pose.bbox)
                self._bundle.reverse_result = re_result; self._stage_begin("8/8 final projection validation", (95, "[8/8] Finalizing projection validation…")); self.reverse_ready.emit(self._bundle); self.progress.emit(100, "Analysis complete")
            else:
                self._stage = "reverse_engineering skipped"; self.progress.emit(100, "Analysis complete")
        except Exception as e:
            self._emit_exception(e)


class Workspace(QWidget):
    def update_results(self, bundle: AnalysisBundle): pass


from gui.canvas import ImageCanvas
from gui.panels import AnalysisPanel, SuggestionPanel


class Analysis2DWorkspace(Workspace):
    """2D analysis workspace; all visual overlay controls live here."""
    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QVBoxLayout(self); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)

        self._overlay_group = QGroupBox("2D Overlays")
        self._overlay_group.setStyleSheet(
            "QGroupBox { margin: 4px 8px 3px 8px; padding-top: 4px; "
            "border: 1px solid #E2E8F0; border-radius: 5px; } "
            "QGroupBox::title { left: 8px; padding: 0 4px; color:#475569; font-size:9pt; }"
        )
        row = QHBoxLayout(self._overlay_group)
        row.setContentsMargins(8, 8, 8, 6)
        row.setSpacing(7)
        self._overlay_controls = []
        specs = [
            ("Skeleton", True, "skeleton"), ("3x3 Grid", True, "thirds"),
            ("Center", True, "center"), ("BBox", True, "bbox"),
            ("Headroom", False, "headroom"), ("Reference Target", True, "reference_target"),
            ("Visual Weight", False, "visual_weight"), ("Reverse Evidence", False, "reverse"),
        ]
        for label, checked, key in specs:
            cb = QCheckBox(label)
            cb.setChecked(checked)
            cb.setProperty("overlay_key", key)
            cb.setStyleSheet("QCheckBox { font-size: 9pt; spacing: 4px; padding: 0px; }")
            cb.stateChanged.connect(self._apply_overlay_options)
            self._overlay_controls.append(cb)
            row.addWidget(cb, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)

        sp = QSplitter(Qt.Horizontal)
        self._ap = AnalysisPanel(); sp.addWidget(self._ap)
        self._cv = ImageCanvas(); sp.addWidget(self._cv)
        self._sp = SuggestionPanel(); sp.addWidget(self._sp)
        sp.setSizes([300, 700, 320]); sp.setStretchFactor(1, 1)
        lo.addWidget(self._overlay_group, 0)
        lo.addWidget(sp, 1)
        self._apply_overlay_options()

    def set_image(self, img: np.ndarray): self._cv.set_image(img)

    def _apply_overlay_options(self, _state=0):
        values = {cb.property("overlay_key"): cb.isChecked() for cb in self._overlay_controls}
        self._cv.set_overlay_options(**values)

    def set_overlay_options(self, **kwargs):
        for cb in self._overlay_controls:
            key = cb.property("overlay_key")
            if key in kwargs:
                cb.blockSignals(True); cb.setChecked(bool(kwargs[key])); cb.blockSignals(False)
        self._apply_overlay_options()

    def update_results(self, bundle: AnalysisBundle):
        if bundle.pose:
            self._cv.set_pose(bundle.pose); vis = sum(1 for lm in bundle.pose.landmarks[:17] if lm.visibility > 0.4); self._ap.update_pose(bundle.pose.detection_confidence, vis)
        if bundle.orientation: self._ap.update_orientation(bundle.orientation)
        if bundle.action: self._ap.update_action(bundle.action)
        if bundle.camera: self._ap.update_camera(bundle.camera); self._cv.set_camera(bundle.camera)
        if bundle.composition: self._ap.update_composition(bundle.composition); self._cv.set_composition(bundle.composition)
        if bundle.suggestions: self._sp.update_suggestions(bundle.suggestions)
        if bundle.reverse_result: self._cv.set_reverse_result(bundle.reverse_result)


class Reverse3DWorkspace(Workspace):
    def __init__(self, parent=None):
        super().__init__(parent); lo = QVBoxLayout(self); lo.setContentsMargins(16, 16, 16, 16); title = QLabel("3D Reverse Engineering Workspace"); title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold)); title.setAlignment(Qt.AlignmentFlag.AlignCenter); lo.addWidget(title); self._info = QLabel("Displays: Subject position, Camera frustum, Ground plane, Candidate solutions\nRequires: OpenGL 3D renderer (future phase)"); self._info.setAlignment(Qt.AlignCenter); lo.addWidget(self._info)
    def update_results(self, bundle: AnalysisBundle):
        if bundle.reverse_result: self._info.setText(f"RE complete. Overall confidence: {bundle.reverse_result.overall_confidence:.0%}\nCandidates: {len(bundle.reverse_result._sim_candidates)}")


class ResultsWorkspace(Workspace):
    def __init__(self, parent=None):
        super().__init__(parent); lo = QVBoxLayout(self); lo.setContentsMargins(16, 16, 16, 16); title = QLabel("Reverse Engineering Report"); title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold)); lo.addWidget(title); self._rl = QLabel("No results yet. Waiting for analysis..."); self._rl.setFont(QFont("Consolas", 10)); self._rl.setAlignment(Qt.AlignTop); self._rl.setWordWrap(True); self._rl.setStyleSheet(f"color: {THEME['text']};"); sc = QScrollArea(); sc.setWidget(self._rl); sc.setWidgetResizable(True); sc.setStyleSheet(f"QScrollArea {{ border: 1px solid {THEME['border']}; background: {THEME['panel']}; }}"); lo.addWidget(sc)
    def update_results(self, bundle: AnalysisBundle): self._rl.setText(bundle.reverse_result.report() if bundle.reverse_result else "Reverse engineering not yet complete...")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Portrait Image Breakdown"); self.setMinimumSize(1200, 700); self.resize(1400, 800)
        from core.pose_detector import PoseDetector
        self._det = PoseDetector(); self._eng = None; self._re_enabled = True
        self._img: Optional[np.ndarray] = None; self._bundle = AnalysisBundle(); self._wk: Optional[AnalysisWorker] = None; self._result_cache: dict[str, AnalysisBundle] = {}
        tb = QToolBar("Main"); tb.setMovable(False); self.addToolBar(tb)
        ao = QAction("Open Image", self); ao.setShortcut(QKeySequence.Open); ao.triggered.connect(self._open); tb.addAction(ao); tb.addSeparator()
        tb.addWidget(QLabel("  Dataset: ")); self._cb = QComboBox(); self._cb.setMinimumWidth(220); self._cb.addItem("Select folder…", ""); self._cb.currentIndexChanged.connect(self._sel); tb.addWidget(self._cb)
        choose = QAction("Choose Folder", self); choose.triggered.connect(self._choose_dataset_folder); tb.addAction(choose); tb.addSeparator()
        self._dataset_folder: Optional[Path] = None
        self._chk_skeleton = None; self._chk_thirds = None; self._chk_center = None; self._chk_bbox = None; self._chk_vweight = None; self._chk_headroom = None; self._chk_reference_target = None
        self._tabs = QTabBar(); [self._tabs.addTab(t) for t in ("2D Analysis", "3D Reverse Engineering", "Results")]; self._tabs.currentChanged.connect(self._sw)
        self._ws = QStackedWidget(); self._w2 = Analysis2DWorkspace(); self._w3 = Reverse3DWorkspace(); self._wr = ResultsWorkspace(); [self._ws.addWidget(w) for w in (self._w2, self._w3, self._wr)]
        cen = QWidget(); ml = QVBoxLayout(cen); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0); ml.addWidget(self._tabs); ml.addWidget(self._ws); self.setCentralWidget(cen)
        self._st = QStatusBar(); self.setStatusBar(self._st); self._progress = QProgressBar(); self._progress.setRange(0, 100); self._progress.setValue(0); self._progress.setTextVisible(True); self._progress.setVisible(False); self._st.addPermanentWidget(self._progress, 1); self._st.showMessage("Ready")
        self._load_dataset_folder(None); self.setAcceptDrops(True)

    def _choose_dataset_folder(self):
        start = str(self._dataset_folder or Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select image folder", start)
        if not path: return
        self._load_dataset_folder(Path(path))

    def _load_dataset_folder(self, path: Optional[Path]):
        self._dataset_folder = Path(path) if path else None
        self._cb.blockSignals(True); self._cb.clear(); self._cb.addItem("Select folder…", "")
        if self._dataset_folder and self._dataset_folder.exists():
            files = [p for p in sorted(self._dataset_folder.iterdir(), key=lambda x: x.name.lower()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
            for f in files: self._cb.addItem(f.name, str(f))
            self._st.showMessage(f"Image folder: {self._dataset_folder} · {len(files)} images")
        else:
            self._st.showMessage("No image folder selected")
        self._cb.blockSignals(False)

    def _ld(self):
        self._load_dataset_folder(self._dataset_folder)

    def _sel(self, index):
        path = self._cb.itemData(index)
        if path and Path(path).exists(): self._la(str(path))

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Image", str(self._dataset_folder or Path.home()), "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if p: self._la(p)

    def _set_progress(self, value: int, message: str): self._progress.setValue(max(0, min(100, int(value)))); self._progress.setVisible(True); self._st.showMessage(message)
    def _finish_progress(self, message: str = "Analysis complete"): self._progress.setValue(100); self._st.showMessage(message); self._progress.setVisible(False)

    def _la(self, path: str):
        from core.image_io import load_image, frame_orientation
        img = load_image(path)
        if img is None: QMessageBox.warning(self, "Error", "Cannot read image"); return
        self._img = img; self._bundle = AnalysisBundle(); self._w2.set_image(img)
        cache_key = _image_hash(img)
        if cache_key in self._result_cache:
            self._bundle = self._result_cache[cache_key]; self._apply_bundle(self._bundle); self._finish_progress(f"Loaded from cache | {frame_orientation(img)} | {os.path.basename(path)}"); return
        self._set_progress(0, f"Preparing analysis | {frame_orientation(img)} | {os.path.basename(path)}")
        if self._wk and self._wk.isRunning(): self._wk.terminate(); self._wk.wait()
        analysis_img = _resize_for_analysis(img, max_side=1600)
        if self._re_enabled and self._eng is None:
            from reverse_engineering.engine import ReverseEngineeringEngine
            self._eng = ReverseEngineeringEngine(enable_simulation=False)
        self._wk = AnalysisWorker(self._det, self._eng, img, analysis_img, enable_re=self._re_enabled)
        self._wk.progress.connect(self._set_progress); self._wk.pose_ready.connect(self._on_pose_ready); self._wk.core_ready.connect(self._on_core_ready); self._wk.reverse_ready.connect(self._on_reverse_ready); self._wk.error.connect(self._err); self._wk.start()

    def _on_pose_ready(self, pose):
        self._bundle.pose = pose; self._w2._cv.set_pose(pose)
        if hasattr(self, "_reference_mode") and self._img is not None: self._reference_mode.set_current(pose, self._img)
        self._st.showMessage("Pose detected")

    def _on_core_ready(self, bundle: AnalysisBundle):
        self._bundle = bundle; self._w2.update_results(bundle); self._result_cache[_image_hash(self._img)] = bundle if self._img is not None else bundle
        action_name = bundle.action.category.value if bundle.action else "?"; self._st.showMessage(f"Core analysis complete | {action_name}")

    def _on_reverse_ready(self, bundle: AnalysisBundle):
        self._bundle = bundle; self._w3.update_results(bundle); self._wr.update_results(bundle); self._w2.update_results(bundle)
        if hasattr(self, '_reference_mode'): self._reference_mode.set_current(bundle.pose, self._img)
        if hasattr(self, '_field_mode') and bundle.action and bundle.orientation and bundle.camera and bundle.composition:
            confidence = float(bundle.reverse_result.overall_confidence) if bundle.reverse_result else None; self._field_mode.set_analysis(bundle.action, bundle.orientation, bundle.camera, bundle.composition, bundle.pose, confidence)
        self._finish_progress("Analysis complete")
        if self._img is not None: self._result_cache[_image_hash(self._img)] = bundle

    def _apply_bundle(self, bundle):
        self._w2.update_results(bundle)
        if bundle.reverse_result: self._w3.update_results(bundle); self._wr.update_results(bundle)
        if hasattr(self, '_reference_mode') and bundle.pose and self._img is not None: self._reference_mode.set_current(bundle.pose, self._img)
        if hasattr(self, '_field_mode') and bundle.action and bundle.orientation and bundle.camera and bundle.composition:
            confidence = float(bundle.reverse_result.overall_confidence) if bundle.reverse_result else None; self._field_mode.set_analysis(bundle.action, bundle.orientation, bundle.camera, bundle.composition, bundle.pose, confidence)

    def _update_overlays(self):
        self._w2._apply_overlay_options()

    def _sw(self, i): self._ws.setCurrentIndex(i)

    def _err(self, msg):
        self._finish_progress("Analysis error"); dialog = QMessageBox(self); dialog.setIcon(QMessageBox.Icon.Critical); dialog.setWindowTitle("Analysis Error — diagnostic mode"); dialog.setText("Analysis failed. The exact stage and complete traceback are shown below."); dialog.setInformativeText(msg.split("\n\n----- FULL TRACEBACK -----", 1)[0]);
        if "----- FULL TRACEBACK -----" in msg: dialog.setDetailedText(msg.split("\n\n----- FULL TRACEBACK -----", 1)[1].lstrip())
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok); dialog.exec()
