"""MainWindow - 2D-first analysis architecture with optional 3D tool."""
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
    QComboBox, QTabBar, QStackedWidget, QProgressBar, QPushButton,
    QGroupBox, QDialog, QDialogButtonBox,
)

from gui.canvas import ImageCanvas
from gui.panels import AnalysisPanel, SuggestionPanel

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


# ---------------------------------------------------------------------------
# Bundle dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Analysis2DBundle:
    """Pure 2D analysis results. No 3D reconstruction state."""
    pose: Optional[object] = None
    orientation: Optional[object] = None
    action: Optional[object] = None
    camera: Optional[object] = None
    composition: Optional[object] = None
    suggestions: Optional[object] = None


AnalysisBundle = Analysis2DBundle


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
        return Path.cwd() / "analysis_errors.log"


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class Analysis2DWorker(QThread):
    pose_ready = Signal(object)
    core_ready = Signal(object)
    progress = Signal(int, str)
    error = Signal(str)

    def __init__(self, det, image: np.ndarray, analysis_image: np.ndarray):
        super().__init__()
        self._det = det
        self._image = image
        self._analysis_image = analysis_image
        self._bundle = Analysis2DBundle()
        self._stage = "initialization"

    def _stage_begin(self, stage: str, progress: Optional[tuple[int, str]] = None):
        self._stage = stage
        if progress is not None:
            self.progress.emit(progress[0], progress[1])

    def _context(self) -> str:
        image = self._analysis_image
        pose = self._bundle.pose
        lines = [
            f"stage: {self._stage}",
            f"python: {platform.python_version()}",
            f"platform: {platform.platform()}",
            f"numpy: {np.__version__}",
            f"opencv: {cv2.__version__}",
            f"worker image shape: {getattr(image, 'shape', None)}",
            f"worker image dtype: {getattr(image, 'dtype', None)}",
            f"worker image type: {type(image).__name__}",
            f"full image shape: {getattr(self._image, 'shape', None)}",
        ]
        if pose is not None:
            landmarks = getattr(pose, "landmarks", None)
            lines.extend([
                f"pose type: {type(pose).__module__}.{type(pose).__name__}",
                f"pose bbox type: {type(getattr(pose, 'bbox', None)).__name__}",
                f"pose bbox: {_safe_repr(getattr(pose, 'bbox', None))}",
                f"pose landmarks type: {type(landmarks).__name__}",
                f"pose landmarks count: {len(landmarks) if hasattr(landmarks, '__len__') else 'n/a'}",
            ])
            if landmarks:
                lines.append(f"pose landmark[0]: {_safe_repr(landmarks[0])}")
                lines.append(f"pose landmark[0] type: {type(landmarks[0]).__module__}.{type(landmarks[0]).__name__}")
        return "\n".join(lines)

    def _emit_exception(self, exc: BaseException):
        tb = traceback.format_exc()
        report = "\n".join([
            f"Portrait Image Breakdown diagnostic @ {datetime.now().isoformat(timespec='seconds')}",
            self._context(), "",
            f"exception type: {type(exc).__module__}.{type(exc).__name__}",
            f"exception message: {exc}", "",
            "FULL TRACEBACK:", tb.rstrip(),
        ])
        log_path = _write_diagnostic_log(report)
        self.error.emit(
            f"Analysis failed at stage: {self._stage}\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"Full traceback was saved to:\n{log_path}\n\n"
            f"----- FULL TRACEBACK -----\n{tb.rstrip()}"
        )

    def run(self):
        try:
            self._stage_begin("1/7 PoseDetector.detect", (5, "[1/7] Detecting subject…"))
            pose = self._det.detect(self._analysis_image)
            if pose is None:
                self._stage = "1/7 PoseDetector.detect (no person detected)"
                self.error.emit("No person detected in image")
                return
            self._bundle.pose = pose
            self.pose_ready.emit(pose)

            self._stage_begin("2/7 2D analyzer imports", (20, "[2/7] Loading 2D analyzers…"))
            from core.orientation import analyze_orientation
            from core.action_classifier import classify_action
            from core.camera_analyzer import analyze_camera
            from core.composition import analyze_composition
            from core.suggestion import generate_suggestions

            self._stage_begin("3/7 analyze_orientation", (24, "[3/7] Analyzing orientation…"))
            orientation = analyze_orientation(pose)

            self._stage_begin("4/7 classify_action", (28, "[4/7] Classifying action…"))
            action = classify_action(pose)

            self._stage_begin("5/7 analyze_camera", (32, "[5/7] Analyzing camera…"))
            camera = analyze_camera(pose, self._analysis_image)

            self._stage_begin("6/7 analyze_composition", (36, "[6/7] Analyzing composition…"))
            composition = analyze_composition(self._analysis_image, pose)

            self._stage_begin("7/7 generate_suggestions", (40, "[7/7] Generating suggestions…"))
            suggestions = generate_suggestions(action, orientation, camera, composition)

            self._bundle.orientation = orientation
            self._bundle.action = action
            self._bundle.camera = camera
            self._bundle.composition = composition
            self._bundle.suggestions = suggestions

            self.core_ready.emit(self._bundle)
            self.progress.emit(100, "2D analysis complete")
        except Exception as e:
            self._emit_exception(e)


class ReconstructionWorker(QThread):
    reconstruction_done = Signal(object)
    progress = Signal(int, str)
    error = Signal(str)

    def __init__(self, eng, image: np.ndarray, pose, bbox):
        super().__init__()
        self._eng = eng
        self._image = image
        self._pose = pose
        self._bbox = bbox
        self._stage = "initialization"

    def _emit_exception(self, exc: BaseException):
        tb = traceback.format_exc()
        self.error.emit(
            f"3D reconstruction failed at stage: {self._stage}\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"----- FULL TRACEBACK -----\n{tb.rstrip()}"
        )

    def run(self):
        try:
            self._stage = "preparing image"
            self.progress.emit(55, "Preparing 3D reconstruction input…")
            re_image = _resize_for_analysis(self._image, max_side=1600)
            self._stage = "ReverseEngineeringEngine.analyze"
            self.progress.emit(65, "Calculating 3D camera geometry…")
            re_result = self._eng.analyze(re_image, self._pose, self._bbox)
            self._stage = "finalizing"
            self.progress.emit(95, "Finalizing projection validation…")
            self.reconstruction_done.emit(re_result)
            self.progress.emit(100, "3D reconstruction complete")
        except Exception as e:
            self._emit_exception(e)


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------

class Workspace(QWidget):
    """Base workspace contract for analysis tabs."""
    def update_results(self, bundle: Analysis2DBundle):
        pass


class Analysis2DWorkspace(Workspace):
    """2D analysis: left analysis panel + center canvas + right suggestion panel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(0)
        self._overlay_group = QGroupBox("2D Overlays")
        self._overlay_group.setStyleSheet(
            "QGroupBox { margin:4px 8px 3px 8px; padding-top:4px; border:1px solid #E2E8F0; border-radius:5px; } "
            "QGroupBox::title { left:8px; padding:0 4px; color:#475569; font-size:9pt; }"
        )
        row = QHBoxLayout(self._overlay_group)
        row.setContentsMargins(8, 8, 8, 6)
        row.setSpacing(7)
        self._overlay_controls = []
        specs = [
            ("Skeleton", True, "skeleton"),
            ("3x3 Grid", True, "thirds"),
            ("Center", True, "center"),
            ("BBox", True, "bbox"),
            ("Headroom", False, "headroom"),
            ("Reference Target", True, "reference_target"),
            ("Visual Weight", False, "visual_weight"),
        ]
        for label, checked, key in specs:
            cb = QCheckBox(label)
            cb.setChecked(checked)
            cb.setProperty("overlay_key", key)
            cb.setStyleSheet("QCheckBox { font-size:9pt; spacing:4px; padding:0px; }")
            cb.stateChanged.connect(self._apply_overlay_options)
            self._overlay_controls.append(cb)
            row.addWidget(cb, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        splitter = QSplitter(Qt.Horizontal)
        self._ap = AnalysisPanel()
        splitter.addWidget(self._ap)
        self._cv = ImageCanvas()
        splitter.addWidget(self._cv)
        self._sp = SuggestionPanel()
        splitter.addWidget(self._sp)
        splitter.setSizes([280, 800, 300])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        lo.addWidget(self._overlay_group, 0)
        lo.addWidget(splitter, 1)
        self._apply_overlay_options()

    def set_image(self, img: np.ndarray):
        self._cv.set_image(img)

    def _apply_overlay_options(self, _state=0):
        self._cv.set_overlay_options(**{
            cb.property("overlay_key"): cb.isChecked()
            for cb in self._overlay_controls
        })

    def set_overlay_options(self, **kwargs):
        for cb in self._overlay_controls:
            key = cb.property("overlay_key")
            if key in kwargs:
                cb.blockSignals(True)
                cb.setChecked(bool(kwargs[key]))
                cb.blockSignals(False)
        self._apply_overlay_options()

    def update_results(self, bundle: Analysis2DBundle):
        if bundle.pose:
            self._cv.set_pose(bundle.pose)
            vis = sum(1 for lm in bundle.pose.landmarks[:17] if lm.visibility > 0.4)
            self._ap.update_pose(bundle.pose.detection_confidence, vis)
        if bundle.orientation:
            self._ap.update_orientation(bundle.orientation)
        if bundle.action:
            self._ap.update_action(bundle.action)
        if bundle.camera:
            self._ap.update_camera(bundle.camera)
            self._cv.set_camera(bundle.camera)
        if bundle.composition:
            self._ap.update_composition(bundle.composition)
            self._cv.set_composition(bundle.composition)
        if bundle.suggestions:
            self._sp.update_suggestions(bundle.suggestions)


class FieldModeWorkspace(Workspace):
    """Field mode: center canvas (image + skeleton + correction guides) + right guidance panel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(0)
        splitter = QSplitter(Qt.Horizontal)
        self._cv = ImageCanvas()
        self._cv.set_overlay_options(
            skeleton=True, thirds=False, center=False, bbox=True,
            headroom=False, visual_weight=False, reference_target=False,
            guidance_corrections=True,
        )
        splitter.addWidget(self._cv)
        self._field_container = QWidget()
        self._field_lo = QVBoxLayout(self._field_container)
        self._field_lo.setContentsMargins(0, 0, 0, 0)
        self._field_lo.setSpacing(0)
        splitter.addWidget(self._field_container)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        lo.addWidget(splitter, 1)

    def set_image(self, img: np.ndarray):
        self._cv.set_image(img)

    def update_results(self, bundle: Analysis2DBundle):
        if bundle.pose:
            self._cv.set_pose(bundle.pose)
        if bundle.camera:
            self._cv.set_camera(bundle.camera)
        if bundle.composition:
            self._cv.set_composition(bundle.composition)
        # Generate and pass guidance corrections for the canvas overlay
        if bundle.action and bundle.orientation and bundle.camera and bundle.composition:
            from core.guidance import generate_guidance
            result = generate_guidance(
                bundle.action, bundle.orientation, bundle.camera, bundle.composition
            )
            self._cv.set_guidance_corrections(list(result.all_actions))
        else:
            self._cv.set_guidance_corrections([])


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, pose_model: str | None = None):
        super().__init__()
        self.setWindowTitle("Portrait Image Breakdown")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)

        self._det = self._initialize_pose_detector(pose_model)
        self._eng = None
        self._re_enabled = False
        self._img: Optional[np.ndarray] = None
        self._bundle = Analysis2DBundle()
        self._wk: Optional[Analysis2DWorker] = None
        self._recon_worker: Optional[ReconstructionWorker] = None
        self._result_cache: dict[str, Analysis2DBundle] = {}

        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open Image", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open)
        toolbar.addAction(open_action)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Dataset: "))
        self._cb = QComboBox()
        self._cb.setMinimumWidth(220)
        self._cb.addItem("Select folder…", "")
        self._cb.currentIndexChanged.connect(self._sel)
        toolbar.addWidget(self._cb)

        self._choose_folder_action = QAction("Browse…", self)
        self._choose_folder_action.triggered.connect(self._choose_dataset_folder)
        toolbar.addAction(self._choose_folder_action)
        toolbar.addSeparator()

        self._dataset_folder: Optional[Path] = None

        # --- Workspace registration: MainWindow owns all tabs + pages ---
        # Tab index ↔ stacked widget index must always match.
        self._tabs = QTabBar()
        self._ws = QStackedWidget()

        # Page 0: 2D Analysis
        self._w2 = Analysis2DWorkspace()
        self._ws.addWidget(self._w2)
        self._tabs.addTab("2D Analysis")

        # Page 1: Field Guidance (现场指导)
        self._w_field = FieldModeWorkspace()
        self._ws.addWidget(self._w_field)
        self._tabs.addTab("现场指导")

        # Page 2: Reference Comparison (图片对比) — widget added later by install_reference_mode
        self._w_ref_container = QWidget()
        self._w_ref_lo = QVBoxLayout(self._w_ref_container)
        self._w_ref_lo.setContentsMargins(0, 0, 0, 0)
        self._w_ref_lo.setSpacing(0)
        self._ws.addWidget(self._w_ref_container)
        self._tabs.addTab("图片对比")

        self._tabs.currentChanged.connect(self._sw)

        self._w3 = None  # 3D workspace, created lazily
        self._reconstruction_dialog = None

        center = QWidget()
        ml = QVBoxLayout(center)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        ml.addWidget(self._tabs)
        ml.addWidget(self._ws)
        self.setCentralWidget(center)

        # Status bar
        self._st = QStatusBar()
        self.setStatusBar(self._st)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)
        self._st.addPermanentWidget(self._progress, 1)
        self._st.showMessage("Ready")

        self._load_dataset_folder(None)
        self.setAcceptDrops(True)

    def _initialize_pose_detector(self, pose_model: str | None = None):
        from core.pose_detector import PoseDetector
        return PoseDetector(model=pose_model)

    def _choose_dataset_folder(self):
        start = str(self._dataset_folder or Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select image folder", start)
        if path:
            self._load_dataset_folder(Path(path))

    def _load_dataset_folder(self, path: Optional[Path]):
        self._dataset_folder = Path(path) if path else None
        self._cb.blockSignals(True)
        self._cb.clear()
        self._cb.addItem("Select folder…", "")
        if self._dataset_folder and self._dataset_folder.exists():
            files = [
                p for p in sorted(self._dataset_folder.iterdir(), key=lambda x: x.name.lower())
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ]
            for f in files:
                self._cb.addItem(f.name, str(f))
            self._st.showMessage(f"Image folder: {self._dataset_folder} · {len(files)} images")
        else:
            self._st.showMessage("No image folder selected")
        self._cb.blockSignals(False)

    def _ld(self):
        self._load_dataset_folder(self._dataset_folder)

    def _sel(self, index):
        path = self._cb.itemData(index)
        if path and Path(path).exists():
            self._la(str(path))

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Image",
            str(self._dataset_folder or Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if p:
            self._la(p)

    def _set_progress(self, value: int, message: str):
        self._progress.setValue(max(0, min(100, int(value))))
        self._progress.setVisible(True)
        self._st.showMessage(message)

    def _finish_progress(self, message: str = "Analysis complete"):
        self._progress.setValue(100)
        self._st.showMessage(message)
        self._progress.setVisible(False)

    def _la(self, path: str):
        from core.image_io import load_image, frame_orientation
        img = load_image(path)
        if img is None:
            QMessageBox.warning(self, "Error", "Cannot read image")
            return
        self._img = img
        self._bundle = Analysis2DBundle()
        self._w2.set_image(img)
        self._w_field.set_image(img)
        cache_key = _image_hash(img)
        if cache_key in self._result_cache:
            self._bundle = self._result_cache[cache_key]
            self._apply_bundle(self._bundle)
            self._finish_progress(
                f"Loaded from cache | {frame_orientation(img)} | {os.path.basename(path)}"
            )
            return
        self._set_progress(0, f"Preparing analysis | {frame_orientation(img)} | {os.path.basename(path)}")
        if self._wk and self._wk.isRunning():
            self._wk.terminate()
            self._wk.wait()
        analysis_img = _resize_for_analysis(img, max_side=1600)
        self._wk = Analysis2DWorker(self._det, img, analysis_img)
        self._wk.progress.connect(self._set_progress)
        self._wk.pose_ready.connect(self._on_pose_ready)
        self._wk.core_ready.connect(self._on_core_ready)
        self._wk.error.connect(self._err)
        self._wk.start()

    def _on_pose_ready(self, pose):
        self._bundle.pose = pose
        self._w2._cv.set_pose(pose)
        self._w_field._cv.set_pose(pose)
        if hasattr(self, "_reference_mode") and self._img is not None:
            self._reference_mode.set_current(pose, self._img)

    def _on_core_ready(self, bundle):
        self._bundle = bundle
        self._w2.update_results(bundle)
        self._w_field.update_results(bundle)
        if self._w3 is not None:
            self._w3.update_results(bundle)
        if self._img is not None:
            self._result_cache[_image_hash(self._img)] = bundle
        self._finish_progress()

    def _apply_bundle(self, bundle):
        self._w2.update_results(bundle)
        self._w_field.update_results(bundle)
        if self._w3 is not None:
            self._w3.update_results(bundle)

    def _err(self, message: str):
        self._progress.setVisible(False)
        QMessageBox.critical(self, "Analysis error", message)

    def _sw(self, index):
        """Switch workspace — simple page switch, no reparenting."""
        self._ws.setCurrentIndex(index)

    def _make_reconstruction_engine(self):
        factory = getattr(self, "_engine_factory", None)
        if factory is not None:
            return factory(enable_simulation=False)
        from reverse_engineering.engine import ReverseEngineeringEngine
        return ReverseEngineeringEngine(enable_simulation=False)

    def open_3d_reconstruction(self):
        if self._img is None or self._bundle.pose is None:
            QMessageBox.information(
                self, "3D 重建",
                "请先打开一张包含人物的图片并完成2D分析。",
            )
            return
        if self._w3 is None:
            from gui.reverse_3d_workspace import Reverse3DWorkspace
            self._w3 = Reverse3DWorkspace()
        if self._reconstruction_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("3D 重建（可选功能）")
            dialog.resize(1280, 820)
            layout = QVBoxLayout(dialog)
            hint = QLabel(
                "3D重建用于探索照片可能的拍摄空间与相机假设；"
                "它不会改变2D分析、现场指令或图片对比结果。"
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#64748B; padding: 4px;")
            layout.addWidget(hint)
            layout.addWidget(self._w3, 1)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.close)
            buttons.accepted.connect(dialog.close)
            layout.addWidget(buttons)
            self._reconstruction_dialog = dialog
        self._reconstruction_dialog.show()
        self._reconstruction_dialog.raise_()
        self._reconstruction_dialog.activateWindow()
        if not hasattr(self, "_re_result") or self._re_result is None:
            self._start_3d_reconstruction()

    def _start_3d_reconstruction(self):
        if self._recon_worker is not None and self._recon_worker.isRunning():
            self._st.showMessage("3D重建正在进行中…")
            return
        image = self._img
        pose = self._bundle.pose
        if image is None or pose is None:
            return
        bbox = getattr(pose, "bbox", None)
        self._set_progress(30, "正在创建3D引擎…")
        if self._eng is None:
            self._eng = self._make_reconstruction_engine()
        self._set_progress(50, "正在准备可选3D重建…")
        self._recon_worker = ReconstructionWorker(self._eng, image, pose, bbox)
        self._recon_worker.progress.connect(self._set_progress)
        self._recon_worker.reconstruction_done.connect(self._on_reconstruction_done)
        self._recon_worker.error.connect(self._err)
        self._recon_worker.start()

    def _on_reconstruction_done(self, re_result):
        self._re_result = re_result
        if self._w3 is not None:
            self._w3.update_results(self._bundle)
        self._finish_progress("3D重建完成")

    def set_overlay_options(self, **kwargs):
        self._w2.set_overlay_options(**kwargs)

    @property
    def current_image(self):
        return self._img

    @property
    def current_path(self):
        return getattr(self, "_current_path", None)
