"""MainWindow - Two-phase analysis architecture (Fast + Full RE)

Architecture:
  Phase 1 (Fast): Pose → Orientation → Action → Camera → Composition → Suggestions
  Phase 2 (Full RE): Image + Pose + Composition → ReverseEngineeringEngine (background)

UI receives staged signals:
  pose_ready   → skeleton visible immediately
  core_ready   → all 2D analysis panels populated
  reverse_ready→ RE results available (3D workspace + advanced overlay)
  error        → display in status bar
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
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
)

# ── Light Theme ────────────────────────────────────────────────────────
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
        QTabBar::tab {{ background: {THEME['surface']}; border: 1px solid {THEME['border']};
                         padding: 6px 16px; margin-right: 2px; color: {THEME['text']}; }}
        QTabBar::tab:selected {{ background: {THEME['panel']}; border-bottom: 2px solid {THEME['accent']}; }}
    """)


# ── Data Bundle ────────────────────────────────────────────────────────
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
    """Fast perceptual hash from downsampled image."""
    small = cv2.resize(image, (64, 64))
    return hashlib.md5(small.tobytes()).hexdigest()


def _resize_for_analysis(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Resize image for fast analysis; preserves aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ── Worker ─────────────────────────────────────────────────────────────
class AnalysisWorker(QThread):
    """Staged analysis worker.

    Signals:
        pose_ready(PoseResult)
        core_ready(AnalysisBundle)
        reverse_ready(AnalysisBundle)
        error(str)
    """
    pose_ready = Signal(object)
    core_ready = Signal(object)
    reverse_ready = Signal(object)
    error = Signal(str)

    def __init__(self, det, eng, image: np.ndarray, analysis_image: np.ndarray,
                 enable_re: bool = True):
        super().__init__()
        self._det = det
        self._eng = eng
        self._image = image
        self._analysis_image = analysis_image
        self._enable_re = enable_re
        self._bundle = AnalysisBundle()

    def run(self):
        try:
            # Phase 1a: Pose detection
            pose = self._det.detect(self._analysis_image)
            if pose is None:
                self.error.emit("No person detected in image")
                return
            self._bundle.pose = pose
            self.pose_ready.emit(pose)

            # Phase 1b: Fast analysis pipeline
            from core.orientation import analyze_orientation
            from core.action_classifier import classify_action
            from core.camera_analyzer import analyze_camera
            from core.composition import analyze_composition
            from core.suggestion import generate_suggestions

            orientation = analyze_orientation(pose)
            action = classify_action(pose)
            camera = analyze_camera(pose)
            composition = analyze_composition(self._analysis_image, pose)
            suggestions = generate_suggestions(action, orientation, camera, composition)

            self._bundle.orientation = orientation
            self._bundle.action = action
            self._bundle.camera = camera
            self._bundle.composition = composition
            self._bundle.suggestions = suggestions
            self.core_ready.emit(self._bundle)

            # Phase 2: Full Reverse Engineering (background)
            if self._enable_re and self._eng is not None:
                re_image = _resize_for_analysis(self._analysis_image, max_side=1600)
                re_result = self._eng.analyze(re_image, pose, pose.bbox)
                self._bundle.reverse_result = re_result
                self.reverse_ready.emit(self._bundle)

        except Exception as e:
            self.error.emit(str(e))


# ── Workspaces ─────────────────────────────────────────────────────────
class Workspace(QWidget):
    """Base workspace."""
    def update_results(self, bundle: AnalysisBundle): pass


from gui.canvas import ImageCanvas
from gui.panels import AnalysisPanel, SuggestionPanel


class Analysis2DWorkspace(Workspace):
    """2D analysis: left=AnalysisPanel, center=canvas, right=SuggestionPanel"""

    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QHBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)

        sp = QSplitter(Qt.Horizontal)
        self._ap = AnalysisPanel()
        sp.addWidget(self._ap)
        self._cv = ImageCanvas()
        sp.addWidget(self._cv)
        self._sp = SuggestionPanel()
        sp.addWidget(self._sp)
        sp.setSizes([300, 700, 320])
        sp.setStretchFactor(1, 1)
        lo.addWidget(sp)

    def set_image(self, img: np.ndarray):
        self._cv.set_image(img)

    def update_results(self, bundle: AnalysisBundle):
        """Update from core analysis bundle (Phase 1)."""
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

    def set_overlay_options(self, **kwargs):
        self._cv.set_overlay_options(**kwargs)


class Reverse3DWorkspace(Workspace):
    """Placeholder for future 3D reverse engineering visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QVBoxLayout(self)
        lo.setContentsMargins(16, 16, 16, 16)

        title = QLabel("3D Reverse Engineering Workspace")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lo.addWidget(title)

        self._info = QLabel(
            "Displays: Subject position, Camera frustum, Ground plane, "
            "Candidate solutions\nRequires: OpenGL 3D renderer (future phase)"
        )
        self._info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lo.addWidget(self._info)

    def update_results(self, bundle: AnalysisBundle):
        if bundle.reverse_result:
            self._info.setText(
                f"RE complete. Overall confidence: "
                f"{bundle.reverse_result.overall_confidence:.0%}\n"
                f"Candidates: {len(bundle.reverse_result._sim_candidates)}"
            )


class ResultsWorkspace(Workspace):
    """Full text report of reverse engineering results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QVBoxLayout(self)
        lo.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Reverse Engineering Report")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        lo.addWidget(title)

        self._rl = QLabel("No results yet. Waiting for analysis...")
        self._rl.setFont(QFont("Consolas", 10))
        self._rl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._rl.setWordWrap(True)
        self._rl.setStyleSheet(f"color: {THEME['text']};")

        sc = QScrollArea()
        sc.setWidget(self._rl)
        sc.setWidgetResizable(True)
        sc.setStyleSheet(f"QScrollArea {{ border: 1px solid {THEME['border']}; background: {THEME['panel']}; }}")
        lo.addWidget(sc)

    def update_results(self, bundle: AnalysisBundle):
        if bundle.reverse_result:
            self._rl.setText(bundle.reverse_result.report())
        else:
            self._rl.setText("Reverse engineering not yet complete...")


# ── Main Window ────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portrait Image Breakdown")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)

        # Core engines
        from core.pose_detector import PoseDetector
        self._det = PoseDetector()
        self._eng = None  # Lazy init for RE engine
        self._re_enabled = True

        # State
        self._img: Optional[np.ndarray] = None
        self._bundle = AnalysisBundle()
        self._wk: Optional[AnalysisWorker] = None
        self._result_cache: dict[str, AnalysisBundle] = {}

        # ── Toolbar ──
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        ao = QAction("Open Image", self)
        ao.setShortcut(QKeySequence.Open)
        ao.triggered.connect(self._open)
        tb.addAction(ao)
        tb.addSeparator()

        tb.addWidget(QLabel("  Dataset: "))
        self._cb = QComboBox()
        self._cb.setMinimumWidth(200)
        self._cb.currentTextChanged.connect(self._sel)
        tb.addWidget(self._cb)
        tb.addSeparator()

        # Overlay toggles
        self._chk_skeleton = QCheckBox("Skeleton")
        self._chk_skeleton.setChecked(True)
        self._chk_skeleton.stateChanged.connect(self._update_overlays)
        tb.addWidget(self._chk_skeleton)

        self._chk_thirds = QCheckBox("3x3 Grid")
        self._chk_thirds.setChecked(True)
        self._chk_thirds.stateChanged.connect(self._update_overlays)
        tb.addWidget(self._chk_thirds)

        self._chk_center = QCheckBox("Center")
        self._chk_center.setChecked(True)
        self._chk_center.stateChanged.connect(self._update_overlays)
        tb.addWidget(self._chk_center)

        self._chk_bbox = QCheckBox("BBox")
        self._chk_bbox.setChecked(True)
        self._chk_bbox.stateChanged.connect(self._update_overlays)
        tb.addWidget(self._chk_bbox)

        self._chk_vweight = QCheckBox("Visual Weight")
        self._chk_vweight.setChecked(False)
        self._chk_vweight.stateChanged.connect(self._update_overlays)
        tb.addWidget(self._chk_vweight)

        self._chk_headroom = QCheckBox("Headroom")
        self._chk_headroom.setChecked(False)
        self._chk_headroom.stateChanged.connect(self._update_overlays)
        tb.addWidget(self._chk_headroom)

        # ── Tabs ──
        self._tabs = QTabBar()
        self._tabs.addTab("2D Analysis")
        self._tabs.addTab("3D Reverse Engineering")
        self._tabs.addTab("Results")
        self._tabs.currentChanged.connect(self._sw)

        self._ws = QStackedWidget()
        self._w2 = Analysis2DWorkspace()
        self._w3 = Reverse3DWorkspace()
        self._wr = ResultsWorkspace()
        self._ws.addWidget(self._w2)
        self._ws.addWidget(self._w3)
        self._ws.addWidget(self._wr)

        cen = QWidget()
        ml = QVBoxLayout(cen)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        ml.addWidget(self._tabs)
        ml.addWidget(self._ws)
        self.setCentralWidget(cen)

        self._st = QStatusBar()
        self.setStatusBar(self._st)
        self._st.showMessage("Ready")

        # Dataset directory
        self._dd = Path(__file__).parent.parent / "dataset"
        if not self._dd.exists():
            self._dd = Path.cwd() / "dataset"
        self._ld()
        self.setAcceptDrops(True)

    # ── Dataset listing ──
    def _ld(self):
        self._cb.blockSignals(True)
        self._cb.clear()
        self._cb.addItem("Select image...")
        if self._dd.exists():
            for f in sorted(
                p.name for p in self._dd.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                self._cb.addItem(f)
        self._cb.blockSignals(False)

    def _sel(self, n):
        if n and n != "Select image...":
            p = self._dd / n
            if p.exists():
                self._la(str(p))

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Image", str(self._dd),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if p:
            self._la(p)

    # ── Load image and start analysis ──
    def _la(self, path: str):
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", "Cannot read image")
            return

        self._img = img
        self._bundle = AnalysisBundle()
        self._w2.set_image(img)

        # Check cache
        cache_key = _image_hash(img)
        if cache_key in self._result_cache:
            self._bundle = self._result_cache[cache_key]
            self._apply_bundle(self._bundle)
            self._st.showMessage("Loaded from cache | " + os.path.basename(path))
            return

        self._st.showMessage("Analyzing: " + os.path.basename(path) + " ...")

        # Terminate previous worker
        if self._wk and self._wk.isRunning():
            self._wk.terminate()
            self._wk.wait()

        # Prepare analysis image
        analysis_img = _resize_for_analysis(img, max_side=1600)

        # Lazy-init RE engine
        if self._re_enabled and self._eng is None:
            from reverse_engineering.engine import ReverseEngineeringEngine
            self._eng = ReverseEngineeringEngine(enable_simulation=False)

        # Launch worker
        self._wk = AnalysisWorker(
            self._det, self._eng, img, analysis_img,
            enable_re=self._re_enabled,
        )
        self._wk.pose_ready.connect(self._on_pose_ready)
        self._wk.core_ready.connect(self._on_core_ready)
        self._wk.reverse_ready.connect(self._on_reverse_ready)
        self._wk.error.connect(self._err)
        self._wk.start()

    # ── Staged signal handlers ──
    def _on_pose_ready(self, pose):
        """Phase 1a: Pose detected — show skeleton immediately."""
        self._bundle.pose = pose
        self._w2._cv.set_pose(pose)
        self._st.showMessage("Pose detected")

    def _on_core_ready(self, bundle: AnalysisBundle):
        """Phase 1b: Fast analysis complete — populate all 2D panels."""
        self._bundle = bundle
        self._w2.update_results(bundle)

        # Cache result
        if self._img is not None:
            cache_key = _image_hash(self._img)
            self._result_cache[cache_key] = bundle

        action_name = bundle.action.category.value if bundle.action else "?"
        self._st.showMessage(f"Core analysis complete | {action_name}")

    def _on_reverse_ready(self, bundle: AnalysisBundle):
        """Phase 2: Full RE complete — update results workspace."""
        self._bundle = bundle
        self._w3.update_results(bundle)
        self._wr.update_results(bundle)

        # Update suggestion panel with RE camera actions
        if bundle.reverse_result:
            self._w2._sp.update_camera_actions(bundle.reverse_result)

        self._st.showMessage("Analysis complete")

    def _err(self, m: str):
        self._st.showMessage("Error: " + m)

    # ── Overlay toggles ──
    def _update_overlays(self):
        self._w2.set_overlay_options(
            skeleton=self._chk_skeleton.isChecked(),
            thirds=self._chk_thirds.isChecked(),
            center=self._chk_center.isChecked(),
            bbox=self._chk_bbox.isChecked(),
            visual_weight=self._chk_vweight.isChecked(),
            headroom=self._chk_headroom.isChecked(),
        )

    def _apply_bundle(self, bundle: AnalysisBundle):
        """Apply a cached bundle to all workspaces."""
        self._w2.update_results(bundle)
        if bundle.reverse_result:
            self._w3.update_results(bundle)
            self._wr.update_results(bundle)
            self._w2._sp.update_camera_actions(bundle.reverse_result)

    # ── Tab switching ──
    def _sw(self, i):
        self._ws.setCurrentIndex(i)

    # ── Drag and drop ──
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        u = e.mimeData().urls()
        if u:
            p = u[0].toLocalFile()
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                self._la(p)

    # ── Cleanup ──
    def closeEvent(self, e):
        self._det.close()
        if self._wk and self._wk.isRunning():
            self._wk.terminate()
            self._wk.wait()
        e.accept()
