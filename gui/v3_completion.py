"""User-facing integration for the final v3 reconstruction workflow."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from reverse_engineering.plane_constraints import PlaneConstraint, PlaneRelation, apply_position_constraint, evaluate_constraint
from reverse_engineering.reference_pose_generation import generate_composition_aware_pose_target
from reverse_engineering.reconstruction_session import load_session, save_session
from reverse_engineering.temporal import TemporalFrameState, TemporalSmoother


class PlaneConstraintPanel(QWidget):
    """UI for positional plane constraints; angular relations remain an API-only primitive until target orientation exists."""

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self._constraints: list[PlaneConstraint] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(5)
        title = QLabel("Plane-aware constraints")
        root.addWidget(title)
        hint = QLabel("用于把人物/物体位置约束到平面或距离平面指定偏移；平行/垂直方向约束等待对象方向模型后启用。")
        hint.setWordWrap(True)
        root.addWidget(hint)
        row = QHBoxLayout()
        self.plane = QComboBox()
        self.relation = QComboBox()
        for relation in (PlaneRelation.ON_PLANE, PlaneRelation.OFFSET):
            self.relation.addItem(relation.label, relation)
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-10.0, 10.0)
        self.offset.setSingleStep(0.05)
        self.offset.setSuffix(" m")
        add = QPushButton("Add")
        add.clicked.connect(self._add)
        for label, widget in (("Plane", self.plane), ("Relation", self.relation), ("Offset", self.offset)):
            row.addWidget(QLabel(label)); row.addWidget(widget, 1)
        row.addWidget(add)
        root.addLayout(row)
        self._list = QListWidget(); self._list.setMaximumHeight(100); root.addWidget(self._list)
        action = QHBoxLayout()
        apply_button = QPushButton("Apply to primary subject"); apply_button.clicked.connect(self._apply)
        evaluate_button = QPushButton("Evaluate"); evaluate_button.clicked.connect(self._evaluate)
        remove_button = QPushButton("Remove"); remove_button.clicked.connect(self._remove)
        action.addWidget(apply_button); action.addWidget(evaluate_button); action.addWidget(remove_button)
        root.addLayout(action)
        self.refresh_planes()

    def refresh_planes(self):
        self.plane.clear()
        for anchor in self.workspace.scene.anchors:
            if getattr(anchor.kind, "value", anchor.kind) == "plane":
                self.plane.addItem(anchor.name, anchor.anchor_id)

    def _add(self):
        plane_id = self.plane.currentData()
        if not plane_id:
            return
        relation = self.relation.currentData()
        constraint = PlaneConstraint(
            constraint_id=f"plane_constraint_{len(self._constraints)+1}",
            plane_anchor_id=str(plane_id),
            relation=relation,
            offset_m=float(self.offset.value()),
        )
        self._constraints.append(constraint)
        self._list.addItem(QListWidgetItem(f"{constraint.relation.label}: {constraint.plane_anchor_id} · {constraint.offset_m:+.2f} m"))
        self.workspace.scene.plane_constraints = [c.to_dict() for c in self._constraints]

    def _remove(self):
        row = self._list.currentRow()
        if row < 0:
            return
        self._constraints.pop(row); self._list.takeItem(row)
        self.workspace.scene.plane_constraints = [c.to_dict() for c in self._constraints]

    def _apply(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._constraints):
            return
        subject = self.workspace.scene.subject
        constraint = self._constraints[row]
        try:
            x, y, z = apply_position_constraint((subject.center_x, subject.center_y, subject.center_z), constraint, self.workspace.scene.anchors)
            subject.center_x, subject.center_y, subject.center_z = x, y, z
            self.workspace._view.update(); self.workspace._refresh_projection()
        except ValueError as exc:
            QMessageBox.warning(self, "Plane constraint", str(exc))

    def _evaluate(self):
        row = self._list.currentRow()
        if row < 0 or row >= len(self._constraints):
            return
        subject = self.workspace.scene.subject
        try:
            result = evaluate_constraint((subject.center_x, subject.center_y, subject.center_z), self._constraints[row], self.workspace.scene.anchors)
            state = "SATISFIED" if result.satisfied else "NOT SATISFIED"
            QMessageBox.information(self, "Plane constraint", f"{state}\nResidual: {result.residual_m:.3f} m\n{result.message}")
        except ValueError as exc:
            QMessageBox.warning(self, "Plane constraint", str(exc))

    def to_dicts(self):
        return [c.to_dict() for c in self._constraints]

    def from_dicts(self, data):
        self._constraints.clear(); self._list.clear()
        for item in data or []:
            try:
                c = PlaneConstraint.from_dict(item)
            except (ValueError, TypeError):
                continue
            if c.relation not in (PlaneRelation.ON_PLANE, PlaneRelation.OFFSET):
                continue
            self._constraints.append(c)
            self._list.addItem(QListWidgetItem(f"{c.relation.label}: {c.plane_anchor_id} · {c.offset_m:+.2f} m"))
        self.workspace.scene.plane_constraints = [c.to_dict() for c in self._constraints]


class TemporalWorker(QThread):
    progressed = Signal(int, float, int)
    completed = Signal()
    failed = Signal(str)

    def __init__(self, video_path, alpha=0.35, sample_fps=4.0):
        super().__init__()
        self.video_path = str(video_path)
        self.alpha = float(alpha)
        self.sample_fps = max(0.5, float(sample_fps))
        self.stop_requested = False

    def run(self):
        detector = None
        capture = None
        try:
            from core.pose_detector import PoseDetector
            detector = PoseDetector(); capture = cv2.VideoCapture(self.video_path)
            if not capture.isOpened():
                raise RuntimeError("Unable to open video")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            stride = max(1, int(round(fps / self.sample_fps)))
            smoother = TemporalSmoother(alpha=self.alpha)
            index = 0
            while not self.stop_requested:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    pose = detector.detect(frame)
                    keypoints = None; confidence = 0.0
                    if pose is not None:
                        keypoints = np.asarray([(lm.x, lm.y) for lm in pose.landmarks[:17]], dtype=float)
                        confidence = float(getattr(pose, "detection_confidence", 0.0))
                    state = smoother.update(TemporalFrameState(index / max(fps, 1e-6), keypoints=keypoints, confidence=confidence))
                    visible = int(np.sum(np.isfinite(state.keypoints).all(axis=1))) if state.keypoints is not None else 0
                    self.progressed.emit(index, state.confidence, visible)
                index += 1
            self.completed.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.release()
            try:
                if detector is not None: detector.close()
            except Exception:
                pass


class TemporalWorkspace(QWidget):
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.worker = None
        root = QVBoxLayout(self); root.setContentsMargins(14, 12, 14, 12)
        root.addWidget(QLabel("V3 Temporal Reconstruction"))
        hint = QLabel("视频序列模式：连续帧人体关键点做时间平滑；TemporalSmoother 同时支持相机状态输入，但本 UI 不会把单帧视频伪装成绝对相机测量。")
        hint.setWordWrap(True); root.addWidget(hint)
        bar = QHBoxLayout(); open_button = QPushButton("Open video"); open_button.clicked.connect(self._open); bar.addWidget(open_button)
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0.05, 1.0); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.35); bar.addWidget(QLabel("Smoothing")); bar.addWidget(self.alpha)
        self.status = QLabel("Idle"); bar.addWidget(self.status, 1); root.addLayout(bar)
        self.frame = QLabel("Frame: —"); self.confidence = QLabel("Temporal confidence: —"); self.visible = QLabel("Visible landmarks: —")
        root.addWidget(self.frame); root.addWidget(self.confidence); root.addWidget(self.visible); root.addStretch(1)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Video (*.mp4 *.mov *.avi *.mkv)")
        if not path: return
        if self.worker and self.worker.isRunning():
            self.worker.stop_requested = True; self.worker.wait(1000)
        self.worker = TemporalWorker(path, alpha=self.alpha.value())
        self.worker.progressed.connect(self._progress); self.worker.completed.connect(lambda: self.status.setText("Sequence complete")); self.worker.failed.connect(lambda text: self.status.setText(text))
        self.status.setText(Path(path).name); self.worker.start()

    def _progress(self, index, confidence, visible):
        self.frame.setText(f"Frame: {int(index)}"); self.confidence.setText(f"Temporal confidence: {confidence:.0%}"); self.visible.setText(f"Visible landmarks: {visible}/17")


def install_v3_completion(window):
    """Install final v3 tools without replacing the existing v2/v2.5 analysis engine."""
    bars = window.findChildren(__import__("PySide6.QtWidgets", fromlist=["QToolBar"]).QToolBar)
    bar = bars[0] if bars else None

    def save_action():
        path, _ = QFileDialog.getSaveFileName(window, "Save reconstruction session", "", "Reconstruction Session (*.pibr.json)")
        if not path: return
        panel = getattr(window._w3, "_plane_constraints_panel", None)
        constraints = panel.to_dicts() if panel is not None else getattr(window._w3.scene, "plane_constraints", [])
        save_session(path, window._w3.scene, image_path=getattr(window, "_current_path", None), image_shape=window._img.shape[:2][::-1] if window._img is not None else None, plane_constraints=constraints)
        window._st.showMessage(f"Saved reconstruction session · {Path(path).name}")

    def load_action():
        path, _ = QFileDialog.getOpenFileName(window, "Load reconstruction session", "", "Reconstruction Session (*.pibr.json)")
        if not path: return
        try:
            scene, data = load_session(path)
            window._w3.scene = scene; window._w3._view.scene = scene
            window._w3._view.update(); window._w3._sync_controls(); window._w3._refresh_projection()
            panel = getattr(window._w3, "_plane_constraints_panel", None)
            if panel is not None: panel.from_dicts(data.get("plane_constraints", []))
            window._st.showMessage(f"Loaded reconstruction session · {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(window, "Load session failed", f"{type(exc).__name__}: {exc}")

    if bar is not None:
        bar.addSeparator(); save = QAction("Save Session", window); save.triggered.connect(save_action); bar.addAction(save)
        load = QAction("Load Session", window); load.triggered.connect(load_action); bar.addAction(load)

    if hasattr(window, "_w3"):
        panel = PlaneConstraintPanel(window._w3)
        scrolls = window._w3.findChildren(__import__("PySide6.QtWidgets", fromlist=["QScrollArea"]).QScrollArea)
        if scrolls:
            inner = scrolls[0].widget(); layout = inner.layout()
            anchors = next((s for s in inner.findChildren(QWidget) if getattr(getattr(s, "button", None), "text", lambda: "")() == "Scene anchors"), None)
            if anchors is not None: layout.insertWidget(layout.indexOf(anchors) + 1, panel)
            else: layout.insertWidget(2, panel)
        window._w3._plane_constraints_panel = panel

    ref = getattr(window, "_reference_mode", None)
    if ref is not None:
        target_box = QGroupBox("Generated composition-aware pose target"); tl = QVBoxLayout(target_box)
        label = QLabel("Ready when a reference and current pose are available."); label.setWordWrap(True); tl.addWidget(label)
        button = QPushButton("Generate target"); tl.addWidget(button)
        def generate():
            if ref._reference_pose is None or ref._reference is None or ref._current_pose is None:
                label.setText("Reference + current pose are required."); return
            width, height = ref._current_image_size
            from reverse_engineering.reference_reconstruction import build_reference_composition
            current = build_reference_composition(ref._current_pose, width, height)
            result = generate_composition_aware_pose_target(ref._reference_pose.landmarks, ref._reference, current)
            if not result.success: label.setText(result.message); return
            target = result.target
            visible = target.visible_count if target is not None else 0
            label.setText(f"{result.message} · target center {target.subject_center[0]/width:.0%}, {target.subject_center[1]/height:.0%} · {visible}/17 landmarks")
        button.clicked.connect(generate); ref.layout().addWidget(target_box)

    temporal = TemporalWorkspace(window); window._tabs.addTab("Temporal"); window._ws.addWidget(temporal); window._temporal_workspace = temporal
    return temporal
