"""User-facing integration for the final v3 reconstruction workflow."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from reverse_engineering.anchor_calibration import estimate_camera_from_anchors
from reverse_engineering.plane_constraints import PlaneConstraint, PlaneRelation, apply_position_constraint, evaluate_constraint
from reverse_engineering.reference_pose_generation import generate_composition_aware_pose_target
from reverse_engineering.reconstruction_session import load_session, save_session
from reverse_engineering.temporal import TemporalFrameState, TemporalSmoother


def _coerce_plane_relation(value) -> PlaneRelation:
    """Accept either the enum or its persisted string value."""
    if isinstance(value, PlaneRelation):
        return value
    try:
        return PlaneRelation(str(value))
    except ValueError:
        # Older/manual callers sometimes pass the display label instead.
        text = str(value).strip().lower()
        labels = {r.label.lower(): r for r in PlaneRelation}
        if text in labels:
            return labels[text]
        return PlaneRelation.ON_PLANE


class PlaneConstraintPanel(QWidget):
    """UI for conservative positional plane constraints on the primary subject."""

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self._constraints: list[PlaneConstraint] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(5)
        title = QLabel("Plane-aware constraints")
        root.addWidget(title)
        hint = QLabel("把人物/物体位置约束到平面或距离平面指定偏移；方向约束仍保持为 API primitive，不会伪装成位置测量。")
        hint.setWordWrap(True)
        root.addWidget(hint)
        row = QHBoxLayout()
        self.plane = QComboBox()
        self.relation = QComboBox()
        for relation in (PlaneRelation.ON_PLANE, PlaneRelation.OFFSET):
            self.relation.addItem(relation.label, relation.value)
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

    @staticmethod
    def _constraint_text(constraint: PlaneConstraint) -> str:
        relation = _coerce_plane_relation(getattr(constraint, "relation", PlaneRelation.ON_PLANE))
        return f"{relation.label}: {constraint.plane_anchor_id} · {float(constraint.offset_m):+.2f} m"

    def refresh_planes(self):
        current = self.plane.currentData()
        self.plane.blockSignals(True)
        self.plane.clear()
        for anchor in self.workspace.scene.anchors:
            if getattr(anchor.kind, "value", anchor.kind) == "plane":
                self.plane.addItem(anchor.name, anchor.anchor_id)
        if current is not None:
            index = self.plane.findData(current)
            if index >= 0: self.plane.setCurrentIndex(index)
        self.plane.blockSignals(False)

    def sync_from_scene(self):
        """Synchronize calibrated plane anchors into this panel without duplicating manual constraints."""
        self.refresh_planes()
        by_plane = {c.plane_anchor_id: c for c in self._constraints}
        changed = False
        for anchor in self.workspace.scene.anchors:
            is_plane = getattr(anchor.kind, "value", anchor.kind) == "plane"
            calibrated = is_plane and len(getattr(anchor, "image_points", ())) >= 4
            if not calibrated:
                continue
            existing = by_plane.get(anchor.anchor_id)
            if existing is None:
                constraint = PlaneConstraint(
                    constraint_id=f"anchor_calibration_{anchor.anchor_id}",
                    plane_anchor_id=str(anchor.anchor_id),
                    relation=PlaneRelation.ON_PLANE,
                    offset_m=0.0,
                    source="anchor_calibration",
                )
                self._constraints.append(constraint)
                by_plane[anchor.anchor_id] = constraint
                changed = True
            elif _coerce_plane_relation(existing.relation) != PlaneRelation.ON_PLANE and existing.source == "anchor_calibration":
                existing.relation = PlaneRelation.ON_PLANE
                existing.offset_m = 0.0
                changed = True
        if changed:
            self._rebuild_list()
        self.workspace.scene.plane_constraints = [c.to_dict() for c in self._constraints]

    def _rebuild_list(self):
        self._list.blockSignals(True)
        self._list.clear()
        for constraint in self._constraints:
            self._list.addItem(QListWidgetItem(self._constraint_text(constraint)))
        self._list.blockSignals(False)

    def _add(self):
        plane_id = self.plane.currentData()
        if not plane_id:
            return
        relation = _coerce_plane_relation(self.relation.currentData())
        constraint = PlaneConstraint(
            constraint_id=f"plane_constraint_{len(self._constraints)+1}",
            plane_anchor_id=str(plane_id),
            relation=relation,
            offset_m=float(self.offset.value()),
        )
        self._constraints.append(constraint)
        self._list.addItem(QListWidgetItem(self._constraint_text(constraint)))
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
                c.relation = _coerce_plane_relation(c.relation)
            except (ValueError, TypeError):
                continue
            if _coerce_plane_relation(c.relation) not in (PlaneRelation.ON_PLANE, PlaneRelation.OFFSET):
                continue
            self._constraints.append(c)
            self._list.addItem(QListWidgetItem(self._constraint_text(c)))
        self.workspace.scene.plane_constraints = [c.to_dict() for c in self._constraints]
        self.refresh_planes()


class AnchorCameraHypothesisPanel(QWidget):
    """Cross-check explicit anchor PnP against the active scene camera without mutating it."""

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self._last_result = None
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(5)
        title = QLabel("Anchored camera cross-check")
        root.addWidget(title)
        hint = QLabel("使用当前已绑定的 2D/3D anchors 做独立 PnP 假设，并与当前 SceneCamera 分开显示。不会自动覆盖活动相机。")
        hint.setWordWrap(True)
        root.addWidget(hint)
        row = QHBoxLayout()
        solve = QPushButton("Solve from anchors")
        solve.clicked.connect(self.solve)
        row.addWidget(solve)
        self.status = QLabel("No anchor solve yet")
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        root.addLayout(row)

        plane_row = QHBoxLayout()
        plane_row.addWidget(QLabel("Plane overlay"))
        self.plane_combo = QComboBox()
        self.plane_combo.addItem("Auto / solved plane", "")
        self.plane_combo.currentIndexChanged.connect(self._plane_overlay_changed)
        plane_row.addWidget(self.plane_combo, 1)
        self.show_plane = QCheckBox("Show")
        self.show_plane.setChecked(True)
        self.show_plane.toggled.connect(self._toggle_plane_overlay)
        plane_row.addWidget(self.show_plane)
        root.addLayout(plane_row)

        self.position = QLabel("Camera position: —")
        self.rmse = QLabel("Reprojection RMSE: —")
        self.delta = QLabel("Active camera: unchanged")
        for label in (self.position, self.rmse, self.delta):
            label.setStyleSheet("color:#475569;")
            label.setWordWrap(True)
            root.addWidget(label)
        self._refresh_plane_choices()

    def _refresh_plane_choices(self, preferred_id: str = ""):
        current = self.plane_combo.currentData()
        self.plane_combo.blockSignals(True)
        self.plane_combo.clear()
        self.plane_combo.addItem("Auto / solved plane", "")
        for anchor in self.workspace.scene.anchors:
            if getattr(anchor.kind, "value", anchor.kind) == "plane":
                calibrated = len(getattr(anchor, "image_points", ())) >= 4
                suffix = " · calibrated" if calibrated else ""
                self.plane_combo.addItem(f"{anchor.name}{suffix}", anchor.anchor_id)
        target = preferred_id or (current if current else "")
        index = self.plane_combo.findData(target)
        self.plane_combo.setCurrentIndex(index if index >= 0 else 0)
        self.plane_combo.blockSignals(False)

    def _selected_plane(self):
        selected = self.plane_combo.currentData()
        if selected:
            return self.workspace.scene.anchor_by_id(str(selected))
        for anchor in self.workspace.scene.anchors:
            if getattr(anchor.kind, "value", anchor.kind) == "plane" and len(getattr(anchor, "image_points", ())) >= 4:
                return anchor
        return None

    def _sync_selected_plane(self):
        plane = self._selected_plane()
        if plane is None:
            return
        plane.visible = bool(self.show_plane.isChecked())
        self.workspace._view.update()
        self.workspace._refresh_projection()

    def _plane_overlay_changed(self, _index):
        self._sync_selected_plane()

    def _toggle_plane_overlay(self, checked):
        self._sync_selected_plane()

    def solve(self):
        image = getattr(self.workspace, "_source_image", None)
        if image is None:
            self.status.setText("需要先加载当前照片。")
            return None
        self._refresh_plane_choices()
        height, width = image.shape[:2]
        camera = self.workspace.scene.camera
        result = estimate_camera_from_anchors(
            self.workspace.scene.anchors,
            width,
            height,
            focal_length_mm=float(camera.focal_length_mm),
            sensor_width_mm=float(camera.sensor_width_mm),
        )
        self._last_result = result
        if not result.success:
            self.status.setText(result.message)
            self.position.setText("Camera position: —")
            self.rmse.setText("Reprojection RMSE: —")
            self.delta.setText("Active camera: unchanged")
            return result
        px, py, pz = result.position
        self.status.setText(result.message)
        self.position.setText(f"Camera position hypothesis: ({px:.2f}, {py:.2f}, {pz:.2f}) m")
        self.rmse.setText(f"Reprojection RMSE: {result.reprojection_rmse_px:.2f} px · {result.point_count} correspondences")
        self.delta.setText(
            f"Active camera: unchanged · reference hypothesis: {getattr(self.workspace, '_hypothesis', None).confidence:.0%} confidence"
            if getattr(self.workspace, "_hypothesis", None) is not None
            else "Active camera: unchanged · no reference-camera hypothesis yet"
        )
        self._sync_selected_plane()
        return result


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
        title = QLabel("V3 Temporal Reconstruction"); title.setStyleSheet("font-size:14pt; font-weight:600;"); root.addWidget(title)
        hint = QLabel("视频序列模式：连续帧人体关键点做时间平滑；不会把单帧视频伪装成绝对相机运动。")
        hint.setWordWrap(True); root.addWidget(hint)
        bar = QHBoxLayout(); open_button = QPushButton("Open video"); open_button.clicked.connect(self._open); bar.addWidget(open_button)
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0.05, 1.0); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.35); bar.addWidget(QLabel("Smoothing")); bar.addWidget(self.alpha)
        stop_button = QPushButton("Stop"); stop_button.clicked.connect(self._stop); bar.addWidget(stop_button)
        self.status = QLabel("Idle"); bar.addWidget(self.status, 1); root.addLayout(bar)
        self.frame = QLabel("Frame: —"); self.confidence = QLabel("Temporal confidence: —"); self.visible = QLabel("Visible landmarks: —")
        root.addWidget(self.frame); root.addWidget(self.confidence); root.addWidget(self.visible); root.addStretch(1)

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop_requested = True
            self.status.setText("Stopping…")

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Video (*.mp4 *.mov *.avi *.mkv)")
        if not path: return
        self._stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(1000)
        self.worker = TemporalWorker(path, alpha=self.alpha.value())
        self.worker.progressed.connect(self._progress); self.worker.completed.connect(lambda: self.status.setText("Sequence complete")); self.worker.failed.connect(lambda text: self.status.setText(text))
        self.status.setText(Path(path).name); self.worker.start()

    def _progress(self, index, confidence, visible):
        self.frame.setText(f"Frame: {int(index)}"); self.confidence.setText(f"Temporal confidence: {confidence:.0%}"); self.visible.setText(f"Visible landmarks: {visible}/17")

    def close_worker(self):
        self._stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(1500)


def _workspace_reference_metadata(window):
    reference = getattr(window, "_reference_mode", None)
    return {
        "reference_image_path": getattr(reference, "_reference_path", None) if reference is not None else None,
        "reference_active": bool(getattr(reference, "_reference", None)) if reference is not None else False,
    }


def install_v3_completion(window):
    """Install and connect the complete v3 desktop workflow."""
    bars = window.findChildren(__import__("PySide6.QtWidgets", fromlist=["QToolBar"]).QToolBar)
    bar = bars[0] if bars else None

    def save_action():
        path, _ = QFileDialog.getSaveFileName(window, "Save reconstruction session", "", "Reconstruction Session (*.pibr.json)")
        if not path: return
        panel = getattr(window._w3, "_plane_constraints_panel", None)
        constraints = panel.to_dicts() if panel is not None else getattr(window._w3.scene, "plane_constraints", [])
        metadata = _workspace_reference_metadata(window)
        save_session(
            path,
            window._w3.scene,
            image_path=getattr(window, "_current_path", None),
            image_shape=window._img.shape[:2][::-1] if window._img is not None else None,
            metadata=metadata,
            plane_constraints=constraints,
        )
        window._st.showMessage(f"Saved reconstruction session · {Path(path).name}")

    def load_action():
        path, _ = QFileDialog.getOpenFileName(window, "Load reconstruction session", "", "Reconstruction Session (*.pibr.json)")
        if not path: return
        try:
            scene, data = load_session(path)
            window._w3.scene = scene
            window._w3._view.set_scene(scene)
            window._w3._sync_controls()
            window._w3._populate_people()
            window._w3._populate_anchors()
            window._w3._populate_candidates()
            window._w3._refresh_projection()
            panel = getattr(window._w3, "_plane_constraints_panel", None)
            if panel is not None:
                panel.from_dicts(data.get("plane_constraints", []))
                panel.sync_from_scene()
            anchor_panel = getattr(window._w3, "_anchor_camera_hypothesis", None)
            if anchor_panel is not None:
                anchor_panel._refresh_plane_choices()
                anchor_panel.solve()
            metadata = dict(data.get("metadata", {}))
            reference = getattr(window, "_reference_mode", None)
            reference_path = metadata.get("reference_image_path")
            if reference is not None and reference_path:
                reference.load_reference_path(reference_path)
            window._current_path = data.get("image", {}).get("path") or getattr(window, "_current_path", None)
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

        anchor_panel = AnchorCameraHypothesisPanel(window._w3)
        if scrolls:
            inner = scrolls[0].widget(); layout = inner.layout()
            constraint_index = layout.indexOf(panel)
            layout.insertWidget(constraint_index + 1, anchor_panel)
        window._w3._anchor_camera_hypothesis = anchor_panel

        window._w3.camera_edited.connect(anchor_panel.solve)

        # Anchor calibration and any later scene edits use the same SceneModel;
        # keep the constraints/plane overlay panels in sync with that source of truth.
        original_anchor_selected = getattr(window._w3, "_anchor_selected", None)
        if original_anchor_selected is not None:
            original_anchor_selected_ref = original_anchor_selected
            def sync_anchor_selected(row):
                original_anchor_selected_ref(row)
                panel.sync_from_scene()
                anchor_panel._refresh_plane_choices()
            window._w3._anchor_selected = sync_anchor_selected
            window._w3._anchors.currentRowChanged.disconnect(original_anchor_selected_ref)
            window._w3._anchors.currentRowChanged.connect(window._w3._anchor_selected)

        panel.sync_from_scene()

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
            if not result.success:
                label.setText(result.message); return
            target = result.target
            ref._last_target = target
            ref._sync_canvas_target()
            visible = target.visible_count if target is not None else 0
            label.setText(
                f"{result.message} · target center {target.subject_center[0]/width:.0%}, {target.subject_center[1]/height:.0%} · {visible}/17 landmarks"
            )
        button.clicked.connect(generate); ref.layout().addWidget(target_box)

    temporal = TemporalWorkspace(window); window._tabs.addTab("Temporal"); window._ws.addWidget(temporal); window._temporal_workspace = temporal
    return temporal
