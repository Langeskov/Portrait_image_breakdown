"""V3 UI integration: plane constraints, anchor camera solve, and temporal workspace.

The application runtime owns session save/load and the main toolbar. This module
owns reconstruction-specific panels only, avoiding duplicate toolbar actions.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QScrollArea,
    QWidget,
)

from reverse_engineering.anchor_calibration import estimate_camera_from_anchors
from reverse_engineering.plane_constraints import (
    PlaneConstraint,
    PlaneRelation,
    apply_position_constraint,
    evaluate_constraint,
)
from reverse_engineering.reference_pose_generation import generate_composition_aware_pose_target
from reverse_engineering.temporal import TemporalFrameState, TemporalSmoother


def coerce_plane_relation(value) -> PlaneRelation:
    """Normalize enum, persisted value, or display label into PlaneRelation."""
    if isinstance(value, PlaneRelation):
        return value
    text = str(value).strip()
    try:
        return PlaneRelation(text)
    except ValueError:
        labels = {relation.label.lower(): relation for relation in PlaneRelation}
        return labels.get(text.lower(), PlaneRelation.ON_PLANE)


class PlaneConstraintPanel(QWidget):
    """Manage positional constraints against scene plane anchors."""

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self._constraints: list[PlaneConstraint] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(5)
        root.addWidget(QLabel("Plane-aware constraints"))

        hint = QLabel(
            "把人物/物体位置约束到平面或距离平面指定偏移；方向关系保留为 API primitive。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748B;")
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
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add)
        for label, widget in (("Plane", self.plane), ("Relation", self.relation), ("Offset", self.offset)):
            row.addWidget(QLabel(label))
            row.addWidget(widget, 1)
        row.addWidget(add_button)
        root.addLayout(row)

        self._list = QListWidget()
        self._list.setMaximumHeight(100)
        root.addWidget(self._list)

        actions = QHBoxLayout()
        apply_button = QPushButton("Apply to primary subject")
        apply_button.clicked.connect(self._apply)
        evaluate_button = QPushButton("Evaluate")
        evaluate_button.clicked.connect(self._evaluate)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._remove)
        actions.addWidget(apply_button)
        actions.addWidget(evaluate_button)
        actions.addWidget(remove_button)
        root.addLayout(actions)
        self.refresh_planes()

    @staticmethod
    def _constraint_text(constraint: PlaneConstraint) -> str:
        relation = coerce_plane_relation(constraint.relation)
        return f"{relation.label}: {constraint.plane_anchor_id} · {float(constraint.offset_m):+.2f} m"

    def refresh_planes(self):
        current = self.plane.currentData()
        self.plane.blockSignals(True)
        self.plane.clear()
        for anchor in self.workspace.scene.anchors:
            if getattr(anchor.kind, "value", anchor.kind) == AnchorKindValue.PLANE:
                self.plane.addItem(anchor.name, anchor.anchor_id)
        if current is not None:
            index = self.plane.findData(current)
            if index >= 0:
                self.plane.setCurrentIndex(index)
        self.plane.blockSignals(False)

    def sync_from_scene(self):
        """Synchronize calibration-derived constraints with the canonical SceneModel."""
        self.refresh_planes()
        plane_ids = {
            anchor.anchor_id
            for anchor in self.workspace.scene.anchors
            if getattr(anchor.kind, "value", anchor.kind) == AnchorKindValue.PLANE
        }
        calibrated_ids = {
            anchor.anchor_id
            for anchor in self.workspace.scene.anchors
            if getattr(anchor.kind, "value", anchor.kind) == AnchorKindValue.PLANE
            and len(getattr(anchor, "image_points", ())) >= 4
        }

        cleaned: list[PlaneConstraint] = []
        for constraint in self._constraints:
            relation = coerce_plane_relation(constraint.relation)
            constraint.relation = relation
            # Calibration-generated constraints are derived state. Remove them
            # as soon as their supporting plane is no longer calibrated.
            if constraint.source == "anchor_calibration" and (
                constraint.plane_anchor_id not in plane_ids
                or constraint.plane_anchor_id not in calibrated_ids
            ):
                continue
            cleaned.append(constraint)
        self._constraints = cleaned

        by_plane = {c.plane_anchor_id: c for c in self._constraints}
        changed = len(cleaned) != len(self._constraints)
        for anchor_id in calibrated_ids:
            if anchor_id in by_plane:
                continue
            constraint = PlaneConstraint(
                constraint_id=f"anchor_calibration_{anchor_id}",
                plane_anchor_id=anchor_id,
                relation=PlaneRelation.ON_PLANE,
                offset_m=0.0,
                source="anchor_calibration",
            )
            self._constraints.append(constraint)
            by_plane[anchor_id] = constraint
            changed = True

        self._rebuild_list()
        self.workspace.scene.plane_constraints = [constraint.to_dict() for constraint in self._constraints]

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
        constraint = PlaneConstraint(
            constraint_id=f"plane_constraint_{len(self._constraints) + 1}",
            plane_anchor_id=str(plane_id),
            relation=coerce_plane_relation(self.relation.currentData()),
            offset_m=float(self.offset.value()),
        )
        self._constraints.append(constraint)
        self._rebuild_list()
        self.workspace.scene.plane_constraints = [c.to_dict() for c in self._constraints]

    def _remove(self):
        row = self._list.currentRow()
        if 0 <= row < len(self._constraints):
            self._constraints.pop(row)
            self._rebuild_list()
            self.workspace.scene.plane_constraints = [c.to_dict() for c in self._constraints]

    def _apply(self):
        row = self._list.currentRow()
        if not 0 <= row < len(self._constraints):
            return
        subject = self.workspace.scene.subject
        try:
            subject.center_x, subject.center_y, subject.center_z = apply_position_constraint(
                (subject.center_x, subject.center_y, subject.center_z),
                self._constraints[row],
                self.workspace.scene.anchors,
            )
            self.workspace.refresh_scene_view()
        except ValueError as exc:
            QMessageBox.warning(self, "Plane constraint", str(exc))

    def _evaluate(self):
        row = self._list.currentRow()
        if not 0 <= row < len(self._constraints):
            return
        subject = self.workspace.scene.subject
        try:
            result = evaluate_constraint(
                (subject.center_x, subject.center_y, subject.center_z),
                self._constraints[row],
                self.workspace.scene.anchors,
            )
            state = "SATISFIED" if result.satisfied else "NOT SATISFIED"
            QMessageBox.information(
                self,
                "Plane constraint",
                f"{state}\nResidual: {result.residual_m:.3f} m\n{result.message}",
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Plane constraint", str(exc))

    def to_dicts(self):
        return [constraint.to_dict() for constraint in self._constraints]

    def from_dicts(self, data):
        self._constraints.clear()
        for item in data or []:
            try:
                constraint = PlaneConstraint.from_dict(item)
                constraint.relation = coerce_plane_relation(constraint.relation)
            except (TypeError, ValueError):
                continue
            if constraint.relation in (PlaneRelation.ON_PLANE, PlaneRelation.OFFSET):
                self._constraints.append(constraint)
        self._rebuild_list()
        self.workspace.scene.plane_constraints = [c.to_dict() for c in self._constraints]
        self.refresh_planes()


class AnchorCameraHypothesisPanel(QWidget):
    """Independent PnP cross-check plus optional selected-plane overlay."""

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self._last_result = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(5)
        root.addWidget(QLabel("Anchored camera cross-check"))

        hint = QLabel(
            "用当前已绑定的 2D/3D anchors 做独立 PnP 假设；不会自动覆盖活动相机。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748B;")
        root.addWidget(hint)

        solve_row = QHBoxLayout()
        solve_button = QPushButton("Solve from anchors")
        solve_button.clicked.connect(self.solve)
        solve_row.addWidget(solve_button)
        self.status = QLabel("No anchor solve yet")
        self.status.setWordWrap(True)
        solve_row.addWidget(self.status, 1)
        root.addLayout(solve_row)

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
        self.refresh_plane_choices()

    def refresh_plane_choices(self, preferred_id: str = ""):
        current = self.plane_combo.currentData()
        self.plane_combo.blockSignals(True)
        self.plane_combo.clear()
        self.plane_combo.addItem("Auto / solved plane", "")
        for anchor in self.workspace.scene.anchors:
            if getattr(anchor.kind, "value", anchor.kind) != AnchorKindValue.PLANE:
                continue
            suffix = " · calibrated" if len(getattr(anchor, "image_points", ())) >= 4 else ""
            self.plane_combo.addItem(f"{anchor.name}{suffix}", anchor.anchor_id)
        target = preferred_id or current or ""
        index = self.plane_combo.findData(target)
        self.plane_combo.setCurrentIndex(index if index >= 0 else 0)
        self.plane_combo.blockSignals(False)

    def _selected_plane(self):
        selected = self.plane_combo.currentData()
        if selected:
            return self.workspace.scene.anchor_by_id(str(selected))
        for anchor in self.workspace.scene.anchors:
            if (
                getattr(anchor.kind, "value", anchor.kind) == AnchorKindValue.PLANE
                and len(getattr(anchor, "image_points", ())) >= 4
            ):
                return anchor
        return None

    def _sync_selected_plane(self):
        plane = self._selected_plane()
        if plane is None:
            return
        plane.visible = bool(self.show_plane.isChecked())
        self.workspace.refresh_scene_view()

    def _plane_overlay_changed(self, _index):
        self._sync_selected_plane()

    def _toggle_plane_overlay(self, _checked):
        self._sync_selected_plane()

    def solve(self):
        image = getattr(self.workspace, "_source_image", None)
        if image is None:
            self.status.setText("需要先加载当前照片。")
            return None
        self.refresh_plane_choices()
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
        self.rmse.setText(
            f"Reprojection RMSE: {result.reprojection_rmse_px:.2f} px · {result.point_count} correspondences"
        )
        hypothesis = getattr(self.workspace, "_hypothesis", None)
        self.delta.setText(
            f"Active camera: unchanged · reference hypothesis: {hypothesis.confidence:.0%} confidence"
            if hypothesis is not None
            else "Active camera: unchanged · no reference-camera hypothesis yet"
        )
        self._sync_selected_plane()
        return result


class TemporalWorker(QThread):
    """Sample a video and smooth 17-point pose observations over time."""

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

            detector = PoseDetector()
            capture = cv2.VideoCapture(self.video_path)
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
                    keypoints = None
                    confidence = 0.0
                    if pose is not None:
                        keypoints = np.asarray([(lm.x, lm.y) for lm in pose.landmarks[:17]], dtype=float)
                        confidence = float(getattr(pose, "detection_confidence", 0.0))
                    state = smoother.update(
                        TemporalFrameState(index / max(fps, 1e-6), keypoints=keypoints, confidence=confidence)
                    )
                    visible = (
                        int(np.sum(np.isfinite(state.keypoints).all(axis=1)))
                        if state.keypoints is not None
                        else 0
                    )
                    self.progressed.emit(index, state.confidence, visible)
                index += 1
            self.completed.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.release()
            try:
                if detector is not None:
                    detector.close()
            except Exception:
                pass


class TemporalWorkspace(QWidget):
    """Video-sequence temporal reconstruction workspace."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.worker = None
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("V3 Temporal Reconstruction")
        title.setStyleSheet("font-size:14pt; font-weight:600;")
        root.addWidget(title)
        hint = QLabel("连续帧人体关键点做时间平滑；不会把单帧视频伪装成绝对相机运动。")
        hint.setWordWrap(True)
        root.addWidget(hint)

        bar = QHBoxLayout()
        open_button = QPushButton("Open video")
        open_button.clicked.connect(self._open)
        bar.addWidget(open_button)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.05, 1.0)
        self.alpha.setSingleStep(0.05)
        self.alpha.setValue(0.35)
        bar.addWidget(QLabel("Smoothing"))
        bar.addWidget(self.alpha)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self._stop)
        bar.addWidget(stop_button)
        self.status = QLabel("Idle")
        bar.addWidget(self.status, 1)
        root.addLayout(bar)

        self.frame = QLabel("Frame: —")
        self.confidence = QLabel("Temporal confidence: —")
        self.visible = QLabel("Visible landmarks: —")
        root.addWidget(self.frame)
        root.addWidget(self.confidence)
        root.addWidget(self.visible)
        root.addStretch(1)

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop_requested = True
            self.status.setText("Stopping…")

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "", "Video (*.mp4 *.mov *.avi *.mkv)"
        )
        if not path:
            return
        self._stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(1000)
        self.worker = TemporalWorker(path, alpha=self.alpha.value())
        self.worker.progressed.connect(self._progress)
        self.worker.completed.connect(lambda: self.status.setText("Sequence complete"))
        self.worker.failed.connect(lambda text: self.status.setText(text))
        self.status.setText(Path(path).name)
        self.worker.start()

    def _progress(self, index, confidence, visible):
        self.frame.setText(f"Frame: {int(index)}")
        self.confidence.setText(f"Temporal confidence: {confidence:.0%}")
        self.visible.setText(f"Visible landmarks: {visible}/17")

    def close_worker(self):
        self._stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(1500)


def install_v3_completion(window):
    """Install reconstruction panels into the canonical V3 workspace."""
    workspace = getattr(window, "_w3", None)
    if workspace is None:
        return None

    scrolls = workspace.findChildren(QScrollArea)
    if not scrolls:
        return None
    inner = scrolls[0].widget()
    layout = inner.layout()
    if layout is None:
        return None

    constraint_panel = PlaneConstraintPanel(workspace)
    layout.insertWidget(2, constraint_panel)
    workspace._plane_constraints_panel = constraint_panel

    anchor_panel = AnchorCameraHypothesisPanel(workspace)
    layout.insertWidget(3, anchor_panel)
    workspace._anchor_camera_hypothesis = anchor_panel
    workspace.camera_edited.connect(anchor_panel.solve)

    if hasattr(workspace, "_anchors"):
        workspace._anchors.currentRowChanged.connect(
            lambda _row: (constraint_panel.sync_from_scene(), anchor_panel.refresh_plane_choices())
        )

    constraint_panel.sync_from_scene()

    reference = getattr(window, "_reference_mode", None)
    if reference is not None:
        target_box = QGroupBox("Generated composition-aware pose target")
        target_layout = QVBoxLayout(target_box)
        label = QLabel("Ready when a reference and current pose are available.")
        label.setWordWrap(True)
        target_layout.addWidget(label)
        button = QPushButton("Generate target")
        button.clicked.connect(
            lambda: _generate_target(reference, label)
        )
        target_layout.addWidget(button)
        reference.layout().addWidget(target_box)

    temporal = TemporalWorkspace(window)
    window._tabs.addTab("Temporal")
    window._ws.addWidget(temporal)
    window._temporal_workspace = temporal
    return temporal


def _generate_target(reference, label):
    if reference._reference_pose is None or reference._reference is None or reference._current_pose is None:
        label.setText("Reference + current pose are required.")
        return
    width, height = reference._current_image_size
    from reverse_engineering.reference_reconstruction import build_reference_composition

    current = build_reference_composition(reference._current_pose, width, height)
    result = generate_composition_aware_pose_target(
        reference._reference_pose.landmarks,
        reference._reference,
        current,
    )
    if not result.success:
        label.setText(result.message)
        return
    target = result.target
    reference._last_target = target
    reference._sync_canvas_target()
    visible = target.visible_count if target is not None else 0
    label.setText(
        f"{result.message} · target center "
        f"{target.subject_center[0] / width:.0%}, {target.subject_center[1] / height:.0%} · "
        f"{visible}/17 landmarks"
    )


# Keep this local alias to avoid importing the enum into every legacy helper call.
from reverse_engineering.scene_anchors import AnchorKind
AnchorKindValue = AnchorKind


__all__ = [
    "AnchorCameraHypothesisPanel",
    "PlaneConstraintPanel",
    "TemporalWorker",
    "TemporalWorkspace",
    "coerce_plane_relation",
    "install_v3_completion",
]
