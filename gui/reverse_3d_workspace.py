"""Reference-aware v3 reconstruction workspace.

This replaces the older compact inspector without changing the underlying
SceneModel, projection engine, or anchor calibration contract. The workspace
adds a conservative reference-camera hypothesis and selected-anchor overlays
in the 2D projection preview.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QPointF, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton, QScrollArea,
    QGroupBox, QFormLayout, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QPushButton, QComboBox, QSizePolicy, QSplitter, QFrame,
)

from gui.reverse_3d import SceneView, ProjectionPreview
from reverse_engineering.projection import build_camera_model
from reverse_engineering.reference_camera import (
    ReferenceCameraHypothesis,
    estimate_reference_camera_hypothesis,
)
from reverse_engineering.scene import SceneCamera, SceneModel
from reverse_engineering.scene_anchors import AnchorKind, SceneAnchor


ACCENT = QColor("#2563EB")
TEXT = QColor("#334155")
MUTED = QColor("#64748B")
PLANE = QColor(37, 99, 235, 48)
ANCHOR = QColor("#D97706")
REF = QColor("#7C3AED")


class CollapsibleSection(QWidget):
    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        self.button = QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(self.button)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 6, 8, 7)
        self.body_layout.setSpacing(6)
        root.addWidget(self.body)
        self.body.setVisible(expanded)
        self.button.toggled.connect(self._toggle)
        self.button.setStyleSheet(
            "QToolButton { border: 1px solid #D9DDE3; border-radius: 6px; "
            "padding: 7px 9px; background: #FAFBFC; color: #1F2937; "
            "text-align: left; font-weight: 600; }"
            "QToolButton:hover { background: #F3F4F6; }"
        )

    def _toggle(self, checked: bool):
        self.body.setVisible(checked)
        self.button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)


class AnchorSceneView(SceneView):
    """SceneView with visible/editable scene-anchor overlays."""

    def paintEvent(self, event):
        super().paintEvent(event)
        anchors = [
            a for a in getattr(self.scene, "anchors", ())
            if a.enabled and getattr(a, "visible", True)
        ]
        if not anchors:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for anchor in anchors:
            center = self._project(anchor.position)
            if anchor.kind == AnchorKind.PLANE:
                corners = anchor.corners()
                if len(corners) == 4:
                    painter.setPen(QPen(ACCENT, 1.3, Qt.DashLine))
                    painter.setBrush(QBrush(PLANE))
                    painter.drawPolygon(QPolygonF([self._project(p) for p in corners]))
                    normal = anchor.normalized_normal()
                    end = np.asarray(anchor.position, dtype=float) + normal * 0.8
                    painter.setPen(QPen(ANCHOR, 2))
                    painter.drawLine(center, self._project(end))
            else:
                painter.setPen(QPen(ANCHOR, 2))
                painter.setBrush(QBrush(QColor("#FFFFFF")))
                painter.drawEllipse(center, 5, 5)
            painter.setPen(QPen(TEXT, 1))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(center + QPointF(8, -8), anchor.name)
        painter.end()


class AnchorProjectionPreview(ProjectionPreview):
    """Projection preview that highlights the currently selected scene anchor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene: Optional[SceneModel] = None
        self._selected_anchor: Optional[SceneAnchor] = None

    def set_selected_anchor(self, scene: SceneModel, anchor: Optional[SceneAnchor]):
        self._scene = scene
        self._selected_anchor = anchor
        self.update()

    def update_projection(self, scene, observed_bbox=None, observed_points=None):
        self._scene = scene
        if self._selected_anchor is not None:
            current = scene.anchor_by_id(self._selected_anchor.anchor_id)
            self._selected_anchor = current
        super().update_projection(scene, observed_bbox, observed_points)

    def _project_selected(self, width: int, height: int):
        anchor = self._selected_anchor
        scene = self._scene
        if anchor is None or scene is None or self._pixmap is None:
            return None
        try:
            camera = build_camera_model(scene, width, height)
            world = anchor.corners() if anchor.kind == AnchorKind.PLANE else np.asarray([anchor.position], dtype=float)
            points = camera.project_points(np.asarray(world, dtype=float)).reshape(-1, 2)
            rotation = camera.extrinsics
            if rotation is None:
                return None
            visible = []
            for point, wp in zip(points, np.asarray(world)):
                cam = rotation
                # CameraModel exposes rvec/tvec through its extrinsics.
                import cv2
                rmat = cv2.Rodrigues(cam.rvec)[0]
                cp = rmat @ wp.reshape(3, 1) + cam.tvec.reshape(3, 1)
                if cp[2, 0] > 1e-6 and np.isfinite(point).all():
                    visible.append(point)
            return np.asarray(visible, dtype=float)
        except (ValueError, TypeError, AttributeError, np.linalg.LinAlgError):
            return None

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._scene is None or self._selected_anchor is None or self._pixmap is None:
            return
        w, h = self._pixmap.width(), self._pixmap.height()
        points = self._project_selected(w, h)
        if points is None or len(points) == 0:
            return

        area = self.rect().adjusted(6, 6, -6, -38)
        scaled = self._pixmap.scaled(area.size().toSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = area.x() + (area.width() - scaled.width()) * 0.5
        oy = area.y() + (area.height() - scaled.height()) * 0.5
        sx = scaled.width() / max(w, 1)
        sy = scaled.height() / max(h, 1)
        qpoints = [QPointF(ox + p[0] * sx, oy + p[1] * sy) for p in points]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(ANCHOR, 2.4 if self._selected_anchor.kind == AnchorKind.PLANE else 2.2, Qt.DashLine))
        painter.setBrush(QBrush(QColor(217, 119, 6, 35)))
        if self._selected_anchor.kind == AnchorKind.PLANE and len(qpoints) >= 3:
            painter.drawPolygon(QPolygonF(qpoints))
        else:
            p = qpoints[0]
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            painter.drawEllipse(p, 6, 6)
            painter.drawLine(p.x() - 10, p.y(), p.x() + 10, p.y())
            painter.drawLine(p.x(), p.y() - 10, p.x(), p.y() + 10)
        painter.setPen(QPen(ANCHOR, 2))
        for p in qpoints:
            painter.drawEllipse(p, 5, 5)
        label = f"SELECTED · {self._selected_anchor.name}"
        label_pos = qpoints[0] + QPointF(8, -9)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(label_pos, label)
        painter.end()


class Reverse3DWorkspace(QWidget):
    """Reference-aware v3 reconstruction inspector."""

    camera_edited = __import__("PySide6.QtCore", fromlist=["Signal"]).Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = SceneModel()
        self._result = None
        self._source_image = None
        self._observed_bbox = None
        self._observed_points = []
        self._anchor_ids: list[str] = []
        self._reference = None
        self._current_composition = None
        self._hypothesis: Optional[ReferenceCameraHypothesis] = None
        self._last_ref_signature = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._view = AnchorSceneView(self.scene)
        left_layout.addWidget(self._view, 1)
        reset = QPushButton("Reset 3D view")
        reset.setMaximumWidth(130)
        reset.clicked.connect(self._view.reset_view)
        left_layout.addWidget(reset, 0, Qt.AlignRight)
        splitter.addWidget(left)

        splitter.addWidget(self._build_inspector())
        splitter.setSizes([980, 430])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

        self._reference_poll = QTimer(self)
        self._reference_poll.setInterval(500)
        self._reference_poll.timeout.connect(self._poll_reference_mode)
        self._reference_poll.start()

    def _build_inspector(self):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setFrameShape(QFrame.NoFrame)
        outer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.setMinimumWidth(390)
        outer.setStyleSheet("QScrollArea { background:#F8FAFC; border:0; }")

        panel = QWidget()
        panel.setMinimumWidth(370)
        panel.setStyleSheet("QGroupBox { border:1px solid #E2E8F0; border-radius:6px; margin-top:5px; padding-top:4px; } QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; color:#475569; }")
        lo = QVBoxLayout(panel)
        lo.setContentsMargins(10, 10, 10, 14)
        lo.setSpacing(7)

        title = QLabel("Camera & scene reconstruction")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        lo.addWidget(title)
        self._confidence = QLabel("No reverse-engineering result yet")
        self._confidence.setStyleSheet("color:#64748B;")
        self._confidence.setWordWrap(True)
        lo.addWidget(self._confidence)

        # Keep Camera and Scene anchors in their existing positions.
        lo.addWidget(self._build_camera_section())
        lo.addWidget(self._build_anchor_section())
        lo.addWidget(self._build_projection_section())
        lo.addWidget(self._build_candidates_section())
        lo.addWidget(self._build_reference_hypothesis_section())

        # Scene people intentionally moves to the very bottom.
        lo.addWidget(self._build_people_section())

        note = QLabel(
            "Reference camera is a hypothesis derived from composition. "
            "It does not replace the active SceneCamera or convert image-space evidence into metric truth."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#64748B; font-size:9pt;")
        lo.addWidget(note)
        lo.addStretch(1)
        outer.setWidget(panel)
        return outer

    def _build_camera_section(self):
        section = CollapsibleSection("Camera", expanded=True)
        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(3, 3, 3, 3)
        self._distance = self._spin(0.1, 50, 0.1, " m")
        self._height = self._spin(0.1, 5, 0.05, " m")
        self._yaw = self._spin(-180, 180, 1, "°")
        self._pitch = self._spin(-90, 90, 1, "°")
        self._roll = self._spin(-180, 180, 1, "°")
        self._focal = self._spin(10, 600, 1, " mm")
        for box in (self._distance, self._height, self._yaw, self._pitch, self._roll, self._focal):
            box.valueChanged.connect(self._camera_spin_changed)
            box.setToolTip("滚轮可微调；修改后实时更新 2D / 3D 投影。")
        for label, box in (("Distance", self._distance), ("Height", self._height), ("Yaw", self._yaw), ("Pitch", self._pitch), ("Roll", self._roll), ("Focal", self._focal)):
            form.addRow(label, box)
        section.body_layout.addWidget(group)
        return section

    def _build_anchor_section(self):
        section = CollapsibleSection("Scene anchors", expanded=True)
        hint = QLabel("场景基准与照片证据分开管理。选择锚点后，其当前相机投影会在 2D preview 中高亮。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748B; font-size:9pt;")
        section.body_layout.addWidget(hint)

        self._anchors = QListWidget()
        self._anchors.setMaximumHeight(118)
        self._anchors.currentRowChanged.connect(self._anchor_selected)
        self._anchors.itemChanged.connect(self._anchor_visibility_changed)
        section.body_layout.addWidget(self._anchors)

        buttons = QHBoxLayout()
        for text, kind in (("+ Point", AnchorKind.POINT), ("+ Plane", AnchorKind.PLANE)):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, k=kind: self._add_anchor(k))
            buttons.addWidget(button)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_anchor)
        buttons.addWidget(remove)
        section.body_layout.addLayout(buttons)

        editor = QGroupBox("Selected anchor")
        form = QFormLayout(editor)
        self._anchor_name = QLabel("—")
        self._anchor_kind = QComboBox()
        self._anchor_kind.addItems(["point", "plane"])
        self._anchor_kind.currentTextChanged.connect(self._anchor_kind_changed)
        self._ax = self._spin(-50, 50, .05, "")
        self._ay = self._spin(-10, 20, .05, "")
        self._az = self._spin(-50, 50, .05, "")
        self._nx = self._spin(-1, 1, .05, "")
        self._ny = self._spin(-1, 1, .05, "")
        self._nz = self._spin(-1, 1, .05, "")
        self._aw = self._spin(.1, 50, .1, " m")
        self._ah = self._spin(.1, 50, .1, " m")
        for box in (self._ax, self._ay, self._az, self._nx, self._ny, self._nz, self._aw, self._ah):
            box.valueChanged.connect(self._anchor_values_changed)
        for label, box in (("Name", self._anchor_name), ("Kind", self._anchor_kind), ("X", self._ax), ("Y", self._ay), ("Z", self._az), ("Normal X", self._nx), ("Normal Y", self._ny), ("Normal Z", self._nz), ("Width", self._aw), ("Height", self._ah)):
            form.addRow(label, box)
        section.body_layout.addWidget(editor)
        return section

    def _build_projection_section(self):
        section = CollapsibleSection("2D projection preview", expanded=True)
        self._preview = AnchorProjectionPreview()
        self._preview.setMinimumHeight(250)
        section.body_layout.addWidget(self._preview)
        self._preview_metrics = QLabel("No projection yet")
        self._preview_metrics.setWordWrap(True)
        self._preview_metrics.setStyleSheet("color:#475569; font-size:9pt;")
        section.body_layout.addWidget(self._preview_metrics)
        return section

    def _build_candidates_section(self):
        section = CollapsibleSection("Candidate solutions", expanded=False)
        self._candidates = QListWidget()
        self._candidates.setMaximumHeight(145)
        self._candidates.currentRowChanged.connect(self._select_candidate)
        section.body_layout.addWidget(self._candidates)
        return section

    def _build_reference_hypothesis_section(self):
        section = CollapsibleSection("Reference Camera Hypothesis", expanded=False)
        self._reference_section = section
        status = QHBoxLayout()
        self._reference_state = QLabel("No reference camera context")
        self._reference_state.setStyleSheet("color:#64748B;")
        status.addWidget(self._reference_state, 1)
        self._reference_conf = QLabel("—")
        self._reference_conf.setStyleSheet("color:#7C3AED; font-weight:600;")
        status.addWidget(self._reference_conf)
        section.body_layout.addLayout(status)

        group = QGroupBox()
        form = QFormLayout(group)
        self._ref_distance = QLabel("—")
        self._ref_delta = QLabel("—")
        self._ref_yaw = QLabel("—")
        self._ref_pitch = QLabel("—")
        self._ref_focal = QLabel("—")
        self._ref_support = QLabel("—")
        self._ref_support.setWordWrap(True)
        form.addRow("Reference distance", self._ref_distance)
        form.addRow("Distance Δ", self._ref_delta)
        form.addRow("Re-aim yaw", self._ref_yaw)
        form.addRow("Re-aim pitch", self._ref_pitch)
        form.addRow("Focal prior", self._ref_focal)
        form.addRow("Support", self._ref_support)
        section.body_layout.addWidget(group)

        hint = QLabel(
            "这里的 yaw / pitch 是屏幕取景的 re-aim 角，不是直接替换 SceneCamera 的 orbit yaw / pitch。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748B; font-size:9pt;")
        section.body_layout.addWidget(hint)
        return section

    def _build_people_section(self):
        section = CollapsibleSection("Scene people", expanded=False)
        self._people = QListWidget()
        self._people.setMaximumHeight(120)
        section.body_layout.addWidget(self._people)
        hint = QLabel("附加人物只有在存在独立深度证据时才获得 relative-3D；否则标为 2D-only。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748B; font-size:9pt;")
        section.body_layout.addWidget(hint)
        return section

    @staticmethod
    def _spin(lo, hi, step, suffix):
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setDecimals(2 if step < .1 else 1)
        box.setSuffix(suffix)
        return box

    def set_image(self, image):
        self._source_image = image
        if hasattr(self, "_preview"):
            self._preview.set_image(image)
        self._refresh_projection()

    def set_result(self, result, observed_bbox=None, observed_points=None):
        self._result = result
        self._observed_bbox = observed_bbox
        self._observed_points = observed_points or []
        self.scene = SceneModel.from_reverse_result(result)
        self._view.set_scene(self.scene)
        self._sync_controls()
        self._populate_people()
        self._populate_anchors()
        self._populate_candidates()
        if result:
            layout = getattr(result, "multi_person_layout", None)
            count = len(getattr(layout, "people", ())) if layout else len(self._observed_points) or 1
            rel3d = len(getattr(layout, "usable_people", ())) if layout else 0
            self._confidence.setText(f"Overall confidence: {result.overall_confidence:.0%} · People: {count} · Relative 3D: {rel3d}/{count}")
        else:
            self._confidence.setText("No reverse-engineering result yet")
        self._refresh_projection()
        self._update_reference_hypothesis()

    def _sync_controls(self):
        c = self.scene.camera
        for box, value in ((self._distance, c.distance), (self._height, c.height), (self._yaw, c.yaw), (self._pitch, c.pitch), (self._roll, c.roll), (self._focal, c.focal_length_mm)):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)

    def _populate_people(self):
        self._people.clear()
        subjects = list(self.scene.subjects) if self.scene.subjects else [self.scene.subject]
        for subject in subjects:
            role = "primary" if subject is self.scene.subject else "additional"
            depth = f"z={subject.center_z:+.2f} · {subject.depth_confidence:.0%}" if subject.depth_is_relative else "2D-only"
            self._people.addItem(QListWidgetItem(f"P{subject.person_index + 1} · {role} · {depth}"))

    def _populate_candidates(self):
        self._candidates.blockSignals(True)
        self._candidates.clear()
        for i, candidate in enumerate(self.scene.candidate_solutions):
            self._candidates.addItem(QListWidgetItem(
                f"#{i + 1}  {candidate.focal_equiv_35mm:.1f} mm / {candidate.distance:.2f} m / "
                f"h={candidate.height:.2f} m / score={candidate.score:.2f}"
            ))
        if self.scene.candidate_solutions:
            self._candidates.setCurrentRow(self.scene.selected_candidate)
        self._candidates.blockSignals(False)

    def _populate_anchors(self, selected_id: Optional[str] = None):
        self._anchors.blockSignals(True)
        self._anchors.clear()
        self._anchor_ids = []
        for anchor in self.scene.anchors:
            kind = "plane" if anchor.kind == AnchorKind.PLANE else "point"
            state = "shown" if getattr(anchor, "visible", True) else "hidden"
            self._anchor_ids.append(anchor.anchor_id)
            item = QListWidgetItem(f"{anchor.name} · {kind} · {state}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if getattr(anchor, "visible", True) else Qt.Unchecked)
            self._anchors.addItem(item)
        self._anchors.blockSignals(False)
        if self._anchor_ids:
            row = self._anchor_ids.index(selected_id) if selected_id in self._anchor_ids else 0
            self._anchors.setCurrentRow(row)
        else:
            self._anchor_selected(-1)

    def _anchor_visibility_changed(self, item):
        row = self._anchors.row(item)
        if not 0 <= row < len(self._anchor_ids):
            return
        anchor = self.scene.anchor_by_id(self._anchor_ids[row])
        if anchor is None:
            return
        anchor.visible = item.checkState() == Qt.Checked
        kind = "plane" if anchor.kind == AnchorKind.PLANE else "point"
        item.setText(f"{anchor.name} · {kind} · {'shown' if anchor.visible else 'hidden'}")
        self._view.update()
        self._refresh_projection()

    def _anchor_selected(self, row):
        if not 0 <= row < len(self._anchor_ids):
            self._anchor_name.setText("—")
            if hasattr(self, "_preview"):
                self._preview.set_selected_anchor(self.scene, None)
            self._update_reference_hypothesis()
            return
        anchor = self.scene.anchor_by_id(self._anchor_ids[row])
        if anchor is None:
            return
        self._anchor_name.setText(anchor.name)
        self._anchor_kind.blockSignals(True)
        self._anchor_kind.setCurrentText(anchor.kind.value)
        self._anchor_kind.blockSignals(False)
        values = ((self._ax, anchor.position[0]), (self._ay, anchor.position[1]), (self._az, anchor.position[2]), (self._nx, anchor.normal[0]), (self._ny, anchor.normal[1]), (self._nz, anchor.normal[2]), (self._aw, anchor.size[0]), (self._ah, anchor.size[1]))
        for box, value in values:
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)
        plane = anchor.kind == AnchorKind.PLANE
        for box in (self._nx, self._ny, self._nz, self._aw, self._ah):
            box.setEnabled(plane)
        if hasattr(self, "_preview"):
            self._preview.set_selected_anchor(self.scene, anchor)
            self._refresh_projection()
        self._update_reference_hypothesis()

    def _current_anchor(self) -> Optional[SceneAnchor]:
        row = self._anchors.currentRow()
        if not 0 <= row < len(self._anchor_ids):
            return None
        return self.scene.anchor_by_id(self._anchor_ids[row])

    def _add_anchor(self, kind: AnchorKind):
        name = "Scene point" if kind == AnchorKind.POINT else "Scene plane"
        anchor = self.scene.create_anchor(name=name, kind=kind)
        self._populate_anchors(anchor.anchor_id)
        self._view.update()
        self._refresh_projection()

    def _remove_anchor(self):
        anchor = self._current_anchor()
        if anchor is None:
            return
        if self.scene.remove_anchor(anchor.anchor_id):
            self._populate_anchors()
            self._view.update()
            self._refresh_projection()

    def _anchor_kind_changed(self, value):
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        anchor.kind = AnchorKind(value)
        self._anchor_selected(self._anchors.currentRow())
        self._view.update()
        self._refresh_projection()

    def _anchor_values_changed(self):
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        anchor.position = (float(self._ax.value()), float(self._ay.value()), float(self._az.value()))
        anchor.normal = (float(self._nx.value()), float(self._ny.value()), float(self._nz.value()))
        anchor.size = (float(self._aw.value()), float(self._ah.value()))
        self._view.update()
        self._refresh_projection()

    def _select_candidate(self, row):
        if row < 0 or not self.scene.candidate_solutions:
            return
        self.scene.set_candidate(row)
        self._sync_controls()
        self._view.update()
        self._refresh_projection()
        self._update_reference_hypothesis()
        self.camera_edited.emit()

    def _camera_spin_changed(self):
        self.scene.camera = SceneCamera(
            distance=float(self._distance.value()),
            height=float(self._height.value()),
            yaw=float(self._yaw.value()),
            pitch=float(self._pitch.value()),
            roll=float(self._roll.value()),
            focal_length_mm=float(self._focal.value()),
            sensor_width_mm=self.scene.camera.sensor_width_mm,
        )
        self._view.set_scene(self.scene)
        self._refresh_projection()
        self._update_reference_hypothesis()
        self.camera_edited.emit()

    def _refresh_projection(self):
        if not hasattr(self, "_preview"):
            return
        self._preview.update_projection(self.scene, self._observed_bbox, self._observed_points)
        self._preview_metrics.setText(self._preview._metrics)
        anchor = self._current_anchor()
        self._preview.set_selected_anchor(self.scene, anchor)

    def set_reference_context(self, reference, current):
        self._reference = reference
        self._current_composition = current
        self._update_reference_hypothesis()

    def clear_reference_context(self):
        self._reference = None
        self._current_composition = None
        self._hypothesis = None
        self._reference_state.setText("No reference camera context")
        self._reference_conf.setText("—")
        for label in (self._ref_distance, self._ref_delta, self._ref_yaw, self._ref_pitch, self._ref_focal, self._ref_support):
            label.setText("—")

    def _update_reference_hypothesis(self):
        if not hasattr(self, "_reference_state"):
            return
        anchor = self._current_anchor()
        if self._reference is None or self._current_composition is None:
            self.clear_reference_context() if self._reference is None and self._current_composition is None else None
            return
        self._hypothesis = estimate_reference_camera_hypothesis(
            self.scene,
            self._reference,
            self._current_composition,
            selected_anchor=anchor,
        )
        h = self._hypothesis
        if not h.success:
            self._reference_state.setText(h.message)
            self._reference_conf.setText("low confidence")
            return
        self._reference_state.setText("Reference context active")
        self._reference_conf.setText(f"{h.confidence:.0%}")
        self._ref_distance.setText(f"{h.reference_distance_m:.2f} m")
        self._ref_delta.setText(f"{h.distance_delta_m:+.2f} m")
        self._ref_yaw.setText(f"{h.reframe_yaw_deg:+.1f}°")
        self._ref_pitch.setText(f"{h.reframe_pitch_deg:+.1f}°")
        self._ref_focal.setText(f"{h.focal_length_mm:.1f} mm (same focal prior)")
        self._ref_support.setText(h.support + (f" · selected {h.anchor_name}" if h.anchor_name else ""))
        if self._reference_section is not None and not self._reference_section.button.isChecked():
            self._reference_section.button.setChecked(True)

    def _poll_reference_mode(self):
        window = self.window()
        widget = getattr(window, "_reference_mode", None)
        if widget is None:
            return
        ref = getattr(widget, "_reference", None)
        cur = None
        if ref is not None:
            try:
                from reverse_engineering.reference_reconstruction import build_reference_composition
                pose = getattr(widget, "_current_pose", None)
                image = getattr(widget, "_current_image", None)
                if pose is not None and image is not None:
                    cur = build_reference_composition(pose, image.shape[1], image.shape[0])
            except (AttributeError, TypeError, ValueError):
                cur = None
        signature = (id(ref), id(cur)) if ref is not None else None
        if signature == self._last_ref_signature:
            return
        self._last_ref_signature = signature
        if ref is None or cur is None:
            self.clear_reference_context()
        else:
            self.set_reference_context(ref, cur)

    def update_results(self, bundle):
        if getattr(bundle, "reverse_result", None):
            pose = getattr(bundle, "pose", None)
            if pose is not None:
                people = getattr(pose, "persons", None) or [pose]
                if self._source_image is not None:
                    tw, th = self._source_image.shape[1], self._source_image.shape[0]
                    people = [p.rescaled(tw, th) if (p.image_width, p.image_height) != (tw, th) else p for p in people]
                self._observed_points = [[[lm.x, lm.y] for lm in p.landmarks[:17]] for p in people]
                self._observed_bbox = getattr(people[0], "bbox", None)
            self.set_result(bundle.reverse_result, self._observed_bbox, self._observed_points)
        elif getattr(bundle, "pose", None):
            pose = bundle.pose
            people = getattr(pose, "persons", None) or [pose]
            if self._source_image is not None:
                tw, th = self._source_image.shape[1], self._source_image.shape[0]
                people = [p.rescaled(tw, th) if (p.image_width, p.image_height) != (tw, th) else p for p in people]
            self._observed_points = [[[lm.x, lm.y] for lm in p.landmarks[:17]] for p in people]
            self._observed_bbox = getattr(people[0], "bbox", None)
            self._refresh_projection()
