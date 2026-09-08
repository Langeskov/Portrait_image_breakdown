"""Compact v3 reconstruction workspace.

Keeps the existing renderer/projection implementation while moving the
control surface into a scrollable, collapsible inspector. Editable scene
anchors form the first v3 Phase 2.4 room/object reconstruction layer.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton, QScrollArea,
    QGroupBox, QFormLayout, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QPushButton, QComboBox, QSizePolicy, QSplitter, QFrame,
)

from gui.reverse_3d import SceneView, ProjectionPreview
from reverse_engineering.scene import SceneCamera, SceneModel
from reverse_engineering.scene_anchors import AnchorKind, SceneAnchor


ACCENT = QColor("#2563EB")
TEXT = QColor("#334155")
PLANE = QColor(37, 99, 235, 55)
ANCHOR = QColor("#D97706")


class CollapsibleSection(QWidget):
    """Compact inspector section that releases vertical space when closed."""

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
        self.button.setStyleSheet(
            "QToolButton { border: 1px solid #D9DDE3; border-radius: 5px; "
            "padding: 6px 8px; background: #FAFBFC; color: #1F2937; "
            "text-align: left; font-weight: 600; } "
            "QToolButton:hover { background: #F3F4F6; }"
        )
        root.addWidget(self.button)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 6, 8, 6)
        self.body_layout.setSpacing(7)
        root.addWidget(self.body)
        self.body.setVisible(expanded)
        self.button.toggled.connect(self._toggle)

    def _toggle(self, checked: bool):
        self.body.setVisible(checked)
        self.button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)


class AnchorSceneView(SceneView):
    """Existing 3D renderer plus lightweight editable-anchor overlays."""

    def paintEvent(self, event):
        super().paintEvent(event)
        anchors = [a for a in getattr(self.scene, "anchors", ()) if a.enabled]
        if not anchors:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for anchor in anchors:
            center = self._project(anchor.position)
            if anchor.kind == AnchorKind.PLANE:
                corners = anchor.corners()
                if len(corners) == 4:
                    painter.setPen(QPen(ACCENT, 1.4, Qt.DashLine))
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


class Reverse3DWorkspace(QWidget):
    """Scrollable compact reconstruction inspector for v3."""

    camera_edited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = SceneModel()
        self._result = None
        self._source_image = None
        self._observed_bbox = None
        self._observed_points = []
        self._anchor_ids: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

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
        splitter.setSizes([980, 390])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

    def _build_inspector(self):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setFrameShape(QFrame.NoFrame)
        outer.setMinimumWidth(360)
        outer.setStyleSheet("QScrollArea { background: #F8FAFC; border: 0; }")

        panel = QWidget()
        panel.setMinimumWidth(340)
        lo = QVBoxLayout(panel)
        lo.setContentsMargins(10, 10, 10, 14)
        lo.setSpacing(8)

        title = QLabel("Camera & scene reconstruction")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        lo.addWidget(title)
        self._confidence = QLabel("No reverse-engineering result yet")
        self._confidence.setStyleSheet("color:#64748B;")
        self._confidence.setWordWrap(True)
        lo.addWidget(self._confidence)

        lo.addWidget(self._build_camera_section())
        lo.addWidget(self._build_anchor_section())
        lo.addWidget(self._build_people_section())
        lo.addWidget(self._build_projection_section())
        lo.addWidget(self._build_candidates_section())

        self._note = QLabel(
            "v3 Phase 2.4: scene anchors are an editable coordinate scaffold. "
            "They remain separate from observed image evidence until explicitly bound."
        )
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color:#64748B; font-size:9pt;")
        lo.addWidget(self._note)
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
            box.setToolTip("鼠标滚轮可微调；修改后实时更新 2D / 3D 投影。")
        for label, box in (
            ("Distance", self._distance), ("Height", self._height),
            ("Yaw", self._yaw), ("Pitch", self._pitch),
            ("Roll", self._roll), ("Focal", self._focal),
        ):
            form.addRow(label, box)
        section.body_layout.addWidget(group)
        return section

    def _build_anchor_section(self):
        section = CollapsibleSection("Scene anchors", expanded=True)
        hint = QLabel(
            "用于建立墙、地面、桌面等可编辑场景基准。默认 Ground plane 只是坐标框架，不是检测结论。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748B; font-size:9pt;")
        section.body_layout.addWidget(hint)

        self._anchors = QListWidget()
        self._anchors.setMaximumHeight(110)
        self._anchors.currentRowChanged.connect(self._anchor_selected)
        section.body_layout.addWidget(self._anchors)

        buttons = QHBoxLayout()
        add_point = QPushButton("+ Point")
        add_plane = QPushButton("+ Plane")
        remove = QPushButton("Remove")
        add_point.clicked.connect(lambda: self._add_anchor(AnchorKind.POINT))
        add_plane.clicked.connect(lambda: self._add_anchor(AnchorKind.PLANE))
        remove.clicked.connect(self._remove_anchor)
        buttons.addWidget(add_point)
        buttons.addWidget(add_plane)
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
        form.addRow("Name", self._anchor_name)
        form.addRow("Kind", self._anchor_kind)
        form.addRow("X", self._ax)
        form.addRow("Y", self._ay)
        form.addRow("Z", self._az)
        form.addRow("Normal X", self._nx)
        form.addRow("Normal Y", self._ny)
        form.addRow("Normal Z", self._nz)
        form.addRow("Width", self._aw)
        form.addRow("Height", self._ah)
        section.body_layout.addWidget(editor)
        return section

    def _build_people_section(self):
        section = CollapsibleSection("Scene people", expanded=False)
        self._people = QListWidget()
        self._people.setMaximumHeight(120)
        section.body_layout.addWidget(self._people)
        hint = QLabel("附加人物仅在存在独立深度证据时才获得 relative-3D；否则标为 2D-only。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748B; font-size:9pt;")
        section.body_layout.addWidget(hint)
        return section

    def _build_projection_section(self):
        section = CollapsibleSection("2D projection preview", expanded=True)
        self._preview = ProjectionPreview()
        self._preview.setMinimumHeight(230)
        section.body_layout.addWidget(self._preview)
        self._preview_metrics = QLabel("No projection yet")
        self._preview_metrics.setWordWrap(True)
        self._preview_metrics.setStyleSheet("color:#475569; font-size:9pt;")
        section.body_layout.addWidget(self._preview_metrics)
        return section

    def _build_candidates_section(self):
        section = CollapsibleSection("Candidate solutions", expanded=False)
        self._candidates = QListWidget()
        self._candidates.setMaximumHeight(150)
        self._candidates.currentRowChanged.connect(self._select_candidate)
        section.body_layout.addWidget(self._candidates)
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
            self._confidence.setText(
                f"Overall confidence: {result.overall_confidence:.0%} · People: {count} · Relative 3D: {rel3d}/{count}"
            )
        else:
            self._confidence.setText("No reverse-engineering result yet")
        self._refresh_projection()

    def _sync_controls(self):
        c = self.scene.camera
        for box, value in (
            (self._distance, c.distance), (self._height, c.height),
            (self._yaw, c.yaw), (self._pitch, c.pitch),
            (self._roll, c.roll), (self._focal, c.focal_length_mm),
        ):
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
            state = "enabled" if anchor.enabled else "hidden"
            self._anchor_ids.append(anchor.anchor_id)
            self._anchors.addItem(QListWidgetItem(f"{anchor.name} · {kind} · {state}"))
        self._anchors.blockSignals(False)
        if self._anchor_ids:
            row = self._anchor_ids.index(selected_id) if selected_id in self._anchor_ids else 0
            self._anchors.setCurrentRow(row)
        else:
            self._anchor_selected(-1)

    def _anchor_selected(self, row):
        if not 0 <= row < len(self._anchor_ids):
            self._anchor_name.setText("—")
            return
        anchor = self.scene.anchor_by_id(self._anchor_ids[row])
        if anchor is None:
            return
        self._anchor_name.setText(anchor.name)
        self._anchor_kind.blockSignals(True)
        self._anchor_kind.setCurrentText(anchor.kind.value)
        self._anchor_kind.blockSignals(False)
        values = (
            (self._ax, anchor.position[0]), (self._ay, anchor.position[1]), (self._az, anchor.position[2]),
            (self._nx, anchor.normal[0]), (self._ny, anchor.normal[1]), (self._nz, anchor.normal[2]),
            (self._aw, anchor.size[0]), (self._ah, anchor.size[1]),
        )
        for box, value in values:
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)
        plane = anchor.kind == AnchorKind.PLANE
        for box in (self._nx, self._ny, self._nz, self._aw, self._ah):
            box.setEnabled(plane)

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
        self.camera_edited.emit()

    def _refresh_projection(self):
        if not hasattr(self, "_preview"):
            return
        self._preview.update_projection(self.scene, self._observed_bbox, self._observed_points)
        self._preview_metrics.setText(self._preview._metrics)

    def update_results(self, bundle):
        if getattr(bundle, "reverse_result", None):
            pose = getattr(bundle, "pose", None)
            if pose is not None:
                people = getattr(pose, "persons", None) or [pose]
                if self._source_image is not None:
                    tw, th = self._source_image.shape[1], self._source_image.shape[0]
                    people = [p.rescaled(tw, th) if (p.image_width, p.image_height) != (tw, th) else p for p in people]
                self._observed_points = [
                    [[lm.x, lm.y] for lm in p.landmarks[:17]] for p in people
                ]
                self._observed_bbox = getattr(people[0], "bbox", None)
            self.set_result(bundle.reverse_result, self._observed_bbox, self._observed_points)
        elif getattr(bundle, "pose", None):
            pose = bundle.pose
            people = getattr(pose, "persons", None) or [pose]
            if self._source_image is not None:
                tw, th = self._source_image.shape[1], self._source_image.shape[0]
                people = [p.rescaled(tw, th) if (p.image_width, p.image_height) != (tw, th) else p for p in people]
            self._observed_points = [
                [[lm.x, lm.y] for lm in p.landmarks[:17]] for p in people
            ]
            self._observed_bbox = getattr(people[0], "bbox", None)
            self._refresh_projection()
