"""Canonical v3 reconstruction workspace entry point."""
from __future__ import annotations

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QSplitter, QVBoxLayout

from gui.reverse_3d import ProjectionPreview
from gui.reverse_3d_workspace import (
    AnchorProjectionPreview as _BaseAnchorProjectionPreview,
    AnchorSceneView,
    CollapsibleSection,
    Reverse3DWorkspace as _BaseReverse3DWorkspace,
)
from gui.reverse_3d_reference_line import CameraVisualMatchSection, ReferenceLineProjectionPreview, install_visual_camera_match
from gui.reference_line_calibration import (
    CalibratedReferenceLinePreview,
    ReferenceLineCalibrationPanel,
    RollCorrectionController,
    install_reference_line_calibration,
    install_roll_correction,
)
from reverse_engineering.reference_line_calibration import ReferenceLineConstraint


class AnchorProjectionPreview(_BaseAnchorProjectionPreview):
    """Safe selected-anchor overlay for PySide6 and the live projection preview."""

    def paintEvent(self, event):
        # Call ProjectionPreview directly. The historical base implementation
        # used QRect.size().toSize(), which is invalid for PySide6's QSize.
        ProjectionPreview.paintEvent(self, event)

        if self._scene is None or self._selected_anchor is None or self._pixmap is None:
            return

        w, h = self._pixmap.width(), self._pixmap.height()
        points = self._project_selected(w, h)
        if points is None or len(points) == 0:
            return

        area = self.rect().adjusted(6, 6, -6, -38)
        target_size = area.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if scaled.isNull():
            return

        ox = area.x() + (area.width() - scaled.width()) * 0.5
        oy = area.y() + (area.height() - scaled.height()) * 0.5
        sx = scaled.width() / max(w, 1)
        sy = scaled.height() / max(h, 1)
        qpoints = [QPointF(ox + p[0] * sx, oy + p[1] * sy) for p in points]

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        plane = getattr(self._selected_anchor.kind, "value", None) == "plane"
        painter.setPen(QPen(QColor("#D97706"), 2.4 if plane else 2.2, Qt.DashLine))
        painter.setBrush(QBrush(QColor(217, 119, 6, 35)))

        if plane and len(qpoints) >= 3:
            painter.drawPolygon(QPolygonF(qpoints))
            painter.setPen(QPen(QColor("#F59E0B"), 2))
            for point in qpoints:
                painter.drawEllipse(point, 5, 5)
        else:
            point = qpoints[0]
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            painter.drawEllipse(point, 6, 6)
            painter.drawLine(point.x() - 10, point.y(), point.x() + 10, point.y())
            painter.drawLine(point.x(), point.y() - 10, point.x(), point.y() + 10)

        painter.setPen(QPen(QColor("#D97706"), 2))
        painter.drawText(qpoints[0] + QPointF(8, -9), f"SELECTED · {self._selected_anchor.name}")
        painter.end()


class Reverse3DWorkspace(_BaseReverse3DWorkspace):
    """Reference-aware v3 workspace with parallel 3D, 2D, and inspector columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        install_visual_camera_match(self)
        install_reference_line_calibration(self)
        install_roll_correction(self)
        panel = getattr(self, "_reference_line_calibration", None)
        if panel is not None:
            panel.evidence_changed.connect(self._on_reference_line_evidence_changed)
        self._projection_panel = None
        self._move_projection_preview_next_to_3d()
        self._sync_visual_camera_match()
        self._sync_anchor_reference_line()

    def _move_projection_preview_next_to_3d(self):
        """Make 3D, live 2D projection, and parameter inspector visible together."""
        preview = getattr(self, "_preview", None)
        splitter = self.findChild(QSplitter)
        if preview is None or splitter is None or splitter.count() < 2:
            return

        old_body = preview.parentWidget()
        old_section = old_body.parentWidget() if old_body is not None else None
        if old_body is not None:
            old_layout = old_body.layout()
            if old_layout is not None:
                old_layout.removeWidget(preview)
                metrics = getattr(self, "_preview_metrics", None)
                if metrics is not None:
                    old_layout.removeWidget(metrics)
        if isinstance(old_section, CollapsibleSection):
            old_section.setVisible(False)

        inspector = splitter.widget(1)
        if inspector is not None:
            inspector.setParent(None)

        projection = QFrame()
        projection.setObjectName("projectionPanel")
        projection.setFrameShape(QFrame.StyledPanel)
        projection.setMinimumWidth(280)
        projection.setMaximumWidth(360)
        projection.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        projection.setStyleSheet(
            "#projectionPanel { background:#0F172A; border:1px solid #CBD5E1; border-radius:8px; }"
            "#projectionPanel QLabel { background:transparent; color:#E2E8F0; }"
        )
        projection_layout = QVBoxLayout(projection)
        projection_layout.setContentsMargins(8, 7, 8, 7)
        projection_layout.setSpacing(4)

        header = QLabel("2D Projection Preview")
        header.setStyleSheet("font-weight:600; color:#F8FAFC;")
        projection_layout.addWidget(header, 0)

        preview.setParent(projection)
        preview.setMinimumHeight(220)
        preview.setMaximumHeight(420)
        preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        projection_layout.addWidget(preview, 1)

        metrics = getattr(self, "_preview_metrics", None)
        if metrics is not None:
            metrics.setParent(projection)
            metrics.setMaximumHeight(42)
            metrics.setWordWrap(True)
            projection_layout.addWidget(metrics, 0)

        splitter.addWidget(projection)
        if inspector is not None:
            splitter.addWidget(inspector)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([670, 320, 410])
        self._projection_panel = projection

    @property
    def scene_model(self):
        return self.scene

    @property
    def source_image(self):
        return self._source_image

    def set_source_image(self, image):
        self._source_image = image
        preview = getattr(self, "_preview", None)
        if preview is not None and hasattr(preview, "set_image"):
            preview.set_image(image)
        self._refresh_projection()

    def refresh_scene_view(self):
        self._view.update()
        self._refresh_projection()

    def set_camera_value(self, parameter: str, value: float):
        controls = {
            "distance": "_distance", "height": "_height", "yaw": "_yaw",
            "pitch": "_pitch", "roll": "_roll", "focal_length_mm": "_focal",
        }
        control_name = controls.get(parameter)
        if control_name is None or not hasattr(self, control_name):
            raise ValueError(f"unsupported camera parameter: {parameter}")
        getattr(self, control_name).setValue(float(value))

    def _sync_visual_camera_match(self):
        section = getattr(self, "_camera_visual_match", None)
        if section is not None:
            section.sync_from_camera()

    def _sync_controls(self):
        super()._sync_controls()
        self._sync_visual_camera_match()

    def _camera_spin_changed(self):
        super()._camera_spin_changed()
        self._sync_visual_camera_match()

    def _select_candidate(self, row):
        super()._select_candidate(row)
        self._sync_visual_camera_match()

    def _anchor_selected(self, row):
        super()._anchor_selected(row)
        self._sync_anchor_reference_line()
        self._update_reference_hypothesis()

    def _sync_anchor_reference_line(self):
        preview = getattr(self, "_preview", None)
        if not isinstance(preview, CalibratedReferenceLinePreview):
            return
        anchor = self._current_anchor()
        if anchor is None:
            preview.clear_evidence()
            return
        constraint_value = getattr(anchor, "reference_line_constraint", ReferenceLineConstraint.FREE.value)
        try:
            constraint = ReferenceLineConstraint(constraint_value)
        except ValueError:
            constraint = ReferenceLineConstraint.FREE
        panel = getattr(self, "_reference_line_calibration", None)
        if panel is not None:
            panel.constraint.blockSignals(True)
            panel.constraint.setCurrentIndex(panel.constraint.findData(constraint))
            panel.constraint.blockSignals(False)
        preview.set_constraint(constraint)
        preview.set_evidence_points(getattr(anchor, "image_points", ()))
        self._reference_line_evidence = preview.evidence()
        controller = getattr(panel, "_roll_apply_controller", None) if panel is not None else None
        if controller is not None:
            controller.refresh(self._reference_line_evidence)

    def _poll_reference_mode(self):
        window = self.window()
        widget = getattr(window, "_reference_mode", None)
        if widget is None:
            return
        ref = getattr(widget, "_reference", None)
        pose = getattr(widget, "_current_pose", None)
        image = getattr(window, "_img", None)
        if image is None:
            image = getattr(widget, "_current_image", None)
        signature = (id(ref), id(pose), id(image), tuple(image.shape[:2]) if image is not None else None)
        if signature == self._last_ref_signature:
            return
        self._last_ref_signature = signature
        if image is not None:
            self.set_source_image(image)
        if ref is None or pose is None or image is None:
            self.clear_reference_context()
            return
        from reverse_engineering.reference_reconstruction import build_reference_composition
        if int(getattr(pose, "image_width", 0) or 0) != int(image.shape[1]) or int(getattr(pose, "image_height", 0) or 0) != int(image.shape[0]):
            pose = pose.rescaled(int(image.shape[1]), int(image.shape[0]))
        current = build_reference_composition(pose, image.shape[1], image.shape[0])
        self.set_reference_context(ref, current)

    def update_results(self, bundle):
        """Keep the live preview synchronized with the application image."""
        window = self.window()
        image = getattr(window, "_img", None)
        if image is None:
            reference_mode = getattr(window, "_reference_mode", None)
            image = getattr(reference_mode, "_current_image", None)
        if image is not None and image is not self._source_image:
            self.set_source_image(image)
        return super().update_results(bundle)

    def _update_reference_hypothesis(self):
        if not hasattr(self, "_reference_state"):
            return
        anchor = self._current_anchor()
        reference = getattr(self, "_reference", None)
        current = getattr(self, "_current_composition", None)
        if reference is None or current is None:
            if reference is None and current is None:
                self.clear_reference_context()
            return
        from reverse_engineering.reference_camera import estimate_reference_camera_hypothesis
        self._hypothesis = estimate_reference_camera_hypothesis(
            self.scene,
            reference,
            current,
            selected_anchor=anchor,
            line_evidence=getattr(self, "_reference_line_evidence", None),
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
        support = h.support
        if h.line_observed_angle_deg is not None:
            support += f" · line obs {h.line_observed_angle_deg:+.1f}°"
        if h.roll_correction_deg is not None:
            support += f" · roll {h.roll_correction_deg:+.1f}° ({h.line_constraint})"
        self._ref_support.setText(support + (f" · selected {h.anchor_name}" if h.anchor_name else ""))
        if self._reference_section is not None and not self._reference_section.button.isChecked():
            self._reference_section.button.setChecked(True)

    def _on_reference_line_evidence_changed(self, evidence):
        self._reference_line_evidence = evidence
        anchor = self._current_anchor()
        if anchor is not None:
            if evidence is None:
                anchor.image_points = ()
                anchor.reference_line_constraint = ReferenceLineConstraint.FREE.value
            else:
                anchor.image_points = (tuple(evidence.p1), tuple(evidence.p2))
                anchor.reference_line_constraint = evidence.constraint.value
        panel = getattr(self, "_reference_line_calibration", None)
        controller = getattr(panel, "_roll_apply_controller", None) if panel is not None else None
        if controller is not None:
            controller.refresh(evidence)
        self._update_reference_hypothesis()
