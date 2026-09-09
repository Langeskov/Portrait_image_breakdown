"""Canonical v3 reconstruction workspace entry point."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout

from gui.reverse_3d_reference import AnchorProjectionPreview, AnchorSceneView, CollapsibleSection, Reverse3DWorkspace as _BaseReverse3DWorkspace
from gui.reverse_3d_reference_line import CameraVisualMatchSection, ReferenceLineProjectionPreview, install_visual_camera_match
from gui.reference_line_calibration import CalibratedReferenceLinePreview, ReferenceLineCalibrationPanel, install_reference_line_calibration
from gui.reference_line_apply import RollCorrectionController, install_roll_correction
from reverse_engineering.reference_line_calibration import ReferenceLineConstraint


class Reverse3DWorkspace(_BaseReverse3DWorkspace):
    """Reference-aware v3 workspace with direct visual camera matching and line evidence.

    Public methods below form the application boundary. Other widgets should not
    reach into this workspace's private controls or renderer state.

    The 2D projection preview is intentionally presented as a parallel view over
    the 3D canvas. Camera/anchor/candidate/reference controls remain consolidated
    in the right-side inspector.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        install_visual_camera_match(self)
        install_reference_line_calibration(self)
        install_roll_correction(self)
        panel = getattr(self, "_reference_line_calibration", None)
        if panel is not None:
            panel.evidence_changed.connect(self._on_reference_line_evidence_changed)
        self._projection_overlay = None
        self._move_projection_preview_to_canvas()
        self._sync_visual_camera_match()
        self._sync_anchor_reference_line()

    def _move_projection_preview_to_canvas(self):
        """Move the 2D preview out of the inspector into the 3D canvas corner."""
        preview = getattr(self, "_preview", None)
        if preview is None:
            return

        old_body = preview.parentWidget()
        if old_body is not None:
            old_layout = old_body.layout()
            if old_layout is not None:
                old_layout.removeWidget(preview)
                metrics = getattr(self, "_preview_metrics", None)
                if metrics is not None:
                    old_layout.removeWidget(metrics)

            section = old_body.parentWidget()
            if isinstance(section, CollapsibleSection):
                section.setVisible(False)

        from PySide6.QtWidgets import QSplitter
        splitter = self.findChild(QSplitter)
        if splitter is None or splitter.count() < 1:
            return
        canvas = splitter.widget(0)
        if canvas is None:
            return

        overlay = QFrame(canvas)
        overlay.setObjectName("projectionOverlay")
        overlay.setFrameShape(QFrame.StyledPanel)
        overlay.setFrameShadow(QFrame.Raised)
        overlay.setStyleSheet(
            "#projectionOverlay { background:#0F172A; border:1px solid #CBD5E1; "
            "border-radius:8px; }"
            "#projectionOverlay QLabel { color:#E2E8F0; background:transparent; }"
        )
        overlay.setMinimumSize(330, 230)
        overlay.setMaximumSize(460, 340)
        overlay.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QVBoxLayout(overlay)
        layout.setContentsMargins(7, 6, 7, 6)
        layout.setSpacing(4)

        header = QLabel("2D Projection Preview")
        header.setFont(self._view.font())
        header.setStyleSheet("font-weight:600; color:#F8FAFC;")
        layout.addWidget(header, 0)

        preview.setParent(overlay)
        preview.setMinimumHeight(200)
        preview.setMaximumHeight(290)
        preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(preview, 1)

        metrics = getattr(self, "_preview_metrics", None)
        if metrics is not None:
            metrics.setParent(overlay)
            metrics.setMaximumHeight(34)
            metrics.setWordWrap(True)
            layout.addWidget(metrics, 0)

        overlay.raise_()
        overlay.show()
        self._projection_overlay = overlay
        self._position_projection_overlay()

    def _position_projection_overlay(self):
        overlay = self._projection_overlay
        if overlay is None:
            return
        parent = overlay.parentWidget()
        if parent is None:
            return
        margin = 14
        width = min(440, max(330, int(parent.width() * 0.34)))
        width = min(width, max(330, parent.width() - margin * 2))
        height = min(325, max(235, int(parent.height() * 0.38)))
        height = min(height, max(235, parent.height() - 92))
        overlay.resize(width, height)
        overlay.move(max(margin, parent.width() - width - margin), margin)
        overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_projection_overlay()

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
        self._position_projection_overlay()

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
        image = getattr(widget, "_current_image", None)
        signature = (id(ref), id(pose), tuple(image.shape[:2]) if image is not None else None)
        if signature == self._last_ref_signature:
            return
        self._last_ref_signature = signature
        if ref is None or pose is None or image is None:
            self.clear_reference_context()
            return
        from reverse_engineering.reference_reconstruction import build_reference_composition
        if int(getattr(pose, "image_width", 0) or 0) != int(image.shape[1]) or int(getattr(pose, "image_height", 0) or 0) != int(image.shape[0]):
            pose = pose.rescaled(int(image.shape[1]), int(image.shape[0]))
        current = build_reference_composition(pose, image.shape[1], image.shape[0])
        self.set_reference_context(ref, current)

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
        self._hypothesis = estimate_reference_camera_hypothesis(self.scene, reference, current, selected_anchor=anchor, line_evidence=getattr(self, "_reference_line_evidence", None))
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
        if h.line_observed_angle_deg is not None: support += f" · line obs {h.line_observed_angle_deg:+.1f}°"
        if h.roll_correction_deg is not None: support += f" · roll {h.roll_correction_deg:+.1f}° ({h.line_constraint})"
        self._ref_support.setText(support + (f" · selected {h.anchor_name}" if h.anchor_name else ""))
        if self._reference_section is not None and not self._reference_section.button.isChecked(): self._reference_section.button.setChecked(True)

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
        if controller is not None: controller.refresh(evidence)
        self._update_reference_hypothesis()


__all__ = ["AnchorProjectionPreview", "AnchorSceneView", "CameraVisualMatchSection", "CalibratedReferenceLinePreview", "CollapsibleSection", "ReferenceLineCalibrationPanel", "ReferenceLineProjectionPreview", "Reverse3DWorkspace", "RollCorrectionController"]
