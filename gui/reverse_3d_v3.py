"""v3 reconstruction workspace entry point with semantic reference-line controls."""
from __future__ import annotations

from gui.reverse_3d_reference import (
    AnchorProjectionPreview,
    AnchorSceneView,
    CollapsibleSection,
    Reverse3DWorkspace as _BaseReverse3DWorkspace,
)
from gui.reverse_3d_reference_line import (
    CameraVisualMatchSection,
    ReferenceLineProjectionPreview,
    install_visual_camera_match,
)
from gui.reference_line_calibration import (
    CalibratedReferenceLinePreview,
    ReferenceLineCalibrationPanel,
    install_reference_line_calibration,
)
from reverse_engineering.reference_line_calibration import ReferenceLineConstraint


class Reverse3DWorkspace(_BaseReverse3DWorkspace):
    """Reference-aware v3 workspace with direct visual camera matching and line evidence."""

    def __init__(self, parent=None):
        super().__init__(parent)
        install_visual_camera_match(self)
        install_reference_line_calibration(self)
        self._sync_visual_camera_match()
        self._sync_anchor_reference_line()

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
        points = getattr(anchor, "image_points", ())
        preview.set_constraint(constraint)
        preview.set_evidence_points(points)
        self._reference_line_evidence = preview.evidence()

    def _poll_reference_mode(self):
        window = self.window()
        widget = getattr(window, "_reference_mode", None)
        if widget is None:
            return

        ref = getattr(widget, "_reference", None)
        pose = getattr(widget, "_current_pose", None)
        image = getattr(widget, "_current_image", None)
        signature = (
            id(ref),
            id(pose),
            tuple(image.shape[:2]) if image is not None else None,
        )
        if signature == self._last_ref_signature:
            return
        self._last_ref_signature = signature

        if ref is None or pose is None or image is None:
            self.clear_reference_context()
            return

        from reverse_engineering.reference_reconstruction import build_reference_composition
        if (
            int(getattr(pose, "image_width", 0) or 0) != int(image.shape[1])
            or int(getattr(pose, "image_height", 0) or 0) != int(image.shape[0])
        ):
            pose = pose.rescaled(int(image.shape[1]), int(image.shape[0]))
        current = build_reference_composition(pose, image.shape[1], image.shape[0])
        self.set_reference_context(ref, current)

    def _update_reference_hypothesis(self):
        """Extend the base hypothesis with explicit image-line evidence."""
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
        evidence = getattr(self, "_reference_line_evidence", None)
        self._hypothesis = estimate_reference_camera_hypothesis(
            self.scene,
            reference,
            current,
            selected_anchor=anchor,
            line_evidence=evidence,
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
            else:
                anchor.image_points = (tuple(evidence.p1), tuple(evidence.p2))
                anchor.reference_line_constraint = evidence.constraint.value
        self._update_reference_hypothesis()


__all__ = [
    "AnchorProjectionPreview",
    "AnchorSceneView",
    "CameraVisualMatchSection",
    "CalibratedReferenceLinePreview",
    "CollapsibleSection",
    "ReferenceLineCalibrationPanel",
    "ReferenceLineProjectionPreview",
    "Reverse3DWorkspace",
]
