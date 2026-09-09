"""Camera correction actions for image-space reference-line evidence."""
from __future__ import annotations

from typing import Optional
import math

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton

from reverse_engineering.reference_line_calibration import ReferenceLineEvidence, angular_error_deg, line_angle_deg


class RollCorrectionController:
    """Apply/undo a semantic reference-line roll correction on a workspace."""

    def __init__(self, workspace, panel):
        self.workspace = workspace
        self.panel = panel
        self._previous_roll: Optional[float] = None
        self._previous_residual: Optional[float] = None
        self._applied = False
        self._build_controls()

    def _build_controls(self):
        root = self.panel.layout()
        actions = QHBoxLayout()
        self.apply_button = QPushButton("Apply roll correction")
        self.undo_button = QPushButton("Undo")
        self.apply_button.clicked.connect(self.apply)
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setEnabled(False)
        actions.addWidget(self.apply_button)
        actions.addWidget(self.undo_button)
        root.addLayout(actions)

        self.status = QLabel("No camera correction applied")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#475569; font-size:9pt;")
        root.addWidget(self.status)
        self.compare = QLabel("Projection residual: —")
        self.compare.setWordWrap(True)
        self.compare.setStyleSheet("color:#475569; font-size:9pt;")
        root.addWidget(self.compare)
        self.panel._roll_apply_controller = self

    def _current_evidence(self) -> Optional[ReferenceLineEvidence]:
        evidence = getattr(self.workspace, "_reference_line_evidence", None)
        return evidence if isinstance(evidence, ReferenceLineEvidence) else None

    def _projected_residual(self, evidence: Optional[ReferenceLineEvidence]) -> Optional[float]:
        if evidence is None:
            return None
        preview = getattr(self.workspace, "_preview", None)
        pixmap = getattr(preview, "_pixmap", None)
        project_line = getattr(preview, "_project_line", None)
        if pixmap is None or getattr(pixmap, "isNull", lambda: True)() or not callable(project_line):
            return None
        try:
            projected = project_line(pixmap.width(), pixmap.height())
            if projected is None or len(projected) != 2:
                return None
            predicted_angle = line_angle_deg(tuple(projected[0]), tuple(projected[1]))
            return abs(float(angular_error_deg(evidence.observed_angle_deg, predicted_angle)))
        except (TypeError, ValueError, AttributeError, OverflowError):
            return None

    def apply(self):
        evidence = self._current_evidence()
        correction = evidence.correction_deg if evidence is not None else None
        if correction is None:
            self.status.setText("需要 Horizontal 或 Vertical 的两点参考线后才能应用 Roll correction。")
            return

        current_roll = float(self.workspace._roll.value())
        if not self._applied:
            self._previous_roll = current_roll
            self._previous_residual = self._projected_residual(evidence)
        new_roll = current_roll + float(correction)
        self.workspace._roll.setValue(new_roll)
        try:
            self.workspace._refresh_projection()
        except AttributeError:
            pass
        self._applied = True
        self.apply_button.setEnabled(False)
        self.undo_button.setEnabled(True)
        after_residual = self._projected_residual(evidence)
        self.status.setText(
            f"Applied roll correction {correction:+.1f}° · "
            f"Roll {current_roll:+.1f}° → {new_roll:+.1f}°"
        )
        if self._previous_residual is not None and after_residual is not None:
            delta = after_residual - self._previous_residual
            trend = "improved" if delta < -1e-6 else "worsened" if delta > 1e-6 else "unchanged"
            self.compare.setText(
                f"Projection residual {self._previous_residual:.1f}° → {after_residual:.1f}° · {trend} ({delta:+.1f}°)"
            )
        else:
            self.compare.setText("Projection residual: unavailable until a projected plane line is visible.")

    def undo(self):
        if not self._applied or self._previous_roll is None:
            return
        evidence = self._current_evidence()
        previous = self._previous_roll
        self.workspace._roll.setValue(previous)
        try:
            self.workspace._refresh_projection()
        except AttributeError:
            pass
        restored_residual = self._projected_residual(evidence)
        self._previous_roll = None
        self._previous_residual = None
        self._applied = False
        self.apply_button.setEnabled(evidence is not None and evidence.correction_deg is not None)
        self.undo_button.setEnabled(False)
        self.status.setText(f"Restored roll to {previous:+.1f}°")
        self.compare.setText(
            f"Projection residual restored to {restored_residual:.1f}°" if restored_residual is not None else "Projection residual: —"
        )

    def refresh(self, evidence):
        if evidence is None or evidence.correction_deg is None:
            self.apply_button.setEnabled(False)
            if not self._applied:
                self.status.setText("需要 Horizontal 或 Vertical 的两点参考线后才能应用 Roll correction。")
                self.compare.setText("Projection residual: —")
            return
        if not self._applied:
            self.apply_button.setEnabled(True)
            self.status.setText(
                f"Suggested roll correction {evidence.correction_deg:+.1f}° · "
                f"confidence {evidence.confidence:.0%}"
            )
            residual = self._projected_residual(evidence)
            self.compare.setText(
                f"Projection residual: {residual:.1f}°" if residual is not None else "Projection residual: unavailable"
            )


def install_roll_correction(workspace):
    """Attach apply/undo actions to the existing reference-line calibration panel."""
    panel = getattr(workspace, "_reference_line_calibration", None)
    if panel is None:
        return None
    controller = RollCorrectionController(workspace, panel)
    panel.evidence_changed.connect(controller.refresh)
    evidence = getattr(workspace, "_reference_line_evidence", None)
    controller.refresh(evidence)
    return controller
