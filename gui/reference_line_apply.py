"""Camera correction actions for image-space reference-line evidence."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton

from reverse_engineering.reference_line_calibration import ReferenceLineEvidence


class RollCorrectionController:
    """Apply/undo a semantic reference-line roll correction on a workspace."""

    def __init__(self, workspace, panel):
        self.workspace = workspace
        self.panel = panel
        self._previous_roll: Optional[float] = None
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
        self.panel._roll_apply_controller = self

    def _current_evidence(self) -> Optional[ReferenceLineEvidence]:
        evidence = getattr(self.workspace, "_reference_line_evidence", None)
        return evidence if isinstance(evidence, ReferenceLineEvidence) else None

    def apply(self):
        evidence = self._current_evidence()
        correction = evidence.correction_deg if evidence is not None else None
        if correction is None:
            self.status.setText("需要 Horizontal 或 Vertical 的两点参考线后才能应用 Roll correction。")
            return

        current_roll = float(self.workspace._roll.value())
        if not self._applied:
            self._previous_roll = current_roll
        new_roll = current_roll + float(correction)
        # Keep the change inside the same camera-edit path as manual controls.
        self.workspace._roll.setValue(new_roll)
        self._applied = True
        self.apply_button.setEnabled(False)
        self.undo_button.setEnabled(True)
        self.status.setText(
            f"Applied roll correction {correction:+.1f}° · "
            f"Roll {current_roll:+.1f}° → {new_roll:+.1f}°"
        )

    def undo(self):
        if not self._applied or self._previous_roll is None:
            return
        previous = self._previous_roll
        self.workspace._roll.setValue(previous)
        self._previous_roll = None
        self._applied = False
        self.apply_button.setEnabled(True)
        self.undo_button.setEnabled(False)
        self.status.setText(f"Restored roll to {previous:+.1f}°")

    def refresh(self, evidence):
        if evidence is None or evidence.correction_deg is None:
            self.apply_button.setEnabled(False)
            if not self._applied:
                self.status.setText("需要 Horizontal 或 Vertical 的两点参考线后才能应用 Roll correction。")
            return
        if not self._applied:
            self.apply_button.setEnabled(True)
            self.status.setText(
                f"Suggested roll correction {evidence.correction_deg:+.1f}° · "
                f"confidence {evidence.confidence:.0%}"
            )


def install_roll_correction(workspace):
    """Attach apply/undo actions to the existing reference-line calibration panel."""
    panel = getattr(workspace, "_reference_line_calibration", None)
    if panel is None:
        return None
    controller = RollCorrectionController(workspace, panel)
    previous_handler = getattr(panel, "_on_evidence", None)

    def on_evidence(evidence):
        if previous_handler is not None:
            previous_handler(evidence)
        controller.refresh(evidence)

    # Keep the existing signal fan-out intact; add a second observer rather than
    # replacing the panel's internal evidence handling.
    panel.evidence_changed.connect(controller.refresh)
    evidence = getattr(workspace, "_reference_line_evidence", None)
    controller.refresh(evidence)
    return controller
