"""Direct image-space reference-line calibration widgets for v3."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QSize, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.reverse_3d_reference_line import ReferenceLineProjectionPreview
from reverse_engineering.reference_line_calibration import (
    ReferenceLineConstraint,
    ReferenceLineEvidence,
    angular_error_deg,
    line_angle_deg,
)
from reverse_engineering.scene import SceneModel
from reverse_engineering.scene_anchors import AnchorKind, SceneAnchor

OBSERVED_LINE = QColor("#22C55E")
TARGET_LINE = QColor("#7C3AED")
POINT = QColor("#F59E0B")
MUTED = QColor("#64748B")


class CalibratedReferenceLinePreview(ReferenceLineProjectionPreview):
    """Projection preview with user-placed image evidence for the selected anchor."""

    evidence_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._constraint = ReferenceLineConstraint.FREE
        self._manual_points: list[tuple[float, float]] = []
        self._drag_index: Optional[int] = None
        self.setMouseTracking(True)
        self.setToolTip("单击两次选取照片中的参考线两端；拖动端点可微调。")

    def set_constraint(self, constraint: ReferenceLineConstraint):
        self._constraint = constraint
        self._emit_evidence()
        self.update()

    def constraint(self) -> ReferenceLineConstraint:
        return self._constraint

    def set_evidence_points(self, points):
        self._manual_points = [tuple(map(float, p)) for p in (points or [])][:2]
        self._emit_evidence()
        self.update()

    def evidence(self) -> Optional[ReferenceLineEvidence]:
        if len(self._manual_points) != 2:
            return None
        try:
            return ReferenceLineEvidence(
                self._manual_points[0], self._manual_points[1], self._constraint
            )
        except ValueError:
            return None

    def clear_evidence(self):
        self._manual_points = []
        self._drag_index = None
        self._emit_evidence()
        self.update()

    def _emit_evidence(self):
        self.evidence_changed.emit(self.evidence())

    def _image_layout(self):
        if self._pixmap is None:
            return None
        area = QRectF(6, 6, self.width() - 12, self.height() - 38)
        size = QSize(max(1, int(area.width())), max(1, int(area.height())))
        scaled = self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = area.x() + (area.width() - scaled.width()) * 0.5
        oy = area.y() + (area.height() - scaled.height()) * 0.5
        sx = scaled.width() / max(self._pixmap.width(), 1)
        sy = scaled.height() / max(self._pixmap.height(), 1)
        return area, scaled, ox, oy, sx, sy

    def _widget_to_image(self, pos: QPointF):
        layout = self._image_layout()
        if layout is None:
            return None
        _, _, ox, oy, sx, sy = layout
        x = (float(pos.x()) - ox) / max(sx, 1e-9)
        y = (float(pos.y()) - oy) / max(sy, 1e-9)
        if not (0 <= x <= self._pixmap.width() - 1 and 0 <= y <= self._pixmap.height() - 1):
            return None
        return (x, y)

    def _image_to_widget(self, point):
        layout = self._image_layout()
        if layout is None:
            return None
        _, _, ox, oy, sx, sy = layout
        return QPointF(ox + point[0] * sx, oy + point[1] * sy)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._pixmap is None:
            return super().mousePressEvent(event)
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return

        if len(self._manual_points) < 2:
            self._manual_points.append(image_point)
            if len(self._manual_points) == 2:
                self._emit_evidence()
            self.update()
            return

        self._drag_index = min(
            range(2),
            key=lambda i: np.linalg.norm(
                np.asarray(self._manual_points[i]) - np.asarray(image_point)
            ),
        )
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_index is None:
            return
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return
        self._manual_points[self._drag_index] = image_point
        self._emit_evidence()
        self.update()

    def mouseReleaseEvent(self, event):
        self._drag_index = None
        self.unsetCursor()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._manual_points:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        widgets = [self._image_to_widget(p) for p in self._manual_points]
        widgets = [p for p in widgets if p is not None]
        if not widgets:
            painter.end()
            return

        painter.setPen(QPen(OBSERVED_LINE, 2.6))
        if len(widgets) == 2:
            painter.drawLine(widgets[0], widgets[1])
            mid = QPointF((widgets[0].x() + widgets[1].x()) * .5, (widgets[0].y() + widgets[1].y()) * .5)
            painter.setPen(QPen(OBSERVED_LINE, 1))
            painter.drawText(mid + QPointF(8, 15), "OBSERVED REFERENCE")
        painter.setPen(QPen(OBSERVED_LINE, 2))
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        for p in widgets:
            painter.drawEllipse(p, 5, 5)

        if len(widgets) == 2 and self._constraint is not ReferenceLineConstraint.FREE:
            evidence = self.evidence()
            if evidence is not None:
                angle = evidence.observed_angle_deg
                correction = evidence.correction_deg or 0.0
                text = f"{self._constraint.label} · obs {angle:+.1f}° · roll {correction:+.1f}°"
                painter.setPen(QPen(TARGET_LINE, 1.4))
                painter.drawText(widgets[0] + QPointF(8, -10), text)
        painter.end()


class ReferenceLineCalibrationPanel(QWidget):
    """Semantic constraint selector and live image-evidence readout."""

    evidence_changed = Signal(object)

    def __init__(self, preview: CalibratedReferenceLinePreview, parent=None):
        super().__init__(parent)
        self.preview = preview
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 5, 8, 7)
        root.setSpacing(5)

        top = QHBoxLayout()
        title = QLabel("Reference-line evidence")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(QLabel("Constraint"))
        self.constraint = QComboBox()
        self.constraint.addItem("Horizontal", ReferenceLineConstraint.HORIZONTAL)
        self.constraint.addItem("Vertical", ReferenceLineConstraint.VERTICAL)
        self.constraint.addItem("Free", ReferenceLineConstraint.FREE)
        self.constraint.currentIndexChanged.connect(self._constraint_changed)
        top.addWidget(self.constraint)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.preview.clear_evidence)
        top.addWidget(clear)
        root.addLayout(top)

        hint = QLabel(
            "在照片上单击两次选择护栏、床垫边缘等可观察线；绿色为照片证据，紫色为当前几何投影。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED.name()}; font-size:9pt;")
        root.addWidget(hint)

        self.readout = QLabel("Waiting for two image points")
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet("color:#334155;")
        root.addWidget(self.readout)
        preview.evidence_changed.connect(self._on_evidence)

    def _constraint_changed(self, index: int):
        constraint = self.constraint.itemData(index)
        self.preview.set_constraint(constraint)

    def _on_evidence(self, evidence):
        if evidence is None:
            self.readout.setText("等待在照片上选择两个参考点。")
        else:
            self.readout.setText(
                f"Observed {evidence.observed_angle_deg:+.1f}° · "
                + (
                    f"Target {evidence.target_angle_deg:+.1f}° · "
                    f"Roll correction {evidence.correction_deg:+.1f}° · "
                    f"confidence {evidence.confidence:.0%}"
                    if evidence.target_angle_deg is not None
                    else f"length {evidence.length_px:.0f}px"
                )
            )
        self.evidence_changed.emit(evidence)


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


def install_reference_line_calibration(workspace):
    """Replace the passive reference-line preview with an interactive one."""
    old_preview = getattr(workspace, "_preview", None)
    if old_preview is None:
        return None
    parent = old_preview.parentWidget()
    layout = parent.layout() if parent is not None else None
    preview = CalibratedReferenceLinePreview()
    preview.setMinimumHeight(250)
    if layout is not None:
        index = layout.indexOf(old_preview)
        layout.removeWidget(old_preview)
        old_preview.deleteLater()
        layout.insertWidget(index, preview)
    workspace._preview = preview

    panel = ReferenceLineCalibrationPanel(preview)
    if layout is not None:
        index = layout.indexOf(preview)
        layout.insertWidget(index + 1, panel)
    workspace._reference_line_calibration = panel
    workspace._reference_line_evidence = None

    def on_evidence(evidence):
        workspace._reference_line_evidence = evidence
        if hasattr(workspace, "_update_reference_hypothesis"):
            workspace._update_reference_hypothesis()

    panel.evidence_changed.connect(on_evidence)
    workspace._refresh_projection()
    return panel
