"""One-screen photographer field mode.

Field Mode is optimized for looking at a monitor while shooting: one dominant
instruction, compact status indicators, a small queue of secondary cues, and
explicit history/voice actions. Technical analysis remains elsewhere.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from core.cue_history import CueHistory
from core.landmark_quality import assess_landmarks
from core.photographer_cue_modes import CueMode, format_cues
from core.photographer_cues import generate_photographer_cues
from core.voice_output import voice_ready_text, ssml


class FieldModeWidget(QWidget):
    """High-signal field UI with explicit foreground colors for dark surfaces."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = CueHistory(capacity=30)
        self._cues = []

        self.setObjectName("fieldMode")
        self.setStyleSheet(
            """
            QWidget#fieldMode {
                background: #10151c;
                color: #e5e7eb;
            }
            QWidget#fieldMode QLabel {
                background: transparent;
                color: #e5e7eb;
            }
            QWidget#fieldMode QLabel#eyebrow {
                color: #8fa3b8;
                font-size: 11px;
                letter-spacing: 1px;
            }
            QWidget#fieldMode QLabel#primaryCue {
                color: #ffffff;
                padding: 8px;
            }
            QWidget#fieldMode QLabel#status {
                color: #aebdcd;
            }
            QWidget#fieldMode QLabel#metric {
                color: #cbd5e1;
                padding: 5px 8px;
                background: #18202b;
                border-radius: 5px;
            }
            QWidget#fieldMode QFrame#statusCard {
                background: #18202b;
                border: 1px solid #2c3847;
                border-radius: 8px;
            }
            QWidget#fieldMode QListWidget {
                background: #141b24;
                border: 1px solid #293542;
                border-radius: 6px;
                padding: 4px;
                color: #e5e7eb;
            }
            QWidget#fieldMode QListWidget::item {
                color: #e5e7eb;
                padding: 9px 8px;
            }
            QWidget#fieldMode QListWidget::item:selected {
                color: #ffffff;
                background: #263342;
            }
            QWidget#fieldMode QPushButton {
                padding: 7px 12px;
                border: 1px solid #3a4858;
                border-radius: 5px;
                background: #1b2530;
                color: #e5e7eb;
            }
            QWidget#fieldMode QPushButton:hover {
                background: #24303d;
                color: #ffffff;
            }
            QWidget#fieldMode QPushButton:disabled {
                color: #657384;
                background: #161d25;
            }
            QWidget#fieldMode QComboBox {
                padding: 6px 10px;
                border: 1px solid #3a4858;
                border-radius: 5px;
                background: #151e28;
                color: #e5e7eb;
            }
            QWidget#fieldMode QComboBox::drop-down {
                border: none;
                width: 22px;
            }
            QWidget#fieldMode QComboBox QAbstractItemView {
                background: #151e28;
                color: #e5e7eb;
                selection-background-color: #263342;
                selection-color: #ffffff;
                border: 1px solid #3a4858;
            }
            QWidget#fieldMode QComboBox QAbstractItemView::item {
                color: #e5e7eb;
                padding: 6px 8px;
            }
            """
        )

        lo = QVBoxLayout(self)
        lo.setContentsMargins(20, 18, 20, 18)
        lo.setSpacing(12)

        header = QHBoxLayout()
        eyebrow = QLabel("FIELD MODE · SHOOTING ASSISTANCE")
        eyebrow.setObjectName("eyebrow")
        header.addWidget(eyebrow)
        header.addStretch(1)
        output_label = QLabel("Output")
        output_label.setObjectName("outputLabel")
        header.addWidget(output_label)
        self._mode = QComboBox()
        self._mode.addItems([m.value for m in CueMode])
        self._mode.setCurrentText(CueMode.NORMAL.value)
        self._mode.setMinimumWidth(90)
        header.addWidget(self._mode)
        lo.addLayout(header)

        card = QFrame()
        card.setObjectName("statusCard")
        card_lo = QVBoxLayout(card)
        card_lo.setContentsMargins(12, 10, 12, 10)
        self._status = QLabel("Waiting for analysis…")
        self._status.setObjectName("status")
        self._status.setWordWrap(True)
        card_lo.addWidget(self._status)

        metrics = QHBoxLayout()
        self._confidence = QLabel("Confidence —")
        self._confidence.setObjectName("metric")
        self._landmarks = QLabel("Landmarks —")
        self._landmarks.setObjectName("metric")
        self._pose_state = QLabel("Pose —")
        self._pose_state.setObjectName("metric")
        metrics.addWidget(self._confidence)
        metrics.addWidget(self._landmarks)
        metrics.addWidget(self._pose_state)
        metrics.addStretch(1)
        card_lo.addLayout(metrics)
        lo.addWidget(card)

        cue_label = QLabel("SAY THIS NOW")
        cue_label.setObjectName("eyebrow")
        lo.addWidget(cue_label)
        self._primary = QLabel("保持自然，我会根据画面继续调整。")
        self._primary.setObjectName("primaryCue")
        self._primary.setWordWrap(True)
        self._primary.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._primary.setFont(QFont("Segoe UI", 27, QFont.Weight.Bold))
        self._primary.setMinimumHeight(150)
        lo.addWidget(self._primary)

        actions = QHBoxLayout()
        self._undo = QPushButton("Undo")
        self._redo = QPushButton("Redo")
        self._copy = QPushButton("Copy voice output")
        self._ssml = QPushButton("Copy SSML")
        actions.addWidget(self._undo)
        actions.addWidget(self._redo)
        actions.addStretch(1)
        actions.addWidget(self._copy)
        actions.addWidget(self._ssml)
        lo.addLayout(actions)

        secondary_header = QHBoxLayout()
        secondary = QLabel("SECONDARY CUES")
        secondary.setObjectName("eyebrow")
        secondary_header.addWidget(secondary)
        secondary_header.addStretch(1)
        self._count = QLabel("0")
        self._count.setObjectName("metric")
        secondary_header.addWidget(self._count)
        lo.addLayout(secondary_header)

        self._detail = QListWidget()
        self._detail.setMinimumHeight(150)
        lo.addWidget(self._detail, 1)

        self._mode.currentTextChanged.connect(self._render_current)
        self._undo.clicked.connect(self._on_undo)
        self._redo.clicked.connect(self._on_redo)
        self._copy.clicked.connect(self._copy_voice)
        self._ssml.clicked.connect(self._copy_ssml)
        self._refresh_buttons()

    def set_analysis(self, action, orientation, camera, composition, pose=None, confidence=None):
        """Generate and render cues from the complete current analysis bundle."""
        cues = generate_photographer_cues(action, orientation, camera, composition)
        quality = assess_landmarks(pose.landmarks) if pose is not None else None
        self.set_cues(cues, confidence, quality)

    def set_cues(self, cues, confidence=None, quality=None):
        cues = list(cues or [])
        candidate_text = tuple(c.cue for c in cues)
        current = self.history.current
        if current is None or current.cue_text != candidate_text:
            self.history.push(cues)
        self._cues = cues
        self._status.setText("Analysis ready · cue history records only meaningful changes")
        self._confidence.setText(f"Confidence {confidence:.0%}" if confidence is not None else "Confidence —")
        self._landmarks.setText(f"Landmarks {quality.visible_count}/17" if quality is not None else "Landmarks —")
        self._pose_state.setText(f"Pose {quality.pose_state}" if quality is not None else "Pose —")
        self._render_current()

    def _mode_enum(self):
        return next((m for m in CueMode if m.value == self._mode.currentText()), CueMode.NORMAL)

    def _render_current(self):
        snap = self.history.current
        if snap is None:
            self._primary.setText("保持自然，我会根据画面继续调整。")
            self._detail.clear()
            self._count.setText("0")
            self._refresh_buttons()
            return
        self._primary.setText(snap.summary)
        self._detail.clear()
        formatted = format_cues(self._cues, self._mode_enum()) if tuple(c.cue for c in self._cues) == snap.cue_text else list(snap.cue_text)
        for line in formatted[1:6]:
            self._detail.addItem(QListWidgetItem(line))
        self._count.setText(str(max(0, min(5, len(formatted) - 1))))
        self._refresh_buttons()

    def _on_undo(self):
        snap = self.history.undo()
        if snap:
            self._render_snapshot(snap)
        self._refresh_buttons()

    def _on_redo(self):
        snap = self.history.redo()
        if snap:
            self._render_snapshot(snap)
        self._refresh_buttons()

    def _render_snapshot(self, snap):
        self._primary.setText(snap.summary)
        self._detail.clear()
        for line in snap.cue_text[1:6]:
            self._detail.addItem(QListWidgetItem(line))
        self._count.setText(str(min(5, max(0, len(snap.cue_text) - 1))))

    def _refresh_buttons(self):
        self._undo.setEnabled(self.history.can_undo)
        self._redo.setEnabled(self.history.can_redo)

    def _copy_voice(self):
        snap = self.history.current
        QApplication.clipboard().setText(voice_ready_text(snap.summary if snap else ""))

    def _copy_ssml(self):
        snap = self.history.current
        QApplication.clipboard().setText(ssml(snap.summary if snap else ""))

    def voice_payload(self):
        snap = self.history.current
        text = voice_ready_text(snap.summary if snap else "")
        return {"text": text, "ssml": ssml(text)}


def install_field_mode(window):
    """Attach field mode as a fourth workspace."""
    field = FieldModeWidget(window)
    window._field_mode = field
    window._tabs.addTab("Field Mode")
    window._ws.addWidget(field)
    old_update = window._w2.update_results

    def update_results(bundle):
        old_update(bundle)
        if bundle.action and bundle.orientation and bundle.camera and bundle.composition:
            confidence = float(bundle.reverse_result.overall_confidence) if bundle.reverse_result else None
            field.set_analysis(bundle.action, bundle.orientation, bundle.camera, bundle.composition, bundle.pose, confidence)

    window._w2.update_results = update_results
    return field
