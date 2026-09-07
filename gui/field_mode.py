"""One-screen photographer field mode.

Field Mode is optimized for looking at a monitor while shooting: one dominant
instruction, compact status indicators, a small queue of secondary cues, and
explicit history/voice actions. Technical analysis remains elsewhere.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QComboBox, QVBoxLayout, QWidget, QApplication,
)

from core.cue_history import CueHistory
from core.landmark_quality import assess_landmarks
from core.photographer_cue_modes import CueMode, format_cues
from core.photographer_cues import generate_photographer_cues
from core.voice_output import voice_ready_text, ssml


class FieldModeWidget(QWidget):
    """High-signal field UI with an isolated, predictable dark palette."""

    _THEME = {
        "bg": "#10151c", "surface": "#18202b", "surface2": "#141b24",
        "border": "#2c3847", "border2": "#3a4858", "text": "#e5e7eb",
        "text2": "#aebdcd", "muted": "#8fa3b8", "accent": "#60a5fa",
        "accent_text": "#ffffff", "disabled": "#657384",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = CueHistory(capacity=30)
        self._cues = []
        t = self._THEME
        self.setObjectName("fieldMode")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            QWidget#fieldMode {{ background: {t['bg']}; color: {t['text']}; }}
            QWidget#fieldMode QLabel {{ color: {t['text']}; background: transparent; }}
            QWidget#fieldMode QLabel#eyebrow {{ color: {t['muted']}; font-size: 11px; letter-spacing: 1px; }}
            QWidget#fieldMode QLabel#status {{ color: {t['text2']}; }}
            QWidget#fieldMode QLabel#metric {{ color: {t['text']}; padding: 5px 8px; background: {t['surface']}; border-radius: 5px; }}
            QWidget#fieldMode QLabel#primaryCue {{ color: {t['text']}; background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; padding: 14px; }}
            QWidget#fieldMode QFrame#statusCard {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; }}
            QWidget#fieldMode QListWidget {{ color: {t['text']}; background: {t['surface2']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 4px; outline: 0; }}
            QWidget#fieldMode QListWidget::item {{ color: {t['text']}; padding: 9px 8px; border-radius: 4px; }}
            QWidget#fieldMode QListWidget::item:selected {{ color: {t['accent_text']}; background: #264766; }}
            QWidget#fieldMode QPushButton {{ color: {t['text']}; padding: 7px 12px; border: 1px solid {t['border2']}; border-radius: 5px; background: {t['surface']}; }}
            QWidget#fieldMode QPushButton:hover {{ background: #223042; }}
            QWidget#fieldMode QPushButton:pressed {{ background: #273b52; }}
            QWidget#fieldMode QPushButton:disabled {{ color: {t['disabled']}; background: #161d25; border-color: {t['border']}; }}
            QWidget#fieldMode QComboBox {{ color: {t['text']}; padding: 6px 10px; border: 1px solid {t['border2']}; border-radius: 5px; background: {t['surface']}; }}
            QWidget#fieldMode QComboBox:hover {{ border-color: {t['accent']}; }}
            QWidget#fieldMode QComboBox QAbstractItemView {{ color: {t['text']}; background: {t['surface']}; border: 1px solid {t['border2']}; selection-background-color: #264766; selection-color: {t['accent_text']}; padding: 4px; }}
        """)

        lo = QVBoxLayout(self); lo.setContentsMargins(20, 18, 20, 18); lo.setSpacing(12)
        header = QHBoxLayout()
        eyebrow = QLabel("FIELD MODE · SHOOTING ASSISTANCE"); eyebrow.setObjectName("eyebrow")
        header.addWidget(eyebrow); header.addStretch(1)
        output_label = QLabel("Output"); output_label.setObjectName("fieldHeaderLabel"); header.addWidget(output_label)
        self._mode = QComboBox(); self._mode.addItems([m.value for m in CueMode]); self._mode.setCurrentText(CueMode.NORMAL.value); self._mode.setMinimumWidth(90)
        header.addWidget(self._mode); lo.addLayout(header)

        card = QFrame(); card.setObjectName("statusCard"); card_lo = QVBoxLayout(card); card_lo.setContentsMargins(12, 10, 12, 10)
        self._status = QLabel("Waiting for analysis…"); self._status.setObjectName("status"); self._status.setWordWrap(True); card_lo.addWidget(self._status)
        metrics = QHBoxLayout()
        self._confidence = QLabel("Confidence —"); self._confidence.setObjectName("metric")
        self._landmarks = QLabel("Landmarks —"); self._landmarks.setObjectName("metric")
        self._pose_state = QLabel("Pose —"); self._pose_state.setObjectName("metric")
        metrics.addWidget(self._confidence); metrics.addWidget(self._landmarks); metrics.addWidget(self._pose_state); metrics.addStretch(1)
        card_lo.addLayout(metrics); lo.addWidget(card)

        cue_label = QLabel("SAY THIS NOW"); cue_label.setObjectName("eyebrow"); lo.addWidget(cue_label)
        self._primary = QLabel("保持自然，我会根据画面继续调整。"); self._primary.setObjectName("primaryCue"); self._primary.setWordWrap(True); self._primary.setAlignment(Qt.AlignLeft | Qt.AlignVCenter); self._primary.setFont(QFont("Segoe UI", 27, QFont.Weight.Bold)); self._primary.setMinimumHeight(150); lo.addWidget(self._primary)

        actions = QHBoxLayout(); self._undo = QPushButton("Undo"); self._redo = QPushButton("Redo"); self._copy = QPushButton("Copy voice output"); self._ssml = QPushButton("Copy SSML")
        actions.addWidget(self._undo); actions.addWidget(self._redo); actions.addStretch(1); actions.addWidget(self._copy); actions.addWidget(self._ssml); lo.addLayout(actions)

        secondary_header = QHBoxLayout(); secondary_label = QLabel("SECONDARY CUES"); secondary_label.setObjectName("eyebrow"); secondary_header.addWidget(secondary_label); secondary_header.addStretch(1)
        self._count = QLabel("0"); self._count.setObjectName("metric"); secondary_header.addWidget(self._count); lo.addLayout(secondary_header)
        self._detail = QListWidget(); self._detail.setMinimumHeight(150); lo.addWidget(self._detail, 1)

        self._mode.currentTextChanged.connect(lambda _: self._render_current())
        self._undo.clicked.connect(self._on_undo); self._redo.clicked.connect(self._on_redo); self._copy.clicked.connect(self._copy_voice); self._ssml.clicked.connect(self._copy_ssml); self._refresh_buttons()

    def set_analysis(self, action, orientation, camera, composition, pose=None, confidence=None):
        cues = generate_photographer_cues(action, orientation, camera, composition)
        quality = assess_landmarks(pose.landmarks) if pose is not None else None
        self.set_cues(cues, confidence, quality)

    def set_cues(self, cues, confidence: float | None = None, quality=None):
        candidate_text = tuple(c.cue for c in cues or []); current = self.history.current
        if current is None or current.cue_text != candidate_text: self.history.push(list(cues or []))
        self._cues = list(cues or []); self._status.setText("Analysis ready · cue history records only meaningful changes")
        self._confidence.setText(f"Confidence {confidence:.0%}" if confidence is not None else "Confidence —")
        if quality is not None:
            self._landmarks.setText(f"Landmarks {quality.visible_count}/17"); self._pose_state.setText(f"Pose {quality.pose_state}")
        else: self._landmarks.setText("Landmarks —"); self._pose_state.setText("Pose —")
        self._render_current()

    def _mode_enum(self):
        return next((m for m in CueMode if m.value == self._mode.currentText()), CueMode.NORMAL)

    def _render_current(self):
        snap = self.history.current
        if snap is None: self._detail.clear(); self._count.setText("0"); self._refresh_buttons(); return
        self._primary.setText(snap.summary); self._detail.clear()
        formatted = format_cues(self._cues, self._mode_enum()) if tuple(c.cue for c in self._cues) == snap.cue_text else list(snap.cue_text)
        secondary = formatted[1:6]
        for line in secondary: self._detail.addItem(QListWidgetItem(line))
        self._count.setText(str(len(secondary))); self._refresh_buttons()

    def _on_undo(self):
        snap = self.history.undo()
        if snap: self._primary.setText(snap.summary); self._detail.clear(); [self._detail.addItem(QListWidgetItem(line)) for line in snap.cue_text[1:6]]; self._count.setText(str(min(5, max(0, len(snap.cue_text)-1))))
        self._refresh_buttons()

    def _on_redo(self):
        snap = self.history.redo()
        if snap: self._primary.setText(snap.summary); self._detail.clear(); [self._detail.addItem(QListWidgetItem(line)) for line in snap.cue_text[1:6]]; self._count.setText(str(min(5, max(0, len(snap.cue_text)-1))))
        self._refresh_buttons()

    def _refresh_buttons(self): self._undo.setEnabled(self.history.can_undo); self._redo.setEnabled(self.history.can_redo)
    def _copy_voice(self): QApplication.clipboard().setText(voice_ready_text(self.history.current.summary if self.history.current else ""))
    def _copy_ssml(self): QApplication.clipboard().setText(ssml(self.history.current.summary if self.history.current else ""))
    def voice_payload(self):
        text = voice_ready_text(self.history.current.summary if self.history.current else "")
        return {"text": text, "ssml": ssml(text)}


def install_field_mode(window):
    field = FieldModeWidget(window); window._field_mode = field; window._tabs.addTab("Field Mode"); window._ws.addWidget(field); old_update = window._w2.update_results
    def update_results(bundle):
        old_update(bundle)
        if bundle.action and bundle.orientation and bundle.camera and bundle.composition:
            confidence = float(bundle.reverse_result.overall_confidence) if bundle.reverse_result else None
            field.set_analysis(bundle.action, bundle.orientation, bundle.camera, bundle.composition, bundle.pose, confidence)
    window._w2.update_results = update_results
    return field
