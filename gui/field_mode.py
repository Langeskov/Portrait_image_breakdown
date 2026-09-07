"""One-screen photographer field mode and integration helper."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem, QApplication

from core.cue_history import CueHistory
from core.photographer_cue_modes import CueMode, format_cues
from core.photographer_cues import generate_photographer_cues
from core.voice_output import voice_ready_text, ssml


class FieldModeWidget(QWidget):
    cue_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = CueHistory(); self._cues = []
        lo = QVBoxLayout(self); lo.setContentsMargins(18, 14, 18, 14); lo.setSpacing(10)
        title = QLabel("FIELD MODE"); title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold)); lo.addWidget(title)
        self._status = QLabel("Waiting for analysis…"); lo.addWidget(self._status)
        row = QHBoxLayout(); row.addWidget(QLabel("Cue mode")); self._mode = QComboBox(); self._mode.addItems([m.value for m in CueMode]); self._mode.setCurrentText(CueMode.NORMAL.value); row.addWidget(self._mode); row.addStretch(1); lo.addLayout(row)
        self._primary = QLabel("保持自然，我会根据画面继续调整。"); self._primary.setWordWrap(True); self._primary.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold)); self._primary.setMinimumHeight(110); lo.addWidget(self._primary)
        btns = QHBoxLayout(); self._undo = QPushButton("Undo"); self._redo = QPushButton("Redo"); self._copy = QPushButton("Copy voice cue"); btns.addWidget(self._undo); btns.addWidget(self._redo); btns.addStretch(1); btns.addWidget(self._copy); lo.addLayout(btns)
        self._detail = QListWidget(); lo.addWidget(self._detail, 1)
        self._mode.currentTextChanged.connect(lambda _: self._render_current()); self._undo.clicked.connect(self._on_undo); self._redo.clicked.connect(self._on_redo); self._copy.clicked.connect(self._copy_voice)
        self._refresh_buttons()

    def set_cues(self, cues, confidence: float | None = None):
        self._cues = list(cues or [])
        self.history.push(self._cues)
        self._status.setText(f"Analysis confidence: {confidence:.0%}" if confidence is not None else "Analysis ready")
        self._render_current()

    def _mode_enum(self):
        return next((m for m in CueMode if m.value == self._mode.currentText()), CueMode.NORMAL)

    def _render_current(self):
        snap = self.history.current
        if snap is None: return
        self._primary.setText(snap.summary); self._detail.clear()
        if self._cues:
            for line in format_cues(self._cues, self._mode_enum())[:6]: self._detail.addItem(QListWidgetItem(line))
        self._refresh_buttons(); self.cue_changed.emit(voice_ready_text(snap.summary))

    def _on_undo(self):
        snap = self.history.undo()
        if snap:
            self._primary.setText(snap.summary); self._detail.clear(); [self._detail.addItem(QListWidgetItem(x)) for x in snap.cue_text]
        self._refresh_buttons(); self.cue_changed.emit(voice_ready_text(snap.summary if snap else ""))

    def _on_redo(self):
        snap = self.history.redo()
        if snap:
            self._primary.setText(snap.summary); self._detail.clear(); [self._detail.addItem(QListWidgetItem(x)) for x in snap.cue_text]
        self._refresh_buttons(); self.cue_changed.emit(voice_ready_text(snap.summary if snap else ""))

    def _refresh_buttons(self): self._undo.setEnabled(self.history.can_undo); self._redo.setEnabled(self.history.can_redo)

    def _copy_voice(self):
        snap = self.history.current
        if snap: QApplication.clipboard().setText(voice_ready_text(snap.summary))

    def voice_payload(self):
        snap = self.history.current; text = voice_ready_text(snap.summary if snap else "")
        return {"text": text, "ssml": ssml(text)}


def install_field_mode(window):
    """Attach field mode as a fourth one-screen workspace."""
    from core.landmark_quality import assess_landmarks
    field = FieldModeWidget(window); window._field_mode = field
    window._tabs.addTab("Field Mode"); window._ws.addWidget(field)
    old_update = window._w2.update_results

    def update_results(bundle):
        old_update(bundle)
        if bundle.action and bundle.orientation and bundle.camera and bundle.composition:
            cues = generate_photographer_cues(bundle.action, bundle.orientation, bundle.camera, bundle.composition)
            quality = assess_landmarks(bundle.pose.landmarks) if bundle.pose else None
            confidence = min(float(bundle.reverse_result.overall_confidence), quality.anchor_confidence) if bundle.reverse_result and quality else (float(bundle.reverse_result.overall_confidence) if bundle.reverse_result else None)
            field.set_cues(cues, confidence)
            if quality:
                field._status.setText(field._status.text() + f" · landmarks {quality.visible_count}/17 · {quality.pose_state}")
    window._w2.update_results = update_results
    return field
