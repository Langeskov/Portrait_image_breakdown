"""One-screen photographer field mode.

Field Mode is optimized for looking at a monitor while shooting: one dominant
instruction, compact status indicators, a small queue of secondary cues, and
explicit history/voice actions. Technical analysis remains elsewhere.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QComboBox, QVBoxLayout, QWidget, QApplication,
)

from core.cue_history import CueHistory
from core.landmark_quality import assess_landmarks
from core.photographer_cue_modes import CueMode, format_cues
from core.photographer_cues import generate_photographer_cues
from core.voice_output import voice_ready_text, ssml


class FieldModeWidget(QWidget):
    """Right-panel field guidance widget."""

    _THEME = {
        "bg": "#F5F6F8", "surface": "#FFFFFF", "surface2": "#FAFBFC",
        "border": "#D9DDE3", "border2": "#C6CBD3", "text": "#1F2937",
        "text2": "#6B7280", "muted": "#9CA3AF", "accent": "#2563EB",
        "accent_text": "#FFFFFF", "disabled": "#9CA3AF", "hover": "#F3F4F6",
        "pressed": "#E8EEF9", "selected": "#2563EB",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = CueHistory(capacity=30)
        self._cues = []
        self.setMinimumWidth(240)
        self.setMaximumWidth(360)
        self.setObjectName("fieldMode")

        t = self._THEME
        self.setStyleSheet(f"""
            QWidget#fieldMode QLabel {{ color: {t['text']}; background: transparent; }}
            QWidget#fieldMode QLabel#primaryCue {{ color: {t['text']}; background: {t['surface']}; border: 1px solid {t['border']}; padding: 10px; }}
            QWidget#fieldMode QComboBox QAbstractItemView {{ color: {t['text']}; background: {t['surface']}; border: 1px solid {t['border2']}; selection-background-color: {t['selected']}; selection-color: {t['accent_text']}; }}
        """)

        lo = QVBoxLayout(self)
        lo.setContentsMargins(10, 10, 10, 10)
        lo.setSpacing(6)

        # Header
        header = QHBoxLayout()
        eyebrow = QLabel("现场指令")
        eyebrow.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        eyebrow.setStyleSheet(f"color: {t['text']};")
        header.addWidget(eyebrow)
        header.addStretch(1)
        self._mode = QComboBox()
        self._mode.addItems([m.value for m in CueMode])
        self._mode.setCurrentText(CueMode.NORMAL.value)
        self._mode.setMinimumWidth(80)
        header.addWidget(self._mode)
        lo.addLayout(header)

        # Status metrics
        metrics = QHBoxLayout()
        self._confidence = QLabel("置信度 —")
        self._confidence.setStyleSheet(f"color: {t['text2']}; font-size: 9pt;")
        self._landmarks = QLabel("关键点 —")
        self._landmarks.setStyleSheet(f"color: {t['text2']}; font-size: 9pt;")
        metrics.addWidget(self._confidence)
        metrics.addWidget(self._landmarks)
        metrics.addStretch(1)
        lo.addLayout(metrics)

        # Primary cue
        cue_label = QLabel("当前指令")
        cue_label.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        cue_label.setStyleSheet(f"color: {t['text2']};")
        lo.addWidget(cue_label)
        self._primary = QLabel("保持自然，我会根据画面继续调整。")
        self._primary.setObjectName("primaryCue")
        self._primary.setWordWrap(True)
        self._primary.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._primary.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        self._primary.setMinimumHeight(80)
        lo.addWidget(self._primary)

        # Undo/redo + copy
        actions = QHBoxLayout()
        self._undo = QPushButton("撤销")
        self._undo.setStyleSheet(f"QPushButton {{ padding: 4px 8px; border: 1px solid {t['border']}; background: {t['surface']}; color: {t['text']}; font-size: 9pt; }}")
        self._redo = QPushButton("重做")
        self._redo.setStyleSheet(self._undo.styleSheet())
        self._copy = QPushButton("复制")
        self._copy.setStyleSheet(self._undo.styleSheet())
        actions.addWidget(self._undo)
        actions.addWidget(self._redo)
        actions.addStretch(1)
        actions.addWidget(self._copy)
        lo.addLayout(actions)

        # Secondary cues
        secondary_label = QLabel("辅助指令")
        secondary_label.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        secondary_label.setStyleSheet(f"color: {t['text2']};")
        lo.addWidget(secondary_label)
        self._detail = QListWidget()
        self._detail.setStyleSheet(
            f"QListWidget {{ background: {t['surface2']}; border: 1px solid {t['border']}; "
            f"color: {t['text']}; font-size: 9pt; padding: 2px; }}"
            f"QListWidget::item {{ padding: 5px 6px; }}"
            f"QListWidget::item:selected {{ background: {t['accent']}; color: white; }}"
        )
        lo.addWidget(self._detail, 1)

        self._mode.currentTextChanged.connect(self._render_current)
        self._undo.clicked.connect(self._on_undo)
        self._redo.clicked.connect(self._on_redo)
        self._copy.clicked.connect(self._copy_voice)
        self._refresh_buttons()

    def set_analysis(self, action, orientation, camera, composition, pose=None, confidence=None):
        cues = generate_photographer_cues(action, orientation, camera, composition)
        quality = assess_landmarks(pose.landmarks) if pose is not None else None
        self.set_cues(cues, confidence, quality)

    def set_cues(self, cues, confidence: float | None = None, quality=None):
        cue_list = list(cues or [])
        candidate_text = tuple(c.cue for c in cue_list)
        current = self.history.current
        if current is None or current.cue_text != candidate_text:
            self.history.push(cue_list)
        self._cues = cue_list
        self._confidence.setText(f"置信度 {confidence:.0%}" if confidence is not None else "置信度 —")
        if quality is not None:
            self._landmarks.setText(f"关键点 {quality.visible_count}/17")
        else:
            self._landmarks.setText("关键点 —")
        self._render_current()

    def _mode_enum(self):
        return next((m for m in CueMode if m.value == self._mode.currentText()), CueMode.NORMAL)

    def _display_cues_for_snapshot(self, snap):
        cues = snap.cues if getattr(snap, "cues", ()) else ()
        if cues:
            return format_cues(list(cues), self._mode_enum())
        return list(snap.cue_text)

    def _render_current(self):
        snap = self.history.current
        self._detail.clear()
        if snap is None:
            self._primary.setText("保持自然，我会根据画面继续调整。")
            self._refresh_buttons()
            return
        formatted = self._display_cues_for_snapshot(snap)
        # Primary text must respect the selected tone mode
        self._primary.setText(formatted[0] if formatted else snap.summary)
        secondary = formatted[1:]
        for line in secondary:
            self._detail.addItem(QListWidgetItem(line))
        self._refresh_buttons()

    def _on_undo(self):
        self.history.undo()
        self._render_current()

    def _on_redo(self):
        self.history.redo()
        self._render_current()

    def _refresh_buttons(self):
        self._undo.setEnabled(self.history.can_undo)
        self._redo.setEnabled(self.history.can_redo)

    def _copy_voice(self):
        snap = self.history.current
        QApplication.clipboard().setText(voice_ready_text(snap.summary if snap else ""))


def install_field_mode(window):
    """Install field mode widget into the pre-registered field workspace.

    MainWindow.__init__ already created _w_field and added it as page 1.
    This function creates the FieldModeWidget and connects it to the
    analysis data pipeline so guidance cues update when analysis completes.
    """
    field = FieldModeWidget()
    window._field_mode = field

    # Add to the field workspace's right-side container
    window._w_field._field_lo.addWidget(field)

    # Wire into analysis results: when _w2.update_results is called,
    # also feed the same bundle to the field mode widget for cue generation.
    original_update = window._w2.update_results

    def _update_with_field(bundle):
        original_update(bundle)
        if bundle.action and bundle.orientation and bundle.camera and bundle.composition:
            confidence = float(bundle.action.confidence) if bundle.action else None
            field.set_analysis(
                bundle.action, bundle.orientation, bundle.camera,
                bundle.composition, bundle.pose, confidence,
            )

    window._w2.update_results = _update_with_field
    return field
