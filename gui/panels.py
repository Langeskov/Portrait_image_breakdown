"""Analysis panels — light theme, no emoji, correct data contracts.

AnalysisPanel: left sidebar (pose, orientation, action, camera, composition)
SuggestionPanel: right sidebar (next actions, detailed suggestions, creative direction)
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QFrame, QScrollArea,
)

from core.action_classifier import ActionResult
from core.camera_analyzer import CameraResult
from core.composition import CompositionResult
from core.orientation import OrientationResult
from core.suggestion import SuggestionResult, SuggestionPriority

# ── Light theme tokens ──
_C_PANEL = "#FFFFFF"
_C_SURFACE = "#FAFBFC"
_C_BORDER = "#D9DDE3"
_C_TEXT = "#1F2937"
_C_TEXT2 = "#6B7280"
_C_ACCENT = "#2563EB"
_C_SUCCESS = "#16A34A"
_C_WARNING = "#D97706"
_C_DANGER = "#DC2626"


def _header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
    lbl.setStyleSheet(f"color: {_C_TEXT}; padding: 6px 0 2px 0;")
    return lbl


def _stat_row(label: str, value: str = "--") -> QWidget:
    """A row: [label (fixed width)] [value (expanding)]."""
    w = QWidget()
    lo = QHBoxLayout(w)
    lo.setContentsMargins(0, 2, 0, 2)
    lo.setSpacing(8)

    lbl = QLabel(label)
    lbl.setFont(QFont("Microsoft YaHei", 9))
    lbl.setStyleSheet(f"color: {_C_TEXT2};")
    lbl.setFixedWidth(110)

    val = QLabel(value)
    val.setFont(QFont("Consolas", 9))
    val.setStyleSheet(f"color: {_C_TEXT};")
    val.setObjectName("_value_label")  # for findChild lookup

    lo.addWidget(lbl)
    lo.addWidget(val, 1)
    return w


def _get_val(widget: QWidget) -> QLabel:
    """Get the value label from a _stat_row widget."""
    return widget.findChild(QLabel, "_value_label")


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"color: {_C_BORDER};")
    return line


# ══════════════════════════════════════════════════════════════════════
#  AnalysisPanel (left sidebar)
# ══════════════════════════════════════════════════════════════════════
class AnalysisPanel(QWidget):
    """Left panel: 5 groups — Skeleton, Orientation, Action, Camera, Composition."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {_C_PANEL}; }}"
        )

        container = QWidget()
        lo = QVBoxLayout(container)
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setSpacing(6)

        # Title
        title = QLabel("Analysis")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {_C_TEXT}; padding: 4px 0;")
        lo.addWidget(title)
        lo.addWidget(_separator())

        # ── Skeleton ──
        lo.addWidget(_header("Skeleton Detection"))
        self._lbl_pose_conf = _stat_row("Confidence")
        self._lbl_pose_count = _stat_row("Visible keypoints")
        lo.addWidget(self._lbl_pose_conf)
        lo.addWidget(self._lbl_pose_count)
        lo.addWidget(_separator())

        # ── Orientation ──
        lo.addWidget(_header("Body Orientation"))
        self._lbl_facing = _stat_row("Facing")
        self._lbl_tilt = _stat_row("Tilt")
        self._lbl_facing_angle = _stat_row("Rotation angle")
        lo.addWidget(self._lbl_facing)
        lo.addWidget(self._lbl_tilt)
        lo.addWidget(self._lbl_facing_angle)
        lo.addWidget(_separator())

        # ── Action ──
        lo.addWidget(_header("Action Category"))
        self._lbl_action = _stat_row("Current action")
        self._lbl_action_conf = _stat_row("Confidence")
        self._lbl_action_detail = _stat_row("Details")
        lo.addWidget(self._lbl_action)
        lo.addWidget(self._lbl_action_conf)
        lo.addWidget(self._lbl_action_detail)

        # Joint angles
        lo.addWidget(_header("Joint Angles"))
        self._angle_labels: dict[str, QWidget] = {}
        for name in ["Left knee", "Right knee", "Left elbow", "Right elbow",
                      "Left hip", "Right hip"]:
            row = _stat_row(name)
            self._angle_labels[name] = row
            lo.addWidget(row)
        lo.addWidget(_separator())

        # ── Camera ──
        lo.addWidget(_header("Camera Analysis"))
        self._lbl_shot = _stat_row("Shot type")
        self._lbl_cam_angle = _stat_row("Camera angle")
        self._lbl_subject_ratio = _stat_row("Subject ratio")
        self._lbl_dutch = _stat_row("Dutch angle")
        lo.addWidget(self._lbl_shot)
        lo.addWidget(self._lbl_cam_angle)
        lo.addWidget(self._lbl_subject_ratio)
        lo.addWidget(self._lbl_dutch)
        lo.addWidget(_separator())

        # ── Composition ──
        lo.addWidget(_header("Composition Analysis"))
        self._lbl_comp_type = _stat_row("Composition type")
        self._lbl_thirds = _stat_row("Thirds alignment")
        self._lbl_symmetry = _stat_row("Symmetry")
        self._lbl_headroom = _stat_row("Headroom")
        self._lbl_balance = _stat_row("Balance")
        lo.addWidget(self._lbl_comp_type)
        lo.addWidget(self._lbl_thirds)
        lo.addWidget(self._lbl_symmetry)
        lo.addWidget(self._lbl_headroom)
        lo.addWidget(self._lbl_balance)

        lo.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Update methods (correct signatures) ──
    def update_pose(self, conf: float, visible_count: int):
        val = _get_val(self._lbl_pose_conf)
        if val:
            val.setText(f"{conf:.0%}")
        val = _get_val(self._lbl_pose_count)
        if val:
            val.setText(f"{visible_count}/17")

    def update_orientation(self, result: OrientationResult):
        val = _get_val(self._lbl_facing)
        if val:
            val.setText(result.facing.value)
            val.setStyleSheet(f"color: {_C_ACCENT}; font-weight: bold;")
        val = _get_val(self._lbl_tilt)
        if val:
            val.setText(result.tilt.value)
        val = _get_val(self._lbl_facing_angle)
        if val:
            val.setText(f"{result.facing_angle:.1f} deg")

    def update_action(self, result: ActionResult):
        val = _get_val(self._lbl_action)
        if val:
            val.setText(result.category.value)
            val.setStyleSheet(f"color: {_C_SUCCESS}; font-weight: bold;")
        val = _get_val(self._lbl_action_conf)
        if val:
            val.setText(f"{result.confidence:.0%}")
        val = _get_val(self._lbl_action_detail)
        if val:
            val.setText(result.sub_description)

        # Joint angles
        angle_map = {
            "Left knee": "left_knee", "Right knee": "right_knee",
            "Left elbow": "left_elbow", "Right elbow": "right_elbow",
            "Left hip": "left_hip", "Right hip": "right_hip",
        }
        for cn_name, en_key in angle_map.items():
            if en_key in result.joint_angles:
                row = self._angle_labels.get(cn_name)
                if row:
                    val = _get_val(row)
                    if val:
                        val.setText(f"{result.joint_angles[en_key]:.1f} deg")

    def update_camera(self, result: CameraResult):
        val = _get_val(self._lbl_shot)
        if val:
            val.setText(result.shot_type.value)
        val = _get_val(self._lbl_cam_angle)
        if val:
            val.setText(result.camera_angle.value)
        val = _get_val(self._lbl_subject_ratio)
        if val:
            val.setText(f"{result.subject_ratio:.1%}")
        val = _get_val(self._lbl_dutch)
        if val:
            val.setText(f"{result.dutch_angle_deg:.1f} deg")

    def update_composition(self, result: CompositionResult):
        val = _get_val(self._lbl_comp_type)
        if val:
            val.setText(result.primary_type.value)
        val = _get_val(self._lbl_thirds)
        if val:
            val.setText(f"{result.thirds_alignment:.0%}")
            val.setStyleSheet(
                f"color: {_C_SUCCESS};" if result.thirds_alignment > 0.6
                else f"color: {_C_WARNING};"
            )
        val = _get_val(self._lbl_symmetry)
        if val:
            val.setText(f"{result.symmetry_score:.0%}")
        val = _get_val(self._lbl_headroom)
        if val:
            val.setText(f"{result.headroom:.0%}")
        val = _get_val(self._lbl_balance)
        if val:
            val.setText(f"{result.balance_score:.0%}")


# ══════════════════════════════════════════════════════════════════════
#  SuggestionPanel (right sidebar)
# ══════════════════════════════════════════════════════════════════════
class SuggestionPanel(QWidget):
    """Right panel: next actions, detailed suggestions, creative direction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # Title
        title = QLabel("Photography Insight")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {_C_TEXT}; padding: 4px 0;")
        layout.addWidget(title)
        layout.addWidget(_separator())

        # ── Next actions ──
        layout.addWidget(_header("Recommended Next Actions"))

        self._next_actions_container = QWidget()
        self._next_actions_layout = QVBoxLayout(self._next_actions_container)
        self._next_actions_layout.setContentsMargins(0, 0, 0, 0)
        self._next_actions_layout.setSpacing(4)
        layout.addWidget(self._next_actions_container)

        # Placeholder
        self._actions_placeholder = QLabel("Waiting for analysis...")
        self._actions_placeholder.setStyleSheet(f"color: {_C_TEXT2}; font-style: italic; padding: 8px;")
        self._next_actions_layout.addWidget(self._actions_placeholder)

        layout.addWidget(_separator())

        # ── Detailed suggestions ──
        layout.addWidget(_header("Detailed Suggestions"))

        self._suggestions_text = QTextEdit()
        self._suggestions_text.setReadOnly(True)
        self._suggestions_text.setPlaceholderText("Suggestions will appear here after analysis...")
        self._suggestions_text.setStyleSheet(f"""
            QTextEdit {{
                background: {_C_SURFACE};
                color: {_C_TEXT};
                border: 1px solid {_C_BORDER};
                border-radius: 4px;
                padding: 8px;
                font-family: "Microsoft YaHei";
                font-size: 10pt;
            }}
        """)
        layout.addWidget(self._suggestions_text, 1)

        layout.addWidget(_separator())

        # ── Creative direction ──
        layout.addWidget(_header("Creative Direction"))

        self._lbl_creative = QLabel("Waiting for analysis...")
        self._lbl_creative.setWordWrap(True)
        self._lbl_creative.setStyleSheet(f"color: {_C_TEXT2}; font-size: 10pt; padding: 8px;")
        self._lbl_creative.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self._lbl_creative)

    # ── Update from SuggestionResult ──
    def update_suggestions(self, result: SuggestionResult):
        # Hide placeholder (may have been deleted by previous _clear_layout)
        try:
            if self._actions_placeholder and self._actions_placeholder.isVisible():
                self._actions_placeholder.hide()
        except RuntimeError:
            self._actions_placeholder = None

        # Clear old next actions
        self._clear_layout(self._next_actions_container.layout())

        # Next action labels
        for i, action in enumerate(result.next_actions):
            rank = f"{i + 1}."
            lbl = QLabel(f"  {rank} {action}")
            lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            lbl.setStyleSheet(f"""
                QLabel {{
                    background: #EFF6FF;
                    color: #1E40AF;
                    border: 1px solid #93C5FD;
                    border-radius: 6px;
                    padding: 8px 12px;
                }}
            """)
            self._next_actions_container.layout().addWidget(lbl)

        # Detailed suggestions (HTML, no emoji)
        html_parts = []
        for s in result.suggestions:
            color_map = {
                SuggestionPriority.HIGH: _C_DANGER,
                SuggestionPriority.MEDIUM: _C_WARNING,
                SuggestionPriority.LOW: _C_ACCENT,
            }
            priority_label = {
                SuggestionPriority.HIGH: "HIGH",
                SuggestionPriority.MEDIUM: "MED",
                SuggestionPriority.LOW: "LOW",
            }
            color = color_map.get(s.priority, _C_TEXT2)
            label = priority_label.get(s.priority, "")
            html_parts.append(
                f'<div style="margin: 6px 0; padding: 6px; border-left: 3px solid {color};">'
                f'<b style="color: {color};">[{label}] {s.title}</b><br>'
                f'<span style="color: {_C_TEXT2};">{s.description}</span>'
                f'</div>'
            )
        self._suggestions_text.setHtml("".join(html_parts))

        # Creative direction
        self._lbl_creative.setText(result.creative_direction)

    # ── Update from RE camera actions ──
    def update_camera_actions(self, re_result):
        """Update suggestion panel with camera actions from RE result."""
        actions = re_result._camera_actions
        if not actions:
            return

        try:
            if self._actions_placeholder and self._actions_placeholder.isVisible():
                self._actions_placeholder.hide()
        except RuntimeError:
            self._actions_placeholder = None

        # Clear old next actions
        self._clear_layout(self._next_actions_container.layout())

        # Add camera actions
        for i, ca in enumerate(actions[:5]):
            rank = f"{i + 1}."
            lbl = QLabel(f"  {rank} {ca.action} -- {ca.expected_effect}")
            lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            lbl.setStyleSheet(f"""
                QLabel {{
                    background: #EFF6FF;
                    color: #1E40AF;
                    border: 1px solid #93C5FD;
                    border-radius: 4px;
                    padding: 6px 10px;
                }}
            """)
            self._next_actions_container.layout().addWidget(lbl)

        # Update creative direction with RE reasons
        reasons = []
        for ca in actions[:3]:
            reasons.extend(ca.reason[:1])
        self._lbl_creative.setText("; ".join(reasons) if reasons else "No specific direction")

    @staticmethod
    def _clear_layout(layout):
        """Remove all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
