"""Analysis panels — technical evidence plus photographer-ready cues."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QFrame,
    QScrollArea, QToolButton, QSizePolicy,
)

from core.action_classifier import ActionResult
from core.camera_analyzer import CameraResult
from core.composition import CompositionResult
from core.orientation import OrientationResult
from core.photographer_cues import CuePriority
from core.suggestion import SuggestionResult, SuggestionPriority

_C_PANEL = "#FFFFFF"
_C_SURFACE = "#FAFBFC"
_C_BORDER = "#E2E8F0"
_C_TEXT = "#1F2937"
_C_TEXT2 = "#6B7280"
_C_ACCENT = "#2563EB"
_C_SUCCESS = "#16A34A"
_C_WARNING = "#D97706"
_C_DANGER = "#DC2626"


def _header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.DemiBold))
    lbl.setStyleSheet(f"color: {_C_TEXT}; padding: 4px 0 2px 0;")
    return lbl


def _stat_row(label: str, value: str = "--") -> QWidget:
    w = QWidget()
    lo = QHBoxLayout(w)
    lo.setContentsMargins(0, 1, 0, 1)
    lo.setSpacing(8)
    lbl = QLabel(label)
    lbl.setFont(QFont("Microsoft YaHei", 9))
    lbl.setStyleSheet(f"color: {_C_TEXT2};")
    lbl.setFixedWidth(100)
    val = QLabel(value)
    val.setFont(QFont("Consolas", 9))
    val.setStyleSheet(f"color: {_C_TEXT};")
    val.setObjectName("_value_label")
    lo.addWidget(lbl)
    lo.addWidget(val, 1)
    return w


def _get_val(widget: QWidget) -> QLabel:
    return widget.findChild(QLabel, "_value_label")


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"color: {_C_BORDER}; max-height: 1px;")
    return line


# ---------------------------------------------------------------------------
# Collapsible section widget
# ---------------------------------------------------------------------------

class CollapsibleSection(QWidget):
    """A titled section that can be collapsed/expanded."""

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._expanded = expanded

        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(0)

        # Header row with toggle button
        header = QWidget()
        header_lo = QHBoxLayout(header)
        header_lo.setContentsMargins(0, 2, 0, 2)
        header_lo.setSpacing(4)

        self._toggle = QToolButton()
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.setStyleSheet(
            "QToolButton { border: none; padding: 2px; }"
        )
        self._toggle.setFixedSize(16, 16)
        self._toggle.clicked.connect(self._on_toggle)
        header_lo.addWidget(self._toggle)

        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        title_lbl.setStyleSheet(f"color: {_C_TEXT};")
        header_lo.addWidget(title_lbl, 1)
        lo.addWidget(header)

        # Content area
        self._body = QWidget()
        self._body.setVisible(expanded)
        self._body_lo = QVBoxLayout(self._body)
        self._body_lo.setContentsMargins(16, 2, 0, 4)
        self._body_lo.setSpacing(3)
        lo.addWidget(self._body)

    def body_layout(self) -> QVBoxLayout:
        return self._body_lo

    def add_widget(self, widget: QWidget):
        self._body_lo.addWidget(widget)

    def add_layout(self, layout):
        self._body_lo.addLayout(layout)

    def _on_toggle(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )


# ---------------------------------------------------------------------------
# Left panel: Analysis
# ---------------------------------------------------------------------------

class AnalysisPanel(QWidget):
    """Left panel containing the professional technical analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(380)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {_C_PANEL}; }}")
        container = QWidget()
        lo = QVBoxLayout(container)
        lo.setContentsMargins(10, 10, 10, 10)
        lo.setSpacing(2)

        # Title
        title = QLabel("Analysis")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {_C_TEXT}; padding: 2px 0 4px 0;")
        lo.addWidget(title)

        # Section 1: Detection Quality
        sec_detect = CollapsibleSection("检测质量", expanded=True)
        self._lbl_pose_conf = _stat_row("置信度")
        self._lbl_pose_count = _stat_row("可见关键点")
        sec_detect.add_widget(self._lbl_pose_conf)
        sec_detect.add_widget(self._lbl_pose_count)
        lo.addWidget(sec_detect)

        # Section 2: Pose Summary
        sec_pose = CollapsibleSection("姿态摘要", expanded=True)
        self._lbl_facing = _stat_row("朝向")
        self._lbl_tilt = _stat_row("倾斜")
        self._lbl_facing_angle = _stat_row("旋转角度")
        self._lbl_action = _stat_row("动作分类")
        self._lbl_action_conf = _stat_row("分类置信度")
        self._lbl_action_detail = _stat_row("细节")
        for w in [self._lbl_facing, self._lbl_tilt, self._lbl_facing_angle,
                  self._lbl_action, self._lbl_action_conf, self._lbl_action_detail]:
            sec_pose.add_widget(w)
        lo.addWidget(sec_pose)

        # Section 3: Joint Angles
        sec_joints = CollapsibleSection("关节角度", expanded=False)
        self._angle_labels: dict[str, QWidget] = {}
        for name in ["Left knee", "Right knee", "Left elbow", "Right elbow",
                      "Left hip", "Right hip"]:
            row = _stat_row(name)
            self._angle_labels[name] = row
            sec_joints.add_widget(row)
        lo.addWidget(sec_joints)

        # Section 4: Camera & Composition
        sec_cam = CollapsibleSection("相机与构图", expanded=True)
        self._lbl_shot = _stat_row("景别")
        self._lbl_cam_angle = _stat_row("视角")
        self._lbl_subject_ratio = _stat_row("主体占比")
        self._lbl_dutch = _stat_row("倾斜角")
        self._lbl_comp_type = _stat_row("构图类型")
        self._lbl_thirds = _stat_row("三分线对齐")
        self._lbl_symmetry = _stat_row("对称性")
        self._lbl_headroom = _stat_row("头部空间")
        self._lbl_balance = _stat_row("平衡")
        for w in [self._lbl_shot, self._lbl_cam_angle, self._lbl_subject_ratio,
                  self._lbl_dutch, self._lbl_comp_type, self._lbl_thirds,
                  self._lbl_symmetry, self._lbl_headroom, self._lbl_balance]:
            sec_cam.add_widget(w)
        lo.addWidget(sec_cam)

        lo.addStretch(1)
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

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


# ---------------------------------------------------------------------------
# Right panel: Suggestions / Photographer cues
# ---------------------------------------------------------------------------

class SuggestionPanel(QWidget):
    """Right panel: verbal cue first, then detailed professional reasoning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        title = QLabel("Photography Insight")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {_C_TEXT}; padding: 2px 0 4px 0;")
        layout.addWidget(title)

        # Primary cue — clean, no heavy border
        layout.addWidget(_header("摄影师提示"))
        self._cue_primary = QLabel("等待分析…")
        self._cue_primary.setWordWrap(True)
        self._cue_primary.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
        self._cue_primary.setStyleSheet(
            f"color: {_C_TEXT}; padding: 8px 10px; "
            f"background: {_C_SURFACE}; border: 1px solid {_C_BORDER};"
        )
        layout.addWidget(self._cue_primary)

        self._cue_reason = QLabel("")
        self._cue_reason.setWordWrap(True)
        self._cue_reason.setStyleSheet(f"color: {_C_TEXT2}; padding: 2px 4px 4px 4px; font-size: 9pt;")
        layout.addWidget(self._cue_reason)

        layout.addWidget(_separator())

        # Next actions — compact list
        layout.addWidget(_header("建议下一步"))
        self._next_actions_container = QWidget()
        self._next_actions_layout = QVBoxLayout(self._next_actions_container)
        self._next_actions_layout.setContentsMargins(0, 0, 0, 0)
        self._next_actions_layout.setSpacing(3)
        layout.addWidget(self._next_actions_container)
        self._actions_placeholder = QLabel("等待分析…")
        self._actions_placeholder.setStyleSheet(f"color: {_C_TEXT2}; font-style: italic; padding: 4px;")
        self._next_actions_layout.addWidget(self._actions_placeholder)

        layout.addWidget(_separator())

        # Detailed suggestions
        layout.addWidget(_header("详细建议"))
        self._suggestions_text = QTextEdit()
        self._suggestions_text.setReadOnly(True)
        self._suggestions_text.setPlaceholderText("分析完成后显示建议…")
        self._suggestions_text.setStyleSheet(
            f"QTextEdit {{ background: {_C_SURFACE}; color: {_C_TEXT}; "
            f"border: 1px solid {_C_BORDER}; padding: 6px; "
            f"font-family: 'Microsoft YaHei'; font-size: 9pt; }}"
        )
        layout.addWidget(self._suggestions_text, 1)

        layout.addWidget(_separator())

        # Creative direction
        layout.addWidget(_header("创作方向"))
        self._lbl_creative = QLabel("等待分析…")
        self._lbl_creative.setWordWrap(True)
        self._lbl_creative.setStyleSheet(f"color: {_C_TEXT2}; font-size: 9pt; padding: 4px;")
        self._lbl_creative.setFont(QFont("Microsoft YaHei", 9))
        layout.addWidget(self._lbl_creative)

    def update_photographer_cues(self, cues):
        if not cues:
            self._cue_primary.setText("先保持自然，我会根据画面继续调整。")
            self._cue_reason.setText("")
            return
        primary = next((c for c in cues if c.priority == CuePriority.PRIMARY), cues[0])
        self._cue_primary.setText(primary.cue)
        self._cue_reason.setText(primary.reason)

    def update_suggestions(self, result: SuggestionResult):
        self.update_photographer_cues(result.photographer_cues)
        try:
            if self._actions_placeholder and self._actions_placeholder.isVisible():
                self._actions_placeholder.hide()
        except RuntimeError:
            self._actions_placeholder = None
        self._clear_layout(self._next_actions_container.layout())
        for i, action in enumerate(result.next_actions):
            lbl = QLabel(f"  {i + 1}. {action}")
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setStyleSheet(
                f"QLabel {{ color: {_C_TEXT}; padding: 5px 8px; "
                f"background: {_C_SURFACE}; border: 1px solid {_C_BORDER}; }}"
            )
            self._next_actions_container.layout().addWidget(lbl)

        html_parts = []
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
        for s in result.suggestions:
            color = color_map.get(s.priority, _C_TEXT2)
            label = priority_label.get(s.priority, "")
            html_parts.append(
                f'<div style="margin: 4px 0; padding: 4px 6px; '
                f'border-left: 2px solid {color};">'
                f'<b style="color: {color}; font-size: 9pt;">[{label}] {s.title}</b><br>'
                f'<span style="color: {_C_TEXT2}; font-size: 9pt;">{s.description}</span></div>'
            )
        self._suggestions_text.setHtml("".join(html_parts))
        self._lbl_creative.setText(result.creative_direction)

    def update_camera_actions(self, re_result):
        actions = re_result._camera_actions
        if not actions:
            return
        self._clear_layout(self._next_actions_container.layout())
        for i, ca in enumerate(actions[:5]):
            lbl = QLabel(f"  {i + 1}. {ca.action} -- {ca.expected_effect}")
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setStyleSheet(
                f"QLabel {{ color: {_C_TEXT}; padding: 5px 8px; "
                f"background: {_C_SURFACE}; border: 1px solid {_C_BORDER}; }}"
            )
            self._next_actions_container.layout().addWidget(lbl)
        reasons = []
        for ca in actions[:3]:
            reasons.extend(ca.reason[:1])
        self._lbl_creative.setText("; ".join(reasons) if reasons else "No specific direction")

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()