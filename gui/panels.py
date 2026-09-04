"""
分析结果面板组件

显示骨架、朝向、动作、镜头、构图的分析结果, 以及下一动作建议。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QGroupBox, QGridLayout, QFrame, QScrollArea, QSizePolicy,
)

from core.action_classifier import ActionResult
from core.camera_analyzer import CameraResult
from core.composition import CompositionResult
from core.orientation import OrientationResult
from core.suggestion import SuggestionResult, SuggestionPriority


def _make_header(text: str) -> QLabel:
    """创建分组标题"""
    lbl = QLabel(text)
    lbl.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
    lbl.setStyleSheet("color: #E0E0E0; padding: 4px 0;")
    return lbl


def _make_stat(label: str, value: str) -> QWidget:
    """创建一行统计项: 标签 + 值"""
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 2, 0, 2)

    lbl = QLabel(label)
    lbl.setFont(QFont("Microsoft YaHei", 9))
    lbl.setStyleSheet("color: #AAAAAA;")
    lbl.setFixedWidth(100)

    val = QLabel(value)
    val.setFont(QFont("Consolas", 9))
    val.setStyleSheet("color: #E0E0E0;")

    layout.addWidget(lbl)
    layout.addWidget(val, 1)
    return w


def _make_separator() -> QFrame:
    """创建分隔线"""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #444444;")
    return line


class AnalysisPanel(QWidget):
    """
    左侧分析结果面板

    包含5个分组: 骨架、朝向、动作、镜头、构图
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1E1E1E; }")

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)

        # ── 标题 ──
        title = QLabel("📊 分析结果")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF; padding: 4px 0;")
        self._layout.addWidget(title)
        self._layout.addWidget(_make_separator())

        # ── 骨架检测 ──
        self._layout.addWidget(_make_header("🦴 骨架检测"))
        self._lbl_pose_conf = _make_stat("检测置信度", "—")
        self._lbl_pose_count = _make_stat("可见关键点", "—")
        self._layout.addWidget(self._lbl_pose_conf)
        self._layout.addWidget(self._lbl_pose_count)
        self._layout.addWidget(_make_separator())

        # ── 身体朝向 ──
        self._layout.addWidget(_make_header("🧭 身体朝向"))
        self._lbl_facing = _make_stat("朝向", "—")
        self._lbl_tilt = _make_stat("倾斜", "—")
        self._lbl_facing_angle = _make_stat("旋转角度", "—")
        self._layout.addWidget(self._lbl_facing)
        self._layout.addWidget(self._lbl_tilt)
        self._layout.addWidget(self._lbl_facing_angle)
        self._layout.addWidget(_make_separator())

        # ── 动作类别 ──
        self._layout.addWidget(_make_header("🏃 动作类别"))
        self._lbl_action = _make_stat("当前动作", "—")
        self._lbl_action_conf = _make_stat("置信度", "—")
        self._lbl_action_detail = _make_stat("细节", "—")
        self._layout.addWidget(self._lbl_action)
        self._layout.addWidget(self._lbl_action_conf)
        self._layout.addWidget(self._lbl_action_detail)

        # 关节角度(可展开)
        self._lbl_angles_header = _make_header("📐 关节角度")
        self._layout.addWidget(self._lbl_angles_header)
        self._angle_labels: dict[str, QLabel] = {}
        for name in ["左膝", "右膝", "左肘", "右肘", "左髋", "右髋"]:
            lbl = _make_stat(name, "—")
            self._angle_labels[name] = lbl.findChild(QLabel, "") or lbl
            self._layout.addWidget(lbl)
        self._layout.addWidget(_make_separator())

        # ── 镜头分析 ──
        self._layout.addWidget(_make_header("📷 镜头分析"))
        self._lbl_shot = _make_stat("镜头类型", "—")
        self._lbl_cam_angle = _make_stat("拍摄角度", "—")
        self._lbl_subject_ratio = _make_stat("人物占比", "—")
        self._lbl_dutch = _make_stat("画面倾斜", "—")
        self._layout.addWidget(self._lbl_shot)
        self._layout.addWidget(self._lbl_cam_angle)
        self._layout.addWidget(self._lbl_subject_ratio)
        self._layout.addWidget(self._lbl_dutch)
        self._layout.addWidget(_make_separator())

        # ── 构图分析 ──
        self._layout.addWidget(_make_header("🖼️ 构图分析"))
        self._lbl_comp_type = _make_stat("构图类型", "—")
        self._lbl_thirds = _make_stat("三分法对齐", "—")
        self._lbl_symmetry = _make_stat("对称性", "—")
        self._lbl_headroom = _make_stat("头部空间", "—")
        self._lbl_balance = _make_stat("画面平衡", "—")
        self._layout.addWidget(self._lbl_comp_type)
        self._layout.addWidget(self._lbl_thirds)
        self._layout.addWidget(self._lbl_symmetry)
        self._layout.addWidget(self._lbl_headroom)
        self._layout.addWidget(self._lbl_balance)

        self._layout.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def update_pose(self, conf: float, visible_count: int):
        self._lbl_pose_conf.findChild(QLabel, "").setText(f"{conf:.0%}") if self._lbl_pose_conf.findChild(QLabel, "") else None
        # 直接设置第二个label
        children = self._lbl_pose_conf.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{conf:.0%}")

        children = self._lbl_pose_count.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{visible_count}/33")

    def update_orientation(self, result: OrientationResult):
        children = self._lbl_facing.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(result.facing.value)
            children[1].setStyleSheet("color: #4FC3F7;")

        children = self._lbl_tilt.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(result.tilt.value)

        children = self._lbl_facing_angle.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{result.facing_angle:.1f}°")

    def update_action(self, result: ActionResult):
        children = self._lbl_action.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(result.category.value)
            children[1].setStyleSheet("color: #81C784; font-weight: bold;")

        children = self._lbl_action_conf.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{result.confidence:.0%}")

        children = self._lbl_action_detail.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(result.sub_description)

        # 更新关节角度
        angle_map = {
            "左膝": "left_knee", "右膝": "right_knee",
            "左肘": "left_elbow", "右肘": "right_elbow",
            "左髋": "left_hip", "右髋": "right_hip",
        }
        for cn_name, en_key in angle_map.items():
            if en_key in result.joint_angles:
                w = self._angle_labels.get(cn_name)
                if w:
                    children = w.findChildren(QLabel)
                    if len(children) >= 2:
                        children[1].setText(f"{result.joint_angles[en_key]:.1f}°")

    def update_camera(self, result: CameraResult):
        children = self._lbl_shot.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(result.shot_type.value)

        children = self._lbl_cam_angle.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(result.camera_angle.value)

        children = self._lbl_subject_ratio.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{result.subject_ratio:.1%}")

        children = self._lbl_dutch.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{result.dutch_angle_deg:.1f}°")

    def update_composition(self, result: CompositionResult):
        children = self._lbl_comp_type.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(result.primary_type.value)

        children = self._lbl_thirds.findChildren(QLabel)
        if len(children) >= 2:
            val = children[1]
            val.setText(f"{result.thirds_alignment:.0%}")
            val.setStyleSheet("color: #81C784;" if result.thirds_alignment > 0.6 else "color: #FFB74D;")

        children = self._lbl_symmetry.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{result.symmetry_score:.0%}")

        children = self._lbl_headroom.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{result.headroom:.0%}")

        children = self._lbl_balance.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{result.balance_score:.0%}")


class SuggestionPanel(QWidget):
    """
    右侧建议面板

    显示下一动作建议和创意方向
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # 标题
        title = QLabel("💡 下一动作建议")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF; padding: 4px 0;")
        layout.addWidget(title)
        layout.addWidget(_make_separator())

        # 推荐动作
        self._lbl_next_header = _make_header("🎯 推荐下一动作")
        layout.addWidget(self._lbl_next_header)

        self._next_actions_container = QWidget()
        self._next_actions_layout = QVBoxLayout(self._next_actions_container)
        self._next_actions_layout.setContentsMargins(0, 0, 0, 0)
        self._next_actions_layout.setSpacing(4)
        layout.addWidget(self._next_actions_container)

        layout.addWidget(_make_separator())

        # 建议列表
        self._lbl_suggestions_header = _make_header("📋 详细建议")
        layout.addWidget(self._lbl_suggestions_header)

        self._suggestions_text = QTextEdit()
        self._suggestions_text.setReadOnly(True)
        self._suggestions_text.setStyleSheet("""
            QTextEdit {
                background: #252525;
                color: #E0E0E0;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 8px;
                font-family: "Microsoft YaHei";
                font-size: 10pt;
            }
        """)
        layout.addWidget(self._suggestions_text, 1)

        layout.addWidget(_make_separator())

        # 创意方向
        self._lbl_creative_header = _make_header("🎨 创意方向")
        layout.addWidget(self._lbl_creative_header)

        self._lbl_creative = QLabel("等待分析...")
        self._lbl_creative.setWordWrap(True)
        self._lbl_creative.setStyleSheet("color: #B39DDB; font-size: 10pt; padding: 8px;")
        self._lbl_creative.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self._lbl_creative)

    def update_suggestions(self, result: SuggestionResult):
        # 清除旧的下一动作
        while self._next_actions_layout.count():
            item = self._next_actions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 下一动作按钮样式
        for i, action in enumerate(result.next_actions):
            lbl = QLabel(f"  {'🥇🥈🥉'[i] if i < 3 else '•'} {action}")
            lbl.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            lbl.setStyleSheet("""
                QLabel {
                    background: #2A3A2A;
                    color: #A5D6A7;
                    border: 1px solid #4CAF50;
                    border-radius: 6px;
                    padding: 8px 12px;
                }
            """)
            self._next_actions_layout.addWidget(lbl)

        # 建议文本
        html_parts = []
        for s in result.suggestions:
            color_map = {
                SuggestionPriority.HIGH: "#FF8A80",
                SuggestionPriority.MEDIUM: "#FFD180",
                SuggestionPriority.LOW: "#80D8FF",
            }
            color = color_map.get(s.priority, "#E0E0E0")
            html_parts.append(
                f'<div style="margin: 6px 0; padding: 6px; border-left: 3px solid {color};">'
                f'<span style="font-size: 14px;">{s.icon}</span> '
                f'<b style="color: {color};">{s.title}</b><br>'
                f'<span style="color: #CCCCCC;">{s.description}</span>'
                f'</div>'
            )
        self._suggestions_text.setHtml("".join(html_parts))

        # 创意方向
        self._lbl_creative.setText(result.creative_direction)

    def update_camera_result(self, re_result):
        """Update camera panel from ReverseEngineeringResult"""
        cp = re_result.camera_pose
        fl = re_result.focal_length
        children = self._lbl_shot.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(fl.category.value)
        children = self._lbl_cam_angle.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"pitch {cp.camera_pitch.value} deg")
        children = self._lbl_subject_ratio.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"dist {cp.camera_distance.value}m")
        children = self._lbl_dutch.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"roll {cp.camera_roll.value} deg")

    def update_composition_result(self, re_result):
        """Update composition panel from ReverseEngineeringResult"""
        comp = re_result.composition
        children = self._lbl_comp_type.findChildren(QLabel)
        if len(children) >= 2:
            names = [s["name"] for s in comp.styles[:2]]
            children[1].setText(", ".join(names) if names else "N/A")
        children = self._lbl_thirds.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{comp.subject_position[0]:.2f}, {comp.subject_position[1]:.2f}")
        children = self._lbl_balance.findChildren(QLabel)
        if len(children) >= 2:
            children[1].setText(f"{comp.subject_scale:.0%}")
    def update_camera_actions(self, re_result):
        """Update suggestion panel with camera actions from RE result"""
        actions = re_result._camera_actions
        if not actions:
            return
        # Clear old next actions
        while self._next_actions_layout.count():
            item = self._next_actions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Add camera actions
        medals = ["1st", "2nd", "3rd"]
        for i, ca in enumerate(actions[:5]):
            lbl = QLabel("  " + medals[i] + ": " + ca.action + " - " + ca.expected_effect)
            lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            lbl.setStyleSheet("background: #EFF6FF; color: #1E40AF; border: 1px solid #93C5FD; border-radius: 4px; padding: 6px 10px;")
            self._next_actions_layout.addWidget(lbl)
        # Update creative direction
        reasons = []
        for ca in actions[:3]:
            reasons.extend(ca.reason[:1])
        self._lbl_creative.setText("; ".join(reasons) if reasons else "No specific direction")