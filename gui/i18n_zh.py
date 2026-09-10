"""Compact Chinese UI layer and image-input affordances.

The analysis/reconstruction engines intentionally remain language-neutral. This
module translates user-facing Qt widget chrome without touching model output,
technical identifiers, or persisted data.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QCursor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabBar,
    QToolBar,
    QToolButton,
)


# Exact UI chrome translations only. Dynamic analysis prose is deliberately
# omitted so technical terminology and evidence wording remain intact.
ZH = {
    "Open Image": "打开图片",
    "Dataset: ": "数据集：",
    "Select folder…": "选择文件夹…",
    "Choose Folder": "选择文件夹",
    "Select image folder": "选择图片文件夹",
    "Select Image": "选择图片",
    "Images (*.jpg *.jpeg *.png *.bmp *.webp)": "图片 (*.jpg *.jpeg *.png *.bmp *.webp)",
    "2D Analysis": "2D 分析",
    "3D Reverse Engineering": "3D 反向工程",
    "Results": "结果",
    "Ready": "就绪",
    "No image folder selected": "未选择图片文件夹",
    "2D Overlays": "2D 叠加层",
    "Skeleton": "骨架",
    "3x3 Grid": "三分网格",
    "Center": "中心",
    "BBox": "边界框",
    "Headroom": "头部空间",
    "Reference Target": "参考目标",
    "Visual Weight": "视觉重心",
    "Reverse Evidence": "反向工程证据",
    "Reverse Engineering Report": "反向工程报告",
    "No results yet. Waiting for analysis...": "暂无结果，等待分析……",
    "Reverse engineering not yet complete...": "反向工程尚未完成……",
    "Analysis": "分析",
    "Skeleton Detection": "骨架检测",
    "Confidence": "置信度",
    "Visible keypoints": "可见关键点",
    "Body Orientation": "身体朝向",
    "Facing": "面向",
    "Tilt": "倾斜",
    "Rotation angle": "旋转角",
    "Action Category": "动作类别",
    "Current action": "当前动作",
    "Details": "详情",
    "Joint Angles": "关节角度",
    "Left knee": "左膝",
    "Right knee": "右膝",
    "Left elbow": "左肘",
    "Right elbow": "右肘",
    "Left hip": "左髋",
    "Right hip": "右髋",
    "Camera Analysis": "相机分析",
    "Shot type": "景别",
    "Camera angle": "相机角度",
    "Subject ratio": "主体占比",
    "Dutch angle": "荷兰角",
    "Composition Analysis": "构图分析",
    "Composition type": "构图类型",
    "Thirds alignment": "三分法对齐",
    "Symmetry": "对称度",
    "Balance": "平衡度",
    "Photography Insight": "图片分析",
    "Photographer Cue": "摄影师提示",
    "Waiting for analysis...": "等待分析……",
    "Waiting for analysis…": "等待分析……",
    "Recommended Next Actions": "建议的下一步",
    "Detailed Suggestions": "详细建议",
    "Suggestions will appear here after analysis...": "分析后显示建议……",
    "Creative Direction": "创作方向",
    "No specific direction": "暂无具体方向",
    "FIELD MODE · SHOOTING ASSISTANCE": "现场模式 · 拍摄辅助",
    "Output": "输出",
    "Confidence —": "置信度 —",
    "Landmarks —": "关键点 —",
    "Pose —": "姿态 —",
    "SAY THIS NOW": "现在直接说",
    "Undo": "撤销",
    "Redo": "重做",
    "Copy voice output": "复制语音输出",
    "Copy SSML": "复制 SSML",
    "SECONDARY CUES": "辅助提示",
    "REFERENCE RECONSTRUCTION · V3 PHASE 2": "参考重建 · V3 阶段 2",
    "Load Reference": "加载参考图",
    "Clear": "清除",
    "Load a reference photograph to compare composition and pose.": "加载一张参考照片，用于比较构图和姿态。",
    "Reference": "参考图",
    "Current": "当前图",
    "No image": "无图片",
    "Composition delta: —": "构图偏差：—",
    "Semantic anchor: —": "语义锚点：—",
    "TARGET PLAN": "目标方案",
    "Analyze a reference and current frame to generate a target shooting plan.": "分析参考图和当前画面，生成目标拍摄方案。",
    "Camera & scene reconstruction": "相机与场景重建",
    "No reverse-engineering result yet": "暂无反向工程结果",
    "Camera": "相机",
    "Scene anchors": "场景锚点",
    "Candidate solutions": "候选解",
    "Reference Camera Hypothesis": "参考相机假设",
    "Scene people": "场景人物",
    "Selected anchor": "已选锚点",
    "2D projection preview": "2D 投影预览",
    "No projection yet": "暂无投影",
    "Distance": "距离",
    "Height": "高度",
    "Yaw": "偏航",
    "Pitch": "俯仰",
    "Roll": "滚转",
    "Focal": "焦距",
    "Name": "名称",
    "Kind": "类型",
    "Normal X": "法线 X",
    "Normal Y": "法线 Y",
    "Normal Z": "法线 Z",
    "Width": "宽度",
    "Reference distance": "参考距离",
    "Distance Δ": "距离 Δ",
    "Re-aim yaw": "重新瞄准偏航",
    "Re-aim pitch": "重新瞄准俯仰",
    "Focal prior": "焦距先验",
    "Support": "依据",
    "+ Point": "+ 点",
    "+ Plane": "+ 平面",
    "Remove": "移除",
    "Reset 3D view": "重置 3D 视图",
    "Settings": "设置",
    "Problem feedback": "问题反馈",
    "Subject": "主题",
    "Open mail client": "打开邮件客户端",
    "Advanced": "高级",
    "Pose Model": "姿态模型",
    "Current": "当前",
    "Unable to switch pose model.": "无法切换姿态模型。",
    "Pose model": "姿态模型",
    "Calibration: ": "标定：",
    "Anchor Calibration": "锚点标定",
    "Save reconstruction session": "保存重建会话",
    "Load reconstruction session": "加载重建会话",
    "Save Session": "保存会话",
    "Load Session": "加载会话",
    "Reconstruction session": "重建会话",
    "Save / Load preserves the editable V3 scene, anchors, plane constraints and session metadata.": "保存/加载会保留可编辑的 V3 场景、锚点、平面约束和会话元数据。",
    "Save session failed": "保存会话失败",
    "Load session failed": "加载会话失败",
    "Plane-aware constraints": "平面约束",
    "Add": "添加",
    "Plane": "平面",
    "Relation": "关系",
    "Offset": "偏移",
    "Apply to primary subject": "应用到主人物",
    "Evaluate": "评估",
    "No reference camera context": "暂无参考相机上下文",
}


def tr(text: str) -> str:
    """Translate exact UI chrome, leaving technical/dynamic text unchanged."""
    if text is None:
        return text
    return ZH.get(str(text), str(text))


def _translate_widget(widget) -> None:
    """Translate static Qt chrome recursively without touching model output."""
    widget_types = (QLabel, QPushButton, QCheckBox, QGroupBox, QToolButton, QLineEdit, QComboBox, QTabBar)
    for child_type in widget_types:
        for child in widget.findChildren(child_type):
            if isinstance(child, QComboBox):
                for i in range(child.count()):
                    child.setItemText(i, tr(child.itemText(i)))
            elif isinstance(child, QTabBar):
                for i in range(child.count()):
                    child.setTabText(i, tr(child.tabText(i)))
            elif isinstance(child, QLineEdit):
                child.setPlaceholderText(tr(child.placeholderText()))
            else:
                current = child.text() if hasattr(child, "text") else ""
                translated = tr(current)
                if translated != current:
                    child.setText(translated)

    for bar in widget.findChildren(QToolBar):
        bar.setWindowTitle(tr(bar.windowTitle()))
        for action in bar.actions():
            action.setText(tr(action.text()))

    for action in getattr(widget, "actions", lambda: [])():
        action.setText(tr(action.text()))


def translate_tree(root) -> None:
    """Translate an existing widget tree and its window title."""
    try:
        _translate_widget(root)
        title = getattr(root, "windowTitle", lambda: "")()
        if title:
            root.setWindowTitle(tr(title))
    except RuntimeError:
        return


class _TranslationFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.Show, QEvent.Type.Polish, QEvent.Type.ChildAdded):
            try:
                translate_tree(obj)
            except RuntimeError:
                pass
        return False


def _supported_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def install_canvas_image_input(window) -> None:
    """Make the empty main canvas genuinely clickable and drag-and-drop capable."""
    from gui.canvas import ImageCanvas

    canvas = getattr(getattr(window, "_w2", None), "_cv", None)
    if canvas is None:
        return

    if not getattr(ImageCanvas, "_pib_zh_input_installed", False):
        original_paint = ImageCanvas.paintEvent
        original_mouse = getattr(ImageCanvas, "mousePressEvent", None)

        def paint_event(self, event):
            if getattr(self, "_pixmap", None) is None:
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.fillRect(self.rect(), Qt.GlobalColor.white)
                painter.setPen(QPen(Qt.GlobalColor.gray))
                painter.setFont(QFont("Microsoft YaHei", 14))
                painter.drawText(
                    self.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    "拖入图片，或点击此处选择图片",
                )
                painter.end()
                return
            original_paint(self, event)

        def mouse_press(self, event):
            if event.button() == Qt.MouseButton.LeftButton and getattr(self, "_pixmap", None) is None:
                handler = getattr(self, "_pib_open_handler", None)
                if callable(handler):
                    handler()
                event.accept()
                return
            if original_mouse is not None:
                original_mouse(self, event)
            else:
                event.ignore()

        def drag_enter(self, event):
            mime = event.mimeData()
            if mime is not None and mime.hasUrls():
                for url in mime.urls():
                    if url.isLocalFile() and _supported_image(url.toLocalFile()):
                        event.acceptProposedAction()
                        return
            event.ignore()

        def drop_event(self, event):
            mime = event.mimeData()
            if mime is not None and mime.hasUrls():
                for url in mime.urls():
                    if url.isLocalFile() and _supported_image(url.toLocalFile()):
                        path = url.toLocalFile()
                        handler = getattr(self, "_pib_open_handler", None)
                        if callable(handler):
                            handler(path)
                        event.acceptProposedAction()
                        return
            event.ignore()

        ImageCanvas.paintEvent = paint_event
        ImageCanvas.mousePressEvent = mouse_press
        ImageCanvas.dragEnterEvent = drag_enter
        ImageCanvas.dropEvent = drop_event
        ImageCanvas._pib_zh_input_installed = True

    canvas.setAcceptDrops(True)
    canvas.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    canvas._pib_open_handler = lambda path=None: window._open() if path is None else window.load_image(path)


def install_chinese_ui(window) -> None:
    """Install lightweight live translation for existing and future dialogs."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None and not getattr(app, "_pib_zh_filter", None):
        filt = _TranslationFilter(app)
        app.installEventFilter(filt)
        app._pib_zh_filter = filt
    translate_tree(window)
    install_canvas_image_input(window)
