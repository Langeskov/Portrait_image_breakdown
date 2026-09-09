"""Extended Chinese UI translations for V3 workspaces.

Keeps the original translation layer intact while covering late-added panels
whose text is created after the main window is built.
"""
from __future__ import annotations

from . import i18n_zh as _base

_EXTRA = {
    "Reference": "参考图",
    "Current": "当前图",
    "Anchored camera cross-check": "锚点相机交叉校验",
    "Solve from anchors": "使用锚点求解",
    "No anchor solve yet": "尚未进行锚点求解",
    "Plane overlay": "平面叠加",
    "Auto / solved plane": "自动 / 已求解平面",
    "Show": "显示",
    "Camera position: —": "相机位置：—",
    "Reprojection RMSE: —": "重投影 RMSE：—",
    "Active camera: unchanged": "活动相机：未改变",
    "Active camera: unchanged · no reference-camera hypothesis yet": "活动相机：未改变 · 暂无参考相机假设",
    "Generated composition-aware pose target": "生成的构图姿态目标",
    "Ready when a reference and current pose are available.": "准备就绪：加载参考图和当前姿态后即可生成。",
    "Generate target": "生成目标",
    "V3 Temporal Reconstruction": "V3 时序重建",
    "Open video": "打开视频",
    "Smoothing": "平滑",
    "Stop": "停止",
    "Idle": "空闲",
    "Sequence complete": "序列处理完成",
    "Frame: —": "帧：—",
    "Temporal confidence: —": "时序置信度：—",
    "Visible landmarks: —": "可见关键点：—",
    "Stopping…": "正在停止……",
    "Select video": "选择视频",
    "Video (*.mp4 *.mov *.avi *.mkv)": "视频 (*.mp4 *.mov *.avi *.mkv)",
    "Reference + current pose are required.": "需要参考姿态和当前姿态。",
    "target center": "目标中心",
    "landmarks": "关键点",
    "Current image could not be read": "无法读取当前图片",
    "Unable to open video": "无法打开视频",
}

_base.ZH.update(_EXTRA)

install_chinese_ui = _base.install_chinese_ui
translate_tree = _base.translate_tree
tr = _base.tr
