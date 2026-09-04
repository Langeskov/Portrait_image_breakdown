"""
图像画布组件 — 显示图像并叠加骨架/构图辅助线
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

from core.pose_detector import PoseResult, POSE_CONNECTIONS, LandmarkIndex as LI
from core.camera_analyzer import CameraResult
from core.composition import CompositionResult


# ── 颜色定义 ──
COLOR_SKELETON = QColor(0, 255, 128)       # 骨架线条 - 绿色
COLOR_KEYPOINT = QColor(255, 80, 80)       # 关键点 - 红色
COLOR_KEYPOINT_LOW = QColor(128, 128, 128) # 低置信度关键点 - 灰色
COLOR_THIRDS = QColor(255, 255, 0, 80)     # 三分法线 - 半透明黄
COLOR_CENTER = QColor(0, 200, 255, 60)     # 中心线 - 半透明蓝
COLOR_BBOX = QColor(255, 165, 0, 120)      # 人体包围框 - 橙色


class ImageCanvas(QWidget):
    """
    图像显示画布, 支持:
    - 自适应缩放
    - 骨架叠加绘制
    - 构图辅助线(三分法/中心线)
    - 人体包围框
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self._pixmap: Optional[QPixmap] = None
        self._original_image: Optional[np.ndarray] = None
        self._pose: Optional[PoseResult] = None
        self._camera: Optional[CameraResult] = None
        self._composition: Optional[CompositionResult] = None
        self._show_skeleton = True
        self._show_thirds = True
        self._show_bbox = True
        self._scale = 1.0
        self._offset = QPoint(0, 0)

    def set_image(self, image: np.ndarray):
        """设置要显示的BGR图像"""
        self._original_image = image.copy()
        self._update_pixmap()

    def set_pose(self, pose: Optional[PoseResult]):
        """设置骨架检测结果"""
        self._pose = pose
        self.update()

    def set_camera(self, camera: Optional[CameraResult]):
        self._camera = camera
        self.update()

    def set_composition(self, composition: Optional[CompositionResult]):
        self._composition = composition
        self.update()

    def set_overlay_options(self, skeleton=True, thirds=True, bbox=True):
        self._show_skeleton = skeleton
        self._show_thirds = thirds
        self._show_bbox = bbox
        self.update()

    def _update_pixmap(self):
        if self._original_image is None:
            self._pixmap = None
            return

        rgb = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap is None:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Microsoft YaHei", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "拖入图片或点击「打开图片」")
            return

        # 计算缩放和偏移以适应窗口
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        ww = self.width()
        wh = self.height()

        scale_x = ww / pw
        scale_y = wh / ph
        self._scale = min(scale_x, scale_y) * 0.95

        draw_w = int(pw * self._scale)
        draw_h = int(ph * self._scale)
        self._offset = QPoint((ww - draw_w) // 2, (wh - draw_h) // 2)

        # 绘制图像
        target = QRect(self._offset.x(), self._offset.y(), draw_w, draw_h)
        painter.drawPixmap(target, self._pixmap)

        # ── 叠加层 ──
        sx = self._scale
        sy = self._scale
        ox = self._offset.x()
        oy = self._offset.y()

        # 三分法辅助线
        if self._show_thirds and self._composition:
            pen = QPen(COLOR_THIRDS, 1, Qt.DashLine)
            painter.setPen(pen)
            for frac in [1/3, 2/3]:
                # 垂直线
                x = ox + int(draw_w * frac)
                painter.drawLine(x, oy, x, oy + draw_h)
                # 水平线
                y = oy + int(draw_h * frac)
                painter.drawLine(ox, y, ox + draw_w, y)

            # 主体位置标记
            px, py = self._composition.subject_position
            mark_x = ox + int(draw_w * px)
            mark_y = oy + int(draw_h * py)
            pen = QPen(QColor(255, 0, 255, 180), 2)
            painter.setPen(pen)
            painter.drawEllipse(QPoint(mark_x, mark_y), 8, 8)
            painter.drawLine(mark_x - 12, mark_y, mark_x + 12, mark_y)
            painter.drawLine(mark_x, mark_y - 12, mark_x, mark_y + 12)

        # 中心线
        if self._show_thirds:
            pen = QPen(COLOR_CENTER, 1, Qt.DotLine)
            painter.setPen(pen)
            cx = ox + draw_w // 2
            cy = oy + draw_h // 2
            painter.drawLine(cx, oy, cx, oy + draw_h)
            painter.drawLine(ox, cy, ox + draw_w, cy)

        # 骨架
        if self._show_skeleton and self._pose:
            self._draw_skeleton(painter, ox, oy, sx, sy)

        # 人体包围框
        if self._show_bbox and self._pose:
            self._draw_bbox(painter, ox, oy, sx, sy)

        painter.end()

    def _draw_skeleton(self, painter: QPainter, ox, oy, sx, sy):
        """绘制骨架"""
        pose = self._pose
        if pose is None:
            return

        # 获取像素坐标
        pts = {}
        for i, lm in enumerate(pose.landmarks):
            if lm.visibility > 0.3:
                px = ox + int(lm.world_x * pose.image_width * sx)
                py = oy + int(lm.world_y * pose.image_height * sy)
                pts[i] = QPoint(px, py)

        # 绘制连接线
        pen = QPen(COLOR_SKELETON, 2, Qt.SolidLine)
        painter.setPen(pen)
        for a, b in POSE_CONNECTIONS:
            if a in pts and b in pts:
                painter.drawLine(pts[a], pts[b])

        # 绘制关键点
        for idx, pt in pts.items():
            lm = pose.landmarks[idx]
            if lm.visibility > 0.6:
                painter.setBrush(QBrush(COLOR_KEYPOINT))
                painter.setPen(QPen(COLOR_KEYPOINT, 1))
                r = 4
            else:
                painter.setBrush(QBrush(COLOR_KEYPOINT_LOW))
                painter.setPen(QPen(COLOR_KEYPOINT_LOW, 1))
                r = 3

            painter.drawEllipse(pt, r, r)

        # 标注主要关节名称
        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 180))
        label_map = {
            LI.LEFT_SHOULDER.value: "L-肩",
            LI.RIGHT_SHOULDER.value: "R-肩",
            LI.LEFT_ELBOW.value: "L-肘",
            LI.RIGHT_ELBOW.value: "R-肘",
            LI.LEFT_WRIST.value: "L-腕",
            LI.RIGHT_WRIST.value: "R-腕",
            LI.LEFT_HIP.value: "L-髋",
            LI.RIGHT_HIP.value: "R-髋",
            LI.LEFT_KNEE.value: "L-膝",
            LI.RIGHT_KNEE.value: "R-膝",
        }
        for idx_val, label in label_map.items():
            if idx_val in pts:
                pt = pts[idx_val]
                painter.drawText(pt.x() + 6, pt.y() - 6, label)

    def _draw_bbox(self, painter: QPainter, ox, oy, sx, sy):
        """绘制人体包围框"""
        pose = self._pose
        if pose is None:
            return

        visible_pts = []
        for lm in pose.landmarks[:17]:  # only COCO 17 points
            if lm.visibility > 0.4:
                px = ox + int(lm.world_x * pose.image_width * sx)
                py = oy + int(lm.world_y * pose.image_height * sy)
                visible_pts.append((px, py))

        if len(visible_pts) < 4:
            return

        xs = [p[0] for p in visible_pts]
        ys = [p[1] for p in visible_pts]

        margin = 15
        x1, y1 = min(xs) - margin, min(ys) - margin
        x2, y2 = max(xs) + margin, max(ys) + margin

        pen = QPen(COLOR_BBOX, 2, Qt.DashDotLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
