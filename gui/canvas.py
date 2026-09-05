"""ImageCanvas — light-theme image viewer with analysis evidence overlays."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush
from PySide6.QtWidgets import QWidget

from core.pose_detector import PoseResult, POSE_CONNECTIONS, LandmarkIndex as LI
from core.camera_analyzer import CameraResult
from core.composition import CompositionResult
from reverse_engineering.data_types import ReverseEngineeringResult

COLOR_SKELETON = QColor(34, 197, 94)
COLOR_SKELETON_SECONDARY = QColor(59, 130, 246)
COLOR_KEYPOINT = QColor(239, 68, 68)
COLOR_KEYPOINT_LOW = QColor(156, 163, 175)
COLOR_THIRDS = QColor(37, 99, 235, 60)
COLOR_CENTER = QColor(251, 146, 60, 80)
COLOR_BBOX = QColor(245, 158, 11, 160)
COLOR_SUBJECT_MARK = QColor(168, 85, 247)
COLOR_HEADROOM = QColor(20, 184, 166, 100)
COLOR_VWEIGHT = QColor(244, 63, 94, 120)
COLOR_REVERSE = QColor(217, 119, 6, 210)
COLOR_REVERSE_SOFT = QColor(217, 119, 6, 110)
COLOR_REVERSE_TEXT = QColor(146, 64, 14)
CANVAS_BG = QColor(250, 251, 252)
CANVAS_TEXT = QColor(107, 114, 128)


class ImageCanvas(QWidget):
    """Image display canvas with explicit 2D evidence overlay layers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self._pixmap: Optional[QPixmap] = None
        self._original_image: Optional[np.ndarray] = None
        self._pose: Optional[PoseResult] = None
        self._poses: list[PoseResult] = []
        self._camera: Optional[CameraResult] = None
        self._composition: Optional[CompositionResult] = None
        self._reverse: Optional[ReverseEngineeringResult] = None
        self._show_skeleton = True
        self._show_thirds = True
        self._show_center = True
        self._show_bbox = True
        self._show_visual_weight = False
        self._show_headroom = False
        self._show_reverse = False
        self._show_reverse_lines = True
        self._show_reverse_vp = True
        self._show_reverse_axis = True
        self._scale = 1.0
        self._offset = QPoint(0, 0)

    def set_image(self, image: np.ndarray):
        self._original_image = image.copy()
        self._update_pixmap()
        if self._pose is not None:
            self._set_pose_collection(self._pose)

    def _set_pose_collection(self, pose: PoseResult):
        people = getattr(pose, "persons", None) or [pose]
        target_w = self._original_image.shape[1] if self._original_image is not None else pose.image_width
        target_h = self._original_image.shape[0] if self._original_image is not None else pose.image_height
        self._poses = [p.rescaled(target_w, target_h) if (p.image_width, p.image_height) != (target_w, target_h) else p for p in people]
        self._pose = self._poses[0] if self._poses else None

    def set_pose(self, pose: Optional[PoseResult]):
        self._pose = pose
        self._poses = []
        if pose is not None:
            self._set_pose_collection(pose)
        self.update()

    def set_camera(self, camera: Optional[CameraResult]):
        self._camera = camera
        self.update()

    def set_composition(self, composition: Optional[CompositionResult]):
        self._composition = composition
        self.update()

    def set_reverse_result(self, result: Optional[ReverseEngineeringResult]):
        self._reverse = result
        self.update()

    def set_overlay_options(self, skeleton=True, thirds=True, center=True,
                            bbox=True, visual_weight=False, headroom=False,
                            reverse=False, reverse_lines=True,
                            reverse_vp=True, reverse_axis=True):
        self._show_skeleton = skeleton
        self._show_thirds = thirds
        self._show_center = center
        self._show_bbox = bbox
        self._show_visual_weight = visual_weight
        self._show_headroom = headroom
        self._show_reverse = reverse
        self._show_reverse_lines = reverse_lines
        self._show_reverse_vp = reverse_vp
        self._show_reverse_axis = reverse_axis
        self.update()

    def _update_pixmap(self):
        if self._original_image is None:
            self._pixmap = None
            self.update()
            return
        rgb = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), CANVAS_BG)
        if self._pixmap is None:
            painter.setPen(CANVAS_TEXT)
            painter.setFont(QFont("Microsoft YaHei", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "Drag an image here or click Open Image")
            painter.end()
            return

        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        self._scale = min(ww / pw, wh / ph) * 0.95
        dw, dh = int(pw * self._scale), int(ph * self._scale)
        self._offset = QPoint((ww - dw) // 2, (wh - dh) // 2)
        ox, oy = self._offset.x(), self._offset.y()
        painter.drawPixmap(QRect(ox, oy, dw, dh), self._pixmap)
        if self._show_thirds:
            self._draw_thirds_grid(painter, ox, oy, dw, dh)
        if self._show_center:
            self._draw_center_cross(painter, ox, oy, dw, dh)
        if self._show_thirds:
            self._draw_subject_marker(painter, ox, oy, dw, dh)
        if self._show_headroom:
            self._draw_headroom_guide(painter, ox, oy, dw, dh)
        if self._show_reverse and self._reverse:
            self._draw_reverse_overlay(painter, ox, oy, dw, dh)
        if self._show_skeleton:
            for index, pose in enumerate(self._poses or ([self._pose] if self._pose else [])):
                self._draw_skeleton(painter, ox, oy, dw, dh, pose, secondary=index > 0)
        if self._show_bbox:
            for pose in self._poses or ([self._pose] if self._pose else []):
                self._draw_bbox(painter, ox, oy, dw, dh, pose)
        if self._show_visual_weight and self._composition:
            vw_x, vw_y = self._composition.visual_weight
            mx, my = ox + int(dw * vw_x), oy + int(dh * vw_y)
            painter.setPen(QPen(COLOR_VWEIGHT, 2))
            painter.setBrush(QBrush(COLOR_VWEIGHT))
            painter.drawEllipse(QPoint(mx, my), 6, 6)
        painter.end()

    def _draw_reverse_overlay(self, painter, ox, oy, dw, dh):
        result = self._reverse
        if result is None:
            return
        if self._show_reverse_lines:
            painter.setPen(QPen(COLOR_REVERSE_SOFT, 1, Qt.DashLine))
            for x1, y1, x2, y2 in result.perspective.line_segments[:60]:
                painter.drawLine(QPoint(ox + int(x1 * dw), oy + int(y1 * dh)), QPoint(ox + int(x2 * dw), oy + int(y2 * dh)))
        if self._show_reverse_vp:
            painter.setPen(QPen(COLOR_REVERSE, 2))
            for i, (vx, vy) in enumerate(result.perspective.vanishing_points):
                px, py = ox + int(vx * dw), oy + int(vy * dh)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPoint(px, py), 8, 8)
                painter.drawLine(px - 12, py, px + 12, py)
                painter.drawLine(px, py - 12, px, py + 12)
                painter.setFont(QFont("Consolas", 8, QFont.Bold))
                painter.setPen(COLOR_REVERSE_TEXT)
                painter.drawText(px + 10, py - 8, f"VP{i + 1}")
        if self._show_reverse_axis and result.composition:
            sx, sy = result.composition.subject_position
            subject = QPoint(ox + int(sx * dw), oy + int(sy * dh))
            yaw = float(result.camera_pose.camera_yaw.value or 0.0)
            pitch = float(result.camera_pose.camera_pitch.value or 0.0)
            edge_x = float(np.clip(0.5 + (yaw / 45.0) * 0.40, 0.05, 0.95))
            edge_y = 0.08 if pitch < 0 else 0.92
            camera = QPoint(ox + int(edge_x * dw), oy + int(edge_y * dh))
            painter.setPen(QPen(COLOR_REVERSE_SOFT, 2, Qt.DashLine))
            painter.drawLine(subject, camera)
            painter.setPen(QPen(COLOR_REVERSE, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(camera, 7, 7)
            painter.drawLine(camera.x() - 9, camera.y(), camera.x() + 9, camera.y())
            painter.drawLine(camera.x(), camera.y() - 9, camera.x(), camera.y() + 9)
            painter.setFont(QFont("Consolas", 8, QFont.Bold))
            painter.setPen(COLOR_REVERSE_TEXT)
            painter.drawText(camera.x() + 10, camera.y() + 4, "CAMERA (EST.)")
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(COLOR_REVERSE_TEXT)
        painter.drawText(ox + 10, oy + 18, f"REVERSE | confidence {result.overall_confidence:.0%} | estimated")

    def _draw_thirds_grid(self, painter, ox, oy, dw, dh):
        painter.setPen(QPen(COLOR_THIRDS, 1, Qt.DashLine))
        for frac in (1 / 3, 2 / 3):
            painter.drawLine(ox + int(dw * frac), oy, ox + int(dw * frac), oy + dh)
            painter.drawLine(ox, oy + int(dh * frac), ox + dw, oy + int(dh * frac))
        painter.setPen(QPen(COLOR_THIRDS, 1))
        for fx in (1 / 3, 2 / 3):
            for fy in (1 / 3, 2 / 3):
                painter.drawEllipse(QPoint(ox + int(dw * fx), oy + int(dh * fy)), 4, 4)

    def _draw_center_cross(self, painter, ox, oy, dw, dh):
        painter.setPen(QPen(COLOR_CENTER, 1, Qt.DotLine))
        cx, cy = ox + dw // 2, oy + dh // 2
        painter.drawLine(cx, oy, cx, oy + dh)
        painter.drawLine(ox, cy, ox + dw, cy)
        painter.setBrush(QBrush(COLOR_CENTER))
        painter.drawEllipse(QPoint(cx, cy), 3, 3)

    def _draw_subject_marker(self, painter, ox, oy, dw, dh):
        if self._composition:
            px, py = self._composition.subject_position
        elif self._pose:
            px, py = self._compute_subject_center(self._pose)
        else:
            return
        mx, my = ox + int(dw * px), oy + int(dh * py)
        painter.setPen(QPen(COLOR_SUBJECT_MARK, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPoint(mx, my), 10, 10)
        painter.setBrush(QBrush(COLOR_SUBJECT_MARK))
        painter.drawEllipse(QPoint(mx, my), 4, 4)
        painter.setPen(QPen(COLOR_SUBJECT_MARK, 1, Qt.DashDotLine))
        painter.drawLine(mx - 16, my, mx + 16, my)
        painter.drawLine(mx, my - 16, mx, my + 16)

    def _draw_headroom_guide(self, painter, ox, oy, dw, dh):
        if self._composition:
            y_norm = self._composition.headroom
        elif self._pose and self._pose.is_visible(LI.NOSE):
            y_norm = self._pose.landmarks[LI.NOSE].world_y
        else:
            return
        y = oy + int(dh * y_norm)
        painter.setPen(QPen(COLOR_HEADROOM, 1, Qt.DashLine))
        painter.drawLine(ox, y, ox + dw, y)
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(COLOR_HEADROOM)
        painter.drawText(ox + 4, y - 4, "headroom")

    def _draw_skeleton(self, painter, ox, oy, dw, dh, pose: PoseResult, secondary=False):
        pts = {}
        for i, lm in enumerate(pose.landmarks):
            if lm.visibility > 0.3:
                pts[i] = QPoint(ox + int(lm.world_x * dw), oy + int(lm.world_y * dh))
        skeleton_color = COLOR_SKELETON_SECONDARY if secondary else COLOR_SKELETON
        painter.setPen(QPen(skeleton_color, 2))
        for a, b in POSE_CONNECTIONS:
            if a in pts and b in pts:
                painter.drawLine(pts[a], pts[b])
        painter.setBrush(QBrush(skeleton_color))
        for idx, pt in pts.items():
            lm = pose.landmarks[idx]
            col, r = (COLOR_KEYPOINT, 4) if lm.visibility > 0.6 else (COLOR_KEYPOINT_LOW, 3)
            if secondary:
                col, r = skeleton_color, 3
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(col, 1))
            painter.drawEllipse(pt, r, r)

    def _draw_bbox(self, painter, ox, oy, dw, dh, pose: PoseResult):
        if pose.bbox is not None:
            x1, y1, x2, y2 = pose.bbox
            painter.setPen(QPen(COLOR_BBOX, 2, Qt.DashDotLine))
            painter.setBrush(Qt.NoBrush)
            rect = QRect(ox + int(x1 / max(pose.image_width,1) * dw),
                         oy + int(y1 / max(pose.image_height,1) * dh),
                         int((x2-x1) / max(pose.image_width,1) * dw),
                         int((y2-y1) / max(pose.image_height,1) * dh))
            painter.drawRect(rect)
            return
        visible = [(ox + int(lm.world_x * dw), oy + int(lm.world_y * dh)) for lm in pose.landmarks[:17] if lm.visibility > 0.4]
        if len(visible) < 4:
            return
        xs, ys = [p[0] for p in visible], [p[1] for p in visible]
        x1, y1, x2, y2 = min(xs)-15, min(ys)-15, max(xs)+15, max(ys)+15
        painter.setPen(QPen(COLOR_BBOX, 2, Qt.DashDotLine))
        painter.drawRect(x1, y1, x2-x1, y2-y1)

    def _compute_subject_center(self, pose: PoseResult) -> tuple[float, float]:
        visible = [(lm.world_x, lm.world_y) for lm in pose.landmarks[:17] if lm.visibility > 0.4]
        if not visible:
            return 0.5, 0.5
        return float(np.mean([p[0] for p in visible])), float(np.mean([p[1] for p in visible]))
