"""ImageCanvas — displays image with skeleton/composition overlays (light theme)

Supports overlays:
  skeleton       — body skeleton lines + keypoints
  thirds         — 3x3 rule-of-thirds grid (always drawable, no composition needed)
  center         — image center crosshair
  bbox           — subject bounding box
  visual_weight  — visual weight center marker
  headroom       — headroom horizontal guide line
"""

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

# ── Overlay colors (light-theme friendly) ──
COLOR_SKELETON = QColor(34, 197, 94)        # Green
COLOR_KEYPOINT = QColor(239, 68, 68)        # Red
COLOR_KEYPOINT_LOW = QColor(156, 163, 175)  # Gray
COLOR_THIRDS = QColor(37, 99, 235, 60)      # Semi-transparent blue
COLOR_CENTER = QColor(251, 146, 60, 80)     # Semi-transparent orange
COLOR_BBOX = QColor(245, 158, 11, 160)      # Amber
COLOR_SUBJECT_MARK = QColor(168, 85, 247)   # Purple
COLOR_HEADROOM = QColor(20, 184, 166, 100)  # Teal
COLOR_VWEIGHT = QColor(244, 63, 94, 120)    # Rose

# Light theme canvas background
CANVAS_BG = QColor(250, 251, 252)           # #FAFBFC
CANVAS_TEXT = QColor(107, 114, 128)         # #6B7280


class ImageCanvas(QWidget):
    """Image display canvas with overlay support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self._pixmap: Optional[QPixmap] = None
        self._original_image: Optional[np.ndarray] = None
        self._pose: Optional[PoseResult] = None
        self._camera: Optional[CameraResult] = None
        self._composition: Optional[CompositionResult] = None

        # Overlay flags
        self._show_skeleton = True
        self._show_thirds = True
        self._show_center = True
        self._show_bbox = True
        self._show_visual_weight = False
        self._show_headroom = False

        # Drawing state
        self._scale = 1.0
        self._offset = QPoint(0, 0)

    # ── Data setters ──
    def set_image(self, image: np.ndarray):
        """Set the BGR image to display."""
        self._original_image = image.copy()
        self._update_pixmap()

    def set_pose(self, pose: Optional[PoseResult]):
        self._pose = pose
        self.update()

    def set_camera(self, camera: Optional[CameraResult]):
        self._camera = camera
        self.update()

    def set_composition(self, composition: Optional[CompositionResult]):
        self._composition = composition
        self.update()

    # ── Overlay options (unified API) ──
    def set_overlay_options(
        self,
        skeleton: bool = True,
        thirds: bool = True,
        center: bool = True,
        bbox: bool = True,
        visual_weight: bool = False,
        headroom: bool = False,
    ):
        self._show_skeleton = skeleton
        self._show_thirds = thirds
        self._show_center = center
        self._show_bbox = bbox
        self._show_visual_weight = visual_weight
        self._show_headroom = headroom
        self.update()

    # ── Pixmap conversion ──
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

    # ── Paint ──
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Light background
        painter.fillRect(self.rect(), CANVAS_BG)

        if self._pixmap is None:
            painter.setPen(CANVAS_TEXT)
            painter.setFont(QFont("Microsoft YaHei", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Drag an image here or click Open Image")
            painter.end()
            return

        # Calculate fit-to-window scaling
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

        # Draw image
        target = QRect(self._offset.x(), self._offset.y(), draw_w, draw_h)
        painter.drawPixmap(target, self._pixmap)

        # Overlay origin
        ox = self._offset.x()
        oy = self._offset.y()

        # 3x3 Rule-of-thirds grid (always draws if flag set, no composition needed)
        if self._show_thirds:
            self._draw_thirds_grid(painter, ox, oy, draw_w, draw_h)

        # Center crosshair
        if self._show_center:
            self._draw_center_cross(painter, ox, oy, draw_w, draw_h)

        # Subject center marker (from composition or computed from pose)
        if self._show_thirds:
            self._draw_subject_marker(painter, ox, oy, draw_w, draw_h)

        # Headroom guide
        if self._show_headroom:
            self._draw_headroom_guide(painter, ox, oy, draw_w, draw_h)

        # Skeleton
        if self._show_skeleton and self._pose:
            self._draw_skeleton(painter, ox, oy, draw_w, draw_h)

        # Bounding box
        if self._show_bbox and self._pose:
            self._draw_bbox(painter, ox, oy, draw_w, draw_h)

        # Visual weight marker
        if self._show_visual_weight and self._composition:
            vw_x, vw_y = self._composition.visual_weight
            mx = ox + int(draw_w * vw_x)
            my = oy + int(draw_h * vw_y)
            pen = QPen(COLOR_VWEIGHT, 2)
            painter.setPen(pen)
            painter.setBrush(QBrush(COLOR_VWEIGHT))
            painter.drawEllipse(QPoint(mx, my), 6, 6)

        painter.end()

    # ── Overlay drawing methods ──
    def _draw_thirds_grid(self, painter: QPainter, ox, oy, dw, dh):
        """Draw 3x3 rule-of-thirds grid. No dependency on composition result."""
        pen = QPen(COLOR_THIRDS, 1, Qt.DashLine)
        painter.setPen(pen)
        for frac in [1 / 3, 2 / 3]:
            x = ox + int(dw * frac)
            painter.drawLine(x, oy, x, oy + dh)
            y = oy + int(dh * frac)
            painter.drawLine(ox, y, ox + dw, y)

        # Draw intersection point indicators
        for fx in [1 / 3, 2 / 3]:
            for fy in [1 / 3, 2 / 3]:
                ix = ox + int(dw * fx)
                iy = oy + int(dh * fy)
                pen = QPen(COLOR_THIRDS, 1)
                painter.setPen(pen)
                painter.drawEllipse(QPoint(ix, iy), 4, 4)

    def _draw_center_cross(self, painter: QPainter, ox, oy, dw, dh):
        """Draw image center crosshair."""
        pen = QPen(COLOR_CENTER, 1, Qt.DotLine)
        painter.setPen(pen)
        cx = ox + dw // 2
        cy = oy + dh // 2
        painter.drawLine(cx, oy, cx, oy + dh)
        painter.drawLine(ox, cy, ox + dw, cy)
        # Small center dot
        painter.setBrush(QBrush(COLOR_CENTER))
        painter.drawEllipse(QPoint(cx, cy), 3, 3)

    def _draw_subject_marker(self, painter: QPainter, ox, oy, dw, dh):
        """Draw subject center point.

        Uses composition.subject_position if available,
        otherwise computes from visible pose landmarks.
        """
        if self._composition:
            px, py = self._composition.subject_position
        elif self._pose:
            px, py = self._compute_subject_center()
        else:
            return

        mx = ox + int(dw * px)
        my = oy + int(dh * py)

        # Outer ring
        pen = QPen(COLOR_SUBJECT_MARK, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPoint(mx, my), 10, 10)

        # Inner dot
        painter.setBrush(QBrush(COLOR_SUBJECT_MARK))
        painter.drawEllipse(QPoint(mx, my), 4, 4)

        # Crosshair
        pen = QPen(COLOR_SUBJECT_MARK, 1, Qt.DashDotLine)
        painter.setPen(pen)
        painter.drawLine(mx - 16, my, mx + 16, my)
        painter.drawLine(mx, my - 16, mx, my + 16)

    def _draw_headroom_guide(self, painter: QPainter, ox, oy, dw, dh):
        """Draw headroom horizontal guide."""
        if self._composition:
            headroom_y = self._composition.headroom
        elif self._pose and self._pose.is_visible(LI.NOSE):
            headroom_y = self._pose.landmarks[LI.NOSE].world_y
        else:
            return

        y = oy + int(dh * headroom_y)
        pen = QPen(COLOR_HEADROOM, 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(ox, y, ox + dw, y)

        # Label
        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.setPen(COLOR_HEADROOM)
        painter.drawText(ox + 4, y - 4, "headroom")

    def _draw_skeleton(self, painter: QPainter, ox, oy, dw, dh):
        """Draw skeleton overlay using normalized coordinates."""
        pose = self._pose
        if pose is None:
            return

        pts = {}
        for i, lm in enumerate(pose.landmarks):
            if lm.visibility > 0.3:
                # world_x/y are normalized [0,1] — map directly to drawn image rect
                px = ox + int(lm.world_x * dw)
                py = oy + int(lm.world_y * dh)
                pts[i] = QPoint(px, py)

        # Connection lines
        pen = QPen(COLOR_SKELETON, 2, Qt.SolidLine)
        painter.setPen(pen)
        for a, b in POSE_CONNECTIONS:
            if a in pts and b in pts:
                painter.drawLine(pts[a], pts[b])

        # Keypoints
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

        # Joint labels
        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.setPen(QColor(31, 41, 55, 180))  # Dark text on light bg
        label_map = {
            LI.LEFT_SHOULDER.value: "L-Shoulder",
            LI.RIGHT_SHOULDER.value: "R-Shoulder",
            LI.LEFT_ELBOW.value: "L-Elbow",
            LI.RIGHT_ELBOW.value: "R-Elbow",
            LI.LEFT_WRIST.value: "L-Wrist",
            LI.RIGHT_WRIST.value: "R-Wrist",
            LI.LEFT_HIP.value: "L-Hip",
            LI.RIGHT_HIP.value: "R-Hip",
            LI.LEFT_KNEE.value: "L-Knee",
            LI.RIGHT_KNEE.value: "R-Knee",
        }
        for idx_val, label in label_map.items():
            if idx_val in pts:
                pt = pts[idx_val]
                painter.drawText(pt.x() + 6, pt.y() - 6, label)

    def _draw_bbox(self, painter: QPainter, ox, oy, dw, dh):
        """Draw subject bounding box using normalized coordinates."""
        pose = self._pose
        if pose is None:
            return

        visible_pts = []
        for lm in pose.landmarks[:17]:
            if lm.visibility > 0.4:
                px = ox + int(lm.world_x * dw)
                py = oy + int(lm.world_y * dh)
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

    # ── Helpers ──
    def _compute_subject_center(self) -> tuple[float, float]:
        """Compute subject center from visible pose landmarks."""
        if self._pose is None:
            return 0.5, 0.5
        visible = [
            (lm.world_x, lm.world_y)
            for lm in self._pose.landmarks[:17]
            if lm.visibility > 0.4
        ]
        if not visible:
            return 0.5, 0.5
        cx = float(np.mean([p[0] for p in visible]))
        cy = float(np.mean([p[1] for p in visible]))
        return cx, cy
