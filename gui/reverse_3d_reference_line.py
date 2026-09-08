"""Reference-line driven reconstruction controls for v3.

The 3D scene keeps real Plane anchors, while the 2D validation view represents
an active plane by a compact projected reference line instead of drawing the
entire finite plane rectangle. Camera yaw/pitch can also be adjusted directly
from a 2D drag pad so parameter changes stay visually tied to the photograph.
"""
from __future__ import annotations

import math
from typing import Optional, Callable

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.reverse_3d import ProjectionPreview
from reverse_engineering.projection import build_camera_model
from reverse_engineering.scene import SceneModel
from reverse_engineering.scene_anchors import AnchorKind, SceneAnchor


OBSERVED = QColor("#D97706")
PROJECTED = QColor("#38BDF8")
REFERENCE = QColor("#7C3AED")
MUTED = QColor("#64748B")
GRID = QColor("#334155")


class ReferenceLineProjectionPreview(ProjectionPreview):
    """Photo preview with a single compact projection cue for the selected anchor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene: Optional[SceneModel] = None
        self._selected_anchor: Optional[SceneAnchor] = None
        self._show_reference_line = True

    def set_selected_anchor(self, scene: SceneModel, anchor: Optional[SceneAnchor]):
        self._scene = scene
        self._selected_anchor = anchor
        self.update()

    def update_projection(self, scene, observed_bbox=None, observed_points=None):
        self._scene = scene
        if self._selected_anchor is not None:
            self._selected_anchor = scene.anchor_by_id(self._selected_anchor.anchor_id)
        super().update_projection(scene, observed_bbox, observed_points)

    def _project_line(self, width: int, height: int):
        anchor = self._selected_anchor
        scene = self._scene
        if anchor is None or scene is None or self._pixmap is None:
            return None

        try:
            camera = build_camera_model(scene, width, height)
            if anchor.kind == AnchorKind.POINT:
                world = np.asarray([anchor.position], dtype=np.float64)
            else:
                tangent, _, _ = anchor.plane_basis()
                center = np.asarray(anchor.position, dtype=np.float64)
                half = max(0.75, min(float(anchor.size[0]), 8.0) * 0.5)
                world = np.asarray([center - tangent * half, center + tangent * half], dtype=np.float64)

            projected = camera.project_points(world).reshape(-1, 2)
            rvec = camera.extrinsics.rvec if camera.extrinsics is not None else None
            tvec = camera.extrinsics.tvec if camera.extrinsics is not None else None
            if rvec is None or tvec is None:
                return None
            rmat = cv2.Rodrigues(rvec)[0]
            camera_points = (rmat @ world.T + tvec.reshape(3, 1)).T
            valid = (
                (camera_points[:, 2] > 1e-6)
                & np.isfinite(projected).all(axis=1)
            )
            if not valid.all():
                return None
            if anchor.kind == AnchorKind.POINT:
                return (projected[0],)
            return (projected[0], projected[1])
        except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError, cv2.error):
            return None

    @staticmethod
    def _clip_segment(a: np.ndarray, b: np.ndarray, width: float, height: float):
        """Clip a 2D infinite-ish segment against the image rectangle."""
        x0, y0 = float(a[0]), float(a[1])
        x1, y1 = float(b[0]), float(b[1])
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) + abs(dy) < 1e-9:
            return None

        candidates = []
        def add(t):
            if -1e-9 <= t <= 1.000000001:
                x, y = x0 + t * dx, y0 + t * dy
                if -1e-6 <= x <= width + 1e-6 and -1e-6 <= y <= height + 1e-6:
                    candidates.append((x, y))

        for x in (0.0, width):
            add((x - x0) / dx) if abs(dx) > 1e-9 else None
        for y in (0.0, height):
            add((y - y0) / dy) if abs(dy) > 1e-9 else None
        if len(candidates) < 2:
            if 0 <= x0 <= width and 0 <= y0 <= height and 0 <= x1 <= width and 0 <= y1 <= height:
                return np.asarray([a, b], dtype=float)
            return None
        unique = []
        for point in candidates:
            if not any(np.linalg.norm(np.asarray(point) - np.asarray(q)) < 1e-5 for q in unique):
                unique.append(point)
        if len(unique) < 2:
            return None
        return np.asarray([unique[0], unique[-1]], dtype=float)

    def _image_layout(self):
        if self._pixmap is None:
            return None
        area = QRectF(6, 6, self.width() - 12, self.height() - 38)
        scaled = self._pixmap.scaled(area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = area.x() + (area.width() - scaled.width()) * 0.5
        oy = area.y() + (area.height() - scaled.height()) * 0.5
        return area, scaled, ox, oy, scaled.width() / max(self._pixmap.width(), 1), scaled.height() / max(self._pixmap.height(), 1)

    def paintEvent(self, event):
        # Re-use the stable base renderer for observed skeleton / bbox and person projections.
        super().paintEvent(event)
        if not self._show_reference_line or self._scene is None or self._selected_anchor is None or self._pixmap is None:
            return

        layout = self._image_layout()
        if layout is None:
            return
        _, _, ox, oy, sx, sy = layout
        w, h = self._pixmap.width(), self._pixmap.height()
        projected = self._project_line(w, h)
        if projected is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        if self._selected_anchor.kind == AnchorKind.POINT:
            p = QPointF(ox + projected[0][0] * sx, oy + projected[0][1] * sy)
            painter.setPen(QPen(OBSERVED, 2))
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            painter.drawEllipse(p, 5, 5)
            painter.drawLine(p.x() - 9, p.y(), p.x() + 9, p.y())
            painter.drawLine(p.x(), p.y() - 9, p.x(), p.y() + 9)
            painter.drawText(p + QPointF(8, -8), f"POINT · {self._selected_anchor.name}")
        else:
            clipped = self._clip_segment(projected[0], projected[1], float(w - 1), float(h - 1))
            if clipped is None:
                painter.end()
                return
            a = QPointF(ox + clipped[0, 0] * sx, oy + clipped[0, 1] * sy)
            b = QPointF(ox + clipped[1, 0] * sx, oy + clipped[1, 1] * sy)
            painter.setPen(QPen(REFERENCE, 2.6, Qt.DashLine))
            painter.drawLine(a, b)
            mid = QPointF((a.x() + b.x()) * 0.5, (a.y() + b.y()) * 0.5)
            painter.setPen(QPen(REFERENCE, 1))
            painter.drawText(mid + QPointF(7, -7), f"PLANE REFERENCE · {self._selected_anchor.name}")
        painter.end()


class CameraMatchPad(QWidget):
    """Small direct-manipulation pad for visual yaw/pitch matching."""

    adjusted = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._last = None
        self.setToolTip("拖动中心点：左右调整 Yaw，上下调整 Pitch。")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._last is None:
            return
        current = event.position()
        delta = current - self._last
        self._last = current
        self.adjusted.emit(float(delta.x()) * 0.20, float(-delta.y()) * 0.20)
        self.update()

    def mouseReleaseEvent(self, event):
        self._last = None
        self.unsetCursor()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0F172A"))
        r = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(QPen(GRID, 1))
        for i in range(1, 4):
            x = r.left() + r.width() * i / 4
            y = r.top() + r.height() * i / 4
            painter.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            painter.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        painter.setPen(QPen(QColor("#94A3B8"), 1.4))
        painter.drawLine(QPointF(r.center().x(), r.top()), QPointF(r.center().x(), r.bottom()))
        painter.drawLine(QPointF(r.left(), r.center().y()), QPointF(r.right(), r.center().y()))
        painter.setBrush(QBrush(REFERENCE))
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawEllipse(r.center(), 7, 7)
        painter.setPen(QPen(QColor("#CBD5E1"), 1))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(r.left(), r.bottom() + 1, "← Yaw →")
        painter.save()
        painter.translate(r.left() + 10, r.center().y())
        painter.rotate(-90)
        painter.drawText(QPointF(-20, 0), "Pitch")
        painter.restore()
        painter.end()


class CameraVisualMatchSection(QWidget):
    """Compact visual controls inserted between Camera and Scene anchors."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self._updating = False
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 7, 8, 7)
        root.setSpacing(6)
        title = QLabel("Visual camera match")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        root.addWidget(title)
        hint = QLabel("直接拖动面板匹配画面方向；距离用滑杆微调，变化实时反映到照片投影。")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED.name()}; font-size:9pt;")
        root.addWidget(hint)

        row = QHBoxLayout()
        self.pad = CameraMatchPad()
        self.pad.adjusted.connect(self._pad_adjusted)
        row.addWidget(self.pad, 1)
        info = QVBoxLayout()
        self._yaw_label = QLabel("Yaw +0.0°")
        self._pitch_label = QLabel("Pitch +0.0°")
        self._distance_label = QLabel("Distance 4.00 m")
        for label in (self._yaw_label, self._pitch_label, self._distance_label):
            label.setStyleSheet("color:#334155;")
            info.addWidget(label)
        info.addStretch(1)
        row.addLayout(info)
        root.addLayout(row)

        distance_row = QHBoxLayout()
        distance_row.addWidget(QLabel("Distance"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(10, 500)
        self.slider.valueChanged.connect(self._distance_changed)
        distance_row.addWidget(self.slider, 1)
        root.addLayout(distance_row)

        self.sync_from_camera()

    def sync_from_camera(self):
        if not hasattr(self.owner, "_distance"):
            return
        self._updating = True
        try:
            c = self.owner.scene.camera
            self.slider.setValue(int(round(float(c.distance) * 10.0)))
            self._yaw_label.setText(f"Yaw {float(c.yaw):+.1f}°")
            self._pitch_label.setText(f"Pitch {float(c.pitch):+.1f}°")
            self._distance_label.setText(f"Distance {float(c.distance):.2f} m")
        finally:
            self._updating = False

    def _pad_adjusted(self, dyaw, dpitch):
        if self._updating:
            return
        owner = self.owner
        owner._yaw.setValue(float(owner._yaw.value()) + dyaw)
        owner._pitch.setValue(float(owner._pitch.value()) + dpitch)
        self.sync_from_camera()

    def _distance_changed(self, value):
        if self._updating:
            return
        self.owner._distance.setValue(float(value) / 10.0)
        self.sync_from_camera()


def install_visual_camera_match(workspace):
    """Install the visual camera control and reference-line preview on a workspace."""
    section = CameraVisualMatchSection(workspace)
    anchor_section = next(
        (s for s in workspace.findChildren(QWidget) if getattr(getattr(s, "button", None), "text", lambda: "")() == "Scene anchors"),
        None,
    )
    if anchor_section is not None and anchor_section.parentWidget() is not None:
        layout = anchor_section.parentWidget().layout()
        index = layout.indexOf(anchor_section)
        layout.insertWidget(index, section)
    else:
        section.setParent(workspace)
        section.show()

    old_preview = getattr(workspace, "_preview", None)
    if old_preview is not None:
        parent = old_preview.parentWidget()
        layout = parent.layout() if parent is not None else None
        new_preview = ReferenceLineProjectionPreview()
        if layout is not None:
            index = layout.indexOf(old_preview)
            layout.removeWidget(old_preview)
            old_preview.deleteLater()
            layout.insertWidget(index, new_preview)
        else:
            new_preview.setParent(old_preview.parentWidget())
        workspace._preview = new_preview
        workspace._preview.setMinimumHeight(250)
        workspace._refresh_projection()

    workspace._camera_visual_match = section
    return section
