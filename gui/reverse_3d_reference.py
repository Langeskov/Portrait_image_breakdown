"""Reference-aware reconstruction workspace adapter and PySide6-safe preview."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF

import gui.reverse_3d_workspace as _workspace
from gui.reverse_3d_workspace import (
    AnchorProjectionPreview as _BaseAnchorProjectionPreview,
    AnchorSceneView,
    CollapsibleSection,
    Reverse3DWorkspace as _BaseReverse3DWorkspace,
)


class AnchorProjectionPreview(_BaseAnchorProjectionPreview):
    """Fix PySide6 QSize handling and reliably render selected plane anchors."""

    def paintEvent(self, event):
        # Call ProjectionPreview.paintEvent directly. The immediate parent
        # implementation also paints this overlay and contains the invalid
        # QSize.toSize() call, so super() is intentionally skipped here.
        from gui.reverse_3d import ProjectionPreview
        ProjectionPreview.paintEvent(self, event)

        if self._scene is None or self._selected_anchor is None or self._pixmap is None:
            return

        w, h = self._pixmap.width(), self._pixmap.height()
        points = self._project_selected(w, h)
        if points is None or len(points) == 0:
            return

        area = self.rect().adjusted(6, 6, -6, -38)
        target_size = area.size()  # QRectF.size() is already a QSize in PySide6.
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if scaled.isNull():
            return

        ox = area.x() + (area.width() - scaled.width()) * 0.5
        oy = area.y() + (area.height() - scaled.height()) * 0.5
        sx = scaled.width() / max(w, 1)
        sy = scaled.height() / max(h, 1)
        qpoints = [QPointF(ox + p[0] * sx, oy + p[1] * sy) for p in points]

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#D97706"), 2.4 if self._selected_anchor.kind.value == "plane" else 2.2, Qt.DashLine))
        painter.setBrush(QBrush(QColor(217, 119, 6, 35)))

        if self._selected_anchor.kind.value == "plane" and len(qpoints) >= 3:
            painter.drawPolygon(QPolygonF(qpoints))
            painter.setPen(QPen(QColor("#F59E0B"), 2))
            for p in qpoints:
                painter.drawEllipse(p, 5, 5)
        else:
            p = qpoints[0]
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            painter.drawEllipse(p, 6, 6)
            painter.drawLine(p.x() - 10, p.y(), p.x() + 10, p.y())
            painter.drawLine(p.x(), p.y() - 10, p.x(), p.y() + 10)

        painter.setPen(QPen(QColor("#D97706"), 2))
        painter.setFont(painter.font())
        painter.drawText(qpoints[0] + QPointF(8, -9), f"SELECTED · {self._selected_anchor.name}")
        painter.end()


# Reverse3DWorkspace resolves AnchorProjectionPreview from the defining
# module's globals, so install the safe subclass before constructing it.
_workspace.AnchorProjectionPreview = AnchorProjectionPreview


class Reverse3DWorkspace(_BaseReverse3DWorkspace):
    """Avoid rebuilding the reference hypothesis on every polling tick."""

    def _poll_reference_mode(self):
        window = self.window()
        widget = getattr(window, "_reference_mode", None)
        if widget is None:
            return

        ref = getattr(widget, "_reference", None)
        pose = getattr(widget, "_current_pose", None)
        image = getattr(widget, "_current_image", None)
        signature = (
            id(ref),
            id(pose),
            tuple(image.shape[:2]) if image is not None else None,
        )
        if signature == self._last_ref_signature:
            return
        self._last_ref_signature = signature

        if ref is None or pose is None or image is None:
            self.clear_reference_context()
            return

        from reverse_engineering.reference_reconstruction import build_reference_composition
        if (
            int(getattr(pose, "image_width", 0) or 0) != int(image.shape[1])
            or int(getattr(pose, "image_height", 0) or 0) != int(image.shape[0])
        ):
            pose = pose.rescaled(int(image.shape[1]), int(image.shape[0]))
        current = build_reference_composition(pose, image.shape[1], image.shape[0])
        self.set_reference_context(ref, current)


__all__ = [
    "AnchorProjectionPreview",
    "AnchorSceneView",
    "CollapsibleSection",
    "Reverse3DWorkspace",
]
