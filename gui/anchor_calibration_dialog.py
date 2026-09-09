"""Image-first manual scene-anchor calibration UI for V3."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from reverse_engineering.anchor_calibration import estimate_camera_from_anchors
from reverse_engineering.scene_anchors import AnchorKind


class AnchorImageCanvas(QWidget):
    """Image canvas used for click-to-place and drag-to-edit anchor points."""

    point_clicked = Signal(float, float)
    point_dragged = Signal(int, float, float)

    def __init__(self, scene, image=None, parent=None):
        super().__init__(parent)
        self.scene = scene
        self._pixmap = QPixmap()
        self.selected_anchor_id: Optional[str] = None
        self._drag_point_index: Optional[int] = None
        self.setMinimumSize(520, 420)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if image is not None:
            self.set_image(image)

    def set_image(self, image):
        if image is None:
            self._pixmap = QPixmap()
        elif isinstance(image, QPixmap):
            self._pixmap = image
        else:
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[2] >= 3:
                rgb = array[:, :, :3][:, :, ::-1].copy()
                h, w = rgb.shape[:2]
                qimage = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888)
                self._pixmap = QPixmap.fromImage(qimage.copy())
            elif array.ndim == 2:
                h, w = array.shape[:2]
                qimage = QImage(array.data, w, h, int(array.strides[0]), QImage.Format.Format_Grayscale8)
                self._pixmap = QPixmap.fromImage(qimage.copy())
        self.update()

    def set_scene(self, scene):
        self.scene = scene
        self.update()

    def set_selected_anchor(self, anchor_id: Optional[str]):
        self.selected_anchor_id = anchor_id
        self._drag_point_index = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()

    def _image_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / max(iw, 1), self.height() / max(ih, 1))
        dw, dh = iw * scale, ih * scale
        return QRectF((self.width() - dw) * 0.5, (self.height() - dh) * 0.5, dw, dh)

    def _image_to_widget(self, point):
        rect = self._image_rect()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if rect.isEmpty() or iw <= 0 or ih <= 0:
            return QPointF()
        return QPointF(
            rect.left() + float(point[0]) / iw * rect.width(),
            rect.top() + float(point[1]) / ih * rect.height(),
        )

    def _widget_to_image(self, position):
        rect = self._image_rect()
        if rect.isEmpty() or not rect.contains(position):
            return None
        iw, ih = self._pixmap.width(), self._pixmap.height()
        x = (position.x() - rect.left()) / rect.width() * iw
        y = (position.y() - rect.top()) / rect.height() * ih
        return (
            float(np.clip(x, 0.0, max(iw - 1.0, 0.0))),
            float(np.clip(y, 0.0, max(ih - 1.0, 0.0))),
        )

    def _hit_selected_point(self, position, radius_px: float = 12.0) -> Optional[int]:
        if self.scene is None or not self.selected_anchor_id:
            return None
        anchor = self.scene.anchor_by_id(str(self.selected_anchor_id))
        if anchor is None or getattr(anchor, "locked", False) or not getattr(anchor, "visible", True):
            return None
        radius2 = radius_px * radius_px
        best_index = None
        best_distance = radius2
        for index, point in enumerate(getattr(anchor, "image_points", ())):
            widget_point = self._image_to_widget(point)
            distance = float(
                (widget_point.x() - position.x()) ** 2
                + (widget_point.y() - position.y()) ** 2
            )
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            drag_index = self._hit_selected_point(event.position())
            if drag_index is not None:
                self._drag_point_index = drag_index
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            point = self._widget_to_image(event.position())
            if point is not None:
                self.point_clicked.emit(*point)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_point_index is not None and (event.buttons() & Qt.LeftButton):
            point = self._widget_to_image(event.position())
            if point is not None:
                self.point_dragged.emit(self._drag_point_index, *point)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_point_index is not None:
            self._drag_point_index = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111827"))
        image_rect = self._image_rect()
        if not self._pixmap.isNull():
            painter.drawPixmap(image_rect.toRect(), self._pixmap)
        if self.scene is None:
            painter.end()
            return

        for anchor in getattr(self.scene, "anchors", ()):
            points = getattr(anchor, "image_points", ())
            if not getattr(anchor, "visible", True) or not points:
                continue
            selected = anchor.anchor_id == self.selected_anchor_id
            widget_points = [self._image_to_widget(point) for point in points]
            pen_color = QColor("#F59E0B") if selected else QColor("#38BDF8")
            painter.setPen(QPen(pen_color, 2 if selected else 1.2))
            if anchor.kind == AnchorKind.PLANE and len(widget_points) == 4:
                fill = QColor(37, 99, 235, 45) if selected else QColor(56, 189, 248, 28)
                painter.setBrush(QBrush(fill))
                painter.drawPolygon(QPolygonF(widget_points))
            else:
                painter.setBrush(QBrush(QColor("#FFFFFF")))
            for index, point in enumerate(widget_points):
                radius = 6 if selected and index == self._drag_point_index else (5 if selected else 4)
                painter.drawEllipse(point, radius, radius)
                if selected:
                    painter.drawText(point + QPointF(7, -7), f"P{index + 1}")
            if selected and anchor.kind == AnchorKind.PLANE and len(widget_points) == 4:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor("#F59E0B"), 2, Qt.DashLine))
                painter.drawPolygon(QPolygonF(widget_points))
        painter.end()


class AnchorCalibrationDialog(QDialog):
    """Bind scene anchors to image-space observations and solve a camera hypothesis."""

    def __init__(self, scene, image_size, image=None, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.width, self.height = int(image_size[0]), int(image_size[1])
        self.setWindowTitle("Scene Anchor Calibration")
        self.resize(1180, 720)
        self.setMinimumSize(980, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        intro = QLabel(
            "在左侧原图上点选参考位置。已有点可直接拖动；右侧坐标会实时同步。"
            "平面需要 4 个图像点。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#64748B;")
        root.addWidget(intro)

        content = QHBoxLayout()
        content.setSpacing(10)
        self.canvas = AnchorImageCanvas(scene, image=image)
        self.canvas.point_clicked.connect(self._canvas_point_clicked)
        self.canvas.point_dragged.connect(self._canvas_point_dragged)
        content.addWidget(self.canvas, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 0, 4, 0)
        panel_layout.setSpacing(7)

        binding = QGroupBox("Manual binding")
        form = QFormLayout(binding)
        self.anchor_combo = QComboBox()
        self.anchor_ids: list[str] = []
        for anchor in scene.anchors:
            if anchor.enabled:
                self.anchor_ids.append(anchor.anchor_id)
                self.anchor_combo.addItem(f"{anchor.name} · {anchor.kind.value}", anchor.anchor_id)
        self.anchor_combo.currentIndexChanged.connect(self._load_anchor)
        form.addRow("Anchor", self.anchor_combo)

        self.kind_label = QLabel("—")
        form.addRow("Kind", self.kind_label)

        self.point_count = QSpinBox()
        self.point_count.setRange(1, 4)
        self.point_count.setValue(1)
        self.point_count.valueChanged.connect(self._refresh_point_fields)
        form.addRow("Image points", self.point_count)

        self.x_fields: list[QDoubleSpinBox] = []
        self.y_fields: list[QDoubleSpinBox] = []
        for index in range(4):
            x_box = QDoubleSpinBox()
            y_box = QDoubleSpinBox()
            x_box.setRange(0, max(self.width - 1, 1))
            y_box.setRange(0, max(self.height - 1, 1))
            x_box.setDecimals(1)
            y_box.setDecimals(1)
            x_box.valueChanged.connect(lambda value, i=index: self._point_field_changed(i))
            y_box.valueChanged.connect(lambda value, i=index: self._point_field_changed(i))
            self.x_fields.append(x_box)
            self.y_fields.append(y_box)
            form.addRow(f"P{index + 1} X", x_box)
            form.addRow(f"P{index + 1} Y", y_box)
        panel_layout.addWidget(binding)

        hint = QLabel("点击空白处会写入下一个点；拖动已有点或修改坐标都会立即更新。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748B; font-size:9pt;")
        panel_layout.addWidget(hint)

        actions = QHBoxLayout()
        apply_button = QPushButton("Apply points")
        apply_button.clicked.connect(self._apply_points)
        actions.addWidget(apply_button)
        clear_button = QPushButton("Clear all points")
        clear_button.clicked.connect(self._clear_all_points)
        actions.addWidget(clear_button)
        panel_layout.addLayout(actions)

        overlay = QGroupBox("Overlay")
        overlay_layout = QVBoxLayout(overlay)
        overlay_hint = QLabel("显示开关只影响画面，不改变校准数据。")
        overlay_hint.setWordWrap(True)
        overlay_hint.setStyleSheet("color:#64748B; font-size:9pt;")
        overlay_layout.addWidget(overlay_hint)
        self.overlay_list = QListWidget()
        self.overlay_list.setMaximumHeight(125)
        self.overlay_list.itemChanged.connect(self._overlay_changed)
        overlay_layout.addWidget(self.overlay_list)
        panel_layout.addWidget(overlay)

        solve = QPushButton("Estimate camera hypothesis")
        solve.clicked.connect(self._solve)
        panel_layout.addWidget(solve)

        self.result_label = QLabel("No camera hypothesis yet")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color:#475569;")
        panel_layout.addWidget(self.result_label)
        panel_layout.addStretch(1)

        scroll.setWidget(panel)
        content.addWidget(scroll, 1)
        root.addLayout(content, 1)

        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        root.addWidget(close)

        self._populate_overlay_list()
        if self.anchor_ids:
            self._load_anchor(0)
        else:
            self.point_count.setEnabled(False)

    def _current_anchor(self):
        data = self.anchor_combo.currentData()
        return self.scene.anchor_by_id(str(data)) if data is not None else None

    def _populate_overlay_list(self):
        self.overlay_list.blockSignals(True)
        self.overlay_list.clear()
        for anchor in self.scene.anchors:
            if not anchor.enabled:
                continue
            item = QListWidgetItem(anchor.name)
            item.setData(Qt.ItemDataRole.UserRole, anchor.anchor_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if getattr(anchor, "visible", True) else Qt.Unchecked)
            self.overlay_list.addItem(item)
        self.overlay_list.blockSignals(False)

    def _overlay_changed(self, item):
        anchor = self.scene.anchor_by_id(str(item.data(Qt.ItemDataRole.UserRole)))
        if anchor is None:
            return
        anchor.visible = item.checkState() == Qt.Checked
        self.canvas.update()

    def _load_anchor(self, index):
        anchor = self._current_anchor()
        if anchor is None:
            return
        self.canvas.set_selected_anchor(anchor.anchor_id)
        self.kind_label.setText(anchor.kind.value)
        count = 4 if anchor.kind == AnchorKind.PLANE else max(1, min(4, len(anchor.image_points)))
        self.point_count.blockSignals(True)
        self.point_count.setValue(count)
        self.point_count.blockSignals(False)
        self._refresh_point_fields(count)
        self._load_points_into_fields(anchor.image_points)
        self.canvas.update()

    def _refresh_point_fields(self, count):
        anchor = self._current_anchor()
        is_plane = anchor is not None and anchor.kind == AnchorKind.PLANE
        if is_plane:
            self.point_count.blockSignals(True)
            self.point_count.setValue(4)
            self.point_count.blockSignals(False)
            self.point_count.setEnabled(False)
            count = 4
        else:
            self.point_count.setEnabled(True)
            count = max(1, min(4, int(count)))
        for index, (x_box, y_box) in enumerate(zip(self.x_fields, self.y_fields)):
            enabled = index < count
            x_box.setEnabled(enabled)
            y_box.setEnabled(enabled)

    def _load_points_into_fields(self, points):
        self._updating_fields = True
        try:
            for index, (x_box, y_box) in enumerate(zip(self.x_fields, self.y_fields)):
                if index < len(points):
                    x_box.setValue(float(points[index][0]))
                    y_box.setValue(float(points[index][1]))
                else:
                    x_box.setValue(0.0)
                    y_box.setValue(0.0)
        finally:
            self._updating_fields = False

    def _draft_points(self):
        """Return the current field values as a sparse point list."""
        anchor = self._current_anchor()
        if anchor is None:
            return []
        count = 4 if anchor.kind == AnchorKind.PLANE else int(self.point_count.value())
        result = []
        for index in range(count):
            x = float(self.x_fields[index].value())
            y = float(self.y_fields[index].value())
            if index == 0 or x != 0.0 or y != 0.0 or index < len(anchor.image_points):
                result.append((x, y))
            else:
                break
        return result

    def _write_fields_to_anchor(self, status: Optional[str] = None):
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        points = self._draft_points()
        anchor.image_points = tuple(points)
        if points:
            anchor.visible = True
        if status:
            self.result_label.setText(status)
        self.canvas.update()

    def _point_field_changed(self, index):
        if getattr(self, "_updating_fields", False):
            return
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        # A typed point is immediately materialized. This fixes the old state in
        # which P1 existed only in the widgets until Apply points was pressed.
        if anchor.kind != AnchorKind.PLANE and index >= self.point_count.value() - 1 and self.point_count.value() < 4:
            self.point_count.blockSignals(True)
            self.point_count.setValue(index + 2)
            self.point_count.blockSignals(False)
            self._refresh_point_fields(self.point_count.value())
        self._write_fields_to_anchor(f"Live calibrated P{index + 1} · {anchor.name}")

    def _next_point_index(self):
        anchor = self._current_anchor()
        if anchor is None:
            return 0
        count = 4 if anchor.kind == AnchorKind.PLANE else int(self.point_count.value())
        for index in range(count):
            if index >= len(anchor.image_points):
                return index
        return min(count - 1, 3)

    def _canvas_point_clicked(self, x, y):
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        index = self._next_point_index()
        self.x_fields[index].setValue(x)
        self.y_fields[index].setValue(y)
        if anchor.kind != AnchorKind.PLANE and index + 1 >= self.point_count.value() and self.point_count.value() < 4:
            self.point_count.blockSignals(True)
            self.point_count.setValue(index + 2)
            self.point_count.blockSignals(False)
            self._refresh_point_fields(self.point_count.value())
        next_index = min(index + 1, 3)
        if next_index < len(self.x_fields) and self.x_fields[next_index].isEnabled():
            self.x_fields[next_index].setFocus()
        self.canvas.update()

    def _canvas_point_dragged(self, index, x, y):
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        points = list(anchor.image_points)
        if index < 0 or index >= len(points):
            return
        points[index] = (float(x), float(y))
        anchor.image_points = tuple(points)
        self._load_points_into_fields(anchor.image_points)
        self.result_label.setText(f"Live calibrated P{index + 1} · {anchor.name}")
        self.canvas.update()

    def _apply_points(self):
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        self._write_fields_to_anchor(f"Applied {len(anchor.image_points)} image point(s) to {anchor.name}.")

    def _clear_all_points(self):
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        anchor.image_points = ()
        self._load_points_into_fields(())
        if anchor.kind == AnchorKind.PLANE:
            self.point_count.blockSignals(True)
            self.point_count.setValue(4)
            self.point_count.blockSignals(False)
        else:
            self.point_count.blockSignals(True)
            self.point_count.setValue(1)
            self.point_count.blockSignals(False)
        self._refresh_point_fields(self.point_count.value())
        self.result_label.setText(f"Cleared all image points · {anchor.name}")
        self.canvas.update()

    def _solve(self):
        self._write_fields_to_anchor()
        result = estimate_camera_from_anchors(
            self.scene.anchors,
            self.width,
            self.height,
            focal_length_mm=float(self.scene.camera.focal_length_mm),
            sensor_width_mm=float(self.scene.camera.sensor_width_mm),
        )
        if not result.success:
            QMessageBox.warning(self, "Camera hypothesis", result.message)
            return
        px, py, pz = result.position
        self.result_label.setText(
            f"{result.message}\n"
            f"Camera position hypothesis: ({px:.2f}, {py:.2f}, {pz:.2f}) m\n"
            "This is a hypothesis only; the active SceneCamera is not changed."
        )
