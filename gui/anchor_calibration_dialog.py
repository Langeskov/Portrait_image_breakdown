"""Image-first manual scene-anchor calibration UI for v3."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from reverse_engineering.anchor_calibration import estimate_camera_from_anchors
from reverse_engineering.scene_anchors import AnchorKind


class AnchorImageCanvas(QWidget):
    """Display the source image and visible manual anchor observations."""

    point_clicked = Signal(float, float)

    def __init__(self, scene, image=None, parent=None):
        super().__init__(parent)
        self.scene = scene
        self._image = None
        self._pixmap = QPixmap()
        self.selected_anchor_id: Optional[str] = None
        self.setMinimumSize(520, 420)
        self.setMouseTracking(True)
        if image is not None:
            self.set_image(image)

    def set_image(self, image):
        self._image = image
        if image is None:
            self._pixmap = QPixmap()
            self.update()
            return
        if isinstance(image, QPixmap):
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
        if rect.isEmpty():
            return QPointF()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        x, y = float(point[0]), float(point[1])
        return QPointF(rect.left() + x / iw * rect.width(), rect.top() + y / ih * rect.height())

    def _widget_to_image(self, position):
        rect = self._image_rect()
        if rect.isEmpty() or not rect.contains(position):
            return None
        iw, ih = self._pixmap.width(), self._pixmap.height()
        x = (position.x() - rect.left()) / rect.width() * iw
        y = (position.y() - rect.top()) / rect.height() * ih
        return float(max(0.0, min(iw - 1.0, x))), float(max(0.0, min(ih - 1.0, y)))

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        point = self._widget_to_image(event.position())
        if point is not None:
            self.point_clicked.emit(*point)
        return super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111827"))
        rect = self._image_rect()
        if not self._pixmap.isNull():
            painter.drawPixmap(rect.toRect(), self._pixmap)
        if self.scene is None:
            painter.end()
            return

        for anchor in getattr(self.scene, "anchors", ()):
            if not getattr(anchor, "visible", True) or not anchor.image_points:
                continue
            selected = anchor.anchor_id == self.selected_anchor_id
            pen = QPen(QColor("#F59E0B") if selected else QColor("#38BDF8"), 2 if selected else 1.2)
            painter.setPen(pen)
            points = [self._image_to_widget(p) for p in anchor.image_points]
            if anchor.kind == AnchorKind.PLANE and len(points) == 4:
                painter.setBrush(QBrush(QColor(37, 99, 235, 45) if selected else QColor(56, 189, 248, 28)))
                painter.drawPolygon(QPolygonF(points))
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            for index, point in enumerate(points):
                radius = 5 if selected else 4
                painter.drawEllipse(point, radius, radius)
                if selected:
                    painter.drawText(point + QPointF(7, -7), f"P{index + 1}")
            if selected and anchor.kind == AnchorKind.PLANE and len(points) == 4:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor("#F59E0B"), 2, Qt.DashLine))
                painter.drawPolygon(QPolygonF(points))
        painter.end()


class AnchorCalibrationDialog(QDialog):
    """Bind scene anchors to source-image pixels and estimate a camera hypothesis."""

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
            "在左侧原图上直接点选参考位置。右侧只保留当前锚点与必要参数；"
            "橙色为当前锚点，蓝色为其它已显示锚点。平面需要 4 个图像点。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#64748B;")
        root.addWidget(intro)

        content = QHBoxLayout()
        content.setSpacing(10)
        self.canvas = AnchorImageCanvas(scene, image=image)
        content.addWidget(self.canvas, 3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        scroll.setFrameShape(QScrollArea.NoFrame)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 0, 4, 0)
        panel_layout.setSpacing(7)

        group = QGroupBox("Manual binding")
        form = QFormLayout(group)
        self.anchor_combo = QComboBox()
        self.anchor_ids: list[str] = []
        for anchor in scene.anchors:
            if not anchor.enabled:
                continue
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
            self.x_fields.append(x_box)
            self.y_fields.append(y_box)
            form.addRow(f"P{index + 1} X", x_box)
            form.addRow(f"P{index + 1} Y", y_box)
        panel_layout.addWidget(group)

        click_hint = QLabel("图像点击会写入下一个 P 点；也可直接修改坐标。")
        click_hint.setWordWrap(True)
        click_hint.setStyleSheet("color:#64748B; font-size:9pt;")
        panel_layout.addWidget(click_hint)

        bind = QPushButton("Apply points")
        bind.clicked.connect(self._save_current_anchor)
        panel_layout.addWidget(bind)

        overlay = QGroupBox("Overlay")
        overlay_layout = QVBoxLayout(overlay)
        overlay_hint = QLabel("显示开关只影响画面，不会改变校准参与状态。")
        overlay_hint.setWordWrap(True)
        overlay_hint.setStyleSheet("color:#64748B; font-size:9pt;")
        overlay_layout.addWidget(overlay_hint)
        self.overlay_list = QListWidget()
        self.overlay_list.setMaximumHeight(125)
        self.overlay_list.itemChanged.connect(self._overlay_changed)
        overlay_layout.addWidget(self.overlay_list)
        panel_layout.addWidget(overlay)

        solve = QPushButton("Estimate camera hypothesis")
        solve.setToolTip("Use all bound anchors to create a non-destructive camera hypothesis")
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

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

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
        anchor_id = item.data(Qt.ItemDataRole.UserRole)
        anchor = self.scene.anchor_by_id(str(anchor_id)) if anchor_id is not None else None
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
        default_count = 4 if anchor.kind == AnchorKind.PLANE and len(anchor.image_points) == 4 else max(1, min(4, len(anchor.image_points)))
        self.point_count.blockSignals(True)
        self.point_count.setValue(default_count)
        self.point_count.blockSignals(False)
        for i, (x_box, y_box) in enumerate(zip(self.x_fields, self.y_fields)):
            if i < len(anchor.image_points):
                x, y = anchor.image_points[i]
                x_box.setValue(x)
                y_box.setValue(y)
            else:
                x_box.setValue(0)
                y_box.setValue(0)
        self._refresh_point_fields(default_count)
        self.canvas.update()

    def _refresh_point_fields(self, count):
        anchor = self._current_anchor()
        plane = anchor is not None and anchor.kind == AnchorKind.PLANE
        if plane:
            self.point_count.blockSignals(True)
            self.point_count.setValue(4)
            self.point_count.blockSignals(False)
            self.point_count.setEnabled(False)
            effective = 4
        else:
            self.point_count.setEnabled(True)
            effective = max(1, min(4, int(count)))
        for i, (x_box, y_box) in enumerate(zip(self.x_fields, self.y_fields)):
            enabled = i < effective
            x_box.setEnabled(enabled)
            y_box.setEnabled(enabled)

    def _next_point_index(self):
        anchor = self._current_anchor()
        if anchor is None:
            return 0
        count = 4 if anchor.kind == AnchorKind.PLANE else int(self.point_count.value())
        current = [
            (float(self.x_fields[i].value()), float(self.y_fields[i].value()))
            for i in range(count)
        ]
        stored = list(anchor.image_points)
        for i in range(count):
            if i >= len(stored) or current[i] == (0.0, 0.0):
                return i
        return count - 1

    def _canvas_point_clicked(self, x, y):
        index = self._next_point_index()
        self.x_fields[index].setValue(x)
        self.y_fields[index].setValue(y)
        if index + 1 < (4 if self._current_anchor().kind == AnchorKind.PLANE else int(self.point_count.value())):
            self.x_fields[index + 1].setFocus()
        self.canvas.update()

    def _save_current_anchor(self):
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        count = 4 if anchor.kind == AnchorKind.PLANE else int(self.point_count.value())
        anchor.image_points = tuple(
            (float(self.x_fields[i].value()), float(self.y_fields[i].value()))
            for i in range(count)
        )
        self.canvas.update()
        self.result_label.setText(
            f"Applied {len(anchor.image_points)} image point(s) to {anchor.name}. "
            "Manual binding is retained as user-provided evidence."
        )

    def _solve(self):
        self._save_current_anchor()
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
