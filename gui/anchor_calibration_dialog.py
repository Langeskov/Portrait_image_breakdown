"""Manual image-point binding UI for v3 scene-anchor calibration."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
)

from reverse_engineering.anchor_calibration import estimate_camera_from_anchors
from reverse_engineering.scene_anchors import AnchorKind


class AnchorCalibrationDialog(QDialog):
    """Bind image pixels to an anchor and optionally solve a camera hypothesis."""

    def __init__(self, scene, image_size, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.width, self.height = int(image_size[0]), int(image_size[1])
        self.setWindowTitle("Scene Anchor Calibration")
        self.setMinimumWidth(430)

        root = QVBoxLayout(self)
        intro = QLabel(
            "把场景锚点与当前照片中的像素位置对应起来。这里只记录人工对应关系，"
            "不会把它们标记成自动观测结果。至少绑定 4 个有效对应点后才能生成相机假设。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#64748B;")
        root.addWidget(intro)

        self.anchor_combo = QComboBox()
        self.anchor_ids: list[str] = []
        for anchor in scene.anchors:
            if not anchor.enabled:
                continue
            self.anchor_ids.append(anchor.anchor_id)
            self.anchor_combo.addItem(f"{anchor.name} · {anchor.kind.value}", anchor.anchor_id)
        self.anchor_combo.currentIndexChanged.connect(self._load_anchor)

        group = QGroupBox("Selected anchor")
        form = QFormLayout(group)
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
            x_box.setRange(0, max(self.width, 1))
            y_box.setRange(0, max(self.height, 1))
            x_box.setDecimals(2)
            y_box.setDecimals(2)
            self.x_fields.append(x_box)
            self.y_fields.append(y_box)
            form.addRow(f"P{index + 1} X", x_box)
            form.addRow(f"P{index + 1} Y", y_box)
        root.addWidget(group)

        bind = QPushButton("Bind image points to anchor")
        bind.clicked.connect(self._save_current_anchor)
        root.addWidget(bind)

        solve = QPushButton("Estimate camera hypothesis from all bound anchors")
        solve.clicked.connect(self._solve)
        root.addWidget(solve)

        self.result_label = QLabel("No camera hypothesis yet")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color:#475569;")
        root.addWidget(self.result_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        if self.anchor_ids:
            self._load_anchor(0)
        else:
            self.point_count.setEnabled(False)

    def _current_anchor(self):
        data = self.anchor_combo.currentData()
        return self.scene.anchor_by_id(str(data)) if data is not None else None

    def _load_anchor(self, index):
        anchor = self._current_anchor()
        if anchor is None:
            return
        self.kind_label.setText(anchor.kind.value)
        default_count = 4 if anchor.kind == AnchorKind.PLANE and len(anchor.image_points) == 4 else max(1, len(anchor.image_points))
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

    def _refresh_point_fields(self, count):
        anchor = self._current_anchor()
        plane = anchor is not None and anchor.kind == AnchorKind.PLANE
        effective = 4 if plane else min(1, int(count))
        if plane:
            self.point_count.setValue(4)
            self.point_count.setEnabled(False)
        else:
            self.point_count.setEnabled(True)
            effective = int(count)
        for i, (x_box, y_box) in enumerate(zip(self.x_fields, self.y_fields)):
            enabled = i < effective
            x_box.setEnabled(enabled)
            y_box.setEnabled(enabled)

    def _save_current_anchor(self):
        anchor = self._current_anchor()
        if anchor is None or anchor.locked:
            return
        count = 4 if anchor.kind == AnchorKind.PLANE else int(self.point_count.value())
        anchor.image_points = tuple(
            (float(self.x_fields[i].value()), float(self.y_fields[i].value()))
            for i in range(count)
        )
        self.result_label.setText(
            f"Bound {len(anchor.image_points)} image point(s) to {anchor.name}. "
            "Binding remains manual input."
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
