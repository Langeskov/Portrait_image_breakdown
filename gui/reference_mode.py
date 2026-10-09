"""v3 reference-photo comparison workspace."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QSplitter, QListWidget, QListWidgetItem, QFrame,
)

from core.image_io import load_image, frame_orientation
from core.reference_reconstruction import (
    build_reference_composition,
    compare_pose_to_reference,
    composition_delta,
    reference_summary,
)
from core.reference_targets import build_reference_target_plan


class ReferenceModeWidget(QWidget):
    reference_changed = Signal()

    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self._detector = detector
        self._reference_pose = None
        self._reference_image = None
        self._reference = None
        self._reference_path: str | None = None
        self._current_pose = None
        self._current_image = None
        self._current_image_size = (1, 1)
        self._last_target = None
        self._reference_pixmap = QPixmap()
        self._current_pixmap = QPixmap()

        lo = QVBoxLayout(self)
        lo.setContentsMargins(14, 12, 14, 12)
        lo.setSpacing(9)

        header = QHBoxLayout()
        title = QLabel("图片对比")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #1F2937;")
        header.addWidget(title)
        header.addStretch(1)
        self._swap = QPushButton("交换 A/B")
        self._swap.setStyleSheet("QPushButton { padding: 4px 8px; border: 1px solid #D9DDE3; background: #FFFFFF; color: #1F2937; font-size: 9pt; }")
        self._swap.clicked.connect(self._swap_images)
        header.addWidget(self._swap)
        self._open = QPushButton("加载参考图")
        self._open.setStyleSheet(self._swap.styleSheet())
        self._open.clicked.connect(self._load_reference)
        header.addWidget(self._open)
        self._clear = QPushButton("清除")
        self._clear.setStyleSheet(self._swap.styleSheet())
        self._clear.clicked.connect(self.clear_reference)
        header.addWidget(self._clear)
        lo.addLayout(header)

        self._summary = QLabel("加载参考照片，用于比较构图和姿态。")
        self._summary.setWordWrap(True)
        lo.addWidget(self._summary)

        split = QSplitter(Qt.Horizontal)
        # Reference image A
        ref_container = QWidget()
        ref_lo = QVBoxLayout(ref_container)
        ref_lo.setContentsMargins(0, 0, 0, 0)
        ref_lo.setSpacing(2)
        ref_label = QLabel("参考图 (A)")
        ref_label.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        ref_label.setStyleSheet("color: #6B7280;")
        ref_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ref_lo.addWidget(ref_label)
        self._reference_preview = QLabel("无图片")
        self._reference_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._reference_preview.setMinimumWidth(320)
        self._reference_preview.setFrameShape(QFrame.Shape.StyledPanel)
        self._reference_preview.setStyleSheet("border: 1px solid #D9DDE3; color: #9CA3AF;")
        ref_lo.addWidget(self._reference_preview, 1)
        split.addWidget(ref_container)

        # Current image B
        cur_container = QWidget()
        cur_lo = QVBoxLayout(cur_container)
        cur_lo.setContentsMargins(0, 0, 0, 0)
        cur_lo.setSpacing(2)
        cur_label = QLabel("当前图 (B)")
        cur_label.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        cur_label.setStyleSheet("color: #6B7280;")
        cur_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cur_lo.addWidget(cur_label)
        self._current_preview = QLabel("无图片")
        self._current_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._current_preview.setMinimumWidth(320)
        self._current_preview.setFrameShape(QFrame.Shape.StyledPanel)
        self._current_preview.setStyleSheet("border: 1px solid #D9DDE3; color: #9CA3AF;")
        cur_lo.addWidget(self._current_preview, 1)
        split.addWidget(cur_container)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        lo.addWidget(split, 2)

        stats = QHBoxLayout()
        self._composition = QLabel("构图偏差：—")
        self._anchor = QLabel("语义锚点：—")
        stats.addWidget(self._composition)
        stats.addStretch(1)
        stats.addWidget(self._anchor)
        lo.addLayout(stats)

        target_title = QLabel("目标方案")
        target_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lo.addWidget(target_title)
        self._target = QLabel("分析参考图和当前画面，生成目标拍摄方案。")
        self._target.setWordWrap(True)
        self._target.setFrameShape(QFrame.Shape.StyledPanel)
        lo.addWidget(self._target)

        self._delta_list = QListWidget()
        self._delta_list.setAlternatingRowColors(True)
        lo.addWidget(self._delta_list)

    def _image_to_pixmap(self, image):
        import cv2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return QPixmap.fromImage(
            QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        )

    def _set_pixmap(self, label, image, attr_name):
        if image is None:
            setattr(self, attr_name, QPixmap())
            label.setPixmap(QPixmap())
            label.setText("无图片")
            return
        pixmap = self._image_to_pixmap(image)
        setattr(self, attr_name, pixmap)
        label.setText("")
        self._refresh_previews()
        QTimer.singleShot(0, self._refresh_previews)

    def _refresh_previews(self):
        for label, pixmap in (
            (self._reference_preview, self._reference_pixmap),
            (self._current_preview, self._current_pixmap),
        ):
            if pixmap.isNull():
                continue
            size = label.contentsRect().size()
            if size.width() > 2 and size.height() > 2:
                label.setPixmap(pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_previews()

    def _sync_canvas_target(self):
        window = self.window()
        canvas = getattr(getattr(window, "_w2", None), "_cv", None)
        if canvas is None:
            return
        if self._reference is None or self._current_pose is None or self._current_image is None:
            canvas.clear_reference_target()
            return
        width, height = self._current_image_size
        current = build_reference_composition(self._current_pose, width, height)
        deltas = compare_pose_to_reference(self._reference_pose, self._current_pose, width, height)
        canvas.set_reference_target(self._reference, current, deltas, visible=True)

    def load_reference_path(self, path: str) -> bool:
        path = str(Path(path).expanduser())
        image = load_image(path)
        if image is None:
            self._summary.setText(f"无法读取参考图：{Path(path).name}")
            return False
        detector = self._detector() if callable(self._detector) else self._detector
        if detector is None:
            self._summary.setText("姿态模型仍在加载，请稍后再试。")
            return False
        pose = detector.detect(image)
        if pose is None:
            self._summary.setText(f"参考图未检测到人物：{Path(path).name}")
            return False
        self._reference_path = path
        self._reference_image = image
        self._reference_pose = pose
        self._reference = build_reference_composition(pose, image.shape[1], image.shape[0])
        self._last_target = None
        self._set_pixmap(self._reference_preview, image, "_reference_pixmap")
        self._summary.setText(
            os.path.basename(path) + f" · {frame_orientation(image)} · " + reference_summary(self._reference)
        )
        self._render_compare()
        self._sync_canvas_target()
        self.reference_changed.emit()
        return True

    def _load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择参考图", "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if path:
            self.load_reference_path(path)

    def clear_reference(self):
        self._reference_image = None
        self._reference_pose = None
        self._reference = None
        self._reference_path = None
        self._last_target = None
        self._reference_pixmap = QPixmap()
        self._delta_list.clear()
        self._composition.setText("构图偏差：—")
        self._anchor.setText("语义锚点：—")
        self._target.setText("分析参考图和当前画面，生成目标拍摄方案。")
        self._summary.setText("加载参考照片，用于比较构图和姿态。")
        self._set_pixmap(self._reference_preview, None, "_reference_pixmap")
        self._sync_canvas_target()
        self.reference_changed.emit()

    def _swap_images(self):
        if self._reference_image is None or self._current_image is None:
            return
        ref_img, cur_img = self._reference_image, self._current_image
        ref_pose, cur_pose = self._reference_pose, self._current_pose
        self._reference_image = cur_img
        self._reference_pose = cur_pose
        self._reference = build_reference_composition(cur_pose, cur_img.shape[1], cur_img.shape[0])
        self._set_pixmap(self._reference_preview, cur_img, "_reference_pixmap")
        self._current_image = ref_img
        self._current_pose = ref_pose
        self._current_image_size = ref_img.shape[:2][::-1]
        self._set_pixmap(self._current_preview, ref_img, "_current_pixmap")
        self._render_compare()
        self._sync_canvas_target()
        self.reference_changed.emit()

    def set_current(self, pose, image):
        self._current_pose = pose
        self._current_image = image
        self.reference_changed.emit()
        self._current_image_size = image.shape[:2][::-1] if image is not None else (1, 1)
        self._set_pixmap(self._current_preview, image, "_current_pixmap")
        self._render_compare()
        self._sync_canvas_target()

    def set_current_image(self, image, pose):
        self.set_current(pose, image)

    def _render_compare(self):
        if self._reference is None or self._reference_pose is None or self._current_pose is None:
            return
        width, height = self._current_image_size
        current = build_reference_composition(self._current_pose, width, height)
        delta = composition_delta(self._reference, current)
        self._composition.setText(
            f"构图偏差：中心 Δ {abs(delta['center_dx']):.1%}×{abs(delta['center_dy']):.1%} · "
            f"当前/参考比例 {delta['scale_ratio']:.0%}"
        )
        anchor = next((a for a in self._reference.anchors if a.name == "hip_center"), None)
        self._anchor.setText(f"语义锚点：{anchor.name if anchor else 'bbox_center'}")

        deltas = compare_pose_to_reference(self._reference_pose, self._current_pose, width, height)
        plan = build_reference_target_plan(self._reference, current, deltas)
        self._last_target = plan
        self._target.setText(plan.as_text())

        self._delta_list.clear()
        if not deltas:
            self._delta_list.addItem(QListWidgetItem("当前姿态与参考图的可见关键点已较为接近。"))
            return
        for item in deltas:
            self._delta_list.addItem(QListWidgetItem(item.instruction))


def install_reference_mode(window):
    """Install reference mode widget into the pre-registered container.

    MainWindow.__init__ already created _w_ref_container and added it as page 2.
    This function creates the ReferenceModeWidget and wires the data flow.
    """
    widget = ReferenceModeWidget(lambda: window._det, window)
    window._reference_mode = widget
    window._w_ref_lo.addWidget(widget)

    # Hook into analysis results — reference mode needs pose + image updates
    old_update = window._w2.update_results

    def update_results(bundle):
        old_update(bundle)
        if bundle.pose and window._img is not None:
            widget.set_current(bundle.pose, window._img)

    window._w2.update_results = update_results
    return widget
