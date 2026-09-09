"""v3 reference-photo comparison workspace.

The workspace keeps the reference image as an explicit target and compares the
current analysis against it using pose/composition deltas. It is intentionally
2D-first: camera reconstruction remains evidence-backed and is not replaced by
an ungrounded metric scene claim.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QSplitter, QListWidget, QListWidgetItem, QFrame,
)

from core.image_io import load_image, frame_orientation
from reverse_engineering.reference_reconstruction import (
    build_reference_composition,
    compare_pose_to_reference,
    composition_delta,
    reference_summary,
)
from reverse_engineering.reference_targets import build_reference_target_plan


class ReferenceModeWidget(QWidget):
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
        title = QLabel("REFERENCE RECONSTRUCTION · V3 PHASE 2")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch(1)
        self._open = QPushButton("Load Reference")
        self._open.clicked.connect(self._load_reference)
        header.addWidget(self._open)
        self._clear = QPushButton("Clear")
        self._clear.clicked.connect(self.clear_reference)
        header.addWidget(self._clear)
        lo.addLayout(header)

        self._summary = QLabel("Load a reference photograph to compare composition and pose.")
        self._summary.setWordWrap(True)
        lo.addWidget(self._summary)

        split = QSplitter(Qt.Horizontal)
        self._reference_preview = QLabel("Reference")
        self._reference_preview.setAlignment(Qt.AlignCenter)
        self._reference_preview.setMinimumWidth(360)
        self._reference_preview.setFrameShape(QFrame.StyledPanel)
        self._current_preview = QLabel("Current")
        self._current_preview.setAlignment(Qt.AlignCenter)
        self._current_preview.setMinimumWidth(360)
        self._current_preview.setFrameShape(QFrame.StyledPanel)
        split.addWidget(self._reference_preview)
        split.addWidget(self._current_preview)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        lo.addWidget(split, 2)

        stats = QHBoxLayout()
        self._composition = QLabel("Composition delta: —")
        self._anchor = QLabel("Semantic anchor: —")
        stats.addWidget(self._composition)
        stats.addStretch(1)
        stats.addWidget(self._anchor)
        lo.addLayout(stats)

        target_title = QLabel("TARGET PLAN")
        target_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lo.addWidget(target_title)
        self._target = QLabel("Analyze a reference and current frame to generate a target shooting plan.")
        self._target.setWordWrap(True)
        self._target.setFrameShape(QFrame.StyledPanel)
        lo.addWidget(self._target)

        self._delta_list = QListWidget()
        self._delta_list.setAlternatingRowColors(True)
        lo.addWidget(self._delta_list, 1)

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
            label.setText("No image")
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
        """Load a reference image programmatically for reconstruction-session restore."""
        path = str(Path(path).expanduser())
        image = load_image(path)
        if image is None:
            self._summary.setText(f"Reference image could not be read: {Path(path).name}")
            return False
        pose = self._detector.detect(image)
        if pose is None:
            self._summary.setText(f"No person detected in reference image: {Path(path).name}")
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
        return True

    def _load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
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
        self._composition.setText("Composition delta: —")
        self._anchor.setText("Semantic anchor: —")
        self._target.setText("Analyze a reference and current frame to generate a target shooting plan.")
        self._summary.setText("Load a reference photograph to compare composition and pose.")
        self._set_pixmap(self._reference_preview, None, "_reference_pixmap")
        self._sync_canvas_target()

    def set_current(self, pose, image):
        self._current_pose = pose
        self._current_image = image
        self._current_image_size = image.shape[:2][::-1] if image is not None else (1, 1)
        self._set_pixmap(self._current_preview, image, "_current_pixmap")
        self._render_compare()
        self._sync_canvas_target()

    def _render_compare(self):
        if self._reference is None or self._reference_pose is None or self._current_pose is None:
            return
        width, height = self._current_image_size
        current = build_reference_composition(self._current_pose, width, height)
        delta = composition_delta(self._reference, current)
        self._composition.setText(
            f"Composition delta: center Δ {abs(delta['center_dx']):.1%}×{abs(delta['center_dy']):.1%} · "
            f"current/reference scale {delta['scale_ratio']:.0%}"
        )
        anchor = next((a for a in self._reference.anchors if a.name == "hip_center"), None)
        self._anchor.setText(f"Semantic anchor: {anchor.name if anchor else 'bbox_center'}")

        deltas = compare_pose_to_reference(self._reference_pose, self._current_pose, width, height)
        plan = build_reference_target_plan(self._reference, current, deltas)
        self._last_target = plan
        self._target.setText(plan.as_text())

        self._delta_list.clear()
        if not deltas:
            self._delta_list.addItem(QListWidgetItem("Pose is already close to the reference on the visible landmarks."))
            return
        for item in deltas:
            self._delta_list.addItem(QListWidgetItem(item.instruction))


def install_reference_mode(window):
    """Attach the v3 reference workspace and keep it synchronized with analysis."""
    widget = ReferenceModeWidget(window._det, window)
    window._reference_mode = widget
    window._tabs.addTab("Reference")
    window._ws.addWidget(widget)

    old_update = window._w2.update_results

    def update_results(bundle):
        old_update(bundle)
        if bundle.pose and window._img is not None:
            widget.set_current(bundle.pose, window._img)
    window._w2.update_results = update_results
    return widget
