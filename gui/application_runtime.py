"""Explicit application composition for the desktop GUI."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from PySide6.QtWidgets import QCheckBox, QLabel, QToolBar, QComboBox, QMessageBox
from PySide6.QtGui import QAction
from core.application_services import ApplicationServices
from gui.application_window import ApplicationMainWindow
from gui.reverse_3d_v3 import Reverse3DWorkspace
from gui.field_mode import install_field_mode
from gui.reference_mode import install_reference_mode
from gui.anchor_calibration_dialog import AnchorCalibrationDialog
from gui.v3_completion import install_v3_completion
from reverse_engineering.calibration import BUILTIN_PROFILES

@dataclass
class RuntimeContext:
    services: ApplicationServices
    calibration_profile: str = "Generic"
    current_path: Optional[str] = None

def _install_real_3d_workspace(window: ApplicationMainWindow) -> None:
    old = window._w3
    index = window._ws.indexOf(old)
    replacement = Reverse3DWorkspace(window)
    if index >= 0:
        window._ws.removeWidget(old); old.deleteLater(); window._ws.insertWidget(index, replacement); window._w3 = replacement

def install_v3_toolbar(window: ApplicationMainWindow, context: RuntimeContext) -> None:
    bars = window.findChildren(QToolBar)
    if not bars: return
    bar = bars[0]
    reverse_toggle = QCheckBox("Reverse Evidence")
    def update_reverse_overlay(_state=0):
        window._w2.set_overlay_options(skeleton=window._chk_skeleton.isChecked(), thirds=window._chk_thirds.isChecked(), center=window._chk_center.isChecked(), bbox=window._chk_bbox.isChecked(), visual_weight=window._chk_vweight.isChecked(), headroom=window._chk_headroom.isChecked(), reference_target=window._chk_reference_target.isChecked(), reverse=reverse_toggle.isChecked())
    reverse_toggle.stateChanged.connect(update_reverse_overlay); bar.addWidget(reverse_toggle); bar.addWidget(QLabel("  Calibration: "))
    combo = QComboBox(); combo.addItems(list(BUILTIN_PROFILES.keys())); combo.setCurrentText(context.calibration_profile)
    def on_profile_changed(name):
        context.calibration_profile = name or "Generic"; window.set_calibration_profile(context.calibration_profile)
        if window._current_path: window.load_image(window._current_path)
    combo.currentTextChanged.connect(on_profile_changed); bar.addWidget(combo)
    anchor_action = QAction("Anchor Calibration", window)
    def open_anchor_calibration():
        image = window.current_image
        if image is None:
            QMessageBox.information(window, "Anchor Calibration", "请先加载一张照片并完成至少一次分析。"); return
        dialog = AnchorCalibrationDialog(window._w3.scene_model, (image.shape[1], image.shape[0]), image=image, parent=window)
        dialog.exec(); window._w3.refresh_scene_view()
    anchor_action.triggered.connect(open_anchor_calibration); bar.addAction(anchor_action)

def build_window(services: ApplicationServices) -> tuple[ApplicationMainWindow, RuntimeContext]:
    context = RuntimeContext(services=services)
    window = ApplicationMainWindow(services)
    window._runtime_context = context
    window._calibration_profile = context.calibration_profile
    _install_real_3d_workspace(window)
    install_field_mode(window); install_reference_mode(window); install_v3_completion(window); install_v3_toolbar(window, context)
    return window, context
