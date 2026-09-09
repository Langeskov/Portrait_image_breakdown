"""Explicit application composition for the desktop GUI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import urllib.parse

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMenu, QMessageBox, QPlainTextEdit, QToolBar,
    QToolButton, QVBoxLayout, QWidget, QFileDialog, QPushButton,
)

from core.application_services import ApplicationServices
from gui.application_window import ApplicationMainWindow
from gui.reverse_3d_v3 import Reverse3DWorkspace
from gui.field_mode import install_field_mode
from gui.reference_mode import install_reference_mode
from gui.anchor_calibration_dialog import AnchorCalibrationDialog
from gui.v3_completion import install_v3_completion
from reverse_engineering.calibration import BUILTIN_PROFILES
from reverse_engineering.reconstruction_session import load_session, save_session

FEEDBACK_EMAIL = "lolekseit@foxmail.com"


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
        window._ws.removeWidget(old)
        old.deleteLater()
        window._ws.insertWidget(index, replacement)
        window._w3 = replacement


def _install_settings_menu(window: ApplicationMainWindow) -> None:
    """Expose advanced/debug functions without consuming normal navigation space."""
    bars = window.findChildren(QToolBar)
    if not bars:
        return
    bar = bars[0]
    button = QToolButton(window)
    button.setText("Settings")
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QMenu(button)

    feedback = QAction("Problem feedback", window)

    def open_feedback() -> None:
        dialog = QDialog(window)
        dialog.setWindowTitle("Problem feedback")
        dialog.resize(620, 430)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        subject = QLineEdit("Portrait Image Breakdown feedback")
        body = QPlainTextEdit()
        body.setPlaceholderText(
            "请描述遇到的问题、复现步骤、期望结果，以及任何有帮助的日志信息。"
        )
        form.addRow("Subject", subject)
        form.addRow("Details", body)
        layout.addLayout(form)
        hint = QLabel(
            f"邮件地址：{FEEDBACK_EMAIL}\n"
            "点击发送后会调用系统默认邮件客户端；程序不会保存邮件凭据。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748B;")
        layout.addWidget(hint)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Open mail client")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        query = urllib.parse.urlencode({
            "subject": subject.text().strip() or "Portrait Image Breakdown feedback",
            "body": body.toPlainText().strip(),
        })
        QDesktopServices.openUrl(QUrl(f"mailto:{FEEDBACK_EMAIL}?{query}"))

    feedback.triggered.connect(open_feedback)
    menu.addAction(feedback)
    menu.addSeparator()

    advanced = menu.addMenu("Advanced")
    results = QAction("Results", window)

    def show_results() -> None:
        index = getattr(window, "_results_tab_index", -1)
        if index < 0:
            return
        window._ws.setCurrentIndex(index)
        window._tabs.blockSignals(True)
        window._tabs.setCurrentIndex(index)
        window._tabs.blockSignals(False)

    results.triggered.connect(show_results)
    advanced.addAction(results)
    button.setMenu(menu)
    bar.addWidget(button)
    window._settings_menu = menu
    window._feedback_action = feedback
    window._advanced_results_action = results


def _session_reference_metadata(window) -> dict:
    reference = getattr(window, "_reference_mode", None)
    return {
        "reference_image_path": getattr(reference, "_reference_path", None) if reference is not None else None,
        "reference_active": bool(getattr(reference, "_reference", None)) if reference is not None else False,
    }


def _save_reconstruction_session(window) -> None:
    path, _ = QFileDialog.getSaveFileName(
        window, "Save reconstruction session", "",
        "Reconstruction Session (*.pibr.json)"
    )
    if not path:
        return
    panel = getattr(window._w3, "_plane_constraints_panel", None)
    constraints = panel.to_dicts() if panel is not None else getattr(window._w3.scene, "plane_constraints", [])
    image = getattr(window, "_img", None)
    try:
        save_session(
            path,
            window._w3.scene,
            image_path=getattr(window, "_current_path", None),
            image_shape=image.shape[:2][::-1] if image is not None else None,
            metadata=_session_reference_metadata(window),
            plane_constraints=constraints,
        )
        window._st.showMessage(f"Saved reconstruction session · {Path(path).name}")
    except Exception as exc:
        QMessageBox.critical(window, "Save session failed", f"{type(exc).__name__}: {exc}")


def _load_reconstruction_session(window) -> None:
    path, _ = QFileDialog.getOpenFileName(
        window, "Load reconstruction session", "",
        "Reconstruction Session (*.pibr.json)"
    )
    if not path:
        return
    try:
        scene, data = load_session(path)
        window._w3.scene = scene
        window._w3._view.set_scene(scene)
        window._w3._sync_controls()
        window._w3._populate_people()
        window._w3._populate_anchors()
        window._w3._populate_candidates()
        window._w3._refresh_projection()
        panel = getattr(window._w3, "_plane_constraints_panel", None)
        if panel is not None:
            panel.from_dicts(data.get("plane_constraints", []))
        anchor_panel = getattr(window._w3, "_anchor_camera_hypothesis", None)
        if anchor_panel is not None:
            anchor_panel.solve()
        stored_image = data.get("image", {}).get("path")
        if stored_image:
            window._current_path = stored_image
        reference = getattr(window, "_reference_mode", None)
        reference_path = dict(data.get("metadata", {})).get("reference_image_path")
        if reference is not None and reference_path:
            loader = getattr(reference, "load_reference_path", None)
            if loader is not None:
                loader(reference_path)
        window._st.showMessage(f"Loaded reconstruction session · {Path(path).name}")
    except Exception as exc:
        QMessageBox.critical(window, "Load session failed", f"{type(exc).__name__}: {exc}")


def _install_temporal_session_controls(window) -> None:
    """Keep exactly one Save/Load Session entry point inside Temporal."""
    temporal = getattr(window, "_temporal_workspace", None)
    if temporal is None:
        return
    for bar in window.findChildren(QToolBar):
        for action in list(bar.actions()):
            if action.text() in {"Save Session", "Load Session"}:
                bar.removeAction(action)
    session_box = QWidget(temporal)
    layout = QVBoxLayout(session_box)
    layout.setContentsMargins(0, 10, 0, 0)
    layout.setSpacing(5)
    title = QLabel("Reconstruction session")
    title.setStyleSheet("font-size:11pt; font-weight:600;")
    layout.addWidget(title)
    hint = QLabel("Save / Load preserves the editable V3 reconstruction scene, anchors, plane constraints and session metadata.")
    hint.setWordWrap(True)
    hint.setStyleSheet("color:#64748B;")
    layout.addWidget(hint)
    row = QHBoxLayout()
    save_button = QPushButton("Save Session")
    load_button = QPushButton("Load Session")
    save_button.clicked.connect(lambda: _save_reconstruction_session(window))
    load_button.clicked.connect(lambda: _load_reconstruction_session(window))
    row.addWidget(save_button); row.addWidget(load_button); row.addStretch(1)
    layout.addLayout(row)
    root = temporal.layout()
    if root is not None:
        root.addWidget(session_box)
    window._temporal_session_controls = session_box


def install_v3_toolbar(window: ApplicationMainWindow, context: RuntimeContext) -> None:
    bars = window.findChildren(QToolBar)
    if not bars:
        return
    bar = bars[0]
    bar.addWidget(QLabel("  Calibration: "))
    combo = QComboBox(); combo.addItems(list(BUILTIN_PROFILES.keys())); combo.setCurrentText(context.calibration_profile)

    def on_profile_changed(name):
        context.calibration_profile = name or "Generic"
        window.set_calibration_profile(context.calibration_profile)
        if window._current_path:
            window.load_image(window._current_path)
    combo.currentTextChanged.connect(on_profile_changed); bar.addWidget(combo)

    anchor_action = QAction("Anchor Calibration", window)
    def open_anchor_calibration():
        image = window.current_image
        if image is None:
            QMessageBox.information(window, "Anchor Calibration", "请先加载一张照片并完成至少一次分析。"); return
        dialog = AnchorCalibrationDialog(window._w3.scene_model, (image.shape[1], image.shape[0]), image=image, parent=window)
        dialog.exec(); window._w3.refresh_scene_view()
    anchor_action.triggered.connect(open_anchor_calibration); bar.addAction(anchor_action)

    _install_settings_menu(window)
    results_index = next((i for i in range(window._tabs.count()) if window._tabs.tabText(i) == "Results"), -1)
    if results_index >= 0:
        window._tabs.setTabVisible(results_index, False)
    window._results_tab_index = results_index
    _install_temporal_session_controls(window)


def build_window(services: ApplicationServices) -> tuple[ApplicationMainWindow, RuntimeContext]:
    context = RuntimeContext(services=services)
    window = ApplicationMainWindow(services)
    window._runtime_context = context
    window._calibration_profile = context.calibration_profile
    _install_real_3d_workspace(window)
    install_field_mode(window)
    install_reference_mode(window)
    install_v3_completion(window)
    install_v3_toolbar(window, context)
    return window, context
