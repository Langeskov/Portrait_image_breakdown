"""Application composition for the desktop GUI.

Runtime owns application-level actions such as Settings and session I/O.
Reconstruction-specific widgets are installed by ``v3_completion``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import urllib.parse
import tomllib

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from core.application_services import ApplicationServices
from core.model_config import DEFAULT_POSE_MODEL, pose_model_choices, pose_model_label
from gui.application_window import ApplicationMainWindow
from gui.field_mode import install_field_mode
from gui.reference_mode import install_reference_mode
from gui.anchor_calibration_dialog import AnchorCalibrationDialog
from gui.v3_completion import install_v3_completion
from reverse_engineering.calibration import BUILTIN_PROFILES
from reverse_engineering.reconstruction_session import load_session, save_session

FEEDBACK_EMAIL = "lolekseit@foxmail.com"
PROJECT_URL = "https://github.com/Langeskov/Portrait_image_breakdown"


def _read_app_version() -> str:
    """Read the project version from pyproject.toml with a safe fallback."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        with pyproject.open("rb") as handle:
            version = tomllib.load(handle).get("project", {}).get("version")
        return str(version or "0.3.0")
    except (OSError, tomllib.TOMLDecodeError, TypeError):
        return "0.3.0"


APP_VERSION = _read_app_version()


@dataclass
class RuntimeContext:
    services: ApplicationServices
    calibration_profile: str = "Generic"
    current_path: Optional[str] = None
    pose_model: str = DEFAULT_POSE_MODEL


def _install_settings_menu(window: ApplicationMainWindow, context: RuntimeContext) -> None:
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
        dialog.resize(680, 500)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        repository = QLineEdit(PROJECT_URL)
        repository.setReadOnly(True)
        repository.setToolTip("可直接选中复制项目地址")
        version = QLineEdit(APP_VERSION)
        version.setReadOnly(True)
        version.setToolTip("当前应用版本号")
        subject = QLineEdit("Portrait Image Breakdown feedback")
        body = QPlainTextEdit()
        body.setPlaceholderText("请描述问题、复现步骤、期望结果，以及任何有帮助的日志信息。")
        body.setPlainText(
            f"项目地址：{PROJECT_URL}\n"
            f"版本号：{APP_VERSION}\n\n"
            "问题描述：\n"
        )

        form.addRow("项目地址", repository)
        form.addRow("版本号", version)
        form.addRow("Subject", subject)
        form.addRow("Details", body)
        layout.addLayout(form)

        copy_button = QPushButton("复制项目地址和版本号")

        def copy_metadata() -> None:
            QApplication.clipboard().setText(
                f"Portrait Image Breakdown\n项目地址：{PROJECT_URL}\n版本号：{APP_VERSION}"
            )
            copy_button.setText("已复制")

        copy_button.clicked.connect(copy_metadata)
        layout.addWidget(copy_button)

        hint = QLabel(
            f"邮件地址：{FEEDBACK_EMAIL}\n"
            "项目地址和版本号会自动加入反馈正文，也可以单独复制后粘贴到 Issue。"
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

    model_menu = advanced.addMenu(f"Pose Model · {pose_model_label(context.pose_model)}")
    model_actions = []
    model_group = QActionGroup(window)
    model_group.setExclusive(True)
    for spec in pose_model_choices():
        action = QAction(spec.label, window)
        action.setCheckable(True)
        action.setData(spec.key)
        action.setChecked(spec.key == context.pose_model)
        model_group.addAction(action)
        model_actions.append(action)
        model_menu.addAction(action)

    current_action = QAction(f"Current: {pose_model_label(context.pose_model)}", window)
    current_action.setEnabled(False)
    model_menu.addSeparator()
    model_menu.addAction(current_action)

    def on_model_triggered(action: QAction):
        selected = str(action.data())
        if selected == context.pose_model:
            return
        try:
            window.set_pose_model(selected)
            context.pose_model = selected
            label = pose_model_label(selected)
            model_menu.setTitle(f"Pose Model · {label}")
            current_action.setText(f"Current: {label}")
            window._st.showMessage(f"Pose model: {window.pose_model_name}")
        except Exception as exc:
            QMessageBox.critical(
                window,
                "Pose model",
                f"Unable to switch pose model.\n\n{type(exc).__name__}: {exc}",
            )
            current = next((item for item in model_actions if str(item.data()) == context.pose_model), None)
            if current is not None:
                current.setChecked(True)

    model_group.triggered.connect(on_model_triggered)

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
    window._pose_model_menu = model_menu
    window._pose_model_actions = model_actions
    window._pose_model_group = model_group
    window._pose_model_current_action = current_action


def _session_reference_metadata(window) -> dict:
    reference = getattr(window, "_reference_mode", None)
    return {
        "reference_image_path": getattr(reference, "_reference_path", None) if reference is not None else None,
        "reference_active": bool(getattr(reference, "_reference", None)) if reference is not None else False,
    }


def _save_reconstruction_session(window) -> None:
    path, _ = QFileDialog.getSaveFileName(
        window, "Save reconstruction session", "", "Reconstruction Session (*.pibr.json)"
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
        window, "Load reconstruction session", "", "Reconstruction Session (*.pibr.json)"
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
            panel.sync_from_scene()

        anchor_panel = getattr(window._w3, "_anchor_camera_hypothesis", None)
        if anchor_panel is not None:
            anchor_panel.refresh_plane_choices()
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
    """Expose the single session entry point inside Temporal."""
    temporal = getattr(window, "_temporal_workspace", None)
    if temporal is None:
        return
    box = QWidget(temporal)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 10, 0, 0)
    layout.setSpacing(5)
    title = QLabel("Reconstruction session")
    title.setStyleSheet("font-size:11pt; font-weight:600;")
    layout.addWidget(title)
    hint = QLabel("Save / Load preserves the editable V3 scene, anchors, plane constraints and session metadata.")
    hint.setWordWrap(True)
    hint.setStyleSheet("color:#64748B;")
    layout.addWidget(hint)
    row = QHBoxLayout()
    save_button = QPushButton("Save Session")
    load_button = QPushButton("Load Session")
    save_button.clicked.connect(lambda: _save_reconstruction_session(window))
    load_button.clicked.connect(lambda: _load_reconstruction_session(window))
    row.addWidget(save_button)
    row.addWidget(load_button)
    row.addStretch(1)
    layout.addLayout(row)
    root = temporal.layout()
    if root is not None:
        root.addWidget(box)
    window._temporal_session_controls = box


def install_v3_toolbar(window: ApplicationMainWindow, context: RuntimeContext) -> None:
    bars = window.findChildren(QToolBar)
    if not bars:
        return
    bar = bars[0]
    bar.addWidget(QLabel("  Calibration: "))
    combo = QComboBox()
    calibration_labels = {
        "Generic": "通用",
        "Full Frame 36x24": "全画幅 36×24",
        "APS-C 23.5x15.6": "APS-C 23.5×15.6",
        "Micro Four Thirds 17.3x13": "M4/3 17.3×13",
    }
    for profile_name in BUILTIN_PROFILES:
        combo.addItem(calibration_labels.get(profile_name, profile_name), profile_name)
    combo.setCurrentIndex(max(0, list(BUILTIN_PROFILES.keys()).index(context.calibration_profile)))

    def on_profile_changed(_display_name):
        profile_name = combo.currentData()
        context.calibration_profile = str(profile_name or "Generic")
        window.set_calibration_profile(context.calibration_profile)
        if window._current_path:
            window.load_image(window._current_path)

    combo.currentTextChanged.connect(on_profile_changed)
    bar.addWidget(combo)

    anchor_action = QAction("Anchor Calibration", window)

    def open_anchor_calibration():
        image = window.current_image
        if image is None:
            QMessageBox.information(window, "Anchor Calibration", "请先加载一张照片并完成至少一次分析。")
            return
        dialog = AnchorCalibrationDialog(
            window._w3.scene_model,
            (image.shape[1], image.shape[0]),
            image=image,
            parent=window,
        )
        dialog.exec()
        panel = getattr(window._w3, "_plane_constraints_panel", None)
        if panel is not None:
            panel.sync_from_scene()
        anchor_panel = getattr(window._w3, "_anchor_camera_hypothesis", None)
        if anchor_panel is not None:
            anchor_panel.refresh_plane_choices()
            anchor_panel.solve()
        window._w3._populate_anchors()
        window._w3.refresh_scene_view()

    anchor_action.triggered.connect(open_anchor_calibration)
    bar.addAction(anchor_action)

    _install_settings_menu(window, context)
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
    context.pose_model = window.pose_model
    install_field_mode(window)
    install_reference_mode(window)
    install_v3_completion(window)
    install_v3_toolbar(window, context)
    return window, context
