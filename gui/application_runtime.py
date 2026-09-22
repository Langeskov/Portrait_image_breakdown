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


def build_window(services: ApplicationServices) -> tuple[ApplicationMainWindow, RuntimeContext]:
    context = RuntimeContext(services=services)
    window = ApplicationMainWindow(services)
    window._runtime_context = context
    window._calibration_profile = context.calibration_profile
    context.pose_model = window.pose_model
    install_field_mode(window)
    install_reference_mode(window)
    _install_settings_menu(window, context)
    return window, context
