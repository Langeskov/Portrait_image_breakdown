from gui.i18n_zh import ZH, tr


def test_core_ui_labels_are_translated():
    assert tr("Open Image") == "打开图片"
    assert tr("2D Analysis") == "2D 分析"
    assert tr("3D Reverse Engineering") == "3D 反向工程"
    assert tr("Reference Camera Hypothesis") == "参考相机假设"


def test_technical_tokens_are_left_intact():
    assert "PnP" not in ZH
    assert "EXIF" not in ZH
    assert "YOLO" not in ZH
    assert "focal length" not in ZH


def test_import_hint_has_direct_image_actions():
    source = open("gui/i18n_zh.py", encoding="utf-8").read()
    assert "drag_enter" in source
    assert "drop_event" in source
    assert "mouse_press" in source
    assert "拖入图片，或点击此处选择图片" in source
