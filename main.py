"""Portrait Image Breakdown - Photography Analysis & Reverse Engineering."""
import re
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_v2_engine():
    """Make the v2 engine the canonical engine used by GUI and CLI."""
    import reverse_engineering.engine as engine_module
    from reverse_engineering.engine_v2 import ReverseEngineeringEngineV2
    engine_module.ReverseEngineeringEngine = ReverseEngineeringEngineV2


def _strip_evidence_block(text: str) -> str:
    """Remove both legacy Evidence State renderings from a report."""
    text = re.sub(
        r"\n*={20,}\nEVIDENCE STATE\n={20,}\n"
        r"Observed:\s*\d+.*?\n"
        r"Observed = directly supported by image/metadata\.\n"
        r"Estimated = inferred from available evidence and model confidence\.\n"
        r"Unknown = insufficient evidence; do not treat as a measured value\.\s*",
        "", text, flags=re.DOTALL,
    )
    text = re.sub(
        r"\n*-- Evidence State --\n\s*observed:\s*\d+\n\s*estimated:\s*\d+\n\s*unknown:\s*\d+\s*",
        "", text, flags=re.IGNORECASE,
    )
    return text.rstrip()


def _append_evidence_state(text: str, reverse_result) -> str:
    """Render exactly one Evidence State block from a fresh report string."""
    summary = reverse_result.evidence_summary(); counts = summary["counts"]; base_text = _strip_evidence_block(text)
    return base_text + ("\n\n" + "=" * 55 + "\nEVIDENCE STATE\n" + "=" * 55 + "\n"
        f"Observed: {counts['observed']}  |  Estimated: {counts['estimated']}  |  Unknown: {counts['unknown']}\n"
        "Observed = directly supported by image/metadata.\n"
        "Estimated = inferred from available evidence and model confidence.\n"
        "Unknown = insufficient evidence; do not treat as a measured value.")


def run_gui(image_path=None):
    from PySide6.QtWidgets import QApplication, QCheckBox, QToolBar, QComboBox, QLabel
    _install_v2_engine()
    import gui.main_window as main_window_module
    from gui.main_window import MainWindow, apply_light_theme
    from gui.reverse_3d_v3 import Reverse3DWorkspace as RealReverse3DWorkspace
    from gui.cache import AnalysisCache, image_cache_key
    from gui.field_mode import install_field_mode
    from gui.reference_mode import install_reference_mode
    from gui.anchor_calibration_dialog import AnchorCalibrationDialog
    from reverse_engineering.calibration import BUILTIN_PROFILES
    import reverse_engineering.engine as engine_module

    main_window_module.Reverse3DWorkspace = RealReverse3DWorkspace
    main_window_module._image_hash = image_cache_key
    original_update_results = main_window_module.Analysis2DWorkspace.update_results

    def update_results_with_reverse(self, bundle):
        original_update_results(self, bundle)
        if bundle.reverse_result: self._cv.set_reverse_result(bundle.reverse_result)
    main_window_module.Analysis2DWorkspace.update_results = update_results_with_reverse
    main_window_module.ResultsWorkspace.update_results = lambda self, bundle: (
        self._rl.setText(_append_evidence_state(bundle.reverse_result.report(), bundle.reverse_result))
        if bundle.reverse_result else self._rl.setText("Reverse engineering not yet complete...")
    )

    app = QApplication(sys.argv); app.setStyle("Fusion"); apply_light_theme(app)
    window = MainWindow(); window._result_cache = AnalysisCache(capacity=8); install_field_mode(window); install_reference_mode(window)
    selected_profile = {"name": "Generic"}; EngineV2 = engine_module.ReverseEngineeringEngine

    def engine_factory(enable_simulation=True):
        return EngineV2(enable_simulation=enable_simulation, calibration_profile=selected_profile["name"])
    engine_module.ReverseEngineeringEngine = engine_factory

    original_load = window._la
    def load_with_projection_sync(path):
        window._current_path = str(path); original_load(path)
        if window._img is not None: window._w3.set_image(window._img)
    window._la = load_with_projection_sync

    reverse_toggle = QCheckBox("Reverse Evidence"); reverse_toggle.setChecked(False)
    def apply_reverse_toggle(_state=0):
        window._w2._cv.set_overlay_options(
            skeleton=window._chk_skeleton.isChecked(), thirds=window._chk_thirds.isChecked(), center=window._chk_center.isChecked(),
            bbox=window._chk_bbox.isChecked(), visual_weight=window._chk_vweight.isChecked(), headroom=window._chk_headroom.isChecked(), reverse=reverse_toggle.isChecked())
    reverse_toggle.stateChanged.connect(apply_reverse_toggle)
    bars = window.findChildren(QToolBar)
    if bars:
        bars[0].addWidget(reverse_toggle); bars[0].addWidget(QLabel("  Calibration: "))
        calibration_combo = QComboBox(); calibration_combo.addItems(list(BUILTIN_PROFILES.keys())); calibration_combo.setCurrentText(selected_profile["name"])
        calibration_combo.setToolTip("Camera calibration profile used by 3D reconstruction and projection")
        def on_profile_changed(name):
            selected_profile["name"] = name or "Generic"; window._eng = None
            if getattr(window, "_current_path", None): window._la(window._current_path)
        calibration_combo.currentTextChanged.connect(on_profile_changed); bars[0].addWidget(calibration_combo)
        anchor_action = QAction("Anchor Calibration", window)
        anchor_action.setToolTip("Bind manual image points to scene anchors and estimate a camera hypothesis")
        def open_anchor_calibration():
            if window._img is None:
                QMessageBox.information(window, "Anchor Calibration", "请先加载一张照片并完成至少一次分析。")
                return
            dialog = AnchorCalibrationDialog(window._w3.scene, (window._img.shape[1], window._img.shape[0]), window)
            dialog.exec()
            window._w3._view.update()
        anchor_action.triggered.connect(open_anchor_calibration); bars[0].addAction(anchor_action)
    window.show()
    if image_path and os.path.exists(image_path): window._la(image_path)
    sys.exit(app.exec())


def run_cli(image_path, verbose=False, calibration_profile="Generic"):
    _install_v2_engine()
    from core.image_io import load_image, frame_orientation
    from core.orientation import analyze_orientation
    from core.action_classifier import classify_action
    from core.camera_analyzer import analyze_camera
    from core.composition import analyze_composition
    from core.suggestion import generate_suggestions
    from reverse_engineering.engine import ReverseEngineeringEngine
    from reverse_engineering.intrinsics import read_exif_intrinsics

    image = load_image(image_path)
    if image is None:
        print(f"Error: cannot read {image_path}"); sys.exit(1)
    intrinsics = read_exif_intrinsics(image_path)
    print(f"Analyzing: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Frame orientation: {frame_orientation(image)}")
    print(f"Calibration profile: {calibration_profile}")
    if intrinsics.has_focal_prior: print(f"EXIF intrinsics: focal={intrinsics.focal_length_mm} mm, 35mm eq={intrinsics.focal_length_35mm} mm")
    print("=" * 60)

    det = __import__("core.pose_detector", fromlist=["PoseDetector"]).PoseDetector()
    engine = ReverseEngineeringEngine(calibration_profile=calibration_profile)
    try:
        pose = det.detect(image)
        if pose is None: print("No person detected"); sys.exit(1)
        vis = sum(1 for lm in pose.landmarks[:17] if lm.visibility > 0.4)
        print(f"Skeleton: {pose.detection_confidence:.0%} confidence, {vis}/17 visible keypoints")
        orient = analyze_orientation(pose); print(f"\nOrientation: {orient.facing.value}, {orient.tilt.value}, {orient.facing_angle:.1f} deg")
        action = classify_action(pose); print(f"Action: {action.category.value} ({action.confidence:.0%})")
        camera = analyze_camera(pose); print(f"Camera: {camera.shot_type.value}, {camera.camera_angle.value}, subject={camera.subject_ratio:.1%}")
        composition = analyze_composition(image, pose); print(f"Composition: {composition.primary_type.value}, thirds={composition.thirds_alignment:.0%}")
        suggestions = generate_suggestions(action, orient, camera, composition)
        print(f"\nSuggestions ({len(suggestions.suggestions)}):")
        for s in suggestions.suggestions[:5]: print(f"  [{s.priority.value}] {s.title}: {s.description}")
        print(f"\nNext actions: {', '.join(suggestions.next_actions)}")
        print(f"Creative: {suggestions.creative_direction}")
        print("\n--- Reverse Engineering v2.5 ---")
        result = engine.analyze(image, pose, pose.bbox, intrinsics_evidence=intrinsics)
        print(result.report())
        print("\nCamera Actions:")
        for a in result._camera_actions:
            print(f"  {a.action}: {a.expected_effect}")
            for r in a.reason[:2]: print(f"    - {r}")
    finally:
        det.close()


def main():
    parser = argparse.ArgumentParser(description="Portrait Image Breakdown")
    parser.add_argument("--image", "-i", help="Image path to analyze")
    parser.add_argument("--cli", action="store_true", help="CLI mode")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--calibration-profile", default="Generic", help="Calibration profile: Generic, Full Frame 36x24, APS-C 23.5x15.6, Micro Four Thirds 17.3x13")
    args = parser.parse_args()
    if args.image and args.cli: run_cli(args.image, args.verbose, args.calibration_profile)
    elif args.image: run_gui(args.image)
    else: run_gui()


if __name__ == "__main__": main()
