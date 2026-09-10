"""Portrait Image Breakdown - explicit application entry points."""
from __future__ import annotations

# Nuitka release configuration. The source build remains unchanged when run
# normally with Python, while ``python -m nuitka main.py`` becomes the
# standalone Windows distribution used for packaging.
# Nuitka 4.x currently has a Torch package-config regression around
# ``torch.utils._config_module``. The Windows release script pins the
# compiler to Nuitka 2.8.10, which is a known-good combination for this app.
# nuitka-project: --mode=standalone
# nuitka-project: --enable-plugin=pyside6
# nuitka-project: --include-data-dir={MAIN_DIRECTORY}/model=model
# nuitka-project: --include-data-dir={MAIN_DIRECTORY}/assets=assets
# nuitka-project-if: {OS} == "Windows":
#     nuitka-project: --windows-console-mode=disable

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _strip_evidence_block(text: str) -> str:
    text = re.sub(r"\n*={20,}\nEVIDENCE STATE\n={20,}\nObserved:\s*\d+.*?\nObserved = directly supported by image/metadata\.\nEstimated = inferred from available evidence and model confidence\.\nUnknown = insufficient evidence; do not treat as a measured value\.\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n*-- Evidence State --\n\s*observed:\s*\d+\n\s*estimated:\s*\d+\n\s*unknown:\s*\d+\s*", "", text, flags=re.IGNORECASE)
    return text.rstrip()


def _append_evidence_state(text: str, reverse_result) -> str:
    counts = reverse_result.evidence_summary()["counts"]
    return _strip_evidence_block(text) + ("\n\n" + "=" * 55 + "\nEVIDENCE STATE\n" + "=" * 55 + "\n"
        f"Observed: {counts['observed']}  |  Estimated: {counts['estimated']}  |  Unknown: {counts['unknown']}\n"
        "Observed = directly supported by image/metadata.\n"
        "Estimated = inferred from image, geometry, priors, and model confidence.\n"
        "Unknown = insufficient evidence; do not treat as a measured value.")


def _find_app_icon() -> Path | None:
    """Locate the single bundled application icon."""
    candidate = ROOT / "assets" / "icon.png"
    return candidate if candidate.exists() else None


def run_gui(image_path: str | None = None, calibration_profile: str = "Generic") -> None:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    from core.application_services import ApplicationServices
    from gui.application_runtime import build_window
    from gui.main_window import apply_light_theme
    from gui.i18n_zh_extra import install_chinese_ui
    services = ApplicationServices.create(calibration_profile=calibration_profile)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    icon_path = _find_app_icon()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    apply_light_theme(app)
    window, context = build_window(services)
    if icon_path is not None:
        window.setWindowIcon(QIcon(str(icon_path)))
    install_chinese_ui(window)
    context.calibration_profile = calibration_profile
    context.current_path = str(image_path) if image_path else None
    window.show()
    if image_path and Path(image_path).exists():
        window.load_image(image_path)
    sys.exit(app.exec())


def run_cli(image_path: str, verbose: bool = False, calibration_profile: str = "Generic") -> None:
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
        print(f"Error: cannot read {image_path}"); raise SystemExit(1)
    intrinsics = read_exif_intrinsics(image_path)
    print(f"Analyzing: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Frame orientation: {frame_orientation(image)}")
    print(f"Calibration profile: {calibration_profile}")
    if intrinsics.has_focal_prior:
        print(f"EXIF intrinsics: focal={intrinsics.focal_length_mm} mm, 35mm eq={intrinsics.focal_length_35mm} mm")
    print("=" * 60)
    det = __import__("core.pose_detector", fromlist=["PoseDetector"]).PoseDetector()
    engine = ReverseEngineeringEngine(calibration_profile=calibration_profile)
    try:
        pose = det.detect(image)
        if pose is None:
            print("No person detected"); raise SystemExit(1)
        vis = sum(1 for lm in pose.landmarks[:17] if lm.visibility > 0.4)
        print(f"Skeleton: {pose.detection_confidence:.0%} confidence, {vis}/17 visible keypoints")
        orient = analyze_orientation(pose); print(f"\nOrientation: {orient.facing.value}, {orient.tilt.value}, {orient.facing_angle:.1f} deg")
        action = classify_action(pose); print(f"Action: {action.category.value} ({action.confidence:.0%})")
        camera = analyze_camera(pose); print(f"Camera: {camera.shot_type.value}, {camera.camera_angle.value}, subject={camera.subject_ratio:.1%}")
        composition = analyze_composition(image, pose); print(f"Composition: {composition.primary_type.value}, thirds={composition.thirds_alignment:.0%}")
        suggestions = generate_suggestions(action, orient, camera, composition)
        print(f"\nSuggestions ({len(suggestions.suggestions)}):")
        for suggestion in suggestions.suggestions[:5]: print(f"  [{suggestion.priority.value}] {suggestion.title}: {suggestion.description}")
        print(f"\nNext actions: {', '.join(suggestions.next_actions)}"); print(f"Creative: {suggestions.creative_direction}")
        print("\n--- Reverse Engineering v2.5 ---"); result = engine.analyze(image, pose, pose.bbox, intrinsics_evidence=intrinsics); print(_append_evidence_state(result.report(), result))
        print("\nCamera Actions:")
        for item in result._camera_actions:
            print(f"  {item.action}: {item.expected_effect}")
            for reason in item.reason[:2]: print(f"    - {reason}")
    finally:
        det.close()


def main():
    parser = argparse.ArgumentParser(description="Portrait Image Breakdown")
    parser.add_argument("--image", "-i", help="Image path to analyze")
    parser.add_argument("--cli", action="store_true", help="CLI mode")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--calibration-profile", default="Generic")
    args = parser.parse_args()
    if args.cli:
        if not args.image: parser.error("--cli requires --image")
        run_cli(args.image, args.verbose, args.calibration_profile)
    else:
        run_gui(args.image, args.calibration_profile)


if __name__ == "__main__": main()
