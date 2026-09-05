"""Portrait Image Breakdown - Photography Analysis & Reverse Engineering."""
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_gui(image_path=None):
    from PySide6.QtWidgets import QApplication, QCheckBox, QToolBar
    import gui.main_window as main_window_module
    from gui.main_window import MainWindow, apply_light_theme
    from gui.reverse_3d import Reverse3DWorkspace as RealReverse3DWorkspace
    from gui.cache import AnalysisCache, image_cache_key

    main_window_module.Reverse3DWorkspace = RealReverse3DWorkspace
    main_window_module._image_hash = image_cache_key

    original_update_results = main_window_module.Analysis2DWorkspace.update_results

    def update_results_with_reverse(self, bundle):
        original_update_results(self, bundle)
        if bundle.reverse_result:
            self._cv.set_reverse_result(bundle.reverse_result)

    main_window_module.Analysis2DWorkspace.update_results = update_results_with_reverse

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_light_theme(app)
    window = MainWindow()
    window._result_cache = AnalysisCache(capacity=8)

    # Keep the 3D projection preview synchronized with the currently loaded image.
    original_load = window._la

    def load_with_projection_sync(path):
        original_load(path)
        if window._img is not None:
            window._w3.set_image(window._img)

    window._la = load_with_projection_sync

    reverse_toggle = QCheckBox("Reverse Evidence")
    reverse_toggle.setChecked(False)

    def apply_reverse_toggle(_state=0):
        window._w2._cv.set_overlay_options(
            skeleton=window._chk_skeleton.isChecked(),
            thirds=window._chk_thirds.isChecked(),
            center=window._chk_center.isChecked(),
            bbox=window._chk_bbox.isChecked(),
            visual_weight=window._chk_vweight.isChecked(),
            headroom=window._chk_headroom.isChecked(),
            reverse=reverse_toggle.isChecked(),
        )

    reverse_toggle.stateChanged.connect(apply_reverse_toggle)
    bars = window.findChildren(QToolBar)
    if bars:
        bars[0].addWidget(reverse_toggle)

    window.show()
    if image_path and os.path.exists(image_path):
        window._la(image_path)
    sys.exit(app.exec())


def run_cli(image_path, verbose=False):
    import cv2
    from core.pose_detector import PoseDetector
    from core.orientation import analyze_orientation
    from core.action_classifier import classify_action
    from core.camera_analyzer import analyze_camera
    from core.composition import analyze_composition
    from core.suggestion import generate_suggestions
    from reverse_engineering.engine import ReverseEngineeringEngine

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read {image_path}")
        sys.exit(1)
    print(f"Analyzing: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print("=" * 60)

    det = PoseDetector()
    engine = ReverseEngineeringEngine()
    try:
        pose = det.detect(image)
        if pose is None:
            print("No person detected")
            sys.exit(1)
        vis = sum(1 for lm in pose.landmarks[:17] if lm.visibility > 0.4)
        print(f"Skeleton: {pose.detection_confidence:.0%} confidence, {vis}/17 visible keypoints")
        orient = analyze_orientation(pose)
        print(f"\nOrientation: {orient.facing.value}, {orient.tilt.value}, {orient.facing_angle:.1f} deg")
        action = classify_action(pose)
        print(f"Action: {action.category.value} ({action.confidence:.0%})")
        camera = analyze_camera(pose)
        print(f"Camera: {camera.shot_type.value}, {camera.camera_angle.value}, subject={camera.subject_ratio:.1%}")
        composition = analyze_composition(image, pose)
        print(f"Composition: {composition.primary_type.value}, thirds={composition.thirds_alignment:.0%}")
        suggestions = generate_suggestions(action, orient, camera, composition)
        print(f"\nSuggestions ({len(suggestions.suggestions)}):")
        for s in suggestions.suggestions[:5]:
            print(f"  [{s.priority.value}] {s.title}: {s.description}")
        print(f"\nNext actions: {', '.join(suggestions.next_actions)}")
        print(f"Creative: {suggestions.creative_direction}")
        print("\n--- Reverse Engineering ---")
        result = engine.analyze(image, pose, pose.bbox)
        print(result.report())
        print("\nCamera Actions:")
        for a in result._camera_actions:
            print(f"  {a.action}: {a.expected_effect}")
            for r in a.reason[:2]:
                print(f"    - {r}")
    finally:
        det.close()


def main():
    parser = argparse.ArgumentParser(description="Portrait Image Breakdown")
    parser.add_argument("--image", "-i", help="Image path to analyze")
    parser.add_argument("--cli", action="store_true", help="CLI mode")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if args.image and args.cli:
        run_cli(args.image, args.verbose)
    elif args.image:
        run_gui(args.image)
    else:
        run_gui()


if __name__ == "__main__":
    main()
