"""Portrait Image Breakdown - Photography Analysis & Reverse Engineering

Two-phase architecture:
  Phase 1 (Fast Analysis): Pose -> Orientation -> Action -> Camera -> Composition -> Suggestions
  Phase 2 (Full RE):       Image + Pose + Composition -> ReverseEngineeringEngine (background)
"""
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_gui(image_path=None):
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow, apply_light_theme
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_light_theme(app)
    window = MainWindow()
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
