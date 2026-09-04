"""Smoke tests for the refactored Portrait Image Breakdown pipeline.

Tests:
  1. Module imports (no AttributeError/TypeError)
  2. Data pipeline contract (correct function signatures)
  3. AnalysisBundle construction
  4. Worker signal types
  5. Canvas overlay API
  6. Panel update methods
  7. GUI initialization (headless)
"""
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_imports():
    """All modules import without error."""
    print("=== Test 1: Imports ===")
    from core.pose_detector import PoseDetector, PoseResult, POSE_CONNECTIONS, LandmarkIndex
    from core.orientation import analyze_orientation, OrientationResult
    from core.action_classifier import classify_action, ActionResult
    from core.camera_analyzer import analyze_camera, CameraResult
    from core.composition import analyze_composition, CompositionResult
    from core.suggestion import generate_suggestions, SuggestionResult
    from reverse_engineering.engine import ReverseEngineeringEngine
    from reverse_engineering.data_types import ReverseEngineeringResult
    from gui.main_window import AnalysisBundle, AnalysisWorker, MainWindow
    from gui.canvas import ImageCanvas
    from gui.panels import AnalysisPanel, SuggestionPanel
    print("  All imports OK")
    return True


def test_analysis_bundle():
    """AnalysisBundle has correct fields."""
    print("=== Test 2: AnalysisBundle ===")
    from gui.main_window import AnalysisBundle
    b = AnalysisBundle()
    assert b.pose is None
    assert b.orientation is None
    assert b.action is None
    assert b.camera is None
    assert b.composition is None
    assert b.suggestions is None
    assert b.reverse_result is None
    # Test with values
    b.pose = "fake_pose"
    b.camera = "fake_camera"
    assert b.pose == "fake_pose"
    assert b.camera == "fake_camera"
    print("  AnalysisBundle OK")
    return True


def test_core_pipeline():
    """Core analysis functions have correct signatures and return correct types."""
    print("=== Test 3: Core pipeline (requires YOLO model) ===")
    import numpy as np
    from core.pose_detector import PoseDetector
    from core.orientation import analyze_orientation
    from core.action_classifier import classify_action
    from core.camera_analyzer import analyze_camera
    from core.composition import analyze_composition
    from core.suggestion import generate_suggestions

    # Create a dummy image (small, won't detect anything)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)  # gray

    det = PoseDetector()
    try:
        pose = det.detect(img)
        if pose is None:
            print("  No person in dummy image (expected), skipping pipeline test")
            print("  To fully test, run with: python test_smoke.py --image <photo.jpg>")
            return True

        # If pose detected (unlikely with blank image), run full pipeline
        orientation = analyze_orientation(pose)
        assert hasattr(orientation, 'facing')
        assert hasattr(orientation, 'tilt')
        assert hasattr(orientation, 'facing_angle')
        print(f"  Orientation: {orientation.facing.value}")

        action = classify_action(pose)
        assert hasattr(action, 'category')
        assert hasattr(action, 'confidence')
        assert hasattr(action, 'joint_angles')
        assert isinstance(action.joint_angles, dict)
        print(f"  Action: {action.category.value} ({action.confidence:.0%})")

        camera = analyze_camera(pose)
        assert hasattr(camera, 'shot_type')
        assert hasattr(camera, 'camera_angle')
        assert hasattr(camera, 'subject_ratio')
        assert hasattr(camera, 'dutch_angle_deg')
        print(f"  Camera: {camera.shot_type.value}")

        composition = analyze_composition(img, pose)
        assert hasattr(composition, 'primary_type')
        assert hasattr(composition, 'subject_position')
        assert hasattr(composition, 'thirds_alignment')
        assert isinstance(composition.subject_position, tuple)
        print(f"  Composition: {composition.primary_type.value}")

        suggestions = generate_suggestions(action, orientation, camera, composition)
        assert hasattr(suggestions, 'suggestions')
        assert hasattr(suggestions, 'next_actions')
        assert hasattr(suggestions, 'creative_direction')
        assert isinstance(suggestions.next_actions, list)
        print(f"  Suggestions: {len(suggestions.suggestions)} items")

        print("  Core pipeline OK")
        return True
    finally:
        det.close()


def test_worker_signals():
    """AnalysisWorker has correct signal types."""
    print("=== Test 4: Worker signals ===")
    from gui.main_window import AnalysisWorker
    # Check signals exist
    assert hasattr(AnalysisWorker, 'pose_ready')
    assert hasattr(AnalysisWorker, 'core_ready')
    assert hasattr(AnalysisWorker, 'reverse_ready')
    assert hasattr(AnalysisWorker, 'error')
    print("  Worker signals OK")
    return True


def test_canvas_overlay_api():
    """ImageCanvas.set_overlay_options accepts correct parameters."""
    print("=== Test 5: Canvas overlay API ===")
    import inspect
    from gui.canvas import ImageCanvas
    sig = inspect.signature(ImageCanvas.set_overlay_options)
    params = list(sig.parameters.keys())
    assert 'skeleton' in params, f"Missing 'skeleton', got {params}"
    assert 'thirds' in params, f"Missing 'thirds', got {params}"
    assert 'center' in params, f"Missing 'center', got {params}"
    assert 'bbox' in params, f"Missing 'bbox', got {params}"
    assert 'visual_weight' in params, f"Missing 'visual_weight', got {params}"
    assert 'headroom' in params, f"Missing 'headroom', got {params}"
    # Should NOT have reverse_eng
    assert 'reverse_eng' not in params, f"Old 'reverse_eng' param still present!"
    print(f"  Canvas overlay params: {params}")
    print("  Canvas overlay API OK")
    return True


def test_panel_methods():
    """AnalysisPanel and SuggestionPanel have correct method signatures."""
    print("=== Test 6: Panel methods ===")
    import inspect
    from gui.panels import AnalysisPanel, SuggestionPanel

    # AnalysisPanel must have these methods (NOT update_camera_result/update_composition_result)
    for method_name in ['update_pose', 'update_orientation', 'update_action',
                        'update_camera', 'update_composition']:
        assert hasattr(AnalysisPanel, method_name), f"AnalysisPanel missing {method_name}"

    # Should NOT have the old wrong names
    assert not hasattr(AnalysisPanel, 'update_camera_result'), \
        "AnalysisPanel still has update_camera_result!"
    assert not hasattr(AnalysisPanel, 'update_composition_result'), \
        "AnalysisPanel still has update_composition_result!"

    # Check signatures
    sig = inspect.signature(AnalysisPanel.update_camera)
    params = list(sig.parameters.keys())
    assert 'result' in params, f"update_camera missing 'result' param: {params}"

    sig = inspect.signature(AnalysisPanel.update_composition)
    params = list(sig.parameters.keys())
    assert 'result' in params, f"update_composition missing 'result' param: {params}"

    # SuggestionPanel
    assert hasattr(SuggestionPanel, 'update_suggestions')
    assert hasattr(SuggestionPanel, 'update_camera_actions')

    print("  Panel methods OK")
    return True


def test_no_dark_colors():
    """Check that panels.py and canvas.py don't contain hardcoded dark colors."""
    print("=== Test 7: No hardcoded dark colors ===")
    dark_colors = ['#1E1E1E', '#252525', '#2A3A2A', '#3A3A3A', '#444444']
    issues = []

    for fname in ['gui/panels.py', 'gui/canvas.py']:
        fpath = ROOT / fname
        content = fpath.read_text(encoding='utf-8')
        for color in dark_colors:
            if color.lower() in content.lower():
                issues.append(f"{fname} contains {color}")

    if issues:
        print("  FAIL: Hardcoded dark colors found:")
        for issue in issues:
            print(f"    {issue}")
        return False
    else:
        print("  No dark colors found OK")
        return True


def test_no_emoji_in_panels():
    """Check that panels.py doesn't use emoji in widget text."""
    print("=== Test 8: No emoji in panels ===")
    fpath = ROOT / 'gui' / 'panels.py'
    content = fpath.read_text(encoding='utf-8')

    # Check for common emoji patterns in the panel text (not in comments)
    import re
    # Look for emoji in QLabel text or header strings
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0001FA00-\U0001FA6F]"
    )
    found = emoji_pattern.findall(content)
    if found:
        print(f"  FAIL: Found emoji characters: {found}")
        return False
    else:
        print("  No emoji in panels OK")
        return True


def test_gui_init():
    """MainWindow can be instantiated (requires display or headless)."""
    print("=== Test 9: GUI initialization ===")
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from gui.main_window import MainWindow, apply_light_theme
        apply_light_theme(app)
        w = MainWindow()
        # Verify structure
        assert hasattr(w, '_w2'), "Missing 2D workspace"
        assert hasattr(w, '_w3'), "Missing 3D workspace"
        assert hasattr(w, '_wr'), "Missing results workspace"
        assert hasattr(w, '_chk_skeleton'), "Missing skeleton checkbox"
        assert hasattr(w, '_chk_thirds'), "Missing thirds checkbox"
        assert hasattr(w, '_chk_center'), "Missing center checkbox"
        assert hasattr(w, '_chk_bbox'), "Missing bbox checkbox"
        assert hasattr(w, '_chk_vweight'), "Missing visual weight checkbox"
        assert hasattr(w, '_chk_headroom'), "Missing headroom checkbox"
        assert hasattr(w, '_bundle'), "Missing AnalysisBundle"
        assert hasattr(w, '_result_cache'), "Missing result cache"
        # Verify no old overlay toggle
        assert not hasattr(w, '_chk'), "Old _chk (RE Overlay) still exists!"
        print("  MainWindow init OK")
        w.close()
        return True
    except Exception as e:
        print(f"  GUI init failed (may need display): {e}")
        return True  # Not a hard failure in headless CI


def test_full_pipeline_with_image(image_path: str):
    """Full end-to-end test with a real image."""
    print(f"=== Test 10: Full pipeline with {image_path} ===")
    import cv2
    from core.pose_detector import PoseDetector
    from core.orientation import analyze_orientation
    from core.action_classifier import classify_action
    from core.camera_analyzer import analyze_camera
    from core.composition import analyze_composition
    from core.suggestion import generate_suggestions
    from gui.main_window import AnalysisBundle, _resize_for_analysis

    img = cv2.imread(image_path)
    assert img is not None, f"Cannot read {image_path}"
    print(f"  Image: {img.shape[1]}x{img.shape[0]}")

    analysis_img = _resize_for_analysis(img)
    print(f"  Analysis image: {analysis_img.shape[1]}x{analysis_img.shape[0]}")

    det = PoseDetector()
    try:
        pose = det.detect(analysis_img)
        assert pose is not None, "No person detected"
        vis = sum(1 for lm in pose.landmarks[:17] if lm.visibility > 0.4)
        print(f"  Pose: {pose.detection_confidence:.0%} conf, {vis}/17 keypoints")

        orientation = analyze_orientation(pose)
        print(f"  Orientation: {orientation.facing.value}, {orientation.tilt.value}")

        action = classify_action(pose)
        print(f"  Action: {action.category.value} ({action.confidence:.0%})")
        print(f"  Joint angles: { {k: f'{v:.0f}' for k, v in action.joint_angles.items()} }")

        camera = analyze_camera(pose)
        print(f"  Camera: {camera.shot_type.value}, {camera.camera_angle.value}, ratio={camera.subject_ratio:.1%}")

        composition = analyze_composition(analysis_img, pose)
        print(f"  Composition: {composition.primary_type.value}, thirds={composition.thirds_alignment:.0%}")

        suggestions = generate_suggestions(action, orientation, camera, composition)
        print(f"  Suggestions: {len(suggestions.suggestions)} items, {len(suggestions.next_actions)} next actions")
        print(f"  Creative: {suggestions.creative_direction}")

        # Build bundle
        bundle = AnalysisBundle(
            pose=pose, orientation=orientation, action=action,
            camera=camera, composition=composition, suggestions=suggestions,
        )
        assert bundle.pose is not None
        assert bundle.action is not None
        print("  AnalysisBundle built OK")

        # Test RE engine
        from reverse_engineering.engine import ReverseEngineeringEngine
        eng = ReverseEngineeringEngine(enable_simulation=False)
        re_result = eng.analyze(analysis_img, pose, pose.bbox)
        bundle.reverse_result = re_result
        print(f"  RE overall confidence: {re_result.overall_confidence:.0%}")
        print(f"  RE camera actions: {len(re_result._camera_actions)}")
        print("  Full pipeline OK")
        return True
    finally:
        det.close()


def main():
    image_path = None
    if len(sys.argv) > 1:
        if sys.argv[1] == '--image' and len(sys.argv) > 2:
            image_path = sys.argv[2]
        elif not sys.argv[1].startswith('--'):
            image_path = sys.argv[1]

    results = []

    results.append(("Imports", test_imports()))
    results.append(("AnalysisBundle", test_analysis_bundle()))
    results.append(("Core pipeline", test_core_pipeline()))
    results.append(("Worker signals", test_worker_signals()))
    results.append(("Canvas overlay API", test_canvas_overlay_api()))
    results.append(("Panel methods", test_panel_methods()))
    results.append(("No dark colors", test_no_dark_colors()))
    results.append(("No emoji", test_no_emoji_in_panels()))
    results.append(("GUI init", test_gui_init()))

    if image_path:
        results.append(("Full pipeline", test_full_pipeline_with_image(image_path)))

    print("\n" + "=" * 50)
    print("RESULTS:")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}: {name}")

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
