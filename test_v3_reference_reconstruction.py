import numpy as np

from core.pose_detector import PoseLandmark, PoseResult
from reverse_engineering.reference_reconstruction import (
    build_reference_composition,
    choose_semantic_anchor,
    compare_pose_to_reference,
    composition_delta,
)


def _pose(offset_x=0.0, offset_y=0.0, scale=1.0):
    xy = np.array([
        [50, 20], [45, 18], [55, 18], [40, 20], [60, 20],
        [35, 45], [65, 45], [28, 65], [72, 65], [25, 82], [75, 82],
        [40, 80], [60, 80], [42, 112], [58, 112], [42, 145], [58, 145],
    ], dtype=float)
    xy[:, 0] = 50 + (xy[:, 0] - 50) * scale + offset_x
    xy[:, 1] = 20 + (xy[:, 1] - 20) * scale + offset_y
    landmarks = [PoseLandmark(i, x, y, 0, 0.95, x/200, y/180, 0) for i, (x,y) in enumerate(xy)]
    return PoseResult(landmarks, 200, 180, 0.95, (int(xy[:,0].min()), int(xy[:,1].min()), int(xy[:,0].max()), int(xy[:,1].max())))


def test_reference_prefers_hip_anchor():
    anchor = choose_semantic_anchor(_pose())
    assert anchor is not None
    assert anchor.name == "hip_center"


def test_reference_composition_has_bbox_and_anchor():
    result = build_reference_composition(_pose(), 200, 180)
    names = {a.name for a in result.anchors}
    assert {"bbox_tl", "bbox_center", "bbox_br", "hip_center"}.issubset(names)
    assert result.subject_scale > 0


def test_pose_delta_points_toward_reference():
    reference = _pose()
    current = _pose(offset_x=20, offset_y=5)
    deltas = compare_pose_to_reference(reference, current, 200, 180)
    nose = next(d for d in deltas if d.landmark == "nose")
    assert nose.dx < 0
    assert nose.dy < 0


def test_composition_delta_reports_center_and_scale():
    ref = build_reference_composition(_pose(), 200, 180)
    cur = build_reference_composition(_pose(offset_x=10, scale=0.8), 200, 180)
    delta = composition_delta(ref, cur)
    assert delta["center_dx"] < 0
    assert 0.0 < delta["scale_ratio"] < 1.0
