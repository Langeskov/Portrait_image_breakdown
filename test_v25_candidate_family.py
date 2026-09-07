import numpy as np

from reverse_engineering.geometry import CameraIntrinsics, PoseCandidate, _camera_pose_from_params
from reverse_engineering.scene_constraints import (
    CameraFeasibilityEvidence,
    candidate_feasibility_score,
)
from reverse_engineering.simulation import _dedupe_candidates


def _candidate(focal, distance, height, score):
    _, extr = _camera_pose_from_params(distance, height, 0.0, -8.0, 0.0)
    return PoseCandidate(
        CameraIntrinsics.from_focal_mm(focal, 1600, 1200),
        extr,
        distance,
        height,
        focal,
        score,
        {"mean_reprojection_px": 10.0},
    )


def test_candidate_dedupe_keeps_distinct_focal_distance_families():
    candidates = [
        _candidate(50.0, 3.0, 1.3, 0.90),
        _candidate(70.0, 4.2, 1.4, 0.82),
        _candidate(85.0, 5.0, 1.5, 0.76),
    ]
    ranked = _dedupe_candidates(candidates, 3)
    assert len(ranked) == 3
    assert [c.focal_equiv_35mm for c in ranked] == [50.0, 70.0, 85.0]


def test_feasibility_score_is_one_inside_broad_range():
    evidence = CameraFeasibilityEvidence(
        distance_range_m=(2.0, 5.0),
        height_range_m=(1.0, 1.8),
        confidence=0.7,
        basis=("pose body extent",),
    )
    candidate = _candidate(70.0, 3.5, 1.4, 0.8)
    score = candidate_feasibility_score(candidate, evidence)
    assert score is not None
    assert np.isclose(score, 1.0)
