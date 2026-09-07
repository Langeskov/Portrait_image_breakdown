import numpy as np

from core.pose_detector import PoseLandmark, PoseResult
from reverse_engineering.depth_provider import MonocularDepthProvider
from reverse_engineering.scene_constraints import build_depth_constraint_evidence, candidate_depth_score, DepthConstraintEvidence
from reverse_engineering.geometry import CameraIntrinsics, PoseCandidate, _camera_pose_from_params


def _synthetic_pose():
    landmarks = []
    # Pixel-space landmarks with deterministic vertical structure.
    for i in range(17):
        x = 300.0 + (i % 5) * 40.0
        y = 180.0 + i * 28.0
        landmarks.append(PoseLandmark(i, x, y, 0.0, 0.95, x, y, 0.0))
    return PoseResult(landmarks=landmarks, image_width=800, image_height=800,
                      bbox=(260, 150, 520, 700), detection_confidence=0.95)


def test_depth_constraint_evidence_is_relative_and_conservative():
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(800, dtype=np.uint8)[:, None]
    pose = _synthetic_pose()
    provider = MonocularDepthProvider()
    depth, evidence = build_depth_constraint_evidence(
        image,
        np.array([[lm.x, lm.y, lm.visibility] for lm in pose.landmarks], dtype=float),
        provider,
    )
    assert depth.shape == (800, 800)
    assert evidence.valid_count == 17
    assert evidence.to_dict()["relative_only"] is True
    assert 0.0 <= evidence.confidence <= 0.9


def test_candidate_depth_score_returns_none_for_unusable_evidence():
    intr = CameraIntrinsics.from_focal_mm(50.0, 800, 800)
    _, extr = _camera_pose_from_params(4.0, 1.4, 0.0, 0.0, 0.0)
    candidate = PoseCandidate(intr, extr, 4.0, 1.4, 50.0, 0.5, {})
    pose = _synthetic_pose()
    kp = np.array([[lm.x, lm.y, lm.visibility] for lm in pose.landmarks], dtype=float)
    weak = DepthConstraintEvidence(tuple([0.5] * 17), 0.0, 17, 800, 800)
    assert candidate_depth_score(candidate, kp, 800, 800, weak) is None
