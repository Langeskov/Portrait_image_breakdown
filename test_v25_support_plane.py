import numpy as np

from reverse_engineering.geometry import CameraIntrinsics, PoseCandidate, _camera_pose_from_params
from reverse_engineering.support_plane import SupportPlaneEvidence, candidate_support_plane_score, estimate_support_plane, expected_support_pitch_deg


def _standing_pose():
    kp = np.zeros((17, 3), dtype=float)
    kp[:, 2] = 0.95
    kp[11, :2] = (400, 420); kp[12, :2] = (500, 420)
    kp[13, :2] = (410, 600); kp[14, :2] = (490, 600)
    kp[15, :2] = (410, 760); kp[16, :2] = (490, 760)
    for i in range(17):
        if i not in (11, 12, 13, 14, 15, 16):
            kp[i, :2] = (450 + (i % 2) * 5, 300 + i * 8)
    return kp


def test_support_plane_detected_from_bilateral_lower_limb_geometry():
    evidence = estimate_support_plane(_standing_pose(), 900, 800)
    assert evidence.usable
    assert evidence.visible_ankles == 2
    assert evidence.contact_world_y == 0.0


def test_support_pitch_matches_camera_position_to_contact_plane():
    evidence = SupportPlaneEvidence(True, 0.7, 0.0, 0.95, 2, ("test",))
    expected = expected_support_pitch_deg(4.0, 1.5, evidence)
    assert np.isclose(expected, np.degrees(np.arctan2(-1.5, 4.0)))


def test_candidate_support_score_prefers_coherent_pitch():
    evidence = SupportPlaneEvidence(True, 0.7, 0.0, 0.95, 2, ("test",))
    _, coherent = _camera_pose_from_params(4.0, 1.5, 0.0, expected_support_pitch_deg(4.0, 1.5, evidence), 0.0)
    _, off = _camera_pose_from_params(4.0, 1.5, 0.0, 0.0, 0.0)
    intr = CameraIntrinsics.from_focal_mm(50.0, 900, 800)
    coherent_c = PoseCandidate(intr, coherent, 4.0, 1.5, 50.0, 0.8, {})
    off_c = PoseCandidate(intr, off, 4.0, 1.5, 50.0, 0.8, {})
    assert candidate_support_plane_score(coherent_c, evidence) > candidate_support_plane_score(off_c, evidence)
