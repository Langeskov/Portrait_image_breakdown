import cv2
import numpy as np

from reverse_engineering.geometry import CameraIntrinsics, PoseSolver, canonical_person_points
from reverse_engineering.scene import SceneModel
from reverse_engineering.data_types import (
    CameraPoseResult,
    CompositionResult,
    DepthOfFieldResult,
    EstimatedValue,
    FocalLengthResult,
    MotionBlurResult,
    PerspectiveResult,
    ReverseEngineeringResult,
    ShootingTechniqueResult,
)


def _synthetic_pose(width=1800, height=1200, focal=70.0, distance=6.0, camera_height=1.05):
    points = canonical_person_points()
    intr = CameraIntrinsics.from_focal_mm(focal, width, height)
    position = np.array([0.0, camera_height, -distance])
    forward = -position
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    rotation = np.vstack([right, -up, forward])
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = -rotation @ position
    projected, _ = cv2.projectPoints(points, rvec, tvec, intr.to_matrix(), None)
    image_points = projected.reshape(-1, 2)
    kp = np.c_[image_points, np.ones(len(image_points))]
    bbox = tuple(np.round([
        image_points[:, 0].min(), image_points[:, 1].min(),
        image_points[:, 0].max(), image_points[:, 1].max(),
    ]).astype(int))
    return kp, bbox


def test_camera_fit_returns_ranked_family_and_reasonable_distance():
    kp, bbox = _synthetic_pose()
    candidates = PoseSolver.fit_camera_to_pose(
        kp,
        1800,
        1200,
        subject_bbox=bbox,
        focal_seeds=(35, 50, 70, 85, 105),
        num_candidates=5,
    )
    assert len(candidates) >= 3
    assert all(0.5 <= c.distance <= 20.0 for c in candidates)
    best = candidates[0]
    assert abs(best.distance - 6.0) < 1.5
    assert best.losses["mean_reprojection_px"] < 20.0


def test_scene_model_uses_pose_driven_proxy_for_projection():
    kp, bbox = _synthetic_pose()
    pose_landmarks = [type("LM", (), {"x": float(p[0]), "y": float(p[1]), "visibility": 1.0})() for p in kp]
    cp = CameraPoseResult(
        EstimatedValue(1.05, "m"), EstimatedValue(6.0, "m"),
        EstimatedValue(0.0, "deg"), EstimatedValue(0.0, "deg"), EstimatedValue(0.0, "deg"),
    )
    fl = FocalLengthResult(EstimatedValue("normal", confidence=.8), EstimatedValue(70.0, "mm", confidence=.8))
    result = ReverseEngineeringResult(
        image_size=(1800, 1200), subject_bbox=bbox, subject_keypoints=pose_landmarks,
        subject_scale=.3, edge_lines=[], blur_regions={},
        perspective=PerspectiveResult(EstimatedValue(.2), EstimatedValue("normal"), [], EstimatedValue(0), EstimatedValue(0)),
        camera_pose=cp, focal_length=fl,
        depth_of_field=DepthOfFieldResult(EstimatedValue("unknown"), True, 0, 0, EstimatedValue("unknown")),
        motion_blur=MotionBlurResult(EstimatedValue("none"), None, 0, EstimatedValue("unknown")),
        composition=CompositionResult([], (.5, .5), .3, .1, "balanced", .5),
        shooting_techniques=ShootingTechniqueResult([]), overall_confidence=.5, uncertainties=[],
    )
    scene = SceneModel.from_reverse_result(result)
    fitted = scene.subject.proxy_points()
    assert fitted.shape == (17, 3)
    assert np.isfinite(fitted).sum() > 30
