"""Camera pose estimation from the shared bounded 2D-pose fitting model."""
from __future__ import annotations

from reverse_engineering.data_types import EstimatedValue, CameraPoseResult
from reverse_engineering.geometry import PoseSolver
from core.pose_detector import PoseResult


def estimate_camera_pose(
    pose: PoseResult,
    perspective_vanishing_points=None,
    representative_focal_mm: float = 50.0,
    image=None,
    subject_bbox=None,
) -> CameraPoseResult:
    """Return the same best-fit camera solution used by the 3D workspace.

    Vanishing-point and image-roll experiments are intentionally not used as
    independent overrides here. The camera state must come from the shared
    reprojection objective so Camera reconstruction and Projection Preview stay
    numerically consistent.
    """
    del perspective_vanishing_points, representative_focal_mm, image
    h, w = pose.image_height, pose.image_width
    kp = [[lm.x, lm.y, lm.visibility] for lm in pose.landmarks[:17]]
    candidates = PoseSolver.fit_camera_to_pose(
        kp, w, h,
        subject_bbox=subject_bbox,
        num_candidates=8,
    )
    if candidates:
        best = max(candidates, key=lambda c: c.score)
        return CameraPoseResult(
            camera_height=EstimatedValue(
                round(best.height, 2), unit="m", range_min=0.25, range_max=2.2,
                confidence=min(0.8, best.score),
                basis=["shared 2D pose reprojection fit", "pose-driven 3D proxy"]),
            camera_distance=EstimatedValue(
                round(best.distance, 2), unit="m",
                range_min=round(max(0.5, best.distance * 0.75), 2),
                range_max=round(best.distance * 1.35, 2),
                confidence=min(0.75, best.score),
                basis=["shared reprojection fit", "focal/distance ambiguity retained"]),
            camera_pitch=EstimatedValue(
                round(best.extrinsics.pitch, 1), unit="deg", range_min=-30, range_max=30,
                confidence=min(0.75, best.score), basis=["shared reprojection fit"]),
            camera_yaw=EstimatedValue(
                round(best.extrinsics.yaw, 1), unit="deg", range_min=-30, range_max=30,
                confidence=min(0.7, best.score), basis=["shared reprojection fit"]),
            camera_roll=EstimatedValue(
                round(best.extrinsics.roll, 1), unit="deg", range_min=-12, range_max=12,
                confidence=min(0.65, best.score), basis=["shared reprojection fit", "roll is weakly constrained by 2D pose"]),
        )

    return CameraPoseResult(
        camera_height=EstimatedValue(1.0, unit="m", range_min=0.5, range_max=2.2, confidence=0.1, basis=["fallback"]),
        camera_distance=EstimatedValue(4.0, unit="m", range_min=2.0, range_max=8.0, confidence=0.1, basis=["fallback"]),
        camera_pitch=EstimatedValue(0.0, unit="deg", range_min=-30, range_max=30, confidence=0.1, basis=["fallback"]),
        camera_yaw=EstimatedValue(0.0, unit="deg", range_min=-30, range_max=30, confidence=0.1, basis=["fallback"]),
        camera_roll=EstimatedValue(0.0, unit="deg", range_min=-12, range_max=12, confidence=0.1, basis=["fallback"]),
    )
