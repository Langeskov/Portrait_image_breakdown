"""Camera pose estimation from the shared bounded 2D-pose fitting model."""
from __future__ import annotations

from reverse_engineering.data_types import EstimatedValue, CameraPoseResult
from reverse_engineering.geometry import PoseSolver
from core.pose_detector import PoseResult


def estimate_image_roll(image) -> tuple[float, float, int]:
    """Compatibility wrapper around the shared scene-line roll estimator.

    The v2.5 runtime authority is ``rotation_solver._estimate_line_roll``.
    Keeping this wrapper preserves the historical API without maintaining a
    second roll implementation with different thresholds and semantics.
    """
    if image is None:
        return 0.0, 0.0, 0
    from reverse_engineering.rotation_solver import _estimate_line_roll
    from reverse_engineering.scene_geometry import analyze_scene_geometry
    evidence = analyze_scene_geometry(image)
    roll, confidence, count = _estimate_line_roll(evidence)
    return (float(roll) if roll is not None else 0.0, float(confidence), int(count))


def estimate_camera_pose_candidates(
    pose: PoseResult,
    subject_bbox=None,
    num_candidates: int = 8,
):
    """Return ranked pose-only camera candidates independently of simulation.

    Candidate generation is a first-class output of the reverse-engineering
    pipeline. The optional simulation/refinement stage may add scene/depth
    constraints, but it must not be required for candidates to exist.
    """
    if pose is None:
        return []
    h, w = pose.image_height, pose.image_width
    kp = [[lm.x, lm.y, lm.visibility] for lm in pose.landmarks[:17]]
    return PoseSolver.fit_camera_to_pose(
        kp,
        w,
        h,
        subject_bbox=subject_bbox,
        num_candidates=max(1, int(num_candidates)),
    )


def estimate_camera_pose(
    pose: PoseResult,
    perspective_vanishing_points=None,
    representative_focal_mm: float = 50.0,
    image=None,
    subject_bbox=None,
) -> CameraPoseResult:
    """Return the same shared best-fit camera solution used by the 3D workspace."""
    del perspective_vanishing_points, representative_focal_mm, image
    candidates = estimate_camera_pose_candidates(pose, subject_bbox=subject_bbox, num_candidates=8)
    if candidates:
        best = max(candidates, key=lambda c: c.score)
        return CameraPoseResult(
            camera_height=EstimatedValue(
                round(best.height, 2), unit="m", range_min=0.25, range_max=2.2,
                confidence=min(0.8, best.score),
                basis=["shared 2D pose reprojection fit", "pose-driven 3D proxy"],
            ),
            camera_distance=EstimatedValue(
                round(best.distance, 2), unit="m",
                range_min=round(max(0.5, best.distance * 0.75), 2),
                range_max=round(best.distance * 1.35, 2),
                confidence=min(0.75, best.score),
                basis=["shared reprojection fit", "focal/distance ambiguity retained"],
            ),
            camera_pitch=EstimatedValue(
                round(best.extrinsics.pitch, 1), unit="deg", range_min=-30, range_max=30,
                confidence=min(0.75, best.score), basis=["shared reprojection fit"],
            ),
            camera_yaw=EstimatedValue(
                round(best.extrinsics.yaw, 1), unit="deg", range_min=-30, range_max=30,
                confidence=min(0.7, best.score), basis=["shared reprojection fit"],
            ),
            camera_roll=EstimatedValue(
                round(best.extrinsics.roll, 1), unit="deg", range_min=-12, range_max=12,
                confidence=min(0.65, best.score),
                basis=["shared reprojection fit", "roll is weakly constrained by 2D pose"],
            ),
        )

    return CameraPoseResult(
        camera_height=EstimatedValue(1.0, unit="m", range_min=0.5, range_max=2.2, confidence=0.1, basis=["fallback"]),
        camera_distance=EstimatedValue(4.0, unit="m", range_min=2.0, range_max=8.0, confidence=0.1, basis=["fallback"]),
        camera_pitch=EstimatedValue(0.0, unit="deg", range_min=-30, range_max=30, confidence=0.1, basis=["fallback"]),
        camera_yaw=EstimatedValue(0.0, unit="deg", range_min=-30, range_max=30, confidence=0.1, basis=["fallback"]),
        camera_roll=EstimatedValue(0.0, unit="deg", range_min=-12, range_max=12, confidence=0.1, basis=["fallback"]),
    )
