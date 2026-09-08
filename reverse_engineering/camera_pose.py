"""Camera pose estimation and robust candidate-family generation."""
from __future__ import annotations

import math

import numpy as np

from reverse_engineering.data_types import EstimatedValue, CameraPoseResult
from reverse_engineering.geometry import (
    CameraIntrinsics,
    PoseCandidate,
    PoseSolver,
    REF_PERSON_HEIGHT,
    _camera_pose_from_params,
)
from core.pose_detector import PoseResult


# A single portrait image cannot uniquely determine focal length and distance.
# Candidate generation therefore exposes a bounded, physically plausible family
# rather than whatever extreme local optimum a nonlinear solver happens to find.
DEFAULT_CANDIDATE_FOCALS_MM = (28.0, 35.0, 50.0, 70.0, 85.0, 105.0, 135.0)
MIN_REASONABLE_HEIGHT_M = 0.70
MAX_REASONABLE_HEIGHT_M = 2.10
MIN_REASONABLE_DISTANCE_M = 1.20
MAX_REASONABLE_DISTANCE_M = 15.0
MIN_REASONABLE_FOCAL_MM = 24.0
MAX_REASONABLE_FOCAL_MM = 150.0


def estimate_image_roll(image) -> tuple[float, float, int]:
    """Compatibility wrapper around the shared scene-line roll estimator."""
    if image is None:
        return 0.0, 0.0, 0
    from reverse_engineering.rotation_solver import _estimate_line_roll
    from reverse_engineering.scene_geometry import analyze_scene_geometry

    evidence = analyze_scene_geometry(image)
    roll, confidence, count = _estimate_line_roll(evidence)
    return (float(roll) if roll is not None else 0.0, float(confidence), int(count))


def _bbox_from_pose(pose: PoseResult) -> tuple[float, float, float, float] | None:
    if pose is None:
        return None
    bbox = getattr(pose, "bbox", None)
    if bbox is not None:
        x0, y0, x1, y1 = map(float, bbox)
        if x1 > x0 and y1 > y0:
            return x0, y0, x1, y1

    points = [
        (float(lm.x), float(lm.y))
        for lm in getattr(pose, "landmarks", [])[:17]
        if float(getattr(lm, "visibility", 0.0)) > 0.35
        and math.isfinite(float(lm.x))
        and math.isfinite(float(lm.y))
    ]
    if len(points) < 5:
        return None
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def _candidate_is_physically_reasonable(candidate: PoseCandidate) -> bool:
    """Reject optimizer boundary artefacts before exposing them to the UI."""
    focal = float(candidate.focal_equiv_35mm)
    distance = float(candidate.distance)
    height = float(candidate.height)
    score = float(candidate.score)
    if not all(math.isfinite(v) for v in (focal, distance, height, score)):
        return False
    if not MIN_REASONABLE_FOCAL_MM <= focal <= MAX_REASONABLE_FOCAL_MM:
        return False
    if not MIN_REASONABLE_DISTANCE_M <= distance <= MAX_REASONABLE_DISTANCE_M:
        return False
    if not MIN_REASONABLE_HEIGHT_M <= height <= MAX_REASONABLE_HEIGHT_M:
        return False
    mean_error = candidate.losses.get("mean_reprojection_px")
    if mean_error is not None and float(mean_error) > 35.0:
        return False
    return True


def _analytic_candidate_family(
    pose: PoseResult,
    subject_bbox=None,
    num_candidates: int = 6,
    focal_seeds=DEFAULT_CANDIDATE_FOCALS_MM,
) -> list[PoseCandidate]:
    """Build a stable focal/distance family from the observed body extent.

    This intentionally represents ambiguity instead of false precision. Each
    focal length is paired with the corresponding distance for the same 1.70 m
    person-height prior. Camera height is only a weak framing prior.
    """
    if pose is None:
        return []
    width = int(getattr(pose, "image_width", 0) or 0)
    height_px = int(getattr(pose, "image_height", 0) or 0)
    if width <= 0 or height_px <= 0:
        return []

    bbox = _bbox_from_pose(pose)
    if bbox is None:
        return []
    _, y0, _, y1 = bbox
    body_px = max(float(y1 - y0), 20.0)
    center_y = ((y0 + y1) * 0.5) / max(height_px, 1)

    candidates: list[PoseCandidate] = []
    for focal in focal_seeds:
        focal = float(focal)
        if not MIN_REASONABLE_FOCAL_MM <= focal <= MAX_REASONABLE_FOCAL_MM:
            continue

        intr = CameraIntrinsics.from_focal_mm(focal, width, height_px)
        distance = REF_PERSON_HEIGHT * intr.fy / body_px
        distance = float(np.clip(distance, MIN_REASONABLE_DISTANCE_M, MAX_REASONABLE_DISTANCE_M))

        # Keep the height prior around chest/eye level. Do not let image framing
        # create a near-floor camera merely to improve a numerical fit.
        camera_height = float(np.clip(
            1.25 + (0.5 - center_y) * 0.55,
            MIN_REASONABLE_HEIGHT_M,
            MAX_REASONABLE_HEIGHT_M,
        ))
        _, extrinsics = _camera_pose_from_params(distance, camera_height, 0.0, 0.0, 0.0)

        score = 0.82 - 0.0015 * abs(focal - 70.0)
        losses = {
            "candidate_source": "analytic_pose_family",
            "body_extent_px": round(body_px, 3),
            "focal_distance_ambiguity": True,
            "frame_center_y": round(center_y, 4),
        }
        candidates.append(PoseCandidate(
            intrinsics=intr,
            extrinsics=extrinsics,
            distance=distance,
            height=camera_height,
            focal_equiv_35mm=focal,
            score=float(np.clip(score, 0.68, 0.82)),
            losses=losses,
        ))

    return candidates[: max(1, int(num_candidates))]


def _dedupe_candidates(candidates, num_candidates: int) -> list[PoseCandidate]:
    unique: list[PoseCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda c: (
            -float(c.score),
            float(c.losses.get("mean_reprojection_px", 1e9)),
            abs(float(c.focal_equiv_35mm) - 70.0),
        ),
    ):
        if not _candidate_is_physically_reasonable(candidate):
            continue
        duplicate = any(
            abs(candidate.focal_equiv_35mm - other.focal_equiv_35mm) < 4.0
            and abs(candidate.distance - other.distance) < 0.25
            and abs(candidate.height - other.height) < 0.12
            and abs(candidate.extrinsics.yaw - other.extrinsics.yaw) < 3.0
            and abs(candidate.extrinsics.pitch - other.extrinsics.pitch) < 3.0
            and abs(candidate.extrinsics.roll - other.extrinsics.roll) < 2.0
            for other in unique
        )
        if duplicate:
            continue
        unique.append(candidate)
        if len(unique) >= max(1, int(num_candidates)):
            break
    return unique


def estimate_camera_pose_candidates(
    pose: PoseResult,
    subject_bbox=None,
    num_candidates: int = 8,
):
    """Return a robust candidate family instead of an optimizer singleton.

    Numerical pose fitting remains useful evidence, but a valid analytic family
    is always available as the safety net. Physically pathological numerical
    boundary solutions are never allowed to replace the family.
    """
    if pose is None:
        return []

    requested = max(1, int(num_candidates))
    analytic = _analytic_candidate_family(
        pose,
        subject_bbox=subject_bbox,
        num_candidates=requested,
    )

    numerical: list[PoseCandidate] = []
    try:
        h, w = pose.image_height, pose.image_width
        kp = [[lm.x, lm.y, lm.visibility] for lm in pose.landmarks[:17]]
        numerical = PoseSolver.fit_camera_to_pose(
            kp,
            w,
            h,
            subject_bbox=subject_bbox,
            focal_seeds=DEFAULT_CANDIDATE_FOCALS_MM,
            num_candidates=max(8, requested),
        )
        numerical = [c for c in numerical if _candidate_is_physically_reasonable(c)]
        for candidate in numerical:
            candidate.losses["candidate_source"] = "numerical_pose_fit"
    except (ValueError, RuntimeError, FloatingPointError, TypeError):
        numerical = []

    # Numeric candidates can outrank analytic candidates, but only after the
    # physical gates above. The analytic family prevents zero-candidate output.
    return _dedupe_candidates(numerical + analytic, requested)


def estimate_camera_pose(
    pose: PoseResult,
    perspective_vanishing_points=None,
    representative_focal_mm: float = 50.0,
    image=None,
    subject_bbox=None,
) -> CameraPoseResult:
    """Return the best member of the same candidate family exposed to the UI."""
    del perspective_vanishing_points, representative_focal_mm, image
    candidates = estimate_camera_pose_candidates(pose, subject_bbox=subject_bbox, num_candidates=8)
    if candidates:
        best = max(candidates, key=lambda c: c.score)
        return CameraPoseResult(
            camera_height=EstimatedValue(
                round(best.height, 2), unit="m", range_min=0.25, range_max=2.2,
                confidence=min(0.8, best.score), basis=["robust candidate family", "pose-driven metric proxy"],
            ),
            camera_distance=EstimatedValue(
                round(best.distance, 2), unit="m",
                range_min=round(max(0.5, best.distance * 0.75), 2),
                range_max=round(best.distance * 1.35, 2),
                confidence=min(0.75, best.score), basis=["focal/distance candidate family"],
            ),
            camera_pitch=EstimatedValue(
                round(best.extrinsics.pitch, 1), unit="deg", range_min=-30, range_max=30,
                confidence=min(0.75, best.score), basis=["candidate family"],
            ),
            camera_yaw=EstimatedValue(
                round(best.extrinsics.yaw, 1), unit="deg", range_min=-30, range_max=30,
                confidence=min(0.7, best.score), basis=["candidate family"],
            ),
            camera_roll=EstimatedValue(
                round(best.extrinsics.roll, 1), unit="deg", range_min=-12, range_max=12,
                confidence=min(0.65, best.score), basis=["candidate family; roll weakly constrained without scene geometry"],
            ),
        )

    return CameraPoseResult(
        camera_height=EstimatedValue(1.25, unit="m", range_min=0.7, range_max=2.1, confidence=0.1, basis=["fallback"]),
        camera_distance=EstimatedValue(4.0, unit="m", range_min=2.0, range_max=8.0, confidence=0.1, basis=["fallback"]),
        camera_pitch=EstimatedValue(0.0, unit="deg", range_min=-30, range_max=30, confidence=0.1, basis=["fallback"]),
        camera_yaw=EstimatedValue(0.0, unit="deg", range_min=-30, range_max=30, confidence=0.1, basis=["fallback"]),
        camera_roll=EstimatedValue(0.0, unit="deg", range_min=-12, range_max=12, confidence=0.1, basis=["fallback"]),
    )
