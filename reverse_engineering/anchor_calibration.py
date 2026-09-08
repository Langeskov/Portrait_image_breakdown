"""Camera hypotheses from explicit scene-anchor/image-point correspondences.

This module is intentionally conservative. It never promotes a manual
correspondence into observed EXIF/scene evidence and never mutates the active
camera. A valid result is a hypothesis that the operator may inspect before
applying it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np

from reverse_engineering.geometry import CameraIntrinsics
from reverse_engineering.scene_anchors import AnchorKind, SceneAnchor


@dataclass(frozen=True)
class CameraAnchorHypothesis:
    success: bool
    rvec: np.ndarray
    tvec: np.ndarray
    position: tuple[float, float, float]
    reprojection_rmse_px: float
    point_count: int
    message: str


def _point_correspondences(anchors: Iterable[SceneAnchor]) -> tuple[np.ndarray, np.ndarray]:
    world: list[list[float]] = []
    image: list[list[float]] = []
    for anchor in anchors:
        if not anchor.enabled:
            continue
        if anchor.kind == AnchorKind.POINT and len(anchor.image_points) >= 1:
            world.append(list(anchor.position))
            image.append(list(anchor.image_points[0]))
        elif anchor.kind == AnchorKind.PLANE and len(anchor.image_points) == 4:
            corners = anchor.corners()
            world.extend(corners.tolist())
            image.extend([list(point) for point in anchor.image_points])
    return np.asarray(world, dtype=np.float64), np.asarray(image, dtype=np.float64)


def estimate_camera_from_anchors(
    anchors: Iterable[SceneAnchor],
    width: int,
    height: int,
    focal_length_mm: float = 50.0,
    sensor_width_mm: float = 36.0,
) -> CameraAnchorHypothesis:
    """Solve a camera pose from 4+ explicit 2D/3D anchor correspondences."""
    world, image = _point_correspondences(anchors)
    count = len(world)
    if count < 4:
        return CameraAnchorHypothesis(
            False, np.zeros(3), np.zeros(3), (0.0, 0.0, 0.0),
            float("inf"), count, "Need at least 4 bound correspondences.",
        )
    if not np.isfinite(world).all() or not np.isfinite(image).all():
        return CameraAnchorHypothesis(
            False, np.zeros(3), np.zeros(3), (0.0, 0.0, 0.0),
            float("inf"), count, "Anchor correspondences contain non-finite values.",
        )

    intrinsics = CameraIntrinsics.from_focal_mm(
        float(focal_length_mm), int(width), int(height),
        float(sensor_width_mm),
        float(sensor_width_mm) * int(height) / max(int(width), 1),
    )
    matrix = intrinsics.to_matrix()
    distortion = np.zeros((4, 1), dtype=np.float64)

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            world,
            image,
            matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=4.0,
            confidence=0.995,
            iterationsCount=200,
        )
    except cv2.error as exc:
        return CameraAnchorHypothesis(
            False, np.zeros(3), np.zeros(3), (0.0, 0.0, 0.0),
            float("inf"), count, f"solvePnP failed: {exc}",
        )

    if not ok:
        return CameraAnchorHypothesis(
            False, np.zeros(3), np.zeros(3), (0.0, 0.0, 0.0),
            float("inf"), count, "solvePnP did not find a camera pose.",
        )

    projected, _ = cv2.projectPoints(world, rvec, tvec, matrix, distortion)
    residual = projected.reshape(-1, 2) - image
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    rotation = cv2.Rodrigues(rvec)[0]
    position = -rotation.T @ tvec.reshape(3)
    inlier_count = int(len(inliers)) if inliers is not None else count
    message = f"Anchor camera hypothesis: {inlier_count}/{count} inliers · RMSE {rmse:.2f}px"

    return CameraAnchorHypothesis(
        True,
        rvec.reshape(3),
        tvec.reshape(3),
        tuple(float(v) for v in position),
        rmse,
        count,
        message,
    )
