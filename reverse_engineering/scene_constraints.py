"""Conservative scene/depth constraints for monocular camera fitting.

Depth is treated as relative evidence only. Metric ranges are expressed as
feasibility intervals, not measurements: they encode the uncertainty of a
human-scale prior and are deliberately broad.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from reverse_engineering.depth_provider import DepthProvider
from reverse_engineering.geometry import CameraModel, pose_driven_person_points


@dataclass(frozen=True)
class DepthConstraintEvidence:
    """Relative-depth evidence sampled at visible pose landmarks."""

    landmark_depths: tuple[float, ...]
    confidence: float
    valid_count: int
    image_width: int
    image_height: int

    @property
    def usable(self) -> bool:
        return self.valid_count >= 5 and self.confidence >= 0.2

    def to_dict(self) -> dict:
        return {
            "confidence": round(float(self.confidence), 3),
            "valid_count": int(self.valid_count),
            "image_size": [self.image_width, self.image_height],
            "relative_only": True,
        }


@dataclass(frozen=True)
class CameraFeasibilityEvidence:
    """Broad camera height/distance feasibility intervals.

    These ranges deliberately include uncertainty from human height, imperfect
    pose extents and monocular depth. They are constraints for ranking, not
    reconstructed metric truth.
    """

    distance_range_m: tuple[float, float]
    height_range_m: tuple[float, float]
    confidence: float
    basis: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.confidence >= 0.25

    def contains(self, distance_m: float, height_m: float) -> float:
        """Return 0..1 feasibility for a candidate inside/outside the ranges."""
        dlo, dhi = self.distance_range_m
        hlo, hhi = self.height_range_m
        def margin(value, lo, hi):
            span = max(hi - lo, 1e-6)
            if lo <= value <= hi:
                return 1.0
            return max(0.0, 1.0 - min(abs(value - lo), abs(value - hi)) / (span * 0.75))
        return float(np.clip(0.5 * margin(distance_m, dlo, dhi) + 0.5 * margin(height_m, hlo, hhi), 0.0, 1.0))

    def to_dict(self) -> dict:
        return {
            "distance_range_m": [round(v, 3) for v in self.distance_range_m],
            "height_range_m": [round(v, 3) for v in self.height_range_m],
            "confidence": round(float(self.confidence), 3),
            "basis": list(self.basis),
        }


def build_depth_constraint_evidence(
    image: np.ndarray,
    pose_keypoints: np.ndarray,
    provider: DepthProvider,
) -> tuple[np.ndarray, DepthConstraintEvidence]:
    """Estimate a relative depth map and sample it at visible keypoints."""
    depth = np.asarray(provider.estimate_depth(image), dtype=np.float32)
    h, w = depth.shape[:2]
    kp = np.asarray(pose_keypoints, dtype=float)
    if kp.ndim != 2 or kp.shape[0] < 17 or kp.shape[1] < 3:
        return depth, DepthConstraintEvidence(tuple(), 0.0, 0, w, h)

    samples: list[float] = []
    valid = kp[:17, 2] > 0.35
    for x, y, is_visible in kp[:17]:
        if not is_visible > 0.35 or not np.isfinite([x, y]).all():
            continue
        px = int(np.clip(round(float(x)), 0, w - 1))
        py = int(np.clip(round(float(y)), 0, h - 1))
        value = float(depth[py, px])
        if np.isfinite(value):
            samples.append(value)

    spread = float(np.std(samples)) if samples else 0.0
    confidence = float(np.clip((len(samples) / 12.0) * min(1.0, spread / 0.08), 0.0, 0.9))
    return depth, DepthConstraintEvidence(tuple(samples), confidence, len(samples), w, h)


def estimate_camera_feasibility(
    pose_keypoints: np.ndarray,
    image_w: int,
    image_h: int,
    focal_length_mm: float,
    depth_evidence: DepthConstraintEvidence | None = None,
    reference_height_m: float = 1.70,
) -> CameraFeasibilityEvidence:
    """Build broad distance/height intervals from body extent and weak priors."""
    kp = np.asarray(pose_keypoints, dtype=float)
    valid = kp[:17, 2] > 0.35 if kp.ndim == 2 and kp.shape[0] >= 17 and kp.shape[1] >= 3 else np.array([], dtype=bool)
    if valid.sum() < 5:
        return CameraFeasibilityEvidence((0.8, 12.0), (0.6, 2.2), 0.1, ("insufficient pose evidence",))

    yspan_px = max(float(np.ptp(kp[:17, 1][valid])), 20.0)
    # Keep this interval intentionally wide: person height is unknown and the
    # observed keypoint extent is not the exact physical head-to-foot span.
    sensor_width_mm = 36.0
    sensor_height_mm = sensor_width_mm * image_h / max(image_w, 1)
    fy = float(focal_length_mm) * image_h / max(sensor_height_mm, 1e-6)
    base_distance = reference_height_m * fy / yspan_px
    distance_range = (
        max(0.8, base_distance * 0.55),
        min(20.0, base_distance * 1.75),
    )

    cy_norm = float(np.mean(kp[:17, 1][valid]) / max(image_h, 1))
    # Camera height is anchored to a broad portrait prior and subject placement,
    # never claimed as a direct measurement.
    base_height = reference_height_m * (0.45 + 0.55 * (1.0 - np.clip(cy_norm, 0.0, 1.0)))
    depth_bonus = 0.08 * (depth_evidence.confidence if depth_evidence is not None else 0.0)
    height_half = 0.42 - depth_bonus
    height_range = (
        max(0.45, base_height - height_half),
        min(2.40, base_height + height_half),
    )
    confidence = 0.35 + 0.25 * min(1.0, valid.sum() / 10.0)
    if depth_evidence is not None and depth_evidence.usable:
        confidence += 0.20 * depth_evidence.confidence
    confidence = float(np.clip(confidence, 0.0, 0.85))
    basis = ["pose body extent", "human-height uncertainty"]
    if depth_evidence is not None and depth_evidence.usable:
        basis.append("relative depth consistency")
    return CameraFeasibilityEvidence(distance_range, height_range, confidence, tuple(basis))


def candidate_feasibility_score(candidate, feasibility: CameraFeasibilityEvidence | None) -> Optional[float]:
    if feasibility is None or not feasibility.usable:
        return None
    return feasibility.contains(float(candidate.distance), float(candidate.height))


def candidate_depth_score(
    candidate,
    pose_keypoints: np.ndarray,
    image_w: int,
    image_h: int,
    depth_evidence: DepthConstraintEvidence | None,
) -> Optional[float]:
    """Compare predicted landmark depth ordering to monocular depth ordering."""
    if depth_evidence is None or not depth_evidence.usable:
        return None

    kp = np.asarray(pose_keypoints, dtype=float)
    visible_indices = [i for i in range(min(17, len(kp)))
                       if kp[i, 2] > 0.35 and np.isfinite(kp[i, :2]).all()]
    if len(visible_indices) < 5 or len(depth_evidence.landmark_depths) < len(visible_indices):
        return None

    depth_values: list[float] = []
    sampled_indices: list[int] = []
    cursor = 0
    for i in range(min(17, len(kp))):
        if kp[i, 2] <= 0.35 or not np.isfinite(kp[i, :2]).all():
            continue
        if cursor >= len(depth_evidence.landmark_depths):
            break
        depth_values.append(float(depth_evidence.landmark_depths[cursor]))
        sampled_indices.append(i)
        cursor += 1

    if len(depth_values) < 5:
        return None

    proxy = pose_driven_person_points(kp, image_w, image_h)
    points = np.asarray(proxy[sampled_indices], dtype=float)
    rvec = candidate.extrinsics.rvec.reshape(3)
    tvec = candidate.extrinsics.tvec.reshape(3)
    camera_points = cv2.Rodrigues(rvec)[0] @ points.T + tvec.reshape(3, 1)
    predicted_z = camera_points[2]

    if not np.isfinite(predicted_z).all() or np.std(predicted_z) < 1e-8 or np.std(depth_values) < 1e-8:
        return None

    def _rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks

    a = _rank(np.asarray(predicted_z, dtype=float))
    b = _rank(np.asarray(depth_values, dtype=float))
    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr):
        return None
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))
