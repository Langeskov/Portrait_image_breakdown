"""Conservative scene/depth constraints for monocular camera fitting.

Depth is treated as relative evidence only. The module never converts a
monocular depth map into an absolute metric distance; it uses depth ordering to
rank otherwise-plausible camera candidates and to expose how useful the depth
evidence actually is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def candidate_depth_score(
    candidate,
    pose_keypoints: np.ndarray,
    image_w: int,
    image_h: int,
    depth_evidence: DepthConstraintEvidence | None,
) -> Optional[float]:
    """Compare predicted landmark depth ordering to monocular depth ordering.

    The score is rank-based, so it is invariant to the arbitrary scale and
    offset of a relative monocular depth map. ``None`` means the evidence is
    not usable and the candidate should not be penalized.
    """
    if depth_evidence is None or not depth_evidence.usable:
        return None

    kp = np.asarray(pose_keypoints, dtype=float)
    visible_indices = [i for i in range(min(17, len(kp)))
                       if kp[i, 2] > 0.35 and np.isfinite(kp[i, :2]).all()]
    if len(visible_indices) < 5 or len(depth_evidence.landmark_depths) < len(visible_indices):
        return None

    # Re-sample directly so landmark ordering stays aligned with original IDs.
    depth_values: list[float] = []
    sampled_conf: list[int] = []
    # The evidence tuple is ordered over visible landmarks; rebuild that order.
    cursor = 0
    for i in range(min(17, len(kp))):
        if kp[i, 2] <= 0.35 or not np.isfinite(kp[i, :2]).all():
            continue
        if cursor >= len(depth_evidence.landmark_depths):
            break
        depth_values.append(float(depth_evidence.landmark_depths[cursor]))
        sampled_conf.append(i)
        cursor += 1

    if len(depth_values) < 5:
        return None

    proxy = pose_driven_person_points(kp, image_w, image_h)
    camera = CameraModel(candidate.intrinsics, candidate.extrinsics)
    points = np.asarray(proxy[sampled_conf], dtype=float)
    # Transform into camera coordinates using the exact same extrinsics used by projection.
    rvec = candidate.extrinsics.rvec.reshape(3)
    tvec = candidate.extrinsics.tvec.reshape(3)
    import cv2
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
    # Monocular provider uses larger values for farther points, matching camera Z.
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))
