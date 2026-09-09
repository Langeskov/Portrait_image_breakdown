"""Generate a reproducible image-space pose target from a reference composition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from reverse_engineering.reference_reconstruction import ReferenceComposition


@dataclass(frozen=True)
class GeneratedPoseTarget:
    width: int
    height: int
    subject_center: tuple[float, float]
    subject_scale: float
    keypoints: tuple[tuple[float, float, float], ...]
    visible_count: int
    source: str = "reference composition"

    def as_normalized(self) -> np.ndarray:
        if not self.keypoints:
            return np.empty((0, 3), dtype=float)
        return np.asarray(
            [(x / max(self.width, 1), y / max(self.height, 1), v) for x, y, v in self.keypoints],
            dtype=float,
        )


@dataclass(frozen=True)
class PoseTargetGenerationResult:
    success: bool
    target: Optional[GeneratedPoseTarget]
    message: str


def generate_composition_aware_pose_target(
    reference_landmarks,
    reference: ReferenceComposition,
    current: ReferenceComposition,
    *,
    visibility_threshold: float = 0.35,
) -> PoseTargetGenerationResult:
    """Map reference pose geometry into the current image using reference composition.

    The target keeps the reference subject's normalized internal pose while moving
    its center and scale to the requested reference composition. It therefore
    produces an actionable image-space target without claiming a metric body pose.
    """
    if reference is None or current is None or reference.subject_bbox is None or reference.subject_scale <= 1e-9:
        return PoseTargetGenerationResult(False, None, "Reference composition is insufficient for pose target generation.")

    landmarks = list(reference_landmarks or [])[:17]
    if not landmarks:
        return PoseTargetGenerationResult(False, None, "Reference landmarks are unavailable.")

    rb = reference.subject_bbox
    ref_w = max(float(rb[2] - rb[0]), 1e-9)
    ref_h = max(float(rb[3] - rb[1]), 1e-9)
    target_scale = max(float(reference.subject_scale), 1e-9)
    current_area = max(float(current.width * current.height), 1.0)
    # Preserve bbox aspect ratio while selecting a target area.
    target_bbox_w = (target_scale * current_area) ** 0.5
    target_bbox_h = target_bbox_w * ref_h / ref_w
    cx, cy = reference.subject_center
    output = []
    for landmark in landmarks:
        x = float(getattr(landmark, "x", 0.0))
        y = float(getattr(landmark, "y", 0.0))
        v = float(getattr(landmark, "visibility", 0.0))
        nx = (x - (rb[0] + rb[2]) * 0.5) / ref_w
        ny = (y - (rb[1] + rb[3]) * 0.5) / ref_h
        tx = cx + nx * target_bbox_w
        ty = cy + ny * target_bbox_h
        output.append((float(tx), float(ty), v))

    visible_count = sum(1 for _, _, v in output if v >= visibility_threshold)
    target = GeneratedPoseTarget(
        width=int(current.width),
        height=int(current.height),
        subject_center=(float(cx), float(cy)),
        subject_scale=float(target_scale),
        keypoints=tuple(output),
        visible_count=int(visible_count),
    )
    return PoseTargetGenerationResult(True, target, f"Generated {visible_count}/17 visible target landmarks in current image space.")
