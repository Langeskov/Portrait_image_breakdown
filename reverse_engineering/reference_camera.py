"""Conservative camera hypothesis derived from reference/current composition.

A single reference photograph does not uniquely determine an absolute camera
pose. This module therefore estimates the part that composition can support:
subject-distance change and optical re-aim angles. The active SceneCamera is
never mutated by this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from reverse_engineering.reference_reconstruction import ReferenceComposition
from reverse_engineering.scene import SceneCamera, SceneModel
from reverse_engineering.scene_anchors import AnchorKind, SceneAnchor


@dataclass(frozen=True)
class ReferenceCameraHypothesis:
    success: bool
    confidence: float
    focal_length_mm: float
    current_distance_m: float
    reference_distance_m: float
    distance_delta_m: float
    reframe_yaw_deg: float
    reframe_pitch_deg: float
    roll_deg: float
    scale_ratio: float
    center_dx: float
    center_dy: float
    anchor_name: Optional[str]
    support: str
    message: str


def _norm_center(comp: ReferenceComposition) -> tuple[float, float]:
    return (
        float(comp.subject_center[0]) / max(float(comp.width), 1.0),
        float(comp.subject_center[1]) / max(float(comp.height), 1.0),
    )


def estimate_reference_camera_hypothesis(
    scene: SceneModel,
    reference: ReferenceComposition,
    current: ReferenceComposition,
    selected_anchor: Optional[SceneAnchor] = None,
) -> ReferenceCameraHypothesis:
    """Estimate a reference-camera delta from composition only.

    The subject bbox area is used as a scale cue. Under a fixed focal length
    prior, image area is approximately proportional to ``1 / distance²``. The
    center offset becomes an optical-axis re-aim angle. These are deliberately
    labelled as hypotheses, not recovered metric camera measurements.
    """
    if reference is None or current is None:
        return ReferenceCameraHypothesis(
            False, 0.0, float(scene.camera.focal_length_mm), float(scene.camera.distance),
            float(scene.camera.distance), 0.0, 0.0, 0.0, float(scene.camera.roll),
            1.0, 0.0, 0.0, None, "none", "Reference and current compositions are required.",
        )
    if reference.subject_scale <= 1e-9 or current.subject_scale <= 1e-9:
        return ReferenceCameraHypothesis(
            False, 0.05, float(scene.camera.focal_length_mm), float(scene.camera.distance),
            float(scene.camera.distance), 0.0, 0.0, 0.0, float(scene.camera.roll),
            1.0, 0.0, 0.0, getattr(selected_anchor, "name", None), "weak", 
            "Visible subject extent is insufficient for a stable distance hypothesis.",
        )

    current_distance = max(float(scene.camera.distance), 0.1)
    area_ratio = float(reference.subject_scale / current.subject_scale)
    # reference/current area = (current_distance/reference_distance)^2
    reference_distance = current_distance / math.sqrt(max(area_ratio, 1e-9))
    reference_distance = max(0.25, min(reference_distance, 100.0))

    ref_cx, ref_cy = _norm_center(reference)
    cur_cx, cur_cy = _norm_center(current)
    center_dx = ref_cx - cur_cx
    center_dy = ref_cy - cur_cy

    # Re-aim angle, not SceneCamera.yaw/pitch. SceneCamera yaw/pitch also
    # encode the orbit position around the target, so directly replacing them
    # would not reproduce a screen-space framing offset.
    yaw = center_dx * scene.camera.horizontal_fov_deg
    pitch = -center_dy * scene.camera.vertical_fov_deg
    yaw = max(-35.0, min(35.0, yaw))
    pitch = max(-30.0, min(30.0, pitch))

    scale_stability = min(1.0, max(0.0, 1.0 - abs(math.log(max(area_ratio, 1e-9))) / 2.0))
    center_stability = min(1.0, max(0.0, 1.0 - math.hypot(center_dx, center_dy) / 0.65))
    anchor_bonus = 0.08 if selected_anchor is not None else 0.0
    if selected_anchor is not None and selected_anchor.kind == AnchorKind.PLANE:
        support = "reference composition + selected plane context"
    elif selected_anchor is not None:
        support = "reference composition + selected point context"
    else:
        support = "reference composition only"
    confidence = min(0.82, 0.42 + 0.38 * scale_stability + 0.12 * center_stability + anchor_bonus)

    distance_delta = reference_distance - current_distance
    message = (
        f"Reference camera hypothesis · distance {current_distance:.2f} → {reference_distance:.2f} m "
        f"({distance_delta:+.2f} m) · re-aim yaw {yaw:+.1f}° · pitch {pitch:+.1f}°"
    )
    return ReferenceCameraHypothesis(
        True,
        float(confidence),
        float(scene.camera.focal_length_mm),
        current_distance,
        reference_distance,
        float(distance_delta),
        float(yaw),
        float(pitch),
        float(scene.camera.roll),
        float(area_ratio),
        float(center_dx),
        float(center_dy),
        getattr(selected_anchor, "name", None),
        support,
        message,
    )
