"""Portrait camera reverse-engineering engine v2.

v2 separates the evidence roles: human pose/BBox constrain framing, while
scene geometry constrains camera rotation and provides an independent focal
length prior.  The final candidate family is produced by fusing both sources.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from core.pose_detector import PoseResult, LandmarkIndex as LI
from reverse_engineering.camera_pose import estimate_camera_pose
from reverse_engineering.data_types import (
    CameraAction,
    CameraPoseResult,
    CompositionResult,
    EstimatedValue,
    ReverseEngineeringResult,
)
from reverse_engineering.depth_of_field import analyze_depth_of_field
from reverse_engineering.depth_provider import MonocularDepthProvider
from reverse_engineering.focal_length import estimate_focal_length
from reverse_engineering.motion_blur import analyze_motion_blur
from reverse_engineering.perspective import analyze_perspective
from reverse_engineering.scene_geometry import analyze_scene_geometry
from reverse_engineering.shooting_technique import classify_techniques
from reverse_engineering.simulation import optimize_parameters


def _extract_keypoints_pixels(pose: PoseResult) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.visibility] for lm in pose.landmarks[:17]], dtype=float)


def _analyze_composition_extended(image, pose=None, bbox=None) -> CompositionResult:
    h, w = image.shape[:2]
    if pose is not None:
        visible = [(lm.world_x, lm.world_y) for lm in pose.landmarks[:17] if lm.visibility > 0.4]
        sx = float(np.mean([p[0] for p in visible])) if visible else 0.5
        sy = float(np.mean([p[1] for p in visible])) if visible else 0.5
    else:
        sx, sy = 0.5, 0.5

    if bbox:
        x1, y1, x2, y2 = bbox
        subject_scale = max(0.0, min(1.0, (x2 - x1) * (y2 - y1) / max(w * h, 1)))
    elif pose:
        vp = [(lm.world_x, lm.world_y) for lm in pose.landmarks[:17] if lm.visibility > 0.4]
        subject_scale = max(0.0, min(1.0, (max(p[0] for p in vp) - min(p[0] for p in vp)) * (max(p[1] for p in vp) - min(p[1] for p in vp)))) if len(vp) >= 4 else 0.3
    else:
        subject_scale = 0.3

    headroom = pose.landmarks[LI.NOSE].world_y if pose and pose.is_visible(LI.NOSE) else sy
    look_room = "right" if sx < 0.4 else "left" if sx > 0.6 else "balanced"
    neg_space = 1.0 - sx if look_room == "right" else sx if look_room == "left" else 0.5

    styles = []
    min_dist = min(abs(sx - tx) + abs(sy - ty) for tx in (1 / 3, 2 / 3) for ty in (1 / 3, 2 / 3))
    thirds_conf = max(0.0, 1.0 - min_dist / 0.35)
    if thirds_conf > 0.4:
        styles.append({"name": "rule_of_thirds", "confidence": round(thirds_conf, 2)})
    center_dist = math.hypot(sx - 0.5, sy - 0.5)
    if center_dist < 0.12:
        styles.append({"name": "center", "confidence": round(1.0 - center_dist / 0.12, 2)})

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64)).astype(float)
    symmetry = 1.0 - np.abs(small - np.fliplr(small)).mean() / (small.max() - small.min() + 1)
    if symmetry > 0.7:
        styles.append({"name": "symmetry", "confidence": round(float(symmetry), 2)})
    if neg_space > 0.6:
        styles.append({"name": "negative_space", "confidence": round(neg_space * 0.8, 2)})
    if subject_scale < 0.4:
        styles.append({"name": "depth_layering", "confidence": 0.5})
    styles.sort(key=lambda s: s["confidence"], reverse=True)

    return CompositionResult(
        styles=styles[:5],
        subject_position=(round(sx, 3), round(sy, 3)),
        subject_scale=round(subject_scale, 4),
        headroom=round(headroom, 3),
        look_room=look_room,
        negative_space_ratio=round(neg_space, 3),
    )


def _generate_camera_actions(result, candidates=None):
    actions = []
    sx, _ = result.composition.subject_position
    third_distance = min(abs(sx - tx) for tx in (1 / 3, 2 / 3))
    if third_distance > 0.15:
        actions.append(CameraAction("MOVE_RIGHT" if sx < 0.5 else "MOVE_LEFT", ["subject not near thirds verticals", f"x={sx:.2f}"], "adjust horizontal framing", priority=2))
    if result.composition.subject_scale > 0.7:
        actions.append(CameraAction("MOVE_BACKWARD", [f"subject {result.composition.subject_scale:.0%} of frame"], "reduce subject scale", priority=2))
    elif result.composition.subject_scale < 0.1:
        actions.append(CameraAction("MOVE_FORWARD", [f"subject only {result.composition.subject_scale:.0%}"], "increase subject presence", priority=2))
    if candidates and len(candidates) > 1:
        alt = candidates[1]
        actions.append(CameraAction("CHANGE_FOCAL_LENGTH", [f"alternative {alt.focal_equiv_35mm:.1f}mm at {alt.distance:.2f}m"], "explore another ranked camera solution", priority=1))
    actions.extend([
        CameraAction("WAIT", ["observe subject movement"], "wait for decisive moment", priority=0),
        CameraAction("CAPTURE", ["current solution is acceptable"], "take the shot", priority=0),
    ])
    actions.sort(key=lambda a: a.priority, reverse=True)
    return actions[:8]


def _camera_pose_from_candidate(candidate) -> CameraPoseResult:
    return CameraPoseResult(
        camera_height=EstimatedValue(round(candidate.height, 2), unit="m", range_min=max(0.25, round(candidate.height - 0.25, 2)), range_max=min(2.5, round(candidate.height + 0.25, 2)), confidence=min(0.8, candidate.score), basis=["v2 pose + scene geometry"]),
        camera_distance=EstimatedValue(round(candidate.distance, 2), unit="m", range_min=max(0.5, round(candidate.distance * 0.75, 2)), range_max=round(candidate.distance * 1.25, 2), confidence=min(0.8, candidate.score), basis=["pose framing + scene rotation", "focal/distance ambiguity retained"]),
        camera_pitch=EstimatedValue(round(candidate.extrinsics.pitch, 1), unit="deg", range_min=round(candidate.extrinsics.pitch - 5, 1), range_max=round(candidate.extrinsics.pitch + 5, 1), confidence=min(0.82, candidate.score), basis=["Manhattan vanishing-point geometry"]),
        camera_yaw=EstimatedValue(round(candidate.extrinsics.yaw, 1), unit="deg", range_min=round(candidate.extrinsics.yaw - 5, 1), range_max=round(candidate.extrinsics.yaw + 5, 1), confidence=min(0.82, candidate.score), basis=["Manhattan vanishing-point geometry"]),
        camera_roll=EstimatedValue(round(candidate.extrinsics.roll, 1), unit="deg", range_min=round(candidate.extrinsics.roll - 3, 1), range_max=round(candidate.extrinsics.roll + 3, 1), confidence=min(0.8, candidate.score), basis=["scene horizon / Manhattan geometry"]),
    )


class ReverseEngineeringEngineV2:
    """Scene-aware camera reverse-engineering pipeline."""

    VERSION = "2.0"

    def __init__(self, enable_simulation=True):
        self._enable_simulation = enable_simulation
        self._depth_provider = MonocularDepthProvider()

    def analyze(self, image, pose=None, bbox=None):
        h, w = image.shape[:2]
        scene_evidence = analyze_scene_geometry(image)
        composition = _analyze_composition_extended(image, pose, bbox)
        perspective = analyze_perspective(image, scene_evidence)

        candidates = []
        if pose is not None:
            kp = _extract_keypoints_pixels(pose)
            candidates = optimize_parameters(
                w, h,
                composition.subject_scale,
                composition.subject_position,
                perspective.perspective_strength.value,
                kp,
                num_candidates=6,
                subject_bbox=bbox,
                scene_evidence=scene_evidence,
            )

        if candidates:
            camera_pose = _camera_pose_from_candidate(candidates[0])
            focal_length = estimate_focal_length(
                perspective.perspective_strength.value,
                perspective.perspective_type.value,
                composition.subject_scale,
                candidates,
            )
        elif pose is not None:
            camera_pose = estimate_camera_pose(
                pose,
                perspective.vanishing_points,
                image=image,
                subject_bbox=bbox,
            )
            focal_length = estimate_focal_length(
                perspective.perspective_strength.value,
                perspective.perspective_type.value,
                composition.subject_scale,
            )
        else:
            camera_pose = CameraPoseResult(
                camera_height=EstimatedValue(1.5, unit="m", confidence=.1, basis=["no pose"]),
                camera_distance=EstimatedValue(4.0, unit="m", confidence=.1, basis=["no pose"]),
                camera_pitch=EstimatedValue(0.0, unit="deg", confidence=.1, basis=["no pose"]),
                camera_yaw=EstimatedValue(0.0, unit="deg", confidence=.1, basis=["no pose"]),
                camera_roll=EstimatedValue(0.0, unit="deg", confidence=.1, basis=["no pose"]),
            )
            focal_length = estimate_focal_length(
                perspective.perspective_strength.value,
                perspective.perspective_type.value,
                composition.subject_scale,
            )

        depth_of_field = analyze_depth_of_field(image, pose, bbox)
        motion_blur = analyze_motion_blur(image, bbox)
        shooting_techniques = classify_techniques(
            perspective, camera_pose, focal_length, depth_of_field, motion_blur,
            composition, composition.subject_scale,
        )

        confidence_values = [
            focal_length.category.confidence,
            focal_length.equivalent_35mm.confidence,
            depth_of_field.dof_type.confidence,
            motion_blur.blur_type.confidence,
            camera_pose.camera_height.confidence,
            camera_pose.camera_distance.confidence,
        ]
        overall_confidence = float(np.mean(confidence_values))

        uncertainties = [
            "exact focal length cannot be uniquely determined from a single image",
            "sensor format and crop status may be unknown",
            "aperture is inferred only from blur/depth characteristics",
            "camera height and distance remain coupled without scene scale or depth",
        ]
        if scene_evidence.has_three_directions:
            uncertainties.append("rotation uses Manhattan scene geometry; quality depends on reliable orthogonal architectural lines")
        else:
            uncertainties.append("insufficient orthogonal scene structure for reliable absolute rotation")

        result = ReverseEngineeringResult(
            image_size=(w, h),
            subject_bbox=bbox,
            subject_keypoints=pose.landmarks[:17] if pose else None,
            subject_scale=composition.subject_scale,
            edge_lines=perspective.line_segments,
            blur_regions={},
            perspective=perspective,
            camera_pose=camera_pose,
            focal_length=focal_length,
            depth_of_field=depth_of_field,
            motion_blur=motion_blur,
            composition=composition,
            shooting_techniques=shooting_techniques,
            overall_confidence=overall_confidence,
            uncertainties=uncertainties,
            _sim_candidates=candidates,
        )
        result._camera_actions = _generate_camera_actions(result, candidates)
        return result
