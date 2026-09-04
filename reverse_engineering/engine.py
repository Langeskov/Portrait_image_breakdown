"""
摄影逆向工程主引擎 - ReverseEngineeringEngine

整合所有子模块, 使用几何约束 + 优化替代纯 heuristic。
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from reverse_engineering.data_types import (
    ReverseEngineeringResult, CompositionResult, EstimatedValue,
    CameraAction, CameraPoseResult, FocalLengthResult)
from reverse_engineering.perspective import analyze_perspective
from reverse_engineering.camera_pose import estimate_camera_pose
from reverse_engineering.focal_length import estimate_focal_length
from reverse_engineering.depth_of_field import analyze_depth_of_field
from reverse_engineering.motion_blur import analyze_motion_blur
from reverse_engineering.shooting_technique import classify_techniques
from reverse_engineering.simulation import optimize_parameters
from reverse_engineering.depth_provider import MonocularDepthProvider
from reverse_engineering.geometry import PoseSolver, CameraIntrinsics
from core.pose_detector import PoseResult, LandmarkIndex as LI


def _extract_keypoints_pixels(pose: PoseResult) -> np.ndarray:
    """提取17个COCO关键点的像素坐标 + 置信度"""
    kp = np.zeros((17, 3))
    for i in range(17):
        lm = pose.landmarks[i]
        kp[i] = [lm.x, lm.y, lm.visibility]
    return kp


def _analyze_composition_extended(image, pose=None, bbox=None):
    h, w = image.shape[:2]
    if pose is not None:
        visible = [(lm.world_x, lm.world_y) for lm in pose.landmarks[:17] if lm.visibility > 0.4]
        if visible:
            sx = float(np.mean([p[0] for p in visible]))
            sy = float(np.mean([p[1] for p in visible]))
        else: sx, sy = 0.5, 0.5
    else: sx, sy = 0.5, 0.5
    if bbox:
        x1, y1, x2, y2 = bbox
        subject_scale = (x2-x1) * (y2-y1) / (w*h)
    elif pose:
        vp = [(lm.world_x, lm.world_y) for lm in pose.landmarks[:17] if lm.visibility > 0.4]
        if len(vp) >= 4:
            subject_scale = (max(p[0] for p in vp) - min(p[0] for p in vp)) * (max(p[1] for p in vp) - min(p[1] for p in vp))
        else: subject_scale = 0.3
    else: subject_scale = 0.3
    headroom = pose.landmarks[LI.NOSE].world_y if pose and pose.is_visible(LI.NOSE) else sy
    look_room = "right" if sx < 0.4 else "left" if sx > 0.6 else "balanced"
    neg_space = 1.0 - sx if look_room == "right" else sx if look_room == "left" else 0.5
    styles = []
    thirds_pts = [1/3, 2/3]
    min_dist = min(abs(sx-tx) + abs(sy-ty) for tx in thirds_pts for ty in thirds_pts)
    ts = max(0, 1.0 - min_dist / 0.35)
    if ts > 0.4: styles.append({"name": "rule_of_thirds", "confidence": round(ts, 2)})
    cd = math.sqrt((sx-0.5)**2 + (sy-0.5)**2)
    if cd < 0.12: styles.append({"name": "center", "confidence": round(1.0 - cd/0.12, 2)})
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64)).astype(float)
    flipped = np.fliplr(small)
    sym = 1.0 - np.abs(small - flipped).mean() / (small.max() - small.min() + 1)
    if sym > 0.7: styles.append({"name": "symmetry", "confidence": round(float(sym), 2)})
    if neg_space > 0.6: styles.append({"name": "negative_space", "confidence": round(neg_space*0.8, 2)})
    if subject_scale < 0.4: styles.append({"name": "depth_layering", "confidence": 0.5})
    styles.sort(key=lambda s: s["confidence"], reverse=True)
    return CompositionResult(styles=styles[:5], subject_position=(round(sx, 3), round(sy, 3)),
                             subject_scale=round(subject_scale, 4), headroom=round(headroom, 3),
                             look_room=look_room, negative_space_ratio=round(neg_space, 3))


def _generate_camera_actions(result, sim_candidates=None):
    actions = []
    comp = result.composition
    sx, sy = comp.subject_position
    thirds_pts = [1/3, 2/3]
    min_dist = min(abs(sx-tx) + abs(sy-ty) for tx in thirds_pts for ty in thirds_pts)
    if min_dist > 0.15:
        if sx < 0.4:
            actions.append(CameraAction("MOVE_RIGHT", ["subject not on thirds", f"x={sx:.2f}"], "improve composition", priority=2))
        elif sx > 0.6:
            actions.append(CameraAction("MOVE_LEFT", ["subject not on thirds", f"x={sx:.2f}"], "improve composition", priority=2))
    if comp.subject_scale > 0.7:
        actions.append(CameraAction("MOVE_BACKWARD", [f"subject {comp.subject_scale:.0%} of frame"], "reduce scale, moderate perspective", priority=3))
    elif comp.subject_scale < 0.1:
        actions.append(CameraAction("MOVE_FORWARD", [f"subject only {comp.subject_scale:.0%}"], "increase presence", priority=2))
    fl = result.focal_length
    if fl.category.value == "wide" and comp.subject_scale > 0.3:
        actions.append(CameraAction("ZOOM_IN", ["wide+large subject", f"suggested {fl.equivalent_35mm.range_min}-{fl.equivalent_35mm.range_max}mm"], "reduce distortion", priority=1))
    pitch = result.camera_pose.camera_pitch.value
    if pitch and abs(pitch) > 15:
        d = "LOWER_CAMERA" if pitch > 0 else "RAISE_CAMERA"
        actions.append(CameraAction(d, [f"pitch {pitch:.1f}deg"], "approach eye-level", priority=1))
    # 如果有多个候选解, 添加切换建议
    if sim_candidates and len(sim_candidates) > 1:
        alt = sim_candidates[1]
        actions.append(CameraAction("CHANGE_FOCAL_LENGTH",
            [f"alternative: {alt.focal_equiv_35mm}mm at {alt.distance}m (score={alt.score:.2f})"],
            "try different focal/distance combination", priority=0))
    actions.append(CameraAction("WAIT", ["observe subject movement"], "wait for decisive moment", priority=0))
    actions.append(CameraAction("CAPTURE", ["parameters acceptable"], "take the shot", priority=0))
    actions.sort(key=lambda a: a.priority, reverse=True)
    return actions[:8]


class ReverseEngineeringEngine:
    def __init__(self, enable_simulation=True):
        self._enable_simulation = enable_simulation
        self._depth_provider = MonocularDepthProvider()

    def analyze(self, image, pose=None, bbox=None):
        uncertainties = []
        h, w = image.shape[:2]

        # 1. 构图分析
        composition = _analyze_composition_extended(image, pose, bbox)
        subject_scale = composition.subject_scale

        # 2. 透视分析
        perspective = analyze_perspective(image)

        # 3. 相机位姿估计 (使用几何约束)
        if pose:
            camera_pose = estimate_camera_pose(pose, perspective.vanishing_points)
        else:
            _ev = lambda v, **kw: EstimatedValue(v, **kw)
            camera_pose = CameraPoseResult(
                camera_height=_ev(1.5, unit="m", confidence=0.1, basis=["default"]),
                camera_distance=_ev(3.0, unit="m", confidence=0.1, basis=["default"]),
                camera_pitch=_ev(0.0, unit="deg", confidence=0.1, basis=["default"]),
                camera_yaw=_ev(0.0, unit="deg", confidence=0.1, basis=["default"]),
                camera_roll=_ev(0.0, unit="deg", confidence=0.1, basis=["default"]))
            uncertainties.append("no pose detection - using defaults")

        # 4. 焦段估计 (先用几何候选, 再用 simulation 优化)
        sim_candidates = None
        if self._enable_simulation and pose:
            kp_pixels = _extract_keypoints_pixels(pose)
            geom_candidates = PoseSolver.solve_from_body_geometry(kp_pixels, w, h)
            sim_candidates = optimize_parameters(
                w, h, subject_scale, composition.subject_position,
                perspective.perspective_strength.value, kp_pixels)
            # 合并几何候选和模拟候选
            all_candidates = geom_candidates + sim_candidates
            all_candidates.sort(key=lambda c: c.score, reverse=True)
            focal_length = estimate_focal_length(
                perspective.perspective_strength.value,
                perspective.perspective_type.value,
                subject_scale, all_candidates[:5])
        else:
            focal_length = estimate_focal_length(
                perspective.perspective_strength.value,
                perspective.perspective_type.value, subject_scale)

        # 5. 景深分析
        depth_of_field = analyze_depth_of_field(image, pose, bbox)

        # 6. 运动模糊分析
        motion_blur = analyze_motion_blur(image, bbox)

        # 7. 摄影手法识别
        shooting_techniques = classify_techniques(
            perspective, camera_pose, focal_length,
            depth_of_field, motion_blur, composition, subject_scale)

        # 8. 整体置信度
        conf_vals = [focal_length.category.confidence, focal_length.equivalent_35mm.confidence,
                     depth_of_field.dof_type.confidence, motion_blur.blur_type.confidence,
                     camera_pose.camera_height.confidence, camera_pose.camera_distance.confidence]
        overall_conf = float(np.mean(conf_vals))

        uncertainties.extend([
            "exact focal length cannot be uniquely determined from single image",
            "sensor format unknown - assumed full-frame equivalent",
            "image may have been cropped",
            "aperture estimated from depth-of-field only",
            "background blur may be post-processed"])

        result = ReverseEngineeringResult(
            image_size=(w, h), subject_bbox=bbox,
            subject_keypoints=pose.landmarks[:17] if pose else None,
            subject_scale=subject_scale, edge_lines=[], blur_regions={},
            perspective=perspective, camera_pose=camera_pose, focal_length=focal_length,
            depth_of_field=depth_of_field, motion_blur=motion_blur, composition=composition,
            shooting_techniques=shooting_techniques, overall_confidence=overall_conf,
            uncertainties=uncertainties, _sim_candidates=sim_candidates or [])

        result._camera_actions = _generate_camera_actions(result, sim_candidates)
        return result
