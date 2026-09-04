"""
相机位置估计 - CameraPoseEstimator

使用几何约束求解器 (geometry.py), 替代之前的 magic formula。
保留 heuristic 作为 fallback, 但主要依赖几何计算。
"""
from __future__ import annotations
import math
import numpy as np
from reverse_engineering.data_types import EstimatedValue, CameraPoseResult
from reverse_engineering.geometry import (
    CameraIntrinsics, PoseSolver, CameraModel,
    REF_PERSON_HEIGHT, REF_SHOULDER_WIDTH,
)
from core.pose_detector import PoseResult, LandmarkIndex as LI


def estimate_camera_pose(
    pose: PoseResult,
    perspective_vanishing_points: list[tuple[float, float]] = None,
) -> CameraPoseResult:
    """
    估计相机位姿

    优先使用几何约束求解, fallback 到 heuristic。
    """
    h, w = pose.image_height, pose.image_width

    # 提取像素坐标
    kp_pixels = np.zeros((17, 3))
    for i in range(17):
        lm = pose.landmarks[i]
        kp_pixels[i] = [lm.x, lm.y, lm.visibility]

    # 尝试几何求解
    candidates = PoseSolver.solve_from_body_geometry(kp_pixels, w, h)

    # 如果有消失点, 额外求解旋转
    extrinsics_from_vp = None
    if perspective_vanishing_points and len(perspective_vanishing_points) >= 2:
        extrinsics_from_vp = PoseSolver.solve_from_vanishing_points(
            perspective_vanishing_points, w, h)

    if candidates:
        best = candidates[0]
        # 合并消失点旋转信息
        if extrinsics_from_vp:
            pitch = extrinsics_from_vp.pitch
            yaw = extrinsics_from_vp.yaw
            roll = extrinsics_from_vp.roll
        else:
            pitch = best.extrinsics.pitch
            yaw = best.extrinsics.yaw
            roll = best.extrinsics.roll

        dist = best.distance
        height = best.height
        focal = best.focal_equiv_35mm
        conf = best.score

        return CameraPoseResult(
            camera_height=EstimatedValue(
                height, unit="m",
                range_min=round(max(0.3, height - 0.4), 2),
                range_max=round(min(2.5, height + 0.4), 2),
                confidence=min(conf + 0.1, 0.8),
                basis=["body geometry constraint", f"score={conf:.2f}"]),
            camera_distance=EstimatedValue(
                dist, unit="m",
                range_min=round(max(0.5, dist * 0.7), 1),
                range_max=round(dist * 1.4, 1),
                confidence=conf,
                basis=["body height + shoulder width constraint"]),
            camera_pitch=EstimatedValue(
                round(pitch, 1), unit="deg",
                range_min=round(pitch - 5, 1),
                range_max=round(pitch + 5, 1),
                confidence=min(conf + 0.05, 0.75),
                basis=["geometric projection"]),
            camera_yaw=EstimatedValue(
                round(yaw, 1), unit="deg",
                range_min=round(yaw - 8, 1),
                range_max=round(yaw + 8, 1),
                confidence=0.35 if abs(yaw) < 5 else 0.5,
                basis=["vanishing point" if extrinsics_from_vp else "subject offset"]),
            camera_roll=EstimatedValue(
                round(roll, 1), unit="deg",
                range_min=round(roll - 2, 1),
                range_max=round(roll + 2, 1),
                confidence=0.6,
                basis=["shoulder tilt"]))

    # Fallback: heuristic (保留但标记为低置信度)
    nose_y = pose.landmarks[LI.NOSE].world_y
    ls_x = pose.landmarks[LI.LEFT_SHOULDER].world_x
    rs_x = pose.landmarks[LI.RIGHT_SHOULDER].world_x
    sw = abs(rs_x - ls_x)

    height = max(0.3, min(2.5, 1.5 + (0.5 - nose_y) * 1.5))
    dist = REF_SHOULDER_W / sw * 1.2 if sw > 0.01 else 5.0
    dist = max(0.5, min(30.0, dist))
    pitch = (0.4 - nose_y) * 30

    return CameraPoseResult(
        camera_height=EstimatedValue(
            round(height, 2), unit="m", range_min=round(height-0.3, 2),
            range_max=round(height+0.3, 2), confidence=0.3,
            basis=["heuristic fallback", f"nose_y={nose_y:.2f}"]),
        camera_distance=EstimatedValue(
            round(dist, 1), unit="m", range_min=round(max(0.5, dist*0.7), 1),
            range_max=round(dist*1.4, 1), confidence=0.25,
            basis=["heuristic fallback"]),
        camera_pitch=EstimatedValue(
            round(pitch, 1), unit="deg", range_min=round(pitch-5, 1),
            range_max=round(pitch+5, 1), confidence=0.25,
            basis=["heuristic fallback"]),
        camera_yaw=EstimatedValue(0.0, unit="deg", confidence=0.1, basis=["default"]),
        camera_roll=EstimatedValue(0.0, unit="deg", confidence=0.1, basis=["default"]))
