"""
镜头位置分析模块

根据人体在画面中的比例和位置, 估算:
- 镜头类型: 特写/半身/全身/远景
- 拍摄角度: 平视/俯拍/仰拍
- 焦距估算(基于透视变形)
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum

import numpy as np

from core.pose_detector import PoseResult, LandmarkIndex as LI


class ShotType(Enum):
    EXTREME_CLOSEUP = "大特写"    # 脸部/局部
    CLOSEUP = "特写"              # 肩部以上
    MEDIUM_CLOSEUP = "中近景"     # 胸部以上
    MEDIUM = "中景"               # 膝盖以上
    MEDIUM_LONG = "中全景"        # 全身带空间
    LONG = "全景"                 # 全身+环境
    EXTREME_LONG = "远景"         # 人物很小
    UNKNOWN = "未知"


class CameraAngle(Enum):
    EYE_LEVEL = "平视"
    LOW_ANGLE = "仰拍"
    HIGH_ANGLE = "俯拍"
    BIRD_EYE = "鸟瞰"
    DUTCH_ANGLE = "荷兰角(倾斜)"


@dataclasses.dataclass
class CameraResult:
    shot_type: ShotType           # 镜头类型
    camera_angle: CameraAngle     # 拍摄角度
    subject_ratio: float          # 人物占画面比例 [0, 1]
    subject_center_offset: tuple[float, float]  # 人物中心偏移 (x, y), 0=中心
    dutch_angle_deg: float        # 画面倾斜角度
    detail: str

    @property
    def summary(self) -> str:
        return f"{self.shot_type.value}, {self.camera_angle.value}"


def analyze_camera(pose: PoseResult) -> CameraResult:
    """
    分析镜头位置和类型

    所有计算使用归一化坐标 [0, 1] (通过 get_normalized()).
    """
    h = pose.image_height
    w = pose.image_width

    # ── 获取关键点归一化坐标 ──
    # 收集所有可见关键点的归一化坐标
    visible_pts = []
    for lm in pose.landmarks[:17]:  # 只用COCO 17点
        if lm.visibility > 0.4:
            visible_pts.append([lm.world_x, lm.world_y])  # normalized

    if len(visible_pts) < 6:
        return CameraResult(
            shot_type=ShotType.UNKNOWN,
            camera_angle=CameraAngle.EYE_LEVEL,
            subject_ratio=0.0,
            subject_center_offset=(0.0, 0.0),
            dutch_angle_deg=0.0,
            detail="可见关键点不足, 无法分析镜头",
        )

    pts = np.array(visible_pts)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # 人体包围框(归一化)
    box_w = x_max - x_min
    box_h = y_max - y_min
    box_area = box_w * box_h

    # 人物占画面比例
    subject_ratio = box_area

    # 人物中心(归一化)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    # 中心偏移(归一化到 [-1, 1])
    offset_x = (cx - 0.5) * 2
    offset_y = (cy - 0.5) * 2

    # ── 镜头类型判断 ──
    vertical_ratio = box_h

    if vertical_ratio > 0.85:
        shot_type = ShotType.LONG
    elif vertical_ratio > 0.7:
        shot_type = ShotType.MEDIUM_LONG
    elif vertical_ratio > 0.5:
        shot_type = ShotType.MEDIUM
    elif vertical_ratio > 0.35:
        shot_type = ShotType.MEDIUM_CLOSEUP
    elif vertical_ratio > 0.2:
        shot_type = ShotType.CLOSEUP
    elif vertical_ratio > 0.1:
        shot_type = ShotType.EXTREME_CLOSEUP
    else:
        shot_type = ShotType.EXTREME_LONG

    # ── 拍摄角度判断 ──
    nose_y = pose.landmarks[LI.NOSE].world_y  # normalized
    ls_y = pose.landmarks[LI.LEFT_SHOULDER].world_y
    rs_y = pose.landmarks[LI.RIGHT_SHOULDER].world_y
    shoulder_y = (ls_y + rs_y) / 2

    lh_y = pose.landmarks[LI.LEFT_HIP].world_y
    rh_y = pose.landmarks[LI.RIGHT_HIP].world_y
    hip_y = (lh_y + rh_y) / 2

    la_y = pose.landmarks[LI.LEFT_ANKLE].world_y if pose.is_visible(LI.LEFT_ANKLE) else y_max
    ra_y = pose.landmarks[LI.RIGHT_ANKLE].world_y if pose.is_visible(LI.RIGHT_ANKLE) else y_max
    ankle_y = (la_y + ra_y) / 2

    head_to_shoulder = abs(shoulder_y - nose_y)
    total_height = abs(ankle_y - nose_y)
    head_ratio = head_to_shoulder / total_height if total_height > 0 else 0.15

    if head_ratio > 0.18 or (offset_y > 0.2 and head_ratio > 0.15):
        camera_angle = CameraAngle.LOW_ANGLE
    elif head_ratio < 0.10 or (offset_y < -0.2 and head_ratio < 0.13):
        camera_angle = CameraAngle.HIGH_ANGLE
    elif offset_y < -0.4:
        camera_angle = CameraAngle.BIRD_EYE
    else:
        camera_angle = CameraAngle.EYE_LEVEL

    # ── 荷兰角检测 ──
    ls_x = pose.landmarks[LI.LEFT_SHOULDER].world_x
    rs_x = pose.landmarks[LI.RIGHT_SHOULDER].world_x
    ls_y_norm = pose.landmarks[LI.LEFT_SHOULDER].world_y
    rs_y_norm = pose.landmarks[LI.RIGHT_SHOULDER].world_y
    shoulder_dx = rs_x - ls_x
    shoulder_dy = rs_y_norm - ls_y_norm

    if abs(shoulder_dx) > 0.01 or abs(shoulder_dy) > 0.01:
        raw_angle = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
        dutch_angle = raw_angle
        while dutch_angle > 90:
            dutch_angle -= 180
        while dutch_angle < -90:
            dutch_angle += 180
    else:
        dutch_angle = 0.0

    if abs(dutch_angle) > 5:
        camera_angle = CameraAngle.DUTCH_ANGLE

    # ── 生成描述 ──
    detail_parts = [
        f"镜头: {shot_type.value}",
        f"人物占比: {subject_ratio:.0%}",
        f"角度: {camera_angle.value}",
    ]
    if abs(dutch_angle) > 3:
        detail_parts.append(f"画面倾斜: {dutch_angle:.1f}°")

    return CameraResult(
        shot_type=shot_type,
        camera_angle=camera_angle,
        subject_ratio=subject_ratio,
        subject_center_offset=(round(offset_x, 3), round(offset_y, 3)),
        dutch_angle_deg=round(dutch_angle, 1),
        detail=", ".join(detail_parts),
    )
