"""
镜头位置分析模块

根据人体在画面中的比例和位置, 估算:
- 镜头类型: 特写/半身/全身/远景
- 拍摄角度: 平视/俯拍/仰拍
- 画面旋转(荷兰角): 仅由图像/场景证据估计
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum

import numpy as np

from core.pose_detector import PoseResult, LandmarkIndex as LI


class ShotType(Enum):
    EXTREME_CLOSEUP = "大特写"
    CLOSEUP = "特写"
    MEDIUM_CLOSEUP = "中近景"
    MEDIUM = "中景"
    MEDIUM_LONG = "中全景"
    LONG = "全景"
    EXTREME_LONG = "远景"
    UNKNOWN = "未知"


class CameraAngle(Enum):
    EYE_LEVEL = "平视"
    LOW_ANGLE = "仰拍"
    HIGH_ANGLE = "俯拍"
    BIRD_EYE = "鸟瞰"
    DUTCH_ANGLE = "荷兰角(倾斜)"


@dataclasses.dataclass
class CameraResult:
    shot_type: ShotType
    camera_angle: CameraAngle
    subject_ratio: float
    subject_center_offset: tuple[float, float]
    dutch_angle_deg: float
    detail: str

    @property
    def summary(self) -> str:
        return f"{self.shot_type.value}, {self.camera_angle.value}"


def _estimate_image_roll(image: np.ndarray | None) -> tuple[float, float, int]:
    """Return (roll, confidence, line_count) from scene geometry only."""
    if image is None:
        return 0.0, 0.0, 0
    try:
        from reverse_engineering.camera_pose import estimate_image_roll
        roll, confidence, line_count = estimate_image_roll(image)
        if not np.isfinite(roll) or not np.isfinite(confidence):
            return 0.0, 0.0, int(line_count)
        return float(roll), float(confidence), int(line_count)
    except Exception:
        return 0.0, 0.0, 0


def analyze_camera(pose: PoseResult, image: np.ndarray | None = None) -> CameraResult:
    """Analyze framing, camera elevation and genuine image roll.

    The image roll is deliberately not inferred from the shoulder line: a human
    can lean while the camera remains level. Scene-line roll evidence is only
    accepted when its confidence is sufficient.
    """
    visible_pts = []
    for lm in pose.landmarks[:17]:
        if lm.visibility > 0.4:
            visible_pts.append([lm.world_x, lm.world_y])

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
    box_w = x_max - x_min
    box_h = y_max - y_min
    subject_ratio = box_w * box_h
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    offset_x = (cx - 0.5) * 2
    offset_y = (cy - 0.5) * 2

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

    nose_y = pose.landmarks[LI.NOSE].world_y
    ls_y = pose.landmarks[LI.LEFT_SHOULDER].world_y
    rs_y = pose.landmarks[LI.RIGHT_SHOULDER].world_y
    lh_y = pose.landmarks[LI.LEFT_HIP].world_y
    rh_y = pose.landmarks[LI.RIGHT_HIP].world_y
    shoulder_y = (ls_y + rs_y) / 2
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

    # Do not treat the person's shoulder slope as camera roll.
    dutch_angle, roll_confidence, line_count = _estimate_image_roll(image)
    if abs(dutch_angle) < 5.0 or roll_confidence < 0.25:
        dutch_angle = 0.0

    if abs(dutch_angle) >= 5.0:
        camera_angle = CameraAngle.DUTCH_ANGLE

    detail_parts = [
        f"镜头: {shot_type.value}",
        f"人物占比: {subject_ratio:.0%}",
        f"角度: {camera_angle.value}",
    ]
    if abs(dutch_angle) >= 3.0:
        detail_parts.append(f"画面倾斜: {dutch_angle:.1f}°")
        detail_parts.append(f"旋转证据置信度: {roll_confidence:.0%} ({line_count} lines)")

    return CameraResult(
        shot_type=shot_type,
        camera_angle=camera_angle,
        subject_ratio=subject_ratio,
        subject_center_offset=(round(offset_x, 3), round(offset_y, 3)),
        dutch_angle_deg=round(dutch_angle, 1),
        detail=", ".join(detail_parts),
    )
