"""
身体朝向分析模块

基于肩部/髋部关键点的相对位置, 判断人物朝向:
- 正面(front) / 背面(back)
- 左侧(left) / 右侧(right)
- 以及躯干倾斜(前倾/后仰/左右倾)
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum

import numpy as np

from core.pose_detector import PoseResult, LandmarkIndex as LI


class FacingDirection(Enum):
    FRONT = "正面"
    BACK = "背面"
    LEFT = "左侧"
    RIGHT = "右侧"
    FRONT_LEFT = "正面偏左"
    FRONT_RIGHT = "正面偏右"
    BACK_LEFT = "背面偏左"
    BACK_RIGHT = "背面偏右"


class TiltDirection(Enum):
    UPRIGHT = "直立"
    LEANING_FORWARD = "前倾"
    LEANING_BACKWARD = "后仰"
    LEANING_LEFT = "左倾"
    LEANING_RIGHT = "右倾"


@dataclasses.dataclass
class OrientationResult:
    facing: FacingDirection
    tilt: TiltDirection
    facing_angle: float
    tilt_angle: float
    confidence: float
    detail: str

    @property
    def summary(self) -> str:
        parts = [self.facing.value]
        if self.tilt != TiltDirection.UPRIGHT:
            parts.append(self.tilt.value)
        return " ".join(parts)


def _shoulder_center(pose: PoseResult) -> np.ndarray:
    ls = np.array(pose.get_normalized(LI.LEFT_SHOULDER))
    rs = np.array(pose.get_normalized(LI.RIGHT_SHOULDER))
    return (ls + rs) / 2


def _hip_center(pose: PoseResult) -> np.ndarray:
    lh = np.array(pose.get_normalized(LI.LEFT_HIP))
    rh = np.array(pose.get_normalized(LI.RIGHT_HIP))
    return (lh + rh) / 2


def _shoulder_width(pose: PoseResult) -> float:
    ls = np.array(pose.get_normalized(LI.LEFT_SHOULDER))
    rs = np.array(pose.get_normalized(LI.RIGHT_SHOULDER))
    return float(np.linalg.norm(ls - rs))


def _hip_width(pose: PoseResult) -> float:
    lh = np.array(pose.get_normalized(LI.LEFT_HIP))
    rh = np.array(pose.get_normalized(LI.RIGHT_HIP))
    return float(np.linalg.norm(lh - rh))


def analyze_orientation(pose: PoseResult) -> OrientationResult:
    """Analyze human facing direction separately from body tilt.

    The previous implementation converted raw shoulder separation directly into
    a degree value. That made a perfectly front-facing person report a large
    yaw simply because the shoulders were wide. Yaw is now derived from the
    nose's lateral offset relative to the shoulder span, while the torso vector
    is retained only for tilt estimation.
    """
    key_points = [
        LI.LEFT_SHOULDER,
        LI.RIGHT_SHOULDER,
        LI.LEFT_HIP,
        LI.RIGHT_HIP,
        LI.NOSE,
    ]
    vis_scores = [pose.landmarks[int(idx)].visibility for idx in key_points]
    avg_vis = sum(vis_scores) / len(vis_scores)
    if avg_vis < 0.3:
        return OrientationResult(
            facing=FacingDirection.FRONT,
            tilt=TiltDirection.UPRIGHT,
            facing_angle=0.0,
            tilt_angle=0.0,
            confidence=avg_vis,
            detail="关键点可见度过低, 无法可靠判断朝向",
        )

    ls_x, _ = pose.get_normalized(LI.LEFT_SHOULDER)
    rs_x, _ = pose.get_normalized(LI.RIGHT_SHOULDER)
    nose_x, _ = pose.get_normalized(LI.NOSE)
    shoulder_cx = (ls_x + rs_x) / 2.0
    shoulder_span = max(abs(rs_x - ls_x), 1e-3)
    nose_offset = nose_x - shoulder_cx

    # Positive = face turned toward image right; negative = image left.
    # A centered nose is the key condition for a straight-on portrait.
    yaw_ratio = float(np.clip(nose_offset / (0.5 * shoulder_span), -1.0, 1.0))
    facing_angle = math.degrees(math.atan(yaw_ratio))
    if abs(facing_angle) < 5.0:
        facing_angle = 0.0

    # Nose visibility provides the front-facing evidence. When it is absent or
    # the lateral offset is large, fall back to the existing sign heuristic.
    nose_visible = pose.landmarks[int(LI.NOSE)].visibility > 0.4
    shoulder_dx = ls_x - rs_x
    if nose_visible and abs(facing_angle) < 12.0:
        is_front = True
    else:
        is_front = (shoulder_dx * nose_offset) > 0 or abs(nose_offset) < 0.02

    abs_angle = abs(facing_angle)
    if abs_angle < 12:
        facing = FacingDirection.FRONT if is_front else FacingDirection.BACK
    elif abs_angle < 45:
        if is_front:
            facing = FacingDirection.FRONT_RIGHT if facing_angle > 0 else FacingDirection.FRONT_LEFT
        else:
            facing = FacingDirection.BACK_RIGHT if facing_angle > 0 else FacingDirection.BACK_LEFT
    else:
        facing = FacingDirection.RIGHT if facing_angle > 0 else FacingDirection.LEFT

    # Torso geometry is used only for body tilt, not yaw.
    sc = _shoulder_center(pose)
    hc = _hip_center(pose)
    trunk_vec = sc - hc
    trunk_len = np.linalg.norm(trunk_vec)

    if trunk_len > 0.01:
        tilt_x = trunk_vec[0]
        vertical = np.array([0.0, -1.0])
        cos_angle = float(np.clip(np.dot(trunk_vec / trunk_len, vertical), -1.0, 1.0))
        tilt_angle_val = math.degrees(math.acos(cos_angle))

        if tilt_angle_val < 10:
            tilt = TiltDirection.UPRIGHT
            tilt_angle = 0.0
        elif abs(tilt_x) > abs(trunk_vec[1] + 1):
            tilt = TiltDirection.LEANING_RIGHT if tilt_x > 0 else TiltDirection.LEANING_LEFT
            tilt_angle = tilt_angle_val
        elif trunk_vec[1] > -0.05:
            tilt = TiltDirection.LEANING_FORWARD
            tilt_angle = tilt_angle_val
        else:
            tilt = TiltDirection.UPRIGHT
            tilt_angle = 0.0
    else:
        tilt = TiltDirection.UPRIGHT
        tilt_angle = 0.0

    detail_parts = [f"朝向: {facing.value}"]
    if abs_angle > 5:
        detail_parts.append(f"旋转约{abs_angle:.0f}°")
    if tilt != TiltDirection.UPRIGHT:
        detail_parts.append(f"姿态: {tilt.value}")

    confidence = min(avg_vis, 1.0)
    if abs(facing_angle) < 5.0:
        confidence = min(1.0, confidence + 0.05)
    return OrientationResult(
        facing=facing,
        tilt=tilt,
        facing_angle=round(facing_angle, 1),
        tilt_angle=round(tilt_angle, 1),
        confidence=confidence,
        detail=", ".join(detail_parts),
    )
