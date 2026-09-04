"""
身体朝向分析模块

基于肩部/髋部关键点的相对位置, 判断人物朝向:
- 正面(front) / 背面(back)
- 左侧(left) / 右侧(right)
- 以及俯仰角(前倾/后仰)
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
    facing: FacingDirection       # 水平朝向
    tilt: TiltDirection           # 俯仰/倾斜
    facing_angle: float           # 水平旋转角度 (°), 0=正对, 正=偏右
    tilt_angle: float             # 俯仰角度 (°), 正=前倾
    confidence: float             # 判断置信度 [0, 1]
    detail: str                   # 中文描述

    @property
    def summary(self) -> str:
        parts = [self.facing.value]
        if self.tilt != TiltDirection.UPRIGHT:
            parts.append(self.tilt.value)
        return " ".join(parts)


def _shoulder_center(pose: PoseResult) -> np.ndarray:
    """肩部中心点(归一化坐标)"""
    ls = np.array(pose.get_normalized(LI.LEFT_SHOULDER))
    rs = np.array(pose.get_normalized(LI.RIGHT_SHOULDER))
    return (ls + rs) / 2


def _hip_center(pose: PoseResult) -> np.ndarray:
    """髋部中心点"""
    lh = np.array(pose.get_normalized(LI.LEFT_HIP))
    rh = np.array(pose.get_normalized(LI.RIGHT_HIP))
    return (lh + rh) / 2


def _shoulder_width(pose: PoseResult) -> float:
    """肩宽(归一化像素距离)"""
    ls = np.array(pose.get_normalized(LI.LEFT_SHOULDER))
    rs = np.array(pose.get_normalized(LI.RIGHT_SHOULDER))
    return float(np.linalg.norm(ls - rs))


def _hip_width(pose: PoseResult) -> float:
    """髋宽"""
    lh = np.array(pose.get_normalized(LI.LEFT_HIP))
    rh = np.array(pose.get_normalized(LI.RIGHT_HIP))
    return float(np.linalg.norm(lh - rh))


def analyze_orientation(pose: PoseResult) -> OrientationResult:
    """
    分析人体朝向

    算法:
    1. 肩部左右不对称性 → 水平旋转角度
       - 左肩x > 右肩x → 人物面朝右(相机看到左侧)
       - 反之亦然
    2. 鼻子相对肩部中心的偏移 → 正/背面判断
    3. 肩部中心与髋部中心的垂直对齐 → 前倾/后仰

    参数:
        pose: 骨架检测结果

    返回:
        OrientationResult
    """
    # ── 可见性检查 ──
    key_points = [LI.LEFT_SHOULDER, LI.RIGHT_SHOULDER,
                  LI.LEFT_HIP, LI.RIGHT_HIP, LI.NOSE]
    vis_scores = []
    for idx in key_points:
        lm = pose.landmarks[int(idx)]
        vis_scores.append(lm.visibility)

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

    # ── 水平朝向分析 ──
    ls_x, ls_y = pose.get_normalized(LI.LEFT_SHOULDER)
    rs_x, rs_y = pose.get_normalized(LI.RIGHT_SHOULDER)

    # 肩部水平差: 左肩x - 右肩x
    # 如果人物面向右, 相机看到左侧, 左肩在画面右侧, 右肩在画面左侧
    # 此时 ls_x < rs_x (左肩更靠右)
    shoulder_dx = ls_x - rs_x  # 负值=面朝右, 正值=面朝左

    # 肩宽(用于归一化)
    sw = abs(shoulder_dx) if abs(shoulder_dx) > 0.001 else 0.001

    # 鼻子相对肩部中心的偏移
    nose_x, nose_y = pose.get_normalized(LI.NOSE)
    shoulder_cx = (ls_x + rs_x) / 2
    nose_offset_x = nose_x - shoulder_cx  # 正=鼻子偏右

    # 用鼻子偏移来判断正/背面:
    # 如果鼻子在肩部中心的"前面"(面向方向), 则是正面
    # 面朝右时 shoulder_dx<0, 如果 nose_offset_x<0(鼻子也偏右), 则是正面
    # 面朝左时 shoulder_dx>0, 如果 nose_offset_x>0(鼻子也偏左), 则是正面
    is_front = (shoulder_dx * nose_offset_x) > 0 or abs(nose_offset_x) < 0.02

    # 旋转角度: arcsin(shoulder_dx / 肩宽实际长度)
    # 肩宽在归一化空间中, 需要考虑图像宽高比
    # 简化: 直接用 shoulder_dx 的绝对值作为角度代理
    # shoulder_dx ≈ 0 → 正面, |shoulder_dx| ≈ 0.15 → 侧面
    angle_raw = math.degrees(math.atan2(abs(shoulder_dx), 0.15))
    facing_angle = angle_raw if shoulder_dx < 0 else -angle_raw

    # 判断主方向
    abs_angle = abs(facing_angle)
    if abs_angle < 15:
        if is_front:
            facing = FacingDirection.FRONT
        else:
            facing = FacingDirection.BACK
    elif abs_angle < 45:
        if is_front:
            facing = FacingDirection.FRONT_RIGHT if facing_angle < 0 else FacingDirection.FRONT_LEFT
        else:
            facing = FacingDirection.BACK_RIGHT if facing_angle < 0 else FacingDirection.BACK_LEFT
    else:
        facing = FacingDirection.RIGHT if facing_angle < 0 else FacingDirection.LEFT

    # ── 俯仰/倾斜分析 ──
    sc = _shoulder_center(pose)
    hc = _hip_center(pose)

    # 躯干向量: 从髋部指向肩部
    trunk_vec = sc - hc  # [dx, dy] in normalized space
    trunk_len = np.linalg.norm(trunk_vec)

    if trunk_len > 0.01:
        # 垂直参考: [0, -1] (向上)
        # trunk_vec 的 x 分量 → 左右倾斜
        # trunk_vec 的 y 分量 → 前后倾(因为y轴向下)
        tilt_x = trunk_vec[0]  # 正=向右倾
        tilt_y = trunk_vec[1]  # 正=向下(即前倾, 因为人站着时躯干向上, y负)

        # 俯仰角: 躯干与垂直方向的夹角
        vertical = np.array([0, -1])
        cos_angle = np.dot(trunk_vec / trunk_len, vertical)
        cos_angle = np.clip(cos_angle, -1, 1)
        tilt_angle_val = math.degrees(math.acos(cos_angle))

        # 判断倾斜方向
        if tilt_angle_val < 10:
            tilt = TiltDirection.UPRIGHT
            tilt_angle = 0.0
        elif abs(tilt_x) > abs(tilt_y + 1):  # 水平分量更大
            tilt = TiltDirection.LEANING_RIGHT if tilt_x > 0 else TiltDirection.LEANING_LEFT
            tilt_angle = tilt_angle_val
        else:
            # 看 y 分量: trunk_vec[1] 正=躯干向上更平=后仰, 负=前倾
            # 但通常人直立时 trunk_vec[1] < 0 (肩在髋上方)
            # 如果 |trunk_vec[1]| 变小 → 躯干更平 → 前倾
            if trunk_vec[1] > -0.05:  # 肩部没有明显高于髋部
                tilt = TiltDirection.LEANING_FORWARD
                tilt_angle = tilt_angle_val
            else:
                tilt = TiltDirection.UPRIGHT
                tilt_angle = 0.0
    else:
        tilt = TiltDirection.UPRIGHT
        tilt_angle = 0.0

    # ── 构造描述 ──
    detail_parts = [f"朝向: {facing.value}"]
    if abs_angle > 5:
        detail_parts.append(f"旋转约{abs_angle:.0f}°")
    if tilt != TiltDirection.UPRIGHT:
        detail_parts.append(f"姿态: {tilt.value}")
    detail = ", ".join(detail_parts)

    return OrientationResult(
        facing=facing,
        tilt=tilt,
        facing_angle=facing_angle,
        tilt_angle=tilt_angle,
        confidence=min(avg_vis, 1.0),
        detail=detail,
    )
