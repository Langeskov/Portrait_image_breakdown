"""
动作类别识别模块

基于关节角度和关键点位置关系, 将当前姿态分类为:
站立、行走、跑步、跳跃、蹲坐、躺卧、手臂动作等。
使用规则引擎, 无需训练数据。
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import Optional

import numpy as np

from core.pose_detector import PoseResult, LandmarkIndex as LI


class ActionCategory(Enum):
    STANDING = "站立"
    WALKING = "行走"
    RUNNING = "跑步"
    JUMPING = "跳跃"
    SQUATTING = "蹲/半蹲"
    SITTING = "坐姿"
    LYING = "躺卧"
    ARMS_RAISED = "举手/举臂"
    WAVING = "挥手"
    POINTING = "指向"
    STRETCHING = "伸展"
    BALANCING = "平衡/单脚"
    BOWING = "鞠躬/弯腰"
    FIGHTING_STANCE = "格斗姿态"
    DANCING = "舞蹈姿态"
    UNKNOWN = "未知"


@dataclasses.dataclass
class ActionResult:
    category: ActionCategory
    confidence: float          # [0, 1]
    sub_description: str       # 细节描述
    joint_angles: dict[str, float]  # 关键关节角度
    features: dict[str, float]      # 特征值(用于后续建议)

    @property
    def summary(self) -> str:
        return f"{self.category.value} ({self.confidence:.0%})"


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的角度(度)"""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def _joint_angle(pose: PoseResult, a: LI, b: LI, c: LI) -> float:
    """
    计算关节角度: ∠ABC, 其中B是关节点

    参数:
        a, b, c: 三个关键点索引, b为关节顶点
    返回:
        角度(度), 180°=完全伸直
    """
    ax, ay = pose.get_normalized(a)
    bx, by = pose.get_normalized(b)
    cx, cy = pose.get_normalized(c)

    v1 = np.array([ax - bx, ay - by])
    v2 = np.array([cx - bx, cy - by])

    return _angle_between(v1, v2)


def _limb_angle_vertical(pose: PoseResult, upper: LI, lower: LI) -> float:
    """肢体与垂直方向的夹角(度)"""
    ux, uy = pose.get_normalized(upper)
    lx, ly = pose.get_normalized(lower)
    vec = np.array([lx - ux, ly - uy])  # 指向远端
    vertical = np.array([0, 1])  # 向下
    return _angle_between(vec, vertical)


def classify_action(pose: PoseResult) -> ActionResult:
    """
    识别当前姿态的动作类别

    参数:
        pose: 骨架检测结果

    返回:
        ActionResult
    """
    # ── 计算关键关节角度 ──
    angles: dict[str, float] = {}

    # 膝盖角度 (大腿-小腿)
    angles["left_knee"] = _joint_angle(pose, LI.LEFT_HIP, LI.LEFT_KNEE, LI.LEFT_ANKLE)
    angles["right_knee"] = _joint_angle(pose, LI.RIGHT_HIP, LI.RIGHT_KNEE, LI.RIGHT_ANKLE)

    # 肘部角度 (上臂-前臂)
    angles["left_elbow"] = _joint_angle(pose, LI.LEFT_SHOULDER, LI.LEFT_ELBOW, LI.LEFT_WRIST)
    angles["right_elbow"] = _joint_angle(pose, LI.RIGHT_SHOULDER, LI.RIGHT_ELBOW, LI.RIGHT_WRIST)

    # 髋部角度 (躯干-大腿)
    angles["left_hip"] = _joint_angle(pose, LI.LEFT_SHOULDER, LI.LEFT_HIP, LI.LEFT_KNEE)
    angles["right_hip"] = _joint_angle(pose, LI.RIGHT_SHOULDER, LI.RIGHT_HIP, LI.RIGHT_KNEE)

    # 肩部角度 (躯干-上臂)
    angles["left_shoulder"] = _joint_angle(pose, LI.LEFT_HIP, LI.LEFT_SHOULDER, LI.LEFT_ELBOW)
    angles["right_shoulder"] = _joint_angle(pose, LI.RIGHT_HIP, LI.RIGHT_SHOULDER, LI.RIGHT_ELBOW)

    # ── 计算位置特征 ──
    features: dict[str, float] = {}

    # 身体中心高度(鼻子y, 0=顶部, 1=底部)
    nose_x, nose_y = pose.get_normalized(LI.NOSE)
    features["body_height"] = nose_y

    # 肩部中心y
    ls_y = pose.landmarks[LI.LEFT_SHOULDER].world_y
    rs_y = pose.landmarks[LI.RIGHT_SHOULDER].world_y
    features["shoulder_y"] = (ls_y + rs_y) / 2

    # 髋部中心y
    lh_y = pose.landmarks[LI.LEFT_HIP].world_y
    rh_y = pose.landmarks[LI.RIGHT_HIP].world_y
    features["hip_y"] = (lh_y + rh_y) / 2

    # 躯干长度(肩到髋的距离)
    features["torso_length"] = abs(features["shoulder_y"] - features["hip_y"])

    # 脚踝高度(越高说明脚离地)
    la_y = pose.landmarks[LI.LEFT_ANKLE].world_y
    ra_y = pose.landmarks[LI.RIGHT_ANKLE].world_y
    features["ankle_y_avg"] = (la_y + ra_y) / 2

    # 脚踝高度差(单脚离地)
    features["ankle_y_diff"] = abs(la_y - ra_y)

    # 手腕高度(越高说明手举得越高)
    lw_y = pose.landmarks[LI.LEFT_WRIST].world_y
    rw_y = pose.landmarks[LI.RIGHT_WRIST].world_y
    features["wrist_y_avg"] = (lw_y + rw_y) / 2
    features["wrist_y_min"] = min(lw_y, rw_y)  # 最高点

    # 手腕是否高于肩膀
    features["hands_above_shoulders"] = float(
        features["wrist_y_min"] < features["shoulder_y"] - 0.05
    )

    # 双脚分开程度
    la_x = pose.landmarks[LI.LEFT_ANKLE].world_x
    ra_x = pose.landmarks[LI.RIGHT_ANKLE].world_x
    features["stance_width"] = abs(la_x - ra_x)

    # 膝盖平均角度
    features["knee_angle_avg"] = (angles["left_knee"] + angles["right_knee"]) / 2

    # 膝盖角度差(双腿不对称)
    features["knee_angle_diff"] = abs(angles["left_knee"] - angles["right_knee"])

    # ── 分类规则引擎 ──
    scores: dict[ActionCategory, float] = {cat: 0.0 for cat in ActionCategory}

    # 1. 躺卧: 躯干接近水平
    if features["torso_length"] < 0.15 and features["body_height"] > 0.5:
        scores[ActionCategory.LYING] += 0.8

    # 2. 坐姿: 膝盖弯曲且髋部角度小, 但膝盖不极度弯曲
    knee_avg = features["knee_angle_avg"]
    hip_avg = (angles["left_hip"] + angles["right_hip"]) / 2
    if 60 < knee_avg < 120 and hip_avg < 120:
        scores[ActionCategory.SITTING] += 0.7
        # 坐姿时膝盖通常在90度左右
        if 70 < knee_avg < 110:
            scores[ActionCategory.SITTING] += 0.2

    # 3. 蹲/半蹲: 膝盖深度弯曲
    if knee_avg < 80 and features["hip_y"] > 0.6:
        scores[ActionCategory.SQUATTING] += 0.7
        if knee_avg < 60:
            scores[ActionCategory.SQUATTING] += 0.2

    # 4. 跳跃: 脚踝离地(高于正常站立位置)
    if features["ankle_y_avg"] < 0.75 and features["ankle_y_diff"] < 0.08:
        # 脚踝位置较高且双脚差不多高 → 跳跃
        scores[ActionCategory.JUMPING] += 0.6
        if features["ankle_y_avg"] < 0.7:
            scores[ActionCategory.JUMPING] += 0.2

    # 5. 平衡/单脚: 双脚高度差大
    if features["ankle_y_diff"] > 0.1:
        scores[ActionCategory.BALANCING] += 0.6
        if features["ankle_y_diff"] > 0.15:
            scores[ActionCategory.BALANCING] += 0.2

    # 6. 举手/举臂: 手腕高于肩膀
    if features["hands_above_shoulders"]:
        scores[ActionCategory.ARMS_RAISED] += 0.5
        # 双手都高
        lw_above = lw_y < features["shoulder_y"] - 0.05
        rw_above = rw_y < features["shoulder_y"] - 0.5
        if lw_above and rw_above:
            scores[ActionCategory.ARMS_RAISED] += 0.3

    # 7. 挥手: 一只手高一只手低
    if features["wrist_y_min"] < features["shoulder_y"] - 0.05:
        wrist_diff = abs(lw_y - rw_y)
        if wrist_diff > 0.1:
            scores[ActionCategory.WAVING] += 0.5

    # 8. 指向: 一只手臂伸直, 另一只放松
    if angles["left_elbow"] > 150 or angles["right_elbow"] > 150:
        scores[ActionCategory.POINTING] += 0.4

    # 9. 鞠躬/弯腰: 躯干前倾, 髋部角度大
    if hip_avg > 120 and features["shoulder_y"] > features["hip_y"] - 0.05:
        scores[ActionCategory.BOWING] += 0.6

    # 10. 格斗姿态: 膝盖微弯, 双手举起, 宽站距
    if (70 < knee_avg < 120 and features["hands_above_shoulders"]
            and features["stance_width"] > 0.1):
        scores[ActionCategory.FIGHTING_STANCE] += 0.7

    # 11. 行走: 膝盖角度不对称, 站立位置
    if features["knee_angle_diff"] > 15 and knee_avg > 100:
        scores[ActionCategory.WALKING] += 0.5
        if features["stance_width"] > 0.05:
            scores[ActionCategory.WALKING] += 0.2

    # 12. 跑步: 类似行走但更动态(膝盖弯曲更大, 身体前倾)
    if features["knee_angle_diff"] > 20 and knee_avg < 120:
        scores[ActionCategory.RUNNING] += 0.4

    # 13. 伸展: 手臂和腿部都接近伸直
    if (angles["left_elbow"] > 150 and angles["right_elbow"] > 150
            and angles["left_knee"] > 150 and angles["right_knee"] > 150):
        scores[ActionCategory.STRETCHING] += 0.5

    # 14. 站立: 默认, 膝盖接近伸直
    if knee_avg > 150 and not features["hands_above_shoulders"]:
        scores[ActionCategory.STANDING] += 0.6
        if features["stance_width"] < 0.15:
            scores[ActionCategory.STANDING] += 0.2

    # 15. 舞蹈: 多个特征组合(单脚+手臂动作)
    if features["ankle_y_diff"] > 0.08 and features["hands_above_shoulders"]:
        scores[ActionCategory.DANCING] += 0.5

    # ── 选择最高分 ──
    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    # 如果没有明显特征, 归为站立
    if best_score < 0.3:
        best_cat = ActionCategory.STANDING
        best_score = 0.4

    # 归一化置信度
    confidence = min(best_score, 1.0)

    # 生成描述
    desc_parts = []
    if features["hands_above_shoulders"]:
        desc_parts.append("手臂上举")
    if features["stance_width"] > 0.15:
        desc_parts.append("宽站距")
    elif features["stance_width"] < 0.05:
        desc_parts.append("并脚")
    if features["knee_angle_diff"] > 15:
        desc_parts.append("双腿不对称")

    sub_desc = ", ".join(desc_parts) if desc_parts else "标准姿态"

    return ActionResult(
        category=best_cat,
        confidence=confidence,
        sub_description=sub_desc,
        joint_angles=angles,
        features=features,
    )
