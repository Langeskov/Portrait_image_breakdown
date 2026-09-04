"""
构图分析模块

分析图像构图特征:
- 三分法(人物位置是否在黄金分割点)
- 对称性
- 视觉重心
- 引导线/对角线
- 留白方向
- 头部空间(headroom)
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from core.pose_detector import PoseResult, LandmarkIndex as LI


class CompositionType(Enum):
    RULE_OF_THIRDS = "三分法"
    CENTER = "居中构图"
    SYMMETRY = "对称构图"
    DIAGONAL = "对角线构图"
    FRAME_WITHIN_FRAME = "框中框"
    LEADING_LINES = "引导线"
    GOLDEN_RATIO = "黄金比例"
    OFF_CENTER = "偏离中心"
    UNKNOWN = "未知"


@dataclasses.dataclass
class CompositionResult:
    primary_type: CompositionType       # 主要构图类型
    subject_position: tuple[float, float]  # 主体位置 (x, y) 归一化 [0,1]
    thirds_alignment: float             # 三分法对齐度 [0, 1], 越高越符合
    symmetry_score: float               # 对称性 [0, 1]
    headroom: float                     # 头部上方空间比例
    lead_space: str                     # 面朝方向的留白 ("left"/"right"/"balanced")
    visual_weight: tuple[float, float]  # 视觉重心位置
    balance_score: float                # 画面平衡度 [0, 1]
    suggestions: list[str]              # 构图改进建议
    detail: str

    @property
    def summary(self) -> str:
        return f"{self.primary_type.value}, 对齐度: {self.thirds_alignment:.0%}"


def _thirds_distance(x: float, y: float) -> float:
    """
    计算点到最近三分法交叉点的距离(归一化)

    三分法交叉点: (1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)
    返回: 0=完美在交叉点上, 1=完全偏离
    """
    thirds = [1/3, 2/3]
    min_dist = float('inf')
    for tx in thirds:
        for ty in thirds:
            dist = math.sqrt((x - tx)**2 + (y - ty)**2)
            min_dist = min(min_dist, dist)
    # 最大可能距离 ≈ sqrt(2)/3 ≈ 0.471
    return min(min_dist / 0.471, 1.0)


import math


def _edge_symmetry(image: np.ndarray) -> float:
    """
    计算图像边缘对称性(简化版)

    将图像左右翻转, 计算与原图的相似度
    """
    h, w = image.shape[:2]
    # 缩小以加速
    small = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float)
    flipped = np.fliplr(gray)

    # 归一化差异
    diff = np.abs(gray - flipped)
    max_diff = gray.max() - gray.min() + 1
    symmetry = 1.0 - (diff.mean() / max_diff)
    return float(np.clip(symmetry, 0, 1))


def _detect_diagonal(image: np.ndarray) -> float:
    """
    检测图像中的对角线强度

    使用霍夫线变换检测接近45度的线条
    """
    small = cv2.resize(image, (160, 160))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)

    if lines is None:
        return 0.0

    diagonal_count = 0
    total = len(lines)
    for line in lines:
        # OpenCV 5.x HoughLinesP returns shape (N, 1, 4)
        pts = line[0] if line.ndim == 2 else line
        x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        # 接近45度或135度
        if 30 < angle < 60 or 120 < angle < 150:
            diagonal_count += 1

    return diagonal_count / max(total, 1)


def analyze_composition(
    image: np.ndarray,
    pose: Optional[PoseResult] = None,
) -> CompositionResult:
    """
    分析图像构图

    参数:
        image: BGR图像
        pose: 可选的骨架检测结果(用于人物位置分析)

    返回:
        CompositionResult
    """
    h, w = image.shape[:2]
    suggestions: list[str] = []

    # ── 主体位置 ──
    if pose is not None:
        # 人物中心(归一化)
        all_pts = pose.all_normalized
        visible_mask = np.array([lm.visibility > 0.4 for lm in pose.landmarks])
        if visible_mask.any():
            subject_x = float(all_pts[visible_mask, 0].mean())
            subject_y = float(all_pts[visible_mask, 1].mean())
        else:
            subject_x, subject_y = 0.5, 0.5
    else:
        # 没有骨架信息, 用图像重心替代
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 计算加权质心
        total = gray.sum()
        if total > 0:
            ys, xs = np.mgrid[0:h, 0:w]
            subject_x = float((xs * gray).sum() / total) / w
            subject_y = float((ys * gray).sum() / total) / h
        else:
            subject_x, subject_y = 0.5, 0.5

    # ── 三分法对齐度 ──
    thirds_raw = _thirds_distance(subject_x, subject_y)
    thirds_alignment = 1.0 - thirds_raw  # 转换为: 越高越符合

    if thirds_alignment > 0.7:
        primary_type = CompositionType.RULE_OF_THIRDS
    elif abs(subject_x - 0.5) < 0.1 and abs(subject_y - 0.5) < 0.1:
        primary_type = CompositionType.CENTER
    else:
        primary_type = CompositionType.OFF_CENTER

    # ── 对称性 ──
    symmetry_score = _edge_symmetry(image)

    if symmetry_score > 0.7:
        primary_type = CompositionType.SYMMETRY

    # ── 对角线 ──
    diagonal_score = _detect_diagonal(image)
    if diagonal_score > 0.3 and primary_type == CompositionType.OFF_CENTER:
        primary_type = CompositionType.DIAGONAL

    # ── 头部空间 ──
    if pose is not None and pose.is_visible(LI.NOSE):
        nose_y = pose.landmarks[LI.NOSE].world_y
        headroom = nose_y  # 鼻子上方的空间比例
    else:
        headroom = subject_y

    # ── 留白方向 ──
    if pose is not None:
        # 简化: 看鼻子相对肩部中心的偏移判断朝向
        nose_x = pose.landmarks[LI.NOSE].world_x
        ls_x = pose.landmarks[LI.LEFT_SHOULDER].world_x
        rs_x = pose.landmarks[LI.RIGHT_SHOULDER].world_x
        shoulder_cx = (ls_x + rs_x) / 2

        # 如果人物面朝右(subject_x偏左), 右侧应有留白
        if subject_x < 0.4:
            lead_space = "right"
        elif subject_x > 0.6:
            lead_space = "left"
        else:
            lead_space = "balanced"
    else:
        lead_space = "balanced"

    # ── 视觉重心 ──
    # 简化: 使用图像亮度加权中心
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    # 上半部分通常权重更大(人脸在上方)
    weight_map = gray.copy()
    ys = np.linspace(0, 1, h).reshape(-1, 1)
    weight_map *= (1 - ys * 0.3)  # 上方权重略高

    total_w = weight_map.sum()
    if total_w > 0:
        xs = np.linspace(0, 1, w).reshape(1, -1)
        vw_x = float((xs * weight_map).sum() / total_w)
        vw_y = float((ys * weight_map).sum() / total_w)
    else:
        vw_x, vw_y = 0.5, 0.5

    # ── 画面平衡度 ──
    # 左右两半的视觉重量对比
    left_half = gray[:, :w//2].sum()
    right_half = gray[:, w//2:].sum()
    lr_balance = 1.0 - abs(left_half - right_half) / (left_half + right_half + 1)

    top_half = gray[:h//2, :].sum()
    bottom_half = gray[h//2:, :].sum()
    tb_balance = 1.0 - abs(top_half - bottom_half) / (top_half + bottom_half + 1)

    balance_score = (lr_balance + tb_balance) / 2

    # ── 生成建议 ──
    if thirds_alignment < 0.5:
        suggestions.append("考虑将人物放在三分法交叉点上, 增强视觉张力")

    if headroom < 0.1:
        suggestions.append("头部空间不足, 可以稍微下移镜头或后退一步")
    elif headroom > 0.35:
        suggestions.append("头部空间过多, 画面重心偏低")

    if symmetry_score < 0.4 and primary_type != CompositionType.SYMMETRY:
        suggestions.append("如果追求对称感, 可以调整拍摄角度")

    if balance_score < 0.6:
        suggestions.append("画面左右不平衡, 考虑调整人物位置或添加视觉元素平衡")

    if not suggestions:
        suggestions.append("构图良好, 可以尝试不同角度拍摄获取更多变化")

    # ── 结果 ──
    detail_parts = [
        f"构图: {primary_type.value}",
        f"三分法对齐: {thirds_alignment:.0%}",
        f"对称性: {symmetry_score:.0%}",
        f"头部空间: {headroom:.0%}",
    ]

    return CompositionResult(
        primary_type=primary_type,
        subject_position=(round(subject_x, 3), round(subject_y, 3)),
        thirds_alignment=round(thirds_alignment, 3),
        symmetry_score=round(symmetry_score, 3),
        headroom=round(headroom, 3),
        lead_space=lead_space,
        visual_weight=(round(vw_x, 3), round(vw_y, 3)),
        balance_score=round(balance_score, 3),
        suggestions=suggestions,
        detail=", ".join(detail_parts),
    )
