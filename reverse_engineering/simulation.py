"""
反向验证引擎 - ReverseSimulationEngine

使用 CameraModel 投影进行参数优化, 输出多个候选解。
"""
from __future__ import annotations
import numpy as np
from reverse_engineering.data_types import EstimatedValue
from reverse_engineering.geometry import (
    CameraIntrinsics, CameraModel, CameraExtrinsics, PoseCandidate,
    REF_PERSON_HEIGHT, REF_SHOULDER_WIDTH,
)
import math


def _project_subject_scale(focal_px: float, distance: float, ref_height: float,
                           image_h: int) -> float:
    """预测主体在图像中的归一化高度"""
    if distance < 0.1: return 1.0
    projected_px = ref_height * focal_px / distance
    return projected_px / image_h


def _perspective_loss(focal_mm: float, observed_strength: float) -> float:
    """透视一致性损失"""
    # 广角(小焦距) → 强透视, 长焦(大焦距) → 弱透视
    predicted = max(0, 1.0 - (focal_mm - 18) / 180)
    return (predicted - observed_strength) ** 2


def _composition_loss(pred_cx: float, pred_cy: float, obs_cx: float, obs_cy: float) -> float:
    """构图一致性损失"""
    return ((pred_cx - obs_cx) ** 2 + (pred_cy - obs_cy) ** 2)


def optimize_parameters(
    image_w: int, image_h: int,
    subject_scale: float,
    subject_position: tuple[float, float],
    perspective_strength: float,
    pose_keypoints: np.ndarray = None,
    num_candidates: int = 5,
) -> list[PoseCandidate]:
    """
    通过网格搜索优化相机参数

    输出多个候选解(而非单一最优值), 诚实表达多解性。
    """
    candidates = []

    # 搜索空间
    focal_range = np.linspace(24, 135, 12)  # 35mm等效焦距
    dist_range = np.linspace(1.0, 15.0, 10)  # 距离(m)
    height_range = np.linspace(0.8, 2.0, 5)  # 高度(m)

    obs_cx, obs_cy = subject_position

    for focal_35mm in focal_range:
        # 从35mm等效转换为像素焦距
        focal_px = focal_35mm * image_w / 36.0

        for distance in dist_range:
            # 预测主体比例
            pred_scale = _project_subject_scale(focal_px, distance,
                                                 REF_PERSON_HEIGHT, image_h)
            scale_error = (pred_scale - subject_scale) ** 2

            # 透视损失
            persp_error = _perspective_loss(focal_35mm, perspective_strength)

            # 距离合理性
            dist_penalty = 0.0 if 0.5 < distance < 20 else 1.0

            total_loss = scale_error * 4.0 + persp_error * 2.0 + dist_penalty * 1.0

            for height in height_range:
                # 俯仰角估计
                pitch = math.degrees(math.atan2(height - REF_PERSON_HEIGHT * 0.45, distance))
                pitch = max(-30, min(30, pitch))

                intrinsics = CameraIntrinsics.from_focal_mm(focal_35mm, image_w, image_h)
                extrinsics = CameraExtrinsics(
                    rvec=np.zeros(3), tvec=np.array([0, 0, -distance]),
                    position=np.array([0, height, 0]),
                    pitch=pitch, yaw=0, roll=0)

                score = max(0, 1.0 - total_loss)

                candidates.append(PoseCandidate(
                    intrinsics=intrinsics, extrinsics=extrinsics,
                    distance=round(distance, 2), height=round(height, 2),
                    focal_equiv_35mm=round(focal_35mm),
                    score=round(score, 3),
                    losses={"scale": round(scale_error, 4),
                            "perspective": round(persp_error, 4)}))

    # 按得分排序, 返回 top N
    candidates.sort(key=lambda c: c.score, reverse=True)

    # 去重(相似参数的候选)
    unique = []
    for c in candidates:
        is_dup = False
        for u in unique:
            if (abs(c.focal_equiv_35mm - u.focal_equiv_35mm) < 10 and
                abs(c.distance - u.distance) < 0.5):
                is_dup = True
                break
        if not is_dup:
            unique.append(c)
        if len(unique) >= num_candidates:
            break

    return unique
