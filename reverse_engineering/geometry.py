"""
几何核心 - CameraModel / Projection / PoseSolver

使用标准针孔相机模型和 OpenCV 几何工具, 替代之前的 magic formula。
复用 OpenCV solvePnP / projectPoints 等成熟实现。
"""
from __future__ import annotations
import math
import dataclasses
from typing import Optional
import cv2
import numpy as np

# 标准人体参考尺寸 (meters)
REF_PERSON_HEIGHT = 1.70
REF_SHOULDER_WIDTH = 0.45
REF_HEAD_HEIGHT = 0.25

@dataclasses.dataclass
class ImagePoint:
    """图像像素坐标"""
    x: float
    y: float
    confidence: float = 1.0

@dataclasses.dataclass
class NormalizedPoint:
    """归一化图像坐标 [0, 1]"""
    x: float
    y: float

@dataclasses.dataclass
class WorldPoint:
    """世界坐标 (米), 仅在有几何约束时使用"""
    x: float
    y: float
    z: float
    confidence: float = 0.0

@dataclasses.dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float          # 焦距 x (pixels)
    fy: float          # 焦距 y (pixels)
    cx: float          # 主点 x
    cy: float          # 主点 y
    width: int = 0
    height: int = 0
    sensor_width_mm: float = 36.0   # 全画幅默认
    sensor_height_mm: float = 24.0

    @property
    def fov_x(self) -> float:
        return 2 * math.degrees(math.atan(self.cx / self.fx))

    @property
    def fov_y(self) -> float:
        return 2 * math.degrees(math.atan(self.cy / self.fy))

    @property
    def focal_length_mm(self) -> float:
        """等效35mm焦距"""
        return self.fx * self.sensor_width_mm / self.width if self.width > 0 else 50.0

    def to_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    @classmethod
    def from_focal_mm(cls, focal_mm: float, width: int, height: int,
                      sensor_w: float = 36.0, sensor_h: float = 24.0) -> CameraIntrinsics:
        fx = focal_mm * width / sensor_w
        fy = focal_mm * height / sensor_h
        return cls(fx=fx, fy=fy, cx=width/2, cy=height/2,
                   width=width, height=height,
                   sensor_width_mm=sensor_w, sensor_height_mm=sensor_h)

@dataclasses.dataclass
class CameraExtrinsics:
    """相机外参 (位姿)"""
    rvec: np.ndarray     # 旋转向量 (3,)
    tvec: np.ndarray     # 平移向量 (3,)
    position: np.ndarray # 相机世界位置 (3,)
    pitch: float = 0.0   # 俯仰角 (度)
    yaw: float = 0.0     # 偏航角 (度)
    roll: float = 0.0    # 横滚角 (度)

@dataclasses.dataclass
class ProjectionResult:
    """投影结果"""
    projected_points: np.ndarray   # (N, 2) 投影后的像素坐标
    reproj_error: float            # 重投影误差 (pixels)
    in_front: bool                 # 是否在相机前方

@dataclasses.dataclass
class PoseCandidate:
    """姿态候选解"""
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    distance: float         # 到主体距离 (m)
    height: float           # 相机高度 (m)
    focal_equiv_35mm: float # 等效35mm焦距
    score: float            # 综合得分
    losses: dict            # 各项损失

class CameraModel:
    """针孔相机模型"""

    def __init__(self, intrinsics: CameraIntrinsics, extrinsics: Optional[CameraExtrinsics] = None):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def project_point(self, point_3d: np.ndarray) -> tuple[float, float]:
        """将3D点投影到图像"""
        if self.extrinsics is None:
            return (0.0, 0.0)
        K = self.intrinsics.to_matrix()
        rvec = self.extrinsics.rvec
        tvec = self.extrinsics.tvec
        pts, _ = cv2.projectPoints(point_3d.reshape(1, 3), rvec, tvec, K, None)
        return float(pts[0, 0, 0]), float(pts[0, 0, 1])

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        if self.extrinsics is None:
            return np.zeros((len(points_3d), 2))
        K = self.intrinsics.to_matrix()
        pts, _ = cv2.projectPoints(points_3d, self.extrinsics.rvec,
                                   self.extrinsics.tvec, K, None)
        return pts.reshape(-1, 2)

    def unproject_point(self, px: float, py: float, depth: float = 1.0) -> np.ndarray:
        """将像素坐标反投影到3D"""
        K_inv = np.linalg.inv(self.intrinsics.to_matrix())
        pt = np.array([px, py, 1.0]) * depth
        return K_inv @ pt

class PoseSolver:
    """基于几何约束的姿态求解器"""

    @staticmethod
    def solve_from_vanishing_points(
        vps: list[tuple[float, float]],
        image_w: int, image_h: int,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> Optional[CameraExtrinsics]:
        """
        从消失点求解相机旋转

        原理: 三个正交方向的消失点定义了相机内参和旋转矩阵。
        如果已知内参, 两个消失点即可求解旋转。
        """
        if len(vps) < 2:
            return None

        if intrinsics is None:
            fx = max(image_w, image_h) * 0.8
            intrinsics = CameraIntrinsics(fx=fx, fy=fx, cx=image_w/2, cy=image_h/2,
                                          width=image_w, height=image_h)

        K = intrinsics.to_matrix()
        K_inv = np.linalg.inv(K)

        # 消失点反投影为方向向量
        dirs = []
        for vp_x, vp_y in vps[:3]:
            vp_px = vp_x * image_w
            vp_py = vp_y * image_h
            d = K_inv @ np.array([vp_px, vp_py, 1.0])
            d /= np.linalg.norm(d)
            dirs.append(d)

        if len(dirs) >= 2:
            # 用两个消失点方向构建旋转矩阵
            d1, d2 = dirs[0], dirs[1]
            r1 = d1 / np.linalg.norm(d1)
            r2 = d2 - np.dot(d2, r1) * r1
            r2 /= np.linalg.norm(r2)
            r3 = np.cross(r1, r2)
            R = np.column_stack([r1, r2, r3])

            # 确保是有效旋转矩阵
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R = -R

            rvec, _ = cv2.Rodrigues(R)
            tvec = np.zeros(3)

            # 提取欧拉角
            pitch, yaw, roll = _rotation_to_euler(R)

            position = np.array([0.0, 1.5, 0.0])  # 默认高度

            return CameraExtrinsics(rvec=rvec.flatten(), tvec=tvec,
                                    position=position, pitch=pitch, yaw=yaw, roll=roll)
        return None

    @staticmethod
    def solve_from_body_geometry(
        pose_keypoints: np.ndarray,        # (17, 2) 像素坐标
        image_w: int, image_h: int,
        ref_height: float = REF_PERSON_HEIGHT,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> list[PoseCandidate]:
        """
        从人体关键点几何约束求解相机位姿

        利用:
        - 人体身高约束
        - 肩宽约束
        - 头部尺寸约束
        - 关键点在图像中的分布
        """
        if intrinsics is None:
            fx = max(image_w, image_h) * 0.8
            intrinsics = CameraIntrinsics(fx=fx, fy=fx, cx=image_w/2, cy=image_h/2,
                                          width=image_w, height=image_h)

        candidates = []

        # 获取人体像素尺寸
        visible = pose_keypoints[pose_keypoints[:, 2] > 0.3]
        if len(visible) < 5:
            return candidates

        # 人体包围框像素尺寸
        y_min, y_max = visible[:, 1].min(), visible[:, 1].max()
        x_min, x_max = visible[:, 0].min(), visible[:, 0].max()
        body_h_px = y_max - y_min
        body_w_px = x_max - x_min

        if body_h_px < 10:
            return candidates

        # 肩宽像素 (关键点5=左肩, 6=右肩)
        kp = pose_keypoints
        if kp[5, 2] > 0.3 and kp[6, 2] > 0.3:
            shoulder_px = np.sqrt((kp[6, 0] - kp[5, 0])**2 + (kp[6, 1] - kp[5, 1])**2)
        else:
            shoulder_px = body_w_px * 0.7

        # 从身高估计距离: distance = ref_height * fy / body_h_px
        fy = intrinsics.fy
        fx = intrinsics.fx

        for focal_mult in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            test_fx = fx * focal_mult
            test_fy = fy * focal_mult
            test_intr = CameraIntrinsics(fx=test_fx, fy=test_fy, cx=image_w/2, cy=image_h/2,
                                         width=image_w, height=image_h,
                                         sensor_width_mm=intrinsics.sensor_width_mm,
                                         sensor_height_mm=intrinsics.sensor_height_mm)

            dist_from_height = ref_height * test_fy / body_h_px
            dist_from_shoulder = REF_SHOULDER_WIDTH * test_fx / shoulder_px
            distance = (dist_from_height + dist_from_shoulder) / 2

            # 相机高度估计
            nose_y_px = kp[0, 1] if kp[0, 2] > 0.3 else (kp[5, 1] + kp[6, 1]) / 2
            nose_y_norm = nose_y_px / image_h
            # 平视时鼻子约在0.35~0.45
            height_offset = (0.4 - nose_y_norm) * distance * 0.8
            camera_height = max(0.3, ref_height * 0.5 + height_offset)

            # 俯仰角
            principal_y = (nose_y_px - image_h / 2) / test_fy
            pitch = -math.degrees(math.atan(principal_y))

            # 计算各损失
            # 1. 主体比例误差
            predicted_body_h = ref_height * test_fy / distance
            scale_loss = ((predicted_body_h - body_h_px) / image_h) ** 2

            # 2. 透视一致性
            expected_persp = 1.0 - min(test_intr.focal_length_mm / 200, 1.0)
            persp_loss = 0.0  # 需要外部透视数据

            # 3. 距离合理性
            dist_loss = 0.0 if 0.5 < distance < 30 else 1.0

            total_loss = scale_loss * 3.0 + dist_loss * 2.0
            score = max(0, 1.0 - total_loss)

            equiv_35mm = test_intr.focal_length_mm

            candidates.append(PoseCandidate(
                intrinsics=test_intr,
                extrinsics=CameraExtrinsics(
                    rvec=np.zeros(3), tvec=np.array([0, 0, -distance]),
                    position=np.array([0, camera_height, 0]),
                    pitch=pitch, yaw=0, roll=0),
                distance=round(distance, 2),
                height=round(camera_height, 2),
                focal_equiv_35mm=round(equiv_35mm),
                score=round(score, 3),
                losses={"scale": round(scale_loss, 4), "distance": round(dist_loss, 4)}))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:5]

def _rotation_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵转欧拉角 (pitch, yaw, roll) 度"""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
