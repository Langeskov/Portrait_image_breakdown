"""
骨架检测模块 — 基于 YOLOv26 Pose 的人体姿态估计

YOLOv26 使用 COCO 17关键点格式, 通过检测+姿态估计联合模型完成。
相比 MediaPipe:
- 更高的精度(YOLOv26 架构优势)
- 原生多人支持
- 更好的遮挡处理
- 检测+姿态一体化, 无需单独的人体检测步骤

输出格式与原 MediaPipe 版本兼容(PoseResult 接口不变)。
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ── COCO 17关键点索引(与YOLOv26输出对应) ─────────────────────────────
class LandmarkIndex(IntEnum):
    """
    关键点索引 — 兼容 COCO 17点格式

    YOLOv26 输出17个关键点, 与原 MediaPipe 33点的对应关系:
    - 0~16 直接对应 COCO 格式
    - 17~32 为扩展点(从可见关键点插值计算), 保持接口兼容
    """
    # ── COCO 17关键点(0~16) ──
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # ── 扩展关键点(17~32), 从COCO点插值, 保持接口兼容 ──
    LEFT_EYE_INNER = 17
    RIGHT_EYE_INNER = 18
    LEFT_EYE_OUTER = 19
    RIGHT_EYE_OUTER = 20
    MOUTH_LEFT = 21
    MOUTH_RIGHT = 22
    LEFT_PINKY = 23
    RIGHT_PINKY = 24
    LEFT_INDEX = 25
    RIGHT_INDEX = 26
    LEFT_THUMB = 27
    RIGHT_THUMB = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# ── 骨架连接定义(COCO 17点) ──────────────────────────────────────────
POSE_CONNECTIONS = [
    # 躯干
    (5, 6),    # 左肩-右肩
    (5, 11),   # 左肩-左髋
    (6, 12),   # 右肩-右髋
    (11, 12),  # 左髋-右髋
    # 左臂
    (5, 7), (7, 9),
    # 右臂
    (6, 8), (8, 10),
    # 左腿
    (11, 13), (13, 15),
    # 右腿
    (12, 14), (14, 16),
    # 面部简化
    (0, 1), (0, 2), (1, 3), (2, 4),
]


@dataclasses.dataclass
class PoseLandmark:
    """单个关键点"""
    index: int
    x: float          # 像素坐标 x
    y: float          # 像素坐标 y
    z: float          # 深度(估算)
    visibility: float  # 可见性/置信度 [0, 1]
    world_x: float     # 归一化坐标 [0, 1]
    world_y: float
    world_z: float


@dataclasses.dataclass
class PoseResult:
    """骨架检测结果(接口兼容 MediaPipe 版本)"""
    landmarks: list[PoseLandmark]        # 33个关键点(17个COCO + 16个扩展)
    image_width: int
    image_height: int
    detection_confidence: float          # 整体检测置信度
    bbox: Optional[tuple[int, int, int, int]] = None  # 人体检测框 (x1,y1,x2,y2)

    def get_pixel(self, idx: LandmarkIndex) -> tuple[int, int]:
        lm = self.landmarks[int(idx)]
        return int(lm.x), int(lm.y)

    def get_normalized(self, idx: LandmarkIndex) -> tuple[float, float]:
        lm = self.landmarks[int(idx)]
        return lm.world_x, lm.world_y  # normalized coords

    def get_world(self, idx: LandmarkIndex) -> tuple[float, float, float]:
        lm = self.landmarks[int(idx)]
        return lm.world_x, lm.world_y, lm.world_z

    def is_visible(self, idx: LandmarkIndex, threshold: float = 0.5) -> bool:
        return self.landmarks[int(idx)].visibility >= threshold

    @property
    def all_pixel_coords(self) -> np.ndarray:
        return np.array([[lm.x, lm.y] for lm in self.landmarks])

    @property
    def all_normalized(self) -> np.ndarray:
        return np.array([[lm.world_x, lm.world_y, lm.world_z] for lm in self.landmarks])


def _interpolate_extended_landmarks(
    coco_landmarks: list[PoseLandmark],
    w: int, h: int,
) -> list[PoseLandmark]:
    """
    从 COCO 17个关键点插值计算扩展点, 保持接口兼容

    扩展点(17~32)大部分无法精确推断, 设为低可见度。
    下游分析模块主要使用0~16的COCO点, 扩展点仅在骨架绘制时使用。
    """
    ext = list(coco_landmarks)  # 复制前17个

    def _lerp(idx_a: int, idx_b: int, t: float, new_idx: int) -> PoseLandmark:
        a = coco_landmarks[idx_a]
        b = coco_landmarks[idx_b]
        vis = min(a.visibility, b.visibility) * 0.8
        return PoseLandmark(
            index=new_idx,
            x=a.x + (b.x - a.x) * t,
            y=a.y + (b.y - a.y) * t,
            z=a.z + (b.z - a.z) * t,
            visibility=vis,
            world_x=a.world_x + (b.world_x - a.world_x) * t,
            world_y=a.world_y + (b.world_y - a.world_y) * t,
            world_z=a.world_z + (b.world_z - a.world_z) * t,
        )

    def _invisible(new_idx: int) -> PoseLandmark:
        return PoseLandmark(
            index=new_idx, x=0, y=0, z=0,
            visibility=0.0, world_x=0, world_y=0, world_z=0,
        )

    # 17: LEFT_EYE_INNER ≈ 左眼偏中(无法精确推断)
    ext.append(_lerp(1, 0, 0.3, 17))
    # 18: RIGHT_EYE_INNER
    ext.append(_lerp(2, 0, 0.3, 18))
    # 19: LEFT_EYE_OUTER
    ext.append(_lerp(1, 3, 0.5, 19))
    # 20: RIGHT_EYE_OUTER
    ext.append(_lerp(2, 4, 0.5, 20))
    # 21: MOUTH_LEFT ≈ 左眼与鼻子之间偏下
    ext.append(_lerp(0, 1, 0.4, 21))
    # 22: MOUTH_RIGHT
    ext.append(_lerp(0, 2, 0.4, 22))
    # 23~28: 手指(无法推断, 设为不可见)
    for i in range(23, 29):
        ext.append(_invisible(i))
    # 29: LEFT_HEEL ≈ 左脚踝后方
    ext.append(_lerp(15, 15, 1.0, 29))  # 近似同踝
    # 30: RIGHT_HEEL
    ext.append(_lerp(16, 16, 1.0, 30))
    # 31: LEFT_FOOT_INDEX
    ext.append(_invisible(31))
    # 32: RIGHT_FOOT_INDEX
    ext.append(_invisible(32))

    return ext


def _find_model_path() -> str:
    """查找 YOLOv26 pose 模型文件"""
    candidates = [
        Path.cwd() / "yolo26s-pose.pt",
        Path(__file__).parent.parent / "yolo26s-pose.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # 如果没找到本地文件, 使用自动下载的模型名
    return "yolo26s-pose.pt"


class PoseDetector:
    """
    YOLOv26 Pose 骨架检测器

    参数:
        model_size: 模型大小 "n"/"s"/"m"/"l"/"x", 默认 "s"
        conf: 检测置信度阈值
        iou: NMS IoU阈值
    """

    def __init__(
        self,
        model_size: str = "s",
        conf: float = 0.5,
        iou: float = 0.7,
        model_complexity: int = 1,  # 保留参数兼容, 不影响YOLO
    ):
        model_name = f"yolo26{model_size}-pose.pt"
        self._model = YOLO(model_name)
        self._conf = conf
        self._iou = iou

    def detect(self, image: np.ndarray) -> Optional[PoseResult]:
        """
        检测图像中的人体骨架

        参数:
            image: BGR格式的OpenCV图像

        返回:
            PoseResult 或 None(未检测到人体)
        """
        h, w = image.shape[:2]

        results = self._model.predict(
            image,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )

        r = results[0]

        if r.keypoints is None or len(r.keypoints) == 0:
            return None

        # 取第一个检测到的人(置信度最高的)
        kps_data = r.keypoints.data[0]  # (17, 3) tensor: x, y, conf
        kps_np = kps_data.cpu().numpy()

        # 检测框
        bbox = None
        if r.boxes is not None and len(r.boxes) > 0:
            box = r.boxes.data[0].cpu().numpy()
            bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

        # 构建17个COCO关键点
        coco_landmarks: list[PoseLandmark] = []
        for i in range(17):
            kx, ky, kconf = float(kps_np[i, 0]), float(kps_np[i, 1]), float(kps_np[i, 2])
            coco_landmarks.append(PoseLandmark(
                index=i,
                x=kx,
                y=ky,
                z=0.0,  # YOLOv26 不输出深度
                visibility=kconf,
                world_x=kx / w,  # 归一化坐标
                world_y=ky / h,
                world_z=0.0,
            ))

        # 计算扩展关键点(17~32)
        all_landmarks = _interpolate_extended_landmarks(coco_landmarks, w, h)

        # 整体置信度
        visible_confs = [lm.visibility for lm in coco_landmarks if lm.visibility > 0.3]
        avg_conf = sum(visible_confs) / len(visible_confs) if visible_confs else 0.0

        # 检测框置信度
        if r.boxes is not None and len(r.boxes) > 0:
            box_conf = float(r.boxes.data[0][4].cpu())
            avg_conf = (avg_conf + box_conf) / 2

        return PoseResult(
            landmarks=all_landmarks,
            image_width=w,
            image_height=h,
            detection_confidence=avg_conf,
            bbox=bbox,
        )

    def close(self):
        """释放模型资源"""
        del self._model

    def __del__(self):
        try:
            del self._model
        except Exception:
            pass
