"""
骨架检测模块 — 基于 YOLOv26 Pose 的人体姿态估计

YOLOv26 Pose 使用 COCO 17关键点格式, 通过检测+姿态估计联合模型完成。
输出保留原 PoseResult 接口，同时支持多人检测，并提供统一的坐标重标定接口。
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


class LandmarkIndex(IntEnum):
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


POSE_CONNECTIONS = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]


@dataclasses.dataclass
class PoseLandmark:
    index: int
    x: float
    y: float
    z: float
    visibility: float
    world_x: float
    world_y: float
    world_z: float


@dataclasses.dataclass
class PoseResult:
    landmarks: list[PoseLandmark]
    image_width: int
    image_height: int
    detection_confidence: float
    bbox: Optional[tuple[int, int, int, int]] = None
    # All detected people in this image. The first entry is the primary pose.
    persons: Optional[list["PoseResult"]] = None

    def get_pixel(self, idx: LandmarkIndex) -> tuple[int, int]:
        lm = self.landmarks[int(idx)]
        return int(lm.x), int(lm.y)

    def get_normalized(self, idx: LandmarkIndex) -> tuple[float, float]:
        lm = self.landmarks[int(idx)]
        return lm.world_x, lm.world_y

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

    def rescaled(self, width: int, height: int) -> "PoseResult":
        """Rescale pixel/normalized coordinates to another image size."""
        width, height = int(width), int(height)
        sx = width / max(self.image_width, 1)
        sy = height / max(self.image_height, 1)
        landmarks = [
            dataclasses.replace(
                lm,
                x=lm.x * sx,
                y=lm.y * sy,
                world_x=(lm.x * sx) / max(width, 1),
                world_y=(lm.y * sy) / max(height, 1),
            )
            for lm in self.landmarks
        ]
        bbox = None
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            bbox = (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))
        return PoseResult(landmarks, width, height, self.detection_confidence, bbox, None)


def _interpolate_extended_landmarks(coco_landmarks: list[PoseLandmark]) -> list[PoseLandmark]:
    ext = list(coco_landmarks)

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
        return PoseLandmark(new_idx, 0, 0, 0, 0, 0, 0, 0)

    ext.append(_lerp(1, 0, 0.3, 17))
    ext.append(_lerp(2, 0, 0.3, 18))
    ext.append(_lerp(1, 3, 0.5, 19))
    ext.append(_lerp(2, 4, 0.5, 20))
    ext.append(_lerp(0, 1, 0.4, 21))
    ext.append(_lerp(0, 2, 0.4, 22))
    for i in range(23, 29):
        ext.append(_invisible(i))
    ext.append(_lerp(15, 15, 1.0, 29))
    ext.append(_lerp(16, 16, 1.0, 30))
    ext.append(_invisible(31))
    ext.append(_invisible(32))
    return ext


def _find_model_path() -> str:
    candidates = [Path.cwd() / "yolo26s-pose.pt", Path(__file__).parent.parent / "yolo26s-pose.pt"]
    for p in candidates:
        if p.exists():
            return str(p)
    return "yolo26s-pose.pt"


class PoseDetector:
    def __init__(self, model_size: str = "s", conf: float = 0.5, iou: float = 0.7, model_complexity: int = 1):
        self._model = YOLO(f"yolo26{model_size}-pose.pt")
        self._conf = conf
        self._iou = iou

    def _results_to_poses(self, result, width: int, height: int) -> list[PoseResult]:
        if result.keypoints is None or len(result.keypoints) == 0:
            return []

        count = len(result.keypoints)
        boxes = result.boxes
        order = list(range(count))
        if boxes is not None and len(boxes) == count:
            box_conf = boxes.conf.cpu().numpy()
            order.sort(key=lambda i: float(box_conf[i]), reverse=True)

        poses: list[PoseResult] = []
        for idx in order:
            kps_np = result.keypoints.data[idx].cpu().numpy()
            bbox = None
            box_conf = 0.0
            if boxes is not None and idx < len(boxes):
                box = boxes.data[idx].cpu().numpy()
                bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                box_conf = float(box[4])

            coco: list[PoseLandmark] = []
            for i in range(17):
                kx, ky, kconf = map(float, kps_np[i, :3])
                coco.append(PoseLandmark(
                    index=i, x=kx, y=ky, z=0.0, visibility=kconf,
                    world_x=kx / max(width, 1), world_y=ky / max(height, 1), world_z=0.0,
                ))
            landmarks = _interpolate_extended_landmarks(coco)
            visible = [lm.visibility for lm in coco if lm.visibility > 0.3]
            avg_conf = sum(visible) / len(visible) if visible else 0.0
            if box_conf:
                avg_conf = (avg_conf + box_conf) / 2
            poses.append(PoseResult(landmarks, width, height, avg_conf, bbox))
        return poses

    def detect_all(self, image: np.ndarray) -> list[PoseResult]:
        h, w = image.shape[:2]
        results = self._model.predict(image, conf=self._conf, iou=self._iou, verbose=False)
        return self._results_to_poses(results[0], w, h)

    def detect(self, image: np.ndarray) -> Optional[PoseResult]:
        """Backward-compatible primary-pose API with all people attached."""
        poses = self.detect_all(image)
        if not poses:
            return None
        poses[0].persons = poses
        return poses[0]

    def close(self):
        del self._model

    def __del__(self):
        try:
            del self._model
        except Exception:
            pass
