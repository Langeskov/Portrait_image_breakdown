"""
深度提供者接口 - DepthProvider

统一深度数据接口, 支持多种深度数据源。
当前实现: MonocularDepthProvider (基于图像梯度的相对深度估计)
"""
from __future__ import annotations
import abc
from typing import Optional
import cv2
import numpy as np


class DepthProvider(abc.ABC):
    """深度数据提供者抽象基类"""

    @abc.abstractmethod
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """估计深度图 (H, W), 值越大越远"""
        ...

    @abc.abstractmethod
    def get_depth(self, x: float, y: float) -> Optional[float]:
        """获取指定位置的深度值"""
        ...

    def estimate_relative_depth(self, image: np.ndarray) -> np.ndarray:
        """估计相对深度 (归一化到 0-1)"""
        depth = self.estimate_depth(image)
        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin < 1e-6:
            return np.zeros_like(depth)
        return (depth - dmin) / (dmax - dmin)


class MonocularDepthProvider(DepthProvider):
    """
    基于图像梯度的单目相对深度估计

    不使用深度学习模型, 仅基于:
    - 梯度强度 (边缘通常是深度不连续)
    - 垂直位置 (上方通常更远)
    - 模糊度 (模糊通常更远)

    这是相对深度, 不是绝对深度。
    """

    def __init__(self):
        self._last_depth: Optional[np.ndarray] = None

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)

        # 梯度强度
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient = np.sqrt(gx**2 + gy**2)
        gradient_norm = gradient / (gradient.max() + 1e-6)

        # 垂直位置先验 (上方更远)
        ys = np.linspace(0, 1, h).reshape(-1, 1)
        vertical_prior = np.tile(ys, (1, w))

        # 模糊度 (拉普拉斯方差)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_map = np.abs(laplacian)
        blur_norm = blur_map / (blur_map.max() + 1e-6)

        # 组合: 低梯度 + 高位置 + 低清晰度 = 更远
        depth = (1 - gradient_norm) * 0.4 + vertical_prior * 0.3 + (1 - blur_norm) * 0.3
        self._last_depth = depth.astype(np.float32)
        return self._last_depth

    def get_depth(self, x: float, y: float) -> Optional[float]:
        if self._last_depth is None:
            return None
        h, w = self._last_depth.shape
        px = int(x * w) if x <= 1 else int(x)
        py = int(y * h) if y <= 1 else int(y)
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        return float(self._last_depth[py, px])
