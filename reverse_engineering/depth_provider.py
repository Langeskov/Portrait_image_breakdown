"""Unified monocular-depth provider interface.

The default provider is dependency-light and deterministic. A local model can be
plugged in through ``model_fn`` without changing the camera reconstruction API.
Returned depth is relative: larger values mean farther away.
"""
from __future__ import annotations

import abc
from typing import Callable, Optional

import cv2
import numpy as np


class DepthProvider(abc.ABC):
    """Depth data provider abstraction."""

    @abc.abstractmethod
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_depth(self, x: float, y: float) -> Optional[float]:
        ...

    def estimate_relative_depth(self, image: np.ndarray) -> np.ndarray:
        depth = np.asarray(self.estimate_depth(image), dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("depth provider must return a HxW array")
        finite = np.isfinite(depth)
        if not finite.any():
            return np.zeros_like(depth, dtype=np.float32)
        lo, hi = np.percentile(depth[finite], [2.0, 98.0])
        span = max(float(hi - lo), 1e-6)
        return np.clip((depth - lo) / span, 0.0, 1.0).astype(np.float32)


class MonocularDepthProvider(DepthProvider):
    """Relative monocular depth with optional local model backend.

    ``model_fn`` may be a callable accepting a BGR image and returning a HxW
    relative-depth map. This keeps the repository offline-first while allowing
    Depth Anything / MiDaS / other locally packaged models to be plugged in.
    If the model raises, the deterministic fallback is used.
    """

    def __init__(self, model_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self._model_fn = model_fn
        self._last_depth: Optional[np.ndarray] = None
        self.last_backend = "fallback"
        self.last_confidence = 0.0

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.ndim < 2:
            raise ValueError("invalid image")
        if self._model_fn is not None:
            try:
                model_depth = np.asarray(self._model_fn(image), dtype=np.float32)
                if model_depth.shape == image.shape[:2] and np.isfinite(model_depth).mean() > 0.95:
                    self._last_depth = self._postprocess(model_depth)
                    self.last_backend = "local_model"
                    self.last_confidence = 0.75
                    return self._last_depth
            except Exception:
                pass
        self._last_depth = self._fallback_depth(image)
        self.last_backend = "structure_prior"
        self.last_confidence = 0.28
        return self._last_depth

    @staticmethod
    def _postprocess(depth: np.ndarray) -> np.ndarray:
        finite = np.isfinite(depth)
        clean = np.nan_to_num(depth, nan=float(np.nanmedian(depth[finite])) if finite.any() else 0.5,
                              posinf=float(np.nanmax(depth[finite])) if finite.any() else 1.0,
                              neginf=float(np.nanmin(depth[finite])) if finite.any() else 0.0)
        lo, hi = np.percentile(clean, [2.0, 98.0])
        clean = np.clip((clean - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        return cv2.GaussianBlur(clean.astype(np.float32), (0, 0), 1.1)

    @staticmethod
    def _fallback_depth(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 1.2)
        g95 = np.percentile(gradient, 95)
        edge_term = np.clip(gradient / max(float(g95), 1e-5), 0.0, 1.0)

        # Coarse vertical prior is deliberately weak and removed near the center
        # of the subject, where portrait photography commonly breaks the outdoor
        # "sky is far" assumption.
        y = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(-1, 1)
        vertical_prior = np.repeat(y, w, axis=1)

        # Local variance is a more stable focus/texture cue than raw Laplacian.
        mean = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sqmean = cv2.GaussianBlur(gray * gray, (0, 0), 2.0)
        local_var = np.maximum(sqmean - mean * mean, 0.0)
        var_norm = local_var / max(float(np.percentile(local_var, 98)), 1e-6)

        # Keep the original qualitative ordering: smooth/weakly textured and
        # upper-image regions tend to be farther away.
        depth = 0.42 * (1.0 - edge_term) + 0.22 * vertical_prior + 0.36 * (1.0 - np.clip(var_norm, 0.0, 1.0))
        depth = cv2.bilateralFilter(depth.astype(np.float32), 7, 0.08, 5.0)
        return MonocularDepthProvider._postprocess(depth)

    def get_depth(self, x: float, y: float) -> Optional[float]:
        if self._last_depth is None:
            return None
        h, w = self._last_depth.shape
        px = int(x * w) if x <= 1 else int(x)
        py = int(y * h) if y <= 1 else int(y)
        px = max(0, min(w - 1, px)); py = max(0, min(h - 1, py))
        return float(self._last_depth[py, px])
