"""Deterministic temporal smoothing for v3 sequence/live input."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np


@dataclass(frozen=True)
class TemporalCameraState:
    distance: float
    height: float
    yaw: float
    pitch: float
    roll: float
    focal_length_mm: float


@dataclass(frozen=True)
class TemporalFrameState:
    timestamp: float
    keypoints: Optional[np.ndarray] = None
    camera: Optional[TemporalCameraState] = None
    confidence: float = 0.0


@dataclass(frozen=True)
class TemporalSmoothedState:
    timestamp: float
    keypoints: Optional[np.ndarray]
    camera: Optional[TemporalCameraState]
    confidence: float


class TemporalSmoother:
    """Exponential temporal filter with missing-frame hold behavior."""

    def __init__(self, alpha: float = 0.35, max_gap_seconds: float = 0.75):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.max_gap_seconds = max(0.0, float(max_gap_seconds))
        self._keypoints: Optional[np.ndarray] = None
        self._camera: Optional[np.ndarray] = None
        self._timestamp: Optional[float] = None
        self._confidence = 0.0

    @staticmethod
    def _ema(previous: Optional[np.ndarray], current: Optional[np.ndarray], alpha: float) -> Optional[np.ndarray]:
        if current is None:
            return previous.copy() if previous is not None else None
        arr = np.asarray(current, dtype=float)
        if previous is None or previous.shape != arr.shape:
            return arr.copy()
        valid = np.isfinite(arr)
        result = previous.copy()
        result[valid] = (1.0 - alpha) * previous[valid] + alpha * arr[valid]
        return result

    @staticmethod
    def _camera_array(camera: Optional[TemporalCameraState]) -> Optional[np.ndarray]:
        if camera is None:
            return None
        return np.array([camera.distance, camera.height, camera.yaw, camera.pitch, camera.roll, camera.focal_length_mm], dtype=float)

    @staticmethod
    def _camera_from_array(values: Optional[np.ndarray]) -> Optional[TemporalCameraState]:
        if values is None or len(values) != 6:
            return None
        return TemporalCameraState(*(float(v) for v in values))

    def update(self, frame: TemporalFrameState) -> TemporalSmoothedState:
        timestamp = float(frame.timestamp)
        gap = float(timestamp - self._timestamp) if self._timestamp is not None else 0.0
        if self._timestamp is not None and gap > self.max_gap_seconds and self.max_gap_seconds > 0:
            self.reset()
        alpha = self.alpha
        self._keypoints = self._ema(self._keypoints, frame.keypoints, alpha)
        self._camera = self._ema(self._camera, self._camera_array(frame.camera), alpha)
        incoming_conf = max(0.0, min(1.0, float(frame.confidence)))
        self._confidence = incoming_conf if self._timestamp is None else (1.0 - alpha) * self._confidence + alpha * incoming_conf
        self._timestamp = timestamp
        return TemporalSmoothedState(
            timestamp,
            self._keypoints.copy() if self._keypoints is not None else None,
            self._camera_from_array(self._camera),
            float(self._confidence),
        )

    def reset(self):
        self._keypoints = None
        self._camera = None
        self._timestamp = None
        self._confidence = 0.0


def smooth_sequence(frames: list[TemporalFrameState], alpha: float = 0.35, max_gap_seconds: float = 0.75) -> list[TemporalSmoothedState]:
    smoother = TemporalSmoother(alpha=alpha, max_gap_seconds=max_gap_seconds)
    return [smoother.update(frame) for frame in frames]
