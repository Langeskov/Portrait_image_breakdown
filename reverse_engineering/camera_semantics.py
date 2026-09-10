"""Camera semantics shared by the reconstruction scene model.

These lightweight value objects describe camera position/orbit and optical aim
without depending on the reverse-engineering engine implementation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraOrbit:
    distance_m: float
    height_m: float
    yaw_deg: float
    pitch_deg: float


@dataclass(frozen=True)
class CameraOpticalAim:
    reframe_yaw_deg: float = 0.0
    reframe_pitch_deg: float = 0.0
    roll_deg: float = 0.0

    @classmethod
    def from_scene_camera(cls, camera):
        return cls(0.0, 0.0, float(camera.roll))


__all__ = ["CameraOrbit", "CameraOpticalAim"]
