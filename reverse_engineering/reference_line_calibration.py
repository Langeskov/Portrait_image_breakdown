"""Semantic image-space reference-line evidence for camera calibration.

A scene plane remains 3D world geometry. This module models the observable
image evidence selected by the user: a line segment and an expected direction.
It deliberately reports only the camera degrees of freedom that the line can
support instead of pretending to recover a full camera pose.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple


class ReferenceLineConstraint(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    FREE = "free"

    @property
    def label(self) -> str:
        return {
            ReferenceLineConstraint.HORIZONTAL: "Horizontal",
            ReferenceLineConstraint.VERTICAL: "Vertical",
            ReferenceLineConstraint.FREE: "Free",
        }[self]

    @property
    def target_angle_deg(self) -> Optional[float]:
        if self is ReferenceLineConstraint.HORIZONTAL:
            return 0.0
        if self is ReferenceLineConstraint.VERTICAL:
            return 90.0
        return None


def normalize_line_angle_deg(angle: float) -> float:
    """Normalize an unoriented line angle to [-90, 90)."""
    value = (float(angle) + 90.0) % 180.0 - 90.0
    return value


def line_angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    if math.hypot(dx, dy) < 1e-9:
        raise ValueError("reference line endpoints are coincident")
    # Image-space y grows downward, so positive visual tilt is clockwise.
    return normalize_line_angle_deg(math.degrees(math.atan2(dy, dx)))


def angular_error_deg(observed: float, target: float) -> float:
    """Return the smallest unoriented line-angle error."""
    return normalize_line_angle_deg(float(target) - float(observed))


@dataclass(frozen=True)
class ReferenceLineEvidence:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    constraint: ReferenceLineConstraint = ReferenceLineConstraint.FREE
    source: str = "manual image evidence"

    @property
    def length_px(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])

    @property
    def observed_angle_deg(self) -> float:
        return line_angle_deg(self.p1, self.p2)

    @property
    def target_angle_deg(self) -> Optional[float]:
        return self.constraint.target_angle_deg

    @property
    def correction_deg(self) -> Optional[float]:
        if self.target_angle_deg is None:
            return None
        return angular_error_deg(self.observed_angle_deg, self.target_angle_deg)

    @property
    def confidence(self) -> float:
        if self.length_px < 30.0:
            return 0.25
        if self.length_px < 80.0:
            return 0.55
        return 0.9

    @property
    def supports_roll(self) -> bool:
        return self.constraint in {
            ReferenceLineConstraint.HORIZONTAL,
            ReferenceLineConstraint.VERTICAL,
        }

    def summary(self) -> str:
        angle = self.observed_angle_deg
        if self.target_angle_deg is None:
            return f"Free reference line · observed {angle:+.1f}° · {self.length_px:.0f}px"
        correction = self.correction_deg or 0.0
        return (
            f"{self.constraint.label} line · observed {angle:+.1f}° · "
            f"target {self.target_angle_deg:+.1f}° · roll correction {correction:+.1f}°"
        )
