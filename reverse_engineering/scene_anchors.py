"""Editable scene anchors for v3 room/object reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import numpy as np


class AnchorKind(str, Enum):
    POINT = "point"
    PLANE = "plane"


@dataclass
class SceneAnchor:
    """A user-editable point or plane in scene coordinates."""

    anchor_id: str
    name: str
    kind: AnchorKind = AnchorKind.POINT
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    size: tuple[float, float] = (2.0, 2.0)
    confidence: float = 0.0
    source: str = "manual"
    locked: bool = False
    enabled: bool = True
    image_points: tuple[tuple[float, float], ...] = ()

    def normalized_normal(self) -> np.ndarray:
        n = np.asarray(self.normal, dtype=float)
        length = float(np.linalg.norm(n))
        if length < 1e-9:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        return n / length

    def plane_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normal = self.normalized_normal()
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(normal, reference))) > 0.92:
            reference = np.array([1.0, 0.0, 0.0], dtype=float)
        tangent = np.cross(reference, normal)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        bitangent = np.cross(normal, tangent)
        bitangent /= max(float(np.linalg.norm(bitangent)), 1e-9)
        return tangent, bitangent, normal

    def corners(self) -> np.ndarray:
        """Return four world-space corners for a plane anchor."""
        if self.kind != AnchorKind.PLANE:
            return np.empty((0, 3), dtype=float)
        center = np.asarray(self.position, dtype=float)
        tangent, bitangent, _ = self.plane_basis()
        half_w = max(0.01, float(self.size[0])) * 0.5
        half_h = max(0.01, float(self.size[1])) * 0.5
        return np.array([
            center - tangent * half_w - bitangent * half_h,
            center + tangent * half_w - bitangent * half_h,
            center + tangent * half_w + bitangent * half_h,
            center - tangent * half_w + bitangent * half_h,
        ])

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.anchor_id.strip():
            errors.append("anchor_id must not be empty")
        if not self.name.strip():
            errors.append("name must not be empty")
        position = np.asarray(self.position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            errors.append("position must contain three finite values")
        normal = np.asarray(self.normal, dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
            errors.append("normal must be a non-zero finite vector")
        if self.kind == AnchorKind.PLANE:
            size = np.asarray(self.size, dtype=float)
            if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
                errors.append("plane size must contain two positive finite values")
        if not 0.0 <= float(self.confidence) <= 1.0:
            errors.append("confidence must be between 0 and 1")
        return errors

    def to_dict(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "name": self.name,
            "kind": self.kind.value,
            "position": [float(v) for v in self.position],
            "normal": [float(v) for v in self.normal],
            "size": [float(v) for v in self.size],
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "locked": bool(self.locked),
            "enabled": bool(self.enabled),
            "image_points": [[float(x), float(y)] for x, y in self.image_points],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneAnchor":
        kind = AnchorKind(str(data.get("kind", AnchorKind.POINT.value)))
        return cls(
            anchor_id=str(data.get("anchor_id", "anchor")),
            name=str(data.get("name", "Anchor")),
            kind=kind,
            position=tuple(float(v) for v in data.get("position", (0.0, 0.0, 0.0)))[:3],
            normal=tuple(float(v) for v in data.get("normal", (0.0, 1.0, 0.0)))[:3],
            size=tuple(float(v) for v in data.get("size", (2.0, 2.0)))[:2],
            confidence=float(data.get("confidence", 0.0)),
            source=str(data.get("source", "manual")),
            locked=bool(data.get("locked", False)),
            enabled=bool(data.get("enabled", True)),
            image_points=tuple(
                (float(p[0]), float(p[1]))
                for p in data.get("image_points", ())
                if len(p) >= 2
            ),
        )


def default_scene_anchors() -> list[SceneAnchor]:
    """Return a conservative editable world scaffold for a new scene."""
    return [
        SceneAnchor(
            anchor_id="ground",
            name="Ground plane",
            kind=AnchorKind.PLANE,
            position=(0.0, 0.0, 0.0),
            normal=(0.0, 1.0, 0.0),
            size=(12.0, 12.0),
            confidence=0.0,
            source="scene scaffold",
        )
    ]


def next_anchor_id(anchors: Iterable[SceneAnchor], prefix: str = "anchor") -> str:
    existing = {a.anchor_id for a in anchors}
    if prefix not in existing:
        return prefix
    for index in range(2, 10000):
        candidate = f"{prefix}_{index}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("unable to allocate scene anchor id")


def plane_from_three_points(anchor_id: str, name: str, points: Iterable[Iterable[float]]) -> Optional[SceneAnchor]:
    """Create a plane anchor from three non-collinear world points."""
    pts = np.asarray(list(points), dtype=float)
    if pts.shape != (3, 3) or not np.isfinite(pts).all():
        return None
    a, b, c = pts
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    normal /= norm
    center = (a + b + c) / 3.0
    width = max(float(np.linalg.norm(b - a)), 0.1)
    height = max(float(np.linalg.norm(c - a)), 0.1)
    return SceneAnchor(
        anchor_id=anchor_id,
        name=name,
        kind=AnchorKind.PLANE,
        position=tuple(float(v) for v in center),
        normal=tuple(float(v) for v in normal),
        size=(width, height),
        confidence=0.6,
        source="three-point manual calibration",
    )
