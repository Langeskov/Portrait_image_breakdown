"""Conservative multi-person scene layout from pose + relative depth.

This module deliberately separates image-space person detection from 3D scene
placement. A single monocular image does not provide metric person distances,
so ``relative_z`` is only a normalized ordering coordinate. A person is marked
``usable_3d`` only when the depth provider has sufficient confidence and the
observed depth separation is meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.pose_detector import LandmarkIndex as LI, PoseResult
from reverse_engineering.depth_provider import DepthProvider


@dataclass(frozen=True)
class PersonLayout:
    person_index: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    scale: float
    relative_depth: float
    relative_z: float
    depth_confidence: float
    usable_3d: bool
    keypoints: tuple[tuple[float, float, float], ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "person_index": self.person_index,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "scale": round(float(self.scale), 4),
            "relative_depth": round(float(self.relative_depth), 4),
            "relative_z": round(float(self.relative_z), 4),
            "depth_confidence": round(float(self.depth_confidence), 3),
            "usable_3d": bool(self.usable_3d),
            "keypoint_count": len(self.keypoints),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class MultiPersonLayoutResult:
    people: tuple[PersonLayout, ...] = ()
    depth_backend: str = "unavailable"
    depth_confidence: float = 0.0
    independent_depth: bool = False
    notes: tuple[str, ...] = ()

    @property
    def usable_people(self) -> tuple[PersonLayout, ...]:
        return tuple(p for p in self.people if p.usable_3d)

    def to_dict(self) -> dict:
        return {
            "people": [p.to_dict() for p in self.people],
            "depth_backend": self.depth_backend,
            "depth_confidence": round(float(self.depth_confidence), 3),
            "independent_depth": bool(self.independent_depth),
            "notes": list(self.notes),
        }


def _visible_anchor(person: PoseResult) -> np.ndarray | None:
    for group in ((LI.LEFT_HIP, LI.RIGHT_HIP), (LI.LEFT_SHOULDER, LI.RIGHT_SHOULDER), (LI.NOSE,)):
        points = []
        for idx in group:
            if idx < len(person.landmarks) and person.landmarks[int(idx)].visibility >= 0.4:
                lm = person.landmarks[int(idx)]
                points.append((lm.x, lm.y))
        if points:
            return np.mean(np.asarray(points, dtype=float), axis=0)
    return None


def _person_bbox(person: PoseResult, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    if person.bbox is not None:
        x0, y0, x1, y1 = person.bbox
    else:
        visible = [lm for lm in person.landmarks[:17] if lm.visibility >= 0.35]
        if visible:
            x0, y0 = min(lm.x for lm in visible), min(lm.y for lm in visible)
            x1, y1 = max(lm.x for lm in visible), max(lm.y for lm in visible)
        else:
            x0 = y0 = 0.0
            x1, y1 = float(image_w), float(image_h)
    return (
        float(np.clip(x0 / max(image_w, 1), 0.0, 1.0)),
        float(np.clip(y0 / max(image_h, 1), 0.0, 1.0)),
        float(np.clip(x1 / max(image_w, 1), 0.0, 1.0)),
        float(np.clip(y1 / max(image_h, 1), 0.0, 1.0)),
    )


def _keypoints(person: PoseResult, image_w: int, image_h: int) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            float(lm.x / max(image_w, 1)),
            float(lm.y / max(image_h, 1)),
            float(lm.visibility),
        )
        for lm in person.landmarks[:17]
    )


def _sample_depth(depth_provider: DepthProvider, anchor: np.ndarray) -> float | None:
    try:
        value = depth_provider.get_depth(float(anchor[0]), float(anchor[1]))
        return float(value) if value is not None and np.isfinite(value) else None
    except Exception:
        return None


def build_multi_person_layout(
    people: Iterable[PoseResult],
    image_w: int,
    image_h: int,
    depth_provider: DepthProvider | None = None,
) -> MultiPersonLayoutResult:
    """Build a relative layout without inventing metric camera-to-person range."""
    persons = list(people)
    if not persons:
        return MultiPersonLayoutResult()

    provider_confidence = float(getattr(depth_provider, "last_confidence", 0.0)) if depth_provider is not None else 0.0
    backend = str(getattr(depth_provider, "last_backend", "unavailable")) if depth_provider is not None else "unavailable"
    raw_depths: list[float | None] = []
    records = []
    for index, person in enumerate(persons):
        anchor = _visible_anchor(person)
        depth = _sample_depth(depth_provider, anchor) if (depth_provider is not None and anchor is not None) else None
        raw_depths.append(depth)
        bbox = _person_bbox(person, image_w, image_h)
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        scale = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        records.append((index, person, bbox, (cx, cy), scale))

    finite = np.asarray([d for d in raw_depths if d is not None], dtype=float)
    independent_depth = bool(
        len(finite) >= 2
        and provider_confidence >= 0.60
        and float(np.ptp(finite)) >= 0.08
    )
    if len(finite):
        lo, hi = float(np.min(finite)), float(np.max(finite))
        span = max(hi - lo, 1e-6)
    else:
        lo, span = 0.0, 1.0

    people_out = []
    for index, person, bbox, center, scale in records:
        depth = raw_depths[index]
        if depth is None:
            relative_depth = 0.5
        else:
            relative_depth = float(np.clip((depth - lo) / span, 0.0, 1.0))
        # Positive relative_z = farther from the camera. Keep this a bounded
        # scene coordinate; it is not meters and is never presented as such.
        relative_z = (relative_depth - 0.5) * 2.0
        usable = bool(independent_depth and depth is not None)
        evidence = [
            f"depth backend: {backend}",
            f"depth confidence: {provider_confidence:.0%}",
        ]
        if depth is None:
            evidence.append("anchor depth unavailable; retained image-space layout")
        elif not independent_depth:
            evidence.append("depth separation/confidence insufficient for independent 3D placement")
        else:
            evidence.append("relative depth separation supports non-metric 3D ordering")
        people_out.append(PersonLayout(
            person_index=index,
            bbox=bbox,
            center=(round(float(center[0]), 5), round(float(center[1]), 5)),
            scale=round(float(scale), 5),
            relative_depth=round(float(relative_depth), 5),
            relative_z=round(float(relative_z), 5),
            depth_confidence=round(provider_confidence, 5),
            usable_3d=usable,
            keypoints=_keypoints(person, image_w, image_h),
            evidence=tuple(evidence),
        ))

    notes = [
        "multi-person layout is image-space first; relative_z is normalized, not metric",
        "3D placement is enabled only when local depth evidence is sufficiently confident and separates people",
    ]
    if len(persons) == 1:
        notes.append("single-person frame: multi-person depth layout remains dormant")
    return MultiPersonLayoutResult(tuple(people_out), backend, provider_confidence, independent_depth, tuple(notes))
