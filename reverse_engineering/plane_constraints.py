"""Conservative plane-aware constraints for editable v3 scene objects."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from reverse_engineering.scene_anchors import SceneAnchor, AnchorKind


class PlaneRelation(str, Enum):
    ON_PLANE = "on_plane"
    OFFSET = "offset"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"

    @property
    def label(self) -> str:
        return {
            PlaneRelation.ON_PLANE: "On plane",
            PlaneRelation.OFFSET: "Offset from plane",
            PlaneRelation.PARALLEL: "Parallel to plane",
            PlaneRelation.PERPENDICULAR: "Perpendicular to plane",
        }[self]


@dataclass
class PlaneConstraint:
    constraint_id: str
    plane_anchor_id: str
    target_id: str = "primary_subject"
    relation: PlaneRelation = PlaneRelation.ON_PLANE
    offset_m: float = 0.0
    enabled: bool = True
    source: str = "manual"

    def to_dict(self) -> dict:
        return {
            "constraint_id": self.constraint_id,
            "plane_anchor_id": self.plane_anchor_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "offset_m": float(self.offset_m),
            "enabled": bool(self.enabled),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlaneConstraint":
        relation = PlaneRelation(str(data.get("relation", PlaneRelation.ON_PLANE.value)))
        return cls(
            constraint_id=str(data.get("constraint_id", "constraint")),
            plane_anchor_id=str(data.get("plane_anchor_id", "ground")),
            target_id=str(data.get("target_id", "primary_subject")),
            relation=relation,
            offset_m=float(data.get("offset_m", 0.0)),
            enabled=bool(data.get("enabled", True)),
            source=str(data.get("source", "manual")),
        )


@dataclass(frozen=True)
class PlaneConstraintEvaluation:
    constraint_id: str
    satisfied: bool
    residual_m: float
    message: str


def _plane_or_raise(anchors: Iterable[SceneAnchor], plane_anchor_id: str) -> SceneAnchor:
    for anchor in anchors:
        if anchor.anchor_id == plane_anchor_id:
            if anchor.kind != AnchorKind.PLANE:
                raise ValueError(f"anchor {plane_anchor_id} is not a plane")
            return anchor
    raise ValueError(f"plane anchor not found: {plane_anchor_id}")


def signed_distance_to_plane(position, plane: SceneAnchor) -> float:
    point = np.asarray(position, dtype=float).reshape(3)
    center = np.asarray(plane.position, dtype=float)
    normal = plane.normalized_normal()
    return float(np.dot(point - center, normal))


def project_position_to_plane(position, plane: SceneAnchor, offset_m: float = 0.0) -> tuple[float, float, float]:
    point = np.asarray(position, dtype=float).reshape(3)
    normal = plane.normalized_normal()
    distance = signed_distance_to_plane(point, plane)
    corrected = point - (distance - float(offset_m)) * normal
    return tuple(float(v) for v in corrected)


def align_direction_with_plane(direction, plane: SceneAnchor, relation: PlaneRelation) -> tuple[float, float, float]:
    vector = np.asarray(direction, dtype=float).reshape(3)
    normal = plane.normalized_normal()
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return tuple(float(v) for v in normal)
    unit = vector / norm
    if relation is PlaneRelation.PARALLEL:
        projected = unit - float(np.dot(unit, normal)) * normal
        pnorm = float(np.linalg.norm(projected))
        if pnorm < 1e-9:
            # Choose a stable in-plane axis when the original direction is normal.
            tangent, _, _ = plane.plane_basis()
            projected = tangent
            pnorm = 1.0
        projected /= pnorm
        return tuple(float(v) for v in projected)
    if relation is PlaneRelation.PERPENDICULAR:
        sign = 1.0 if float(np.dot(unit, normal)) >= 0 else -1.0
        return tuple(float(v) for v in normal * sign)
    return tuple(float(v) for v in unit)


def evaluate_constraint(position, constraint: PlaneConstraint, anchors: Iterable[SceneAnchor], *, tolerance_m: float = 0.03) -> PlaneConstraintEvaluation:
    plane = _plane_or_raise(anchors, constraint.plane_anchor_id)
    distance = abs(signed_distance_to_plane(position, plane))
    target = abs(float(constraint.offset_m)) if constraint.relation is PlaneRelation.OFFSET else 0.0
    residual = abs(distance - target)
    satisfied = residual <= max(float(tolerance_m), 0.0)
    if constraint.relation is PlaneRelation.ON_PLANE:
        message = f"distance to {plane.name}: {distance:.3f} m"
    elif constraint.relation is PlaneRelation.OFFSET:
        message = f"distance {distance:.3f} m · target {target:.3f} m"
    else:
        message = f"position component compatible with {constraint.relation.label.lower()} relation"
    return PlaneConstraintEvaluation(constraint.constraint_id, satisfied, float(residual), message)


def apply_position_constraint(position, constraint: PlaneConstraint, anchors: Iterable[SceneAnchor]) -> tuple[float, float, float]:
    plane = _plane_or_raise(anchors, constraint.plane_anchor_id)
    if constraint.relation is PlaneRelation.ON_PLANE:
        return project_position_to_plane(position, plane, 0.0)
    if constraint.relation is PlaneRelation.OFFSET:
        return project_position_to_plane(position, plane, float(constraint.offset_m))
    return tuple(float(v) for v in np.asarray(position, dtype=float).reshape(3))
