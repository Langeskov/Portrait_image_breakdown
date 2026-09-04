"""
核心数据结构 - 区分 Observed / Estimated / Unknown 三层数据
"""
from __future__ import annotations
import dataclasses
from enum import Enum
from typing import Any, Optional

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

@dataclasses.dataclass
class EstimatedValue:
    value: Any
    unit: str = ""
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    confidence: float = 0.0
    basis: list[str] = dataclasses.field(default_factory=list)
    is_observed: bool = False

    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence >= 0.75: return ConfidenceLevel.HIGH
        elif self.confidence >= 0.45: return ConfidenceLevel.MEDIUM
        elif self.confidence > 0: return ConfidenceLevel.LOW
        return ConfidenceLevel.UNKNOWN

    def to_dict(self) -> dict:
        d = {"estimated": self.value, "unit": self.unit,
             "confidence": round(self.confidence, 3),
             "confidence_level": self.confidence_level.value,
             "basis": self.basis, "is_observed": self.is_observed}
        if self.range_min is not None or self.range_max is not None:
            d["range"] = [self.range_min if self.range_min is not None else self.value,
                          self.range_max if self.range_max is not None else self.value]
        return d

@dataclasses.dataclass
class PerspectiveResult:
    perspective_strength: EstimatedValue
    perspective_type: EstimatedValue
    vanishing_points: list[tuple[float, float]]
    vertical_convergence: EstimatedValue
    horizontal_convergence: EstimatedValue

@dataclasses.dataclass
class CameraPoseResult:
    camera_height: EstimatedValue
    camera_distance: EstimatedValue
    camera_pitch: EstimatedValue
    camera_yaw: EstimatedValue
    camera_roll: EstimatedValue

@dataclasses.dataclass
class FocalLengthResult:
    category: EstimatedValue
    equivalent_35mm: EstimatedValue

@dataclasses.dataclass
class DepthOfFieldResult:
    dof_type: EstimatedValue
    subject_sharp: bool
    foreground_blur: float
    background_blur: float
    aperture_range: EstimatedValue

@dataclasses.dataclass
class MotionBlurResult:
    blur_type: EstimatedValue
    blur_direction: Optional[tuple[float, float]]
    blur_magnitude: float
    shutter_range: EstimatedValue

@dataclasses.dataclass
class ShootingTechniqueResult:
    techniques: list[dict]

@dataclasses.dataclass
class CompositionResult:
    styles: list[dict]
    subject_position: tuple[float, float]
    subject_scale: float
    headroom: float
    look_room: str
    negative_space_ratio: float

@dataclasses.dataclass
class CameraAction:
    action: str
    reason: list[str]
    expected_effect: str
    priority: int = 0

@dataclasses.dataclass
class ShootingState:
    camera: dict
    lens: dict
    composition: dict
    depth: dict
    techniques: list[str]
    confidence: float
    uncertainties: list[str]
    camera_actions: list[CameraAction]

@dataclasses.dataclass
class ReverseEngineeringResult:
    image_size: tuple[int, int]
    subject_bbox: Optional[tuple[int, int, int, int]]
    subject_keypoints: Optional[list]
    subject_scale: float
    edge_lines: list
    blur_regions: dict
    perspective: PerspectiveResult
    camera_pose: CameraPoseResult
    focal_length: FocalLengthResult
    depth_of_field: DepthOfFieldResult
    motion_blur: MotionBlurResult
    composition: CompositionResult
    shooting_techniques: ShootingTechniqueResult
    overall_confidence: float
    uncertainties: list[str]
    _sim_candidates: list = dataclasses.field(default_factory=list)
    _camera_actions: list = dataclasses.field(default_factory=list)

    def to_shooting_state(self) -> ShootingState:
        return ShootingState(
            camera={"height": self.camera_pose.camera_height.value,
                    "distance": self.camera_pose.camera_distance.value,
                    "pitch": self.camera_pose.camera_pitch.value,
                    "yaw": self.camera_pose.camera_yaw.value,
                    "roll": self.camera_pose.camera_roll.value},
            lens={"type": self.focal_length.category.value,
                  "equivalent_focal_length": self.focal_length.equivalent_35mm.value,
                  "range": [self.focal_length.equivalent_35mm.range_min,
                            self.focal_length.equivalent_35mm.range_max]},
            composition={"styles": [s["name"] for s in self.composition.styles],
                         "subject_position": self.composition.subject_position,
                         "negative_space_ratio": self.composition.negative_space_ratio},
            depth={"type": self.depth_of_field.dof_type.value,
                   "aperture_range": self.depth_of_field.aperture_range.value},
            techniques=[t["name"] for t in self.shooting_techniques.techniques],
            confidence=self.overall_confidence,
            uncertainties=self.uncertainties,
            camera_actions=self._camera_actions)

    def to_dict(self) -> dict:
        return {
            "observed": {"image_size": list(self.image_size),
                         "subject_bbox": self.subject_bbox,
                         "subject_scale": round(self.subject_scale, 4),
                         "subject_keypoints_count": len(self.subject_keypoints) if self.subject_keypoints else 0},
            "estimated": {
                "perspective": {"strength": self.perspective.perspective_strength.to_dict(),
                                "type": self.perspective.perspective_type.to_dict(),
                                "vanishing_points": self.perspective.vanishing_points},
                "camera_pose": {k: getattr(self.camera_pose, f"camera_{k}").to_dict()
                                for k in ("height", "distance", "pitch", "yaw", "roll")},
                "focal_length": {"category": self.focal_length.category.to_dict(),
                                 "equivalent_35mm": self.focal_length.equivalent_35mm.to_dict()},
                "depth_of_field": {"type": self.depth_of_field.dof_type.to_dict(),
                                   "subject_sharp": self.depth_of_field.subject_sharp,
                                   "background_blur": round(self.depth_of_field.background_blur, 3),
                                   "aperture_range": self.depth_of_field.aperture_range.to_dict()},
                "motion_blur": {"type": self.motion_blur.blur_type.to_dict(),
                                "magnitude": round(self.motion_blur.blur_magnitude, 3),
                                "shutter_range": self.motion_blur.shutter_range.to_dict()},
                "composition": {"styles": self.composition.styles,
                                "subject_position": [round(v, 3) for v in self.composition.subject_position],
                                "subject_scale": round(self.composition.subject_scale, 3),
                                "headroom": round(self.composition.headroom, 3)},
                "shooting_techniques": self.shooting_techniques.techniques},
            "meta": {"overall_confidence": round(self.overall_confidence, 3),
                     "uncertainties": self.uncertainties}}

    def report(self) -> str:
        lines = ["=" * 55, "  PHOTO REVERSE ENGINEERING REPORT", "=" * 55, ""]
        lines.append("-- Camera Position --")
        cp = self.camera_pose
        lines.append(f"  height:    {cp.camera_height.value} m  (range {cp.camera_height.range_min}-{cp.camera_height.range_max} m)")
        lines.append(f"  distance:  {cp.camera_distance.value} m  (range {cp.camera_distance.range_min}-{cp.camera_distance.range_max} m)")
        lines.append(f"  pitch:     {cp.camera_pitch.value} deg")
        lines.append(f"  roll:      {cp.camera_roll.value} deg")
        lines.append("")
        lines.append("-- Lens --")
        fl = self.focal_length
        lines.append(f"  category:  {fl.category.value}  (conf {fl.category.confidence:.0%})")
        lines.append(f"  35mm eq:   {fl.equivalent_35mm.value} mm  (range {fl.equivalent_35mm.range_min}-{fl.equivalent_35mm.range_max})")
        lines.append("")
        lines.append("-- Exposure (estimated) --")
        dof = self.depth_of_field
        lines.append(f"  DOF type:  {dof.dof_type.value}  (conf {dof.dof_type.confidence:.0%})")
        lines.append(f"  aperture:  {dof.aperture_range.value}")
        mb = self.motion_blur
        lines.append(f"  shutter:   {mb.shutter_range.value}")
        lines.append("")
        lines.append("-- Composition --")
        for s in self.composition.styles[:3]:
            lines.append(f"  * {s['name']}  ({s['confidence']:.0%})")
        lines.append(f"  subject: ({self.composition.subject_position[0]:.2f}, {self.composition.subject_position[1]:.2f})")
        lines.append("")
        lines.append("-- Techniques --")
        for t in self.shooting_techniques.techniques[:5]:
            lines.append(f"  * {t['name']}  ({t['confidence']:.0%})")
        lines.append("")
        lines.append(f"-- Overall Confidence: {self.overall_confidence:.0%} --")
        if self.uncertainties:
            lines.append("  Uncertainties:")
            for u in self.uncertainties:
                lines.append(f"    * {u}")
        # Candidate solutions
        if self._sim_candidates:
            lines.append("")
            lines.append("-- Candidate Solutions --")
            for i, c in enumerate(self._sim_candidates[:5]):
                lines.append(f"  #{i+1}: {c.focal_equiv_35mm}mm / {c.distance}m / h={c.height}m  (score {c.score:.2f})")
        return "\n".join(lines)
