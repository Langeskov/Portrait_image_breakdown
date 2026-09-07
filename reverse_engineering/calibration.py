"""Camera calibration profiles for v2.5 intrinsic priors."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CalibrationProfile:
    """Optional camera-specific intrinsic calibration information."""

    name: str
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    pixel_aspect_ratio: float = 1.0
    principal_point_x: Optional[float] = None
    principal_point_y: Optional[float] = None
    default_focal_length_mm: Optional[float] = None
    note: str = ""

    def for_image(self, width: int, height: int) -> "CalibrationProfile":
        """Fill an unspecified principal point with the image center."""
        return replace(
            self,
            principal_point_x=(self.principal_point_x if self.principal_point_x is not None else width / 2.0),
            principal_point_y=(self.principal_point_y if self.principal_point_y is not None else height / 2.0),
        )

    @property
    def has_sensor_size(self) -> bool:
        return bool(self.sensor_width_mm and self.sensor_height_mm)

    @property
    def has_principal_point(self) -> bool:
        return self.principal_point_x is not None and self.principal_point_y is not None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sensor_width_mm": self.sensor_width_mm,
            "sensor_height_mm": self.sensor_height_mm,
            "pixel_aspect_ratio": self.pixel_aspect_ratio,
            "principal_point_x": self.principal_point_x,
            "principal_point_y": self.principal_point_y,
            "default_focal_length_mm": self.default_focal_length_mm,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationProfile":
        return cls(
            name=str(data.get("name", "Custom")),
            sensor_width_mm=data.get("sensor_width_mm"),
            sensor_height_mm=data.get("sensor_height_mm"),
            pixel_aspect_ratio=float(data.get("pixel_aspect_ratio", 1.0)),
            principal_point_x=data.get("principal_point_x"),
            principal_point_y=data.get("principal_point_y"),
            default_focal_length_mm=data.get("default_focal_length_mm"),
            note=str(data.get("note", "")),
        )


BUILTIN_PROFILES = {
    "Generic": CalibrationProfile("Generic", note="No sensor-size prior; safest default."),
    "Full Frame 36x24": CalibrationProfile("Full Frame 36x24", 36.0, 24.0, note="35mm full-frame prior."),
    "APS-C 23.5x15.6": CalibrationProfile("APS-C 23.5x15.6", 23.5, 15.6, note="Common APS-C prior."),
    "Micro Four Thirds 17.3x13": CalibrationProfile("Micro Four Thirds 17.3x13", 17.3, 13.0, note="Micro Four Thirds prior."),
}


def resolve_profile(name: str) -> CalibrationProfile:
    return BUILTIN_PROFILES.get(name, BUILTIN_PROFILES["Generic"])


def load_profile(path: str | Path) -> CalibrationProfile:
    return CalibrationProfile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_profile(profile: CalibrationProfile, path: str | Path) -> None:
    Path(path).write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
