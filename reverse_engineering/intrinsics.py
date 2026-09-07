"""Camera-intrinsics evidence from EXIF and optional calibration metadata.

EXIF is observed evidence. Calibration profiles are explicit priors that can
fill otherwise unknown sensor/principal-point information without pretending
that the image itself revealed those values.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from reverse_engineering.calibration import CalibrationProfile


@dataclass(frozen=True)
class IntrinsicsEvidence:
    focal_length_mm: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    principal_point_x: Optional[float] = None
    principal_point_y: Optional[float] = None
    pixel_aspect_ratio: Optional[float] = None
    make: Optional[str] = None
    model: Optional[str] = None
    lens_model: Optional[str] = None
    source: str = "unknown"
    confidence: float = 0.0
    observed_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    calibration_profile: Optional[str] = None

    @property
    def has_focal_prior(self) -> bool:
        return self.focal_length_mm is not None or self.focal_length_35mm is not None

    def preferred_focal_mm(self) -> Optional[float]:
        if self.focal_length_mm is not None and self.focal_length_mm > 0:
            return float(self.focal_length_mm)
        if self.focal_length_35mm is not None and self.focal_length_35mm > 0:
            return float(self.focal_length_35mm)
        return None

    def to_dict(self) -> dict:
        return {
            "focal_length_mm": self.focal_length_mm,
            "focal_length_35mm": self.focal_length_35mm,
            "sensor_width_mm": self.sensor_width_mm,
            "sensor_height_mm": self.sensor_height_mm,
            "principal_point_x": self.principal_point_x,
            "principal_point_y": self.principal_point_y,
            "pixel_aspect_ratio": self.pixel_aspect_ratio,
            "make": self.make,
            "model": self.model,
            "lens_model": self.lens_model,
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
            "observed_fields": list(self.observed_fields),
            "notes": list(self.notes),
            "calibration_profile": self.calibration_profile,
        }


def _number(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def read_exif_intrinsics(
    path: str | Path,
    profile: Optional[CalibrationProfile] = None,
    image_size: Optional[tuple[int, int]] = None,
) -> IntrinsicsEvidence:
    """Read EXIF and optionally merge a user-selected calibration profile."""
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return IntrinsicsEvidence(source="unavailable", notes=("Pillow is not installed",))

    try:
        with Image.open(path) as image:
            raw = image.getexif()
            tags = {ExifTags.TAGS.get(key, key): value for key, value in raw.items()}
            if image_size is None:
                image_size = (image.width, image.height)
    except Exception as exc:
        return IntrinsicsEvidence(source="unavailable", notes=(f"EXIF read failed: {type(exc).__name__}",))

    focal = _number(tags.get("FocalLength"))
    focal_35 = _number(tags.get("FocalLengthIn35mmFilm"))
    make = str(tags.get("Make")) if tags.get("Make") else None
    model = str(tags.get("Model")) if tags.get("Model") else None
    lens = str(tags.get("LensModel")) if tags.get("LensModel") else None

    fields = []
    if focal is not None:
        fields.append("FocalLength")
    if focal_35 is not None:
        fields.append("FocalLengthIn35mmFilm")
    if make:
        fields.append("Make")
    if model:
        fields.append("Model")
    if lens:
        fields.append("LensModel")

    notes = []
    sensor_width = None
    sensor_height = None
    pp_x = None
    pp_y = None
    pixel_aspect = None
    if focal is not None and focal_35 is not None:
        crop = focal_35 / focal
        if 0.5 < crop < 8.0:
            sensor_width = 36.0 / crop
            notes.append(f"derived sensor width ≈ {sensor_width:.2f} mm from EXIF focal pair")
        confidence = 0.98
    elif focal is not None:
        confidence = 0.90
    elif focal_35 is not None:
        confidence = 0.72
        notes.append("35mm equivalent is observed; physical sensor width remains unknown")
    else:
        confidence = 0.0
        notes.append("no focal-length EXIF field")

    source = "EXIF"
    profile_name = None
    if profile is not None:
        profile_name = profile.name
        if sensor_width is None:
            sensor_width = profile.sensor_width_mm
        if sensor_height is None:
            sensor_height = profile.sensor_height_mm
        if image_size is not None:
            applied = profile.for_image(*image_size)
            pp_x = applied.principal_point_x
            pp_y = applied.principal_point_y
        else:
            pp_x = profile.principal_point_x
            pp_y = profile.principal_point_y
        pixel_aspect = profile.pixel_aspect_ratio
        source = "EXIF+CALIBRATION" if fields else "CALIBRATION"
        notes.append(f"calibration profile applied: {profile.name}")

    return IntrinsicsEvidence(
        focal_length_mm=focal,
        focal_length_35mm=focal_35,
        sensor_width_mm=sensor_width,
        sensor_height_mm=sensor_height,
        principal_point_x=pp_x,
        principal_point_y=pp_y,
        pixel_aspect_ratio=pixel_aspect,
        make=make,
        model=model,
        lens_model=lens,
        source=source,
        confidence=confidence,
        observed_fields=tuple(fields),
        notes=tuple(notes),
        calibration_profile=profile_name,
    )
