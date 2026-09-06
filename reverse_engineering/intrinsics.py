"""Camera-intrinsics evidence from EXIF and optional calibration metadata.

EXIF is treated as an observed source when the camera actually wrote the
metadata.  Values derived from 35mm-equivalent focal length are marked as
estimated because crop factor/sensor format may still be unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class IntrinsicsEvidence:
    focal_length_mm: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    make: Optional[str] = None
    model: Optional[str] = None
    lens_model: Optional[str] = None
    source: str = "unknown"
    confidence: float = 0.0
    observed_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

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
            "make": self.make,
            "model": self.model,
            "lens_model": self.lens_model,
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
            "observed_fields": list(self.observed_fields),
            "notes": list(self.notes),
        }


def _number(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def read_exif_intrinsics(path: str | Path) -> IntrinsicsEvidence:
    """Read focal/camera identity information from an image's EXIF block."""
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return IntrinsicsEvidence(source="unavailable", notes=("Pillow is not installed",))

    try:
        with Image.open(path) as image:
            raw = image.getexif()
            tags = {ExifTags.TAGS.get(key, key): value for key, value in raw.items()}
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
    if focal is not None and focal_35 is not None:
        # This ratio is a crop-factor estimate, not a guaranteed sensor spec.
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

    return IntrinsicsEvidence(
        focal_length_mm=focal,
        focal_length_35mm=focal_35,
        make=make,
        model=model,
        lens_model=lens,
        source="EXIF",
        confidence=confidence,
        observed_fields=tuple(fields),
        notes=tuple(notes),
    )
