"""Image-space reference anchors used by v3 reference reconstruction."""
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class ReferenceImageAnchor:
    """Observable point in an image; not a world-space scene geometry anchor."""
    name: str
    x: float
    y: float
    confidence: float
    source: str

    def to_dict(self) -> dict:
        return asdict(self)
