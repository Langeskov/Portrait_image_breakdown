"""Explicit application-service wiring for the desktop application."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from reverse_engineering.engine_v2 import ReverseEngineeringEngineV2

@dataclass(frozen=True)
class ApplicationServices:
    engine_factory: Callable[..., object]
    image_cache_key: Callable[[object], str]

    @classmethod
    def create(cls, *, calibration_profile: str = "Generic") -> "ApplicationServices":
        def engine_factory(enable_simulation: bool = True, **kwargs):
            return ReverseEngineeringEngineV2(enable_simulation=enable_simulation, calibration_profile=calibration_profile, **kwargs)
        from gui.cache import image_cache_key
        return cls(engine_factory=engine_factory, image_cache_key=image_cache_key)
