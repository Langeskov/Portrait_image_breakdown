"""Compatibility entry point for the current reverse-engineering engine.

The application historically imported ``ReverseEngineeringEngine`` from this
module. v2.5 is now the canonical implementation, so keep the old import path
as a thin alias instead of maintaining two divergent pipelines.
"""
from __future__ import annotations

from reverse_engineering.engine_v2 import ReverseEngineeringEngineV2


# Public compatibility alias.  GUI code and third-party callers importing the
# historical name now receive the same engine covered by the v2.5 regression
# suite, including calibration, depth constraints, image refinement and the
# conservative multi-person layout integration.
ReverseEngineeringEngine = ReverseEngineeringEngineV2

__all__ = ["ReverseEngineeringEngine", "ReverseEngineeringEngineV2"]
