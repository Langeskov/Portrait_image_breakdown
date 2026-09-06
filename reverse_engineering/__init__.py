"""Photography reverse engineering package.

v2 separates observed evidence, scene geometry, framing candidates and
camera-rotation inference.  Public consumers can use ``ReverseEngineeringEngineV2``
as the active pipeline.
"""

from reverse_engineering.engine_v2 import ReverseEngineeringEngineV2

__all__ = ["ReverseEngineeringEngineV2"]
