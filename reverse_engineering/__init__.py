"""Photography reverse engineering package.

v2 separates observed evidence, scene geometry, framing candidates and
camera-rotation inference.
"""

__all__ = ["ReverseEngineeringEngineV2"]


def __getattr__(name):
    if name == "ReverseEngineeringEngineV2":
        from reverse_engineering.engine_v2 import ReverseEngineeringEngineV2
        return ReverseEngineeringEngineV2
    raise AttributeError(name)
