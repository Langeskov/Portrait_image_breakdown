"""Photography reverse engineering package."""
__all__ = ["ReverseEngineeringEngine"]

def __getattr__(name):
    if name == "ReverseEngineeringEngine":
        from reverse_engineering.engine import ReverseEngineeringEngine
        return ReverseEngineeringEngine
    raise AttributeError(name)
