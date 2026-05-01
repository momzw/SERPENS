try:
    from .cerpens_simulation import CerpensSimulation

    __all__ = ["CerpensSimulation"]

except (OSError, ImportError):
    pass