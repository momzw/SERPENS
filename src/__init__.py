try:
    from .cerpens import CerpensSimulation

    __all__ = ["CerpensSimulation"]

except (OSError, ImportError):
    pass