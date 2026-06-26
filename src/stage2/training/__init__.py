"""
Stage-2 training.
"""

from .config import TrainingConfig

__all__ = [
    "TrainingConfig",
]

try:
    from .trainer import Stage2Trainer

    __all__.append("Stage2Trainer")
except ModuleNotFoundError as exc:
    if exc.name != "flash_ipa":
        raise
