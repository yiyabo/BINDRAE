"""
Stage-1 训练模块
"""

from .config import TrainingConfig

__all__ = ['TrainingConfig', 'Stage1Trainer']


def __getattr__(name):
    if name == 'Stage1Trainer':
        from .trainer import Stage1Trainer
        return Stage1Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
