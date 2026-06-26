"""
Stage-2 package (apo -> holo bridge flow).
"""

from .datasets import Stage2Batch, ApoHoloBridgeDataset, create_stage2_dataloader
from . import models as _models
from . import training as _training

TorsionFlowNet = getattr(_models, "TorsionFlowNet", None)
TorsionFlowNetConfig = getattr(_models, "TorsionFlowNetConfig", None)
TrainingConfig = _training.TrainingConfig
Stage2Trainer = getattr(_training, "Stage2Trainer", None)

__all__ = [
    "Stage2Batch",
    "ApoHoloBridgeDataset",
    "create_stage2_dataloader",
    "TrainingConfig",
]

if TorsionFlowNet is not None and TorsionFlowNetConfig is not None:
    __all__.extend([
        "TorsionFlowNet",
        "TorsionFlowNetConfig",
    ])

if Stage2Trainer is not None:
    __all__.append("Stage2Trainer")
