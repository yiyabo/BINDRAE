"""
Stage-1 核心模块集合

包含：
- edge_embed: FlashIPA EdgeEmbedder 适配层
- losses: 训练损失函数
"""

from .edge_embed import (
    EdgeEmbedderAdapter,
    ProjectEdgeConfig,
    create_edge_embedder,
)

from .losses import (
    fape_loss,
    torsion_loss,
    distance_loss,
    clash_penalty,
)

__all__ = [
    'EdgeEmbedderAdapter',
    'ProjectEdgeConfig',
    'create_edge_embedder',
    'fape_loss',
    'torsion_loss',
    'distance_loss',
    'clash_penalty',
]

