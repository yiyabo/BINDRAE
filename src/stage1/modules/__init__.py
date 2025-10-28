"""
Stage-1 核心模块集合

包含：
- edge_embed: 边嵌入封装（FlashIPA EdgeEmbedder）
"""

from .edge_embed import (
    EdgeEmbedderWrapper,
    EdgeEmbedderConfig,
    GaussianRBF,
    create_edge_embedder,
)

__all__ = [
    'EdgeEmbedderWrapper',
    'EdgeEmbedderConfig',
    'GaussianRBF',
    'create_edge_embedder',
]

