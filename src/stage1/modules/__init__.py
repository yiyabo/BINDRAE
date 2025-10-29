"""
Stage-1 核心模块集合

包含：
- edge_embed: FlashIPA EdgeEmbedder 适配层
"""

from .edge_embed import (
    EdgeEmbedderAdapter,
    ProjectEdgeConfig,
    create_edge_embedder,
)

__all__ = [
    'EdgeEmbedderAdapter',
    'ProjectEdgeConfig',
    'create_edge_embedder',
]

