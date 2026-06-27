"""Stage-1 核心模块集合。"""

from .losses import (
    fape_loss,
    torsion_loss,
    distance_loss,
    clash_penalty,
)

__all__ = [
    'fape_loss',
    'torsion_loss',
    'distance_loss',
    'clash_penalty',
]

try:
    from .edge_embed import (
        EdgeEmbedderAdapter,
        ProjectEdgeConfig,
        create_edge_embedder,
    )

    __all__.extend([
        'EdgeEmbedderAdapter',
        'ProjectEdgeConfig',
        'create_edge_embedder',
    ])
except ModuleNotFoundError as exc:
    if exc.name != 'flash_ipa':
        raise
