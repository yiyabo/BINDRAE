"""Time embedding utilities for Stage-2."""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int = 64):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] in [0,1]
        Returns:
            emb: [B, dim]
        """
        device = t.device
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=device, dtype=t.dtype) *
            (-math.log(10000.0) / (half - 1))
        )
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb
