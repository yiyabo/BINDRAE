"""Lightweight latent-change predictor used by Change-Prediction RAE."""

import torch
import torch.nn as nn


class DeltaZPredictor(nn.Module):
    """Predict per-residue latent change from conditioned node features."""

    def __init__(self, c_s: int, hidden: int = 256, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = [nn.LayerNorm(c_s)]
        in_dim = c_s
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, c_s))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s_with_ligand: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        delta_z = self.mlp(s_with_ligand)
        return delta_z * node_mask.unsqueeze(-1)
