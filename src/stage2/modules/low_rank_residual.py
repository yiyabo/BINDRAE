"""Graph-coupled low-rank decoder for endpoint-conditioned path residuals."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn


_BLOCK_COMPONENTS = {
    "rotation": 3,
    "translation": 3,
    "chi": 4,
}


class GraphCoupledLowRankResidualDecoder(nn.Module):
    """Factor a path residual into static spatial bases and global time scores.

    The residue features are expected to come from a graph encoder evaluated on
    fixed endpoint context. With fixed features, each output block has temporal
    rank at most ``rank`` before tangent-normal projection.
    """

    def __init__(
        self,
        residue_dim: int,
        time_dim: int,
        hidden_dim: int,
        rank: int,
        *,
        active_blocks: Iterable[str],
        rotation_gate_bias: float,
        translation_gate_bias: float,
        chi_gate_bias: float,
    ) -> None:
        super().__init__()
        if residue_dim <= 0 or time_dim <= 0 or hidden_dim <= 0:
            raise ValueError("decoder dimensions must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        active = frozenset(active_blocks)
        unknown = active - set(_BLOCK_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown residual blocks: {sorted(unknown)}")
        if not active:
            raise ValueError("At least one residual block must be active")

        self.residue_dim = int(residue_dim)
        self.time_dim = int(time_dim)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.active_blocks = active
        coefficient_dim = residue_dim + time_dim
        gate_biases = {
            "rotation": float(rotation_gate_bias),
            "translation": float(translation_gate_bias),
            "chi": float(chi_gate_bias),
        }

        self.basis_heads = nn.ModuleDict()
        self.coefficient_heads = nn.ModuleDict()
        self.gate_heads = nn.ModuleDict()
        for block, components in _BLOCK_COMPONENTS.items():
            self.basis_heads[block] = nn.Sequential(
                nn.LayerNorm(residue_dim),
                nn.Linear(residue_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, rank * components),
            )
            coefficient_head = nn.Sequential(
                nn.LayerNorm(coefficient_dim),
                nn.Linear(coefficient_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, rank),
            )
            nn.init.zeros_(coefficient_head[-1].weight)
            nn.init.zeros_(coefficient_head[-1].bias)
            self.coefficient_heads[block] = coefficient_head

            gate_head = nn.Sequential(
                nn.LayerNorm(residue_dim),
                nn.Linear(residue_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(gate_head[-1].weight)
            nn.init.constant_(gate_head[-1].bias, gate_biases[block])
            self.gate_heads[block] = gate_head

    @staticmethod
    def _masked_mean(features: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        weight = node_mask.to(dtype=features.dtype).unsqueeze(-1)
        return (features * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _normalize_basis(
        basis: torch.Tensor,
        component_mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        mask = component_mask.to(dtype=basis.dtype).unsqueeze(2)
        masked = basis * mask
        component_count = mask.sum(dim=(1, 3), keepdim=True).clamp(min=1.0)
        rms = torch.sqrt(
            masked.square().sum(dim=(1, 3), keepdim=True) / component_count
        ).clamp(min=eps)
        return masked / rms

    def forward(
        self,
        residue_features: torch.Tensor,
        time_embedding: torch.Tensor,
        node_mask: torch.Tensor,
        chi_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if residue_features.ndim != 3:
            raise ValueError("residue_features must have shape [B, N, D]")
        batch_size, n_residues, feature_dim = residue_features.shape
        if feature_dim != self.residue_dim:
            raise ValueError(
                f"Expected residue feature dim {self.residue_dim}, got {feature_dim}"
            )
        if time_embedding.shape != (batch_size, self.time_dim):
            raise ValueError(
                f"time_embedding must have shape {(batch_size, self.time_dim)}"
            )
        if node_mask.shape != (batch_size, n_residues):
            raise ValueError("node_mask shape mismatch")
        if chi_mask.shape != (batch_size, n_residues, 4):
            raise ValueError("chi_mask shape mismatch")

        node_mask = node_mask.bool()
        chi_mask = chi_mask.bool() & node_mask.unsqueeze(-1)
        pooled = self._masked_mean(residue_features, node_mask)
        coefficient_input = torch.cat([pooled, time_embedding], dim=-1)
        outputs: Dict[str, torch.Tensor] = {}
        active_gates = []

        for block, components in _BLOCK_COMPONENTS.items():
            basis = self.basis_heads[block](residue_features).view(
                batch_size,
                n_residues,
                self.rank,
                components,
            )
            component_mask = (
                chi_mask
                if block == "chi"
                else node_mask.unsqueeze(-1).expand(-1, -1, components)
            )
            basis = self._normalize_basis(basis, component_mask)
            gate = torch.sigmoid(self.gate_heads[block](residue_features))
            gate = gate * node_mask.unsqueeze(-1).to(dtype=gate.dtype)
            active_scale = float(block in self.active_blocks)
            gate = gate * active_scale
            basis = basis * gate.unsqueeze(2)
            coefficients = self.coefficient_heads[block](coefficient_input)
            coefficients = coefficients * active_scale
            residual = torch.einsum("bnkc,bk->bnc", basis, coefficients)
            residual = residual * component_mask.to(dtype=residual.dtype)

            outputs[f"residual_{block}"] = residual
            outputs[f"residual_{block}_gate"] = gate
            outputs[f"residual_{block}_coefficients"] = coefficients
            if block in self.active_blocks:
                active_gates.append(gate)

        outputs["residual_gate"] = torch.stack(active_gates, dim=-1).mean(dim=-1)
        return outputs
