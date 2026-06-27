"""Explicit local residue-ligand interaction prior.

This module is intentionally small and checkpoint-friendly.  It scores each
residue from apo/current side-chain geometry and known-pose ligand tokens, and
is used as a soft local prior rather than as a deterministic holo endpoint.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ligand_utils import LIGAND_TYPE_DIM


@dataclass
class InteractionPriorConfig:
    num_rbf: int = 24
    rbf_max: float = 14.0
    type_sigma: float = 4.5
    target_contact_dist: float = 4.5
    residue_chunk: int = 64
    hidden_dim: int = 192
    num_layers: int = 3
    dropout: float = 0.05

    @property
    def input_dim(self) -> int:
        return 21 + self.num_rbf + self.num_rbf + int(LIGAND_TYPE_DIM) + 4


def rbf_encode(d: torch.Tensor, num_rbf: int, d_max: float) -> torch.Tensor:
    centers = torch.linspace(0.0, float(d_max), int(num_rbf), device=d.device, dtype=d.dtype)
    width = float(d_max) / max(int(num_rbf) - 1, 1)
    return torch.exp(-0.5 * ((d.unsqueeze(-1) - centers) / max(width, 1e-6)).pow(2))


def sidechain_atom_mask(atom14_mask: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    atom_idx = torch.arange(atom14_mask.shape[-1], device=atom14_mask.device).view(1, 1, -1)
    return atom14_mask.bool() & (atom_idx >= 4) & node_mask.bool().unsqueeze(-1)


def min_sidechain_ligand_dist(
    atom14: torch.Tensor,
    atom_mask: torch.Tensor,
    lig_points: torch.Tensor,
    lig_mask: torch.Tensor,
    node_mask: torch.Tensor,
    residue_chunk: int = 64,
) -> torch.Tensor:
    bsz, n_res, n_atom, _ = atom14.shape
    n_lig = lig_points.shape[1]
    out = atom14.new_full((bsz, n_res), float("inf"))
    if n_lig == 0:
        return out
    chunk = max(int(residue_chunk), 1)
    for start in range(0, n_res, chunk):
        end = min(start + chunk, n_res)
        c = end - start
        dists = torch.cdist(
            atom14[:, start:end].reshape(bsz * c, n_atom, 3),
            lig_points[:, None, :, :].expand(bsz, c, n_lig, 3).reshape(bsz * c, n_lig, 3),
        ).reshape(bsz, c, n_atom, n_lig)
        valid = (
            atom_mask[:, start:end, :, None]
            & lig_mask[:, None, None, :].bool()
            & node_mask[:, start:end, None, None].bool()
        )
        dists = dists.masked_fill(~valid.expand_as(dists), float("inf"))
        out[:, start:end] = dists.amin(dim=(-1, -2))
    return out


def build_pair_features(
    atom14: torch.Tensor,
    atom_mask: torch.Tensor,
    ca_coords: torch.Tensor,
    aatype: torch.Tensor,
    lig_points: torch.Tensor,
    lig_types: torch.Tensor,
    lig_mask: torch.Tensor,
    node_mask: torch.Tensor,
    config: InteractionPriorConfig,
) -> torch.Tensor:
    bsz, n_res, n_atom, _ = atom14.shape
    n_lig = lig_points.shape[1]
    dtype = atom14.dtype
    min_dist = atom14.new_full((bsz, n_res), 50.0)
    rbf_sum = atom14.new_zeros((bsz, n_res, int(config.num_rbf)))
    type_sum = atom14.new_zeros((bsz, n_res, lig_types.shape[-1]))
    weight_sum = atom14.new_zeros((bsz, n_res, 1))
    close_count = atom14.new_zeros((bsz, n_res, 1))
    has_ligand = n_lig > 0 and bool(lig_mask.any().item())

    if has_ligand:
        chunk = max(int(config.residue_chunk), 1)
        for start in range(0, n_res, chunk):
            end = min(start + chunk, n_res)
            c = end - start
            dists = torch.cdist(
                atom14[:, start:end].reshape(bsz * c, n_atom, 3),
                lig_points[:, None, :, :].expand(bsz, c, n_lig, 3).reshape(bsz * c, n_lig, 3),
            ).reshape(bsz, c, n_atom, n_lig)
            valid = (
                atom_mask[:, start:end, :, None]
                & lig_mask[:, None, None, :].bool()
                & node_mask[:, start:end, None, None].bool()
            )
            dists = dists.masked_fill(~valid.expand_as(dists), 50.0)
            min_dist[:, start:end] = dists.amin(dim=(-1, -2)).clamp(max=50.0)

            atom_min = dists.amin(dim=-1).clamp(max=float(config.rbf_max))
            atom_valid = atom_mask[:, start:end].float()
            atom_count = atom_valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
            rbf = rbf_encode(atom_min, int(config.num_rbf), float(config.rbf_max))
            rbf_sum[:, start:end] = (rbf * atom_valid.unsqueeze(-1)).sum(dim=2) / atom_count

            pair_weight = torch.exp(-0.5 * (dists / max(float(config.type_sigma), 1e-6)).pow(2))
            pair_weight = pair_weight * valid.float()
            type_weight = pair_weight.sum(dim=2)
            denom = type_weight.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            type_sum[:, start:end] = torch.einsum("bcm,bmd->bcd", type_weight, lig_types.to(dtype)) / denom
            weight_sum[:, start:end] = denom
            close_count[:, start:end, 0] = (dists < float(config.target_contact_dist)).float().sum(dim=(-1, -2))

    if has_ligand:
        ca_d = torch.cdist(ca_coords, lig_points).masked_fill(~lig_mask[:, None, :].bool(), 50.0)
        ca_d = ca_d.amin(dim=-1).clamp(max=50.0)
    else:
        ca_d = ca_coords.new_full((bsz, n_res), 50.0)
    ca_rbf = rbf_encode(ca_d.clamp(max=float(config.rbf_max)), int(config.num_rbf), float(config.rbf_max))
    aa_onehot = F.one_hot(aatype.clamp(0, 20), 21).float()
    scalar = torch.stack(
        [
            min_dist.clamp(max=50.0) / 10.0,
            torch.log1p(weight_sum.squeeze(-1)),
            torch.log1p(close_count.squeeze(-1)),
            lig_mask.float().sum(dim=-1, keepdim=True).expand(-1, n_res) / 128.0,
        ],
        dim=-1,
    )
    return torch.cat([aa_onehot, rbf_sum, ca_rbf, type_sum, scalar], dim=-1)


class InteractionPriorNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(input_dim)]
        dim = input_dim
        for _ in range(max(int(num_layers), 1)):
            layers.extend([
                nn.Linear(dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class InteractionPriorScorer(nn.Module):
    def __init__(self, config: Optional[InteractionPriorConfig] = None):
        super().__init__()
        self.config = config or InteractionPriorConfig()
        self.net = InteractionPriorNet(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.num_layers,
            self.config.dropout,
        )

    def forward(
        self,
        atom14: torch.Tensor,
        atom_mask: torch.Tensor,
        ca_coords: torch.Tensor,
        aatype: torch.Tensor,
        lig_points: torch.Tensor,
        lig_types: torch.Tensor,
        lig_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = build_pair_features(
            atom14,
            atom_mask,
            ca_coords,
            aatype,
            lig_points,
            lig_types,
            lig_mask,
            node_mask,
            self.config,
        )
        return self.net(features)


def interaction_prior_config_from_args(args) -> InteractionPriorConfig:
    return InteractionPriorConfig(
        num_rbf=int(args.num_rbf),
        rbf_max=float(args.rbf_max),
        type_sigma=float(args.type_sigma),
        target_contact_dist=float(args.target_contact_dist),
        residue_chunk=int(args.residue_chunk),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    )


def load_interaction_prior(
    checkpoint_path: str,
    device: torch.device,
    fallback_config: Optional[InteractionPriorConfig] = None,
) -> InteractionPriorScorer:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    if isinstance(args, dict):
        config = InteractionPriorConfig(
            num_rbf=int(args.get("num_rbf", 24)),
            rbf_max=float(args.get("rbf_max", 14.0)),
            type_sigma=float(args.get("type_sigma", 4.5)),
            target_contact_dist=float(args.get("target_contact_dist", 4.5)),
            residue_chunk=int(args.get("residue_chunk", 64)),
            hidden_dim=int(args.get("hidden_dim", 192)),
            num_layers=int(args.get("num_layers", 3)),
            dropout=float(args.get("dropout", 0.05)),
        )
    else:
        config = fallback_config or InteractionPriorConfig()
    model = InteractionPriorScorer(config).to(device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    result = model.net.load_state_dict(state, strict=False)
    if result.unexpected_keys or result.missing_keys:
        raise RuntimeError(
            f"Interaction prior checkpoint mismatch: "
            f"unexpected={result.unexpected_keys}, missing={result.missing_keys}"
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
