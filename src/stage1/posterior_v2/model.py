"""Compact teacher-distilled Stage-1-v2 posterior student."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stage1.models.adapter import ESMAdapter
from src.stage1.models.ligand_condition import LigandConditioner, LigandConditionerConfig


@dataclass
class Stage1PosteriorV2Config:
    esm_dim: int = 1280
    c_s: int = 256
    d_lig: int = 64
    num_heads_cross: int = 8
    num_rbf: int = 24
    rbf_max: float = 16.0
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    output_latent_dim: int = 64
    use_latent_head: bool = False
    warmup_steps: int = 0


def rbf_encode(d: torch.Tensor, num_rbf: int, d_max: float) -> torch.Tensor:
    centers = torch.linspace(0.0, float(d_max), int(num_rbf), device=d.device, dtype=d.dtype)
    width = float(d_max) / max(int(num_rbf) - 1, 1)
    return torch.exp(-0.5 * ((d.unsqueeze(-1) - centers) / max(width, 1e-6)).pow(2))


class Stage1PosteriorV2(nn.Module):
    """Predict local ligand-causal posterior labels from apo + known ligand pose."""

    def __init__(self, config: Optional[Stage1PosteriorV2Config] = None):
        super().__init__()
        self.config = config or Stage1PosteriorV2Config()
        self.esm_adapter = ESMAdapter(
            esm_dim=self.config.esm_dim,
            output_dim=self.config.c_s,
            dropout=self.config.dropout,
        )
        self.ligand_conditioner = LigandConditioner(
            LigandConditionerConfig(
                c_s=self.config.c_s,
                d_lig=self.config.d_lig,
                num_heads=self.config.num_heads_cross,
                dropout=self.config.dropout,
                warmup_steps=self.config.warmup_steps,
            )
        )
        scalar_dim = 8
        pair_dim = self.config.num_rbf * 2 + scalar_dim
        trunk_in = self.config.c_s + pair_dim
        layers = [nn.LayerNorm(trunk_in)]
        dim = trunk_in
        for _ in range(max(int(self.config.num_layers), 1)):
            layers.extend(
                [
                    nn.Linear(dim, int(self.config.hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(self.config.dropout)),
                ]
            )
            dim = int(self.config.hidden_dim)
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict(
            {
                "contact_logit": nn.Linear(dim, 1),
                "approach_logit": nn.Linear(dim, 1),
                "release_logit": nn.Linear(dim, 1),
                "switch_logit": nn.Linear(dim, 1),
                "confidence_logit": nn.Linear(dim, 1),
                "teacher_min_dist": nn.Linear(dim, 1),
                "signed_delta_dist": nn.Linear(dim, 1),
            }
        )
        self.latent_head = (
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(self.config.output_latent_dim)))
            if self.config.use_latent_head
            else None
        )

    def _pair_features(
        self,
        ca_coords: torch.Tensor,
        lig_points: torch.Tensor,
        lig_mask: torch.Tensor,
        w_res: torch.Tensor,
        node_mask: Optional[torch.Tensor],
        torsion_apo: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N, _ = ca_coords.shape
        if lig_points.shape[1] == 0 or not bool(lig_mask.any().item()):
            min_ca_dist = ca_coords.new_full((B, N), 50.0)
        else:
            d = torch.cdist(ca_coords, lig_points)
            d = d.masked_fill(~lig_mask[:, None, :].bool(), 50.0)
            min_ca_dist = d.amin(dim=-1).clamp(max=50.0)

        ca_rbf = rbf_encode(min_ca_dist.clamp(max=float(self.config.rbf_max)), self.config.num_rbf, self.config.rbf_max)
        w_rbf = rbf_encode((1.0 - w_res.clamp(0.0, 1.0)) * float(self.config.rbf_max), self.config.num_rbf, self.config.rbf_max)
        if torsion_apo is None:
            chi_scalar = ca_coords.new_zeros(B, N, 4)
        else:
            chi = torsion_apo[..., 3:7]
            chi_scalar = torch.stack([torch.sin(chi[..., 0]), torch.cos(chi[..., 0]), torch.sin(chi[..., 1]), torch.cos(chi[..., 1])], dim=-1)
        scalar = torch.stack(
            [
                min_ca_dist.clamp(max=50.0) / 10.0,
                w_res.float(),
                torch.log1p(lig_mask.float().sum(dim=-1, keepdim=True).expand(-1, N)) / 5.0,
                (node_mask.float() if node_mask is not None else torch.ones_like(w_res)),
            ],
            dim=-1,
        )
        pair = torch.cat([ca_rbf, w_rbf, scalar, chi_scalar], dim=-1)
        return pair

    def forward(
        self,
        esm: torch.Tensor,
        Ca_apo: torch.Tensor,
        lig_points: torch.Tensor,
        lig_types: torch.Tensor,
        lig_mask: torch.Tensor,
        w_res: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        torsion_apo: Optional[torch.Tensor] = None,
        current_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        s = self.esm_adapter(esm)
        s = self.ligand_conditioner(
            s,
            lig_points,
            lig_types,
            node_mask,
            lig_mask,
            current_step=current_step,
        )
        pair = self._pair_features(Ca_apo, lig_points, lig_mask, w_res, node_mask, torsion_apo)
        h = self.trunk(torch.cat([s, pair.to(dtype=s.dtype)], dim=-1))
        out = {name: head(h).squeeze(-1).float() for name, head in self.heads.items()}
        out["teacher_min_dist_raw"] = out["teacher_min_dist"]
        out["teacher_min_dist"] = F.softplus(out["teacher_min_dist"])
        out["contact_prob"] = torch.sigmoid(out["contact_logit"])
        out["approach_prob"] = torch.sigmoid(out["approach_logit"])
        out["release_prob"] = torch.sigmoid(out["release_logit"])
        out["switch_prob"] = torch.sigmoid(out["switch_logit"])
        out["confidence"] = torch.sigmoid(out["confidence_logit"])
        if self.latent_head is not None:
            out["z_post"] = F.normalize(self.latent_head(h).float(), dim=-1)
        if node_mask is not None:
            mask = node_mask.float()
            for key, value in list(out.items()):
                if value.ndim == 2:
                    out[key] = value * mask
                elif value.ndim == 3:
                    out[key] = value * mask.unsqueeze(-1)
        return out
