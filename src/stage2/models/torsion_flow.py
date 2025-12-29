"""Stage-2 TorsionFlowNet (ligand-conditioned, pocket-gated bridge flow)."""

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

# FlashIPA path
import sys
import os
from pathlib import Path

flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.rigid import Rigid

# Project root
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.stage1.models.adapter import ESMAdapter
from src.stage1.models.ipa import FlashIPAModule, FlashIPAModuleConfig
from src.stage1.models.ligand_condition import LigandConditioner, LigandConditionerConfig
from src.stage1.modules.edge_embed import EdgeEmbedderAdapter, ProjectEdgeConfig
from src.stage2.modules.time_embed import SinusoidalTimeEmbedding
from src.stage2.modules import se3_log, rigid_inverse, rigid_compose, wrap_to_pi


@dataclass
class TorsionFlowNetConfig:
    # ESM Adapter
    esm_dim: int = 1280
    c_s: int = 384

    # EdgeEmbedder
    c_p: int = 128
    z_factor_rank: int = 2
    num_rbf: int = 16

    # FlashIPA
    c_hidden: int = 128
    no_heads: int = 8
    depth: int = 3
    no_qk_points: int = 8
    no_v_points: int = 12

    # LigandConditioner
    d_lig: int = 64
    num_heads_cross: int = 8
    warmup_steps: int = 2000

    # Time embedding
    time_dim: int = 64

    # Optional NMA feature dim (0 = disabled)
    nma_dim: int = 0

    # Stage-1 relative features
    delta_f_dim: int = 6
    delta_chi_dim: int = 4

    # Heads
    head_hidden: int = 256

    # Dropout
    dropout: float = 0.1


class TorsionFlowNet(nn.Module):
    """Ligand-conditioned pocket-gated vector field on (rigids, chi)."""

    def __init__(self, config: Optional[TorsionFlowNetConfig] = None):
        super().__init__()
        self.config = config or TorsionFlowNetConfig()

        # 1) ESM adapter
        self.esm_adapter = ESMAdapter(
            esm_dim=self.config.esm_dim,
            output_dim=self.config.c_s,
            dropout=self.config.dropout,
        )

        # 2) Edge embedder
        edge_config = ProjectEdgeConfig(
            c_s=self.config.c_s,
            c_p=self.config.c_p,
            z_factor_rank=self.config.z_factor_rank,
            num_rbf=self.config.num_rbf,
        )
        self.edge_embedder = EdgeEmbedderAdapter(edge_config)

        # 3) FlashIPA trunk
        ipa_config = FlashIPAModuleConfig(
            c_s=self.config.c_s,
            c_z=self.config.c_p,
            c_hidden=self.config.c_hidden,
            no_heads=self.config.no_heads,
            depth=self.config.depth,
            no_qk_points=self.config.no_qk_points,
            no_v_points=self.config.no_v_points,
            z_factor_rank=self.config.z_factor_rank,
            dropout=self.config.dropout,
        )
        self.ipa_module = FlashIPAModule(ipa_config)

        # 4) Ligand conditioner
        ligand_config = LigandConditionerConfig(
            c_s=self.config.c_s,
            d_lig=self.config.d_lig,
            num_heads=self.config.num_heads_cross,
            dropout=self.config.dropout,
            warmup_steps=self.config.warmup_steps,
        )
        self.ligand_conditioner = LigandConditioner(ligand_config)

        # 5) Time embedding
        self.time_embed = SinusoidalTimeEmbedding(self.config.time_dim)

        # 6) Gate + heads
        delta_in_dim = self.config.delta_f_dim + self.config.delta_chi_dim
        gate_in_dim = self.config.c_s + 1 + self.config.time_dim + self.config.nma_dim + delta_in_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, self.config.head_hidden),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden, 1),
        )

        chi_in_dim = self.config.c_s + 8 + 1 + self.config.time_dim + self.config.nma_dim + delta_in_dim
        self.chi_head = nn.Sequential(
            nn.Linear(chi_in_dim, self.config.head_hidden),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden, 4),
        )

        rigid_in_dim = self.config.c_s + 1 + self.config.time_dim + self.config.nma_dim + delta_in_dim
        self.rigid_head = nn.Sequential(
            nn.Linear(rigid_in_dim, self.config.head_hidden),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden, 6),
        )

    def forward(self,
                chi: torch.Tensor,           # [B, N, 4]
                rigids: Rigid,              # Rigid[B, N]
                esm: torch.Tensor,          # [B, N, 1280]
                lig_points: torch.Tensor,   # [B, M, 3]
                lig_types: torch.Tensor,    # [B, M, 20]
                lig_mask: torch.Tensor,     # [B, M]
                w_res: torch.Tensor,        # [B, N]
                t: torch.Tensor,            # [B]
                node_mask: Optional[torch.Tensor] = None,
                nma_features: Optional[torch.Tensor] = None,
                stage1_chi: Optional[torch.Tensor] = None,     # [B, N, 4]
                stage1_rigids: Optional[Rigid] = None,         # Rigid[B, N]
                current_step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with d_chi, d_rigid_rot, d_rigid_trans, gate
        """
        B, N, _ = chi.shape

        # time embedding
        t_emb = self.time_embed(t).unsqueeze(1).expand(B, N, -1)

        # ESM adapter
        s = self.esm_adapter(esm)

        # ligand conditioning (pre-IPA)
        s = self.ligand_conditioner(
            s,
            lig_points,
            lig_types,
            node_mask,
            lig_mask,
            current_step=current_step,
        )

        # edge features (use CA as translation of rigids)
        ca_coords = rigids.get_trans()
        edge_outputs = self.edge_embedder(s, ca_coords, node_mask)
        z_f1 = edge_outputs['z_f1']
        z_f2 = edge_outputs['z_f2']

        # FlashIPA (encoder rigids only; do not overwrite state rigids)
        s_geo, _ = self.ipa_module(
            s,
            rigids,
            z_f1,
            z_f2,
            node_mask,
            ligand_conditioner=self.ligand_conditioner,
            lig_points=lig_points,
            lig_types=lig_types,
            protein_mask=node_mask,
            ligand_mask=lig_mask,
            current_step=current_step,
        )

        if self.config.nma_dim > 0 and nma_features is None:
            raise ValueError("nma_dim > 0 but nma_features is None")
        if self.config.nma_dim == 0 and nma_features is not None:
            raise ValueError("nma_features provided but nma_dim is 0")
        if self.config.nma_dim > 0 and nma_features is not None:
            if nma_features.ndim == 2:
                nma_features = nma_features.unsqueeze(-1)
            if nma_features.shape[-1] != self.config.nma_dim:
                raise ValueError("nma_features last dim does not match nma_dim")

        if (stage1_chi is None) ^ (stage1_rigids is None):
            raise ValueError("stage1_chi and stage1_rigids must be both provided or both None")
        if stage1_chi is None:
            delta_chi = chi.new_zeros(B, N, self.config.delta_chi_dim)
            delta_f = chi.new_zeros(B, N, self.config.delta_f_dim)
        else:
            if stage1_chi.shape != chi.shape:
                raise ValueError("stage1_chi shape mismatch with chi")
            delta_chi = wrap_to_pi(stage1_chi - chi)
            R_curr, t_curr = rigids.get_rots().get_rot_mats(), rigids.get_trans()
            R1, t1 = stage1_rigids.get_rots().get_rot_mats(), stage1_rigids.get_trans()
            R_inv, t_inv = rigid_inverse(R_curr, t_curr)
            R_delta, t_delta = rigid_compose(R_inv, t_inv, R1, t1)
            delta_f = se3_log(R_delta, t_delta)

        # gate input (optionally append NMA features)
        gate_input = torch.cat([s_geo, w_res.unsqueeze(-1), t_emb, delta_f, delta_chi], dim=-1)
        if self.config.nma_dim > 0:
            if nma_features.ndim == 2:
                nma_features = nma_features.unsqueeze(-1)
            gate_input = torch.cat([gate_input, nma_features], dim=-1)

        gate = torch.sigmoid(self.gate_mlp(gate_input))  # [B, N, 1]

        # chi velocity
        sin_cos = torch.stack([torch.sin(chi), torch.cos(chi)], dim=-1).reshape(B, N, 8)
        chi_input = torch.cat([s_geo, sin_cos, w_res.unsqueeze(-1), t_emb, delta_f, delta_chi], dim=-1)
        if self.config.nma_dim > 0:
            chi_input = torch.cat([chi_input, nma_features], dim=-1)
        d_chi = self.chi_head(chi_input)
        d_chi = d_chi * gate

        # rigid velocity
        rigid_input = torch.cat([s_geo, w_res.unsqueeze(-1), t_emb, delta_f, delta_chi], dim=-1)
        if self.config.nma_dim > 0:
            rigid_input = torch.cat([rigid_input, nma_features], dim=-1)
        rigid_vel = self.rigid_head(rigid_input)
        d_rot = rigid_vel[..., :3] * gate.squeeze(-1)
        d_trans = rigid_vel[..., 3:] * gate.squeeze(-1)

        # mask padded residues
        if node_mask is not None:
            mask = node_mask.unsqueeze(-1).float()
            d_chi = d_chi * mask
            d_rot = d_rot * mask
            d_trans = d_trans * mask
            gate = gate * mask

        return {
            "d_chi": d_chi,
            "d_rigid_rot": d_rot,
            "d_rigid_trans": d_trans,
            "gate": gate,
        }
