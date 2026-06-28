"""Stage-2 TorsionFlowNet (ligand-conditioned, pocket-gated bridge flow)."""

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

# Project root & FlashIPA path
import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# FlashIPA path (项目内 vendor 目录)
flash_ipa_path = str(project_root / 'vendor' / 'flash_ipa' / 'src')
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.rigid import Rigid

from src.stage1.models.adapter import ESMAdapter, ESMLayerFusionAdapter
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
    esm_fusion_enabled: bool = False
    esm_num_layers: int = 1
    esm_fusion_mode: str = "sum"
    esm_layer_dropout: float = 0.0

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
    stage1_chi_feature_scale: float = 1.0
    interaction_prior_feature_dim: int = 0
    interaction_prior_feature_scale: float = 1.0

    # REPA-style auxiliary alignment (disabled by default)
    repa_enabled: bool = False
    repa_dim: int = 128
    repa_target_dim: int = 0

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
        if self.config.esm_fusion_enabled:
            self.esm_adapter = ESMLayerFusionAdapter(
                esm_dim=self.config.esm_dim,
                output_dim=self.config.c_s,
                num_layers=self.config.esm_num_layers,
                fusion_mode=self.config.esm_fusion_mode,
                layer_dropout=self.config.esm_layer_dropout,
                dropout=self.config.dropout,
            )
        else:
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

        # Optional REPA-style projection from the geometry trunk hidden state
        # into the fixed posterior/oracle feature space. The target features
        # remain external and frozen; only this student projector is trained.
        self.repa_student_proj = None
        if self.config.repa_enabled:
            if self.config.repa_target_dim <= 0:
                raise ValueError("repa_enabled=True requires repa_target_dim > 0")
            if self.config.repa_dim <= 0:
                raise ValueError("repa_dim must be > 0 when repa is enabled")
            self.repa_student_proj = nn.Sequential(
                nn.LayerNorm(self.config.c_s),
                nn.Linear(self.config.c_s, self.config.repa_dim),
                nn.SiLU(),
                nn.Linear(self.config.repa_dim, self.config.repa_target_dim),
            )

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

        # 6) Optional posterior/prior residue injection.
        # Bias-free layers keep all-zero controls exactly equivalent to no
        # residue-level prior injection.
        delta_in_dim = self.config.delta_f_dim + self.config.delta_chi_dim
        prior_dim = self.config.interaction_prior_feature_dim
        self.prior_residue_proj = None
        if prior_dim > 0:
            self.prior_residue_proj = nn.Sequential(
                nn.Linear(prior_dim, self.config.c_s, bias=False),
                nn.SiLU(),
                nn.Linear(self.config.c_s, self.config.c_s, bias=False),
            )

        # 7) Gate + heads
        gate_in_dim = self.config.c_s + 1 + prior_dim + self.config.time_dim + self.config.nma_dim + delta_in_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, self.config.head_hidden),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden, 1),
        )

        chi_in_dim = self.config.c_s + 8 + 1 + prior_dim + self.config.time_dim + self.config.nma_dim + delta_in_dim
        self.chi_head = nn.Sequential(
            nn.Linear(chi_in_dim, self.config.head_hidden),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden, 4),
        )

        rigid_in_dim = self.config.c_s + 1 + prior_dim + self.config.time_dim + self.config.nma_dim + delta_in_dim
        self.rigid_head = nn.Sequential(
            nn.Linear(rigid_in_dim, self.config.head_hidden),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden, 6),
        )

    def forward(self,
                chi: torch.Tensor,           # [B, N, 4]
                rigids: Rigid,              # Rigid[B, N]
                esm: torch.Tensor,          # [B, N, 1280] or [B, N, K, 1280]
                lig_points: torch.Tensor,   # [B, M, 3]
                lig_types: torch.Tensor,    # [B, M, 20]
                lig_mask: torch.Tensor,     # [B, M]
                w_res: torch.Tensor,        # [B, N]
                t: torch.Tensor,            # [B]
                node_mask: Optional[torch.Tensor] = None,
                nma_features: Optional[torch.Tensor] = None,
                stage1_chi: Optional[torch.Tensor] = None,     # [B, N, 4]
                stage1_rigids: Optional[Rigid] = None,         # Rigid[B, N]
                stage1_chi_mask: Optional[torch.Tensor] = None, # [B, N]
                interaction_prior: Optional[torch.Tensor] = None, # [B, N] or [B, N, D]
                current_step: Optional[int] = None,
                return_repa: bool = False) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with d_chi, d_rigid_rot, d_rigid_trans, gate
        """
        B, N, _ = chi.shape

        prior_feature = None
        if self.config.interaction_prior_feature_dim > 0:
            if interaction_prior is None:
                prior_feature = chi.new_zeros(B, N, self.config.interaction_prior_feature_dim)
            else:
                if interaction_prior.ndim == 2:
                    interaction_prior = interaction_prior.unsqueeze(-1)
                expected = (B, N, self.config.interaction_prior_feature_dim)
                if interaction_prior.shape != expected:
                    raise ValueError(
                        f"interaction_prior must have shape [B, N] or {expected}, "
                        f"got {tuple(interaction_prior.shape)}"
                    )
                prior_feature = interaction_prior.float().to(dtype=chi.dtype)
                prior_feature = prior_feature * float(self.config.interaction_prior_feature_scale)
                if node_mask is not None:
                    prior_feature = prior_feature * node_mask.unsqueeze(-1).float()

        # time embedding
        t_emb = self.time_embed(t).unsqueeze(1).expand(B, N, -1)

        # ESM adapter
        s = self.esm_adapter(esm)
        if prior_feature is not None and self.prior_residue_proj is not None:
            s = s + self.prior_residue_proj(prior_feature)
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
        s_geo, rigids_geo = self.ipa_module(
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
        # Stage-2 uses IPA as a geometry-aware representation trunk and keeps
        # the bridge state in the explicit velocity heads below. The final IPA
        # block still predicts a rigid update; include it as a zero-valued graph
        # dependency so DDP does not treat that last update head as unused.
        ipa_update_dependency = (
            rigids_geo.get_trans().sum()
            + rigids_geo.get_rots().get_rot_mats().sum()
        ) * 0.0

        if self.config.nma_dim > 0 and nma_features is None:
            raise ValueError("nma_dim > 0 but nma_features is None")
        if self.config.nma_dim == 0 and nma_features is not None:
            raise ValueError("nma_features provided but nma_dim is 0")
        if self.config.nma_dim > 0 and nma_features is not None:
            if nma_features.ndim == 2:
                nma_features = nma_features.unsqueeze(-1)
            if nma_features.shape[-1] != self.config.nma_dim:
                raise ValueError("nma_features last dim does not match nma_dim")

        if stage1_chi is None:
            delta_chi = chi.new_zeros(B, N, self.config.delta_chi_dim)
        else:
            if stage1_chi.shape != chi.shape:
                raise ValueError("stage1_chi shape mismatch with chi")
            delta_chi = wrap_to_pi(stage1_chi - chi)
            if stage1_chi_mask is not None:
                delta_chi = delta_chi * stage1_chi_mask.unsqueeze(-1).float()
            delta_chi = delta_chi * self.config.stage1_chi_feature_scale

        if stage1_rigids is None:
            delta_f = chi.new_zeros(B, N, self.config.delta_f_dim)
        else:
            R_curr, t_curr = rigids.get_rots().get_rot_mats(), rigids.get_trans()
            R1, t1 = stage1_rigids.get_rots().get_rot_mats(), stage1_rigids.get_trans()
            R_inv, t_inv = rigid_inverse(R_curr, t_curr)
            R_delta, t_delta = rigid_compose(R_inv, t_inv, R1, t1)
            delta_f = se3_log(R_delta, t_delta)

        # gate input (optionally append NMA features)
        scalar_features = [w_res.unsqueeze(-1)]
        if prior_feature is not None:
            scalar_features.append(prior_feature)

        gate_input = torch.cat([s_geo, *scalar_features, t_emb, delta_f, delta_chi], dim=-1)
        if self.config.nma_dim > 0:
            if nma_features.ndim == 2:
                nma_features = nma_features.unsqueeze(-1)
            gate_input = torch.cat([gate_input, nma_features], dim=-1)

        gate = torch.sigmoid(self.gate_mlp(gate_input))  # [B, N, 1]

        # chi velocity
        sin_cos = torch.stack([torch.sin(chi), torch.cos(chi)], dim=-1).reshape(B, N, 8)
        chi_input = torch.cat([s_geo, sin_cos, *scalar_features, t_emb, delta_f, delta_chi], dim=-1)
        if self.config.nma_dim > 0:
            chi_input = torch.cat([chi_input, nma_features], dim=-1)
        d_chi = self.chi_head(chi_input)
        d_chi = d_chi * gate

        # rigid velocity
        rigid_input = torch.cat([s_geo, *scalar_features, t_emb, delta_f, delta_chi], dim=-1)
        if self.config.nma_dim > 0:
            rigid_input = torch.cat([rigid_input, nma_features], dim=-1)
        rigid_vel = self.rigid_head(rigid_input)
        d_rot = rigid_vel[..., :3] * gate  # gate: [B, N, 1] broadcasts to [B, N, 3]
        d_trans = rigid_vel[..., 3:] * gate

        # mask padded residues
        if node_mask is not None:
            mask = node_mask.unsqueeze(-1).float()
            d_chi = d_chi * mask
            d_rot = d_rot * mask
            d_trans = d_trans * mask
            gate = gate * mask

        out = {
            "d_chi": d_chi + ipa_update_dependency,
            "d_rigid_rot": d_rot,
            "d_rigid_trans": d_trans,
            "gate": gate,
        }
        if self.repa_student_proj is not None:
            repa_student = self.repa_student_proj(s_geo)
            if return_repa:
                out["repa_student"] = repa_student
            else:
                out["d_chi"] = out["d_chi"] + repa_student.sum() * 0.0
        esm_lw = getattr(self.esm_adapter, "last_layer_weights", None)
        if esm_lw is not None:
            out["esm_layer_weights"] = esm_lw
        return out
