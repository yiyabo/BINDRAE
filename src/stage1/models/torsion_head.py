"""
扭转角预测头

功能：
预测主链和侧链扭转角（输出sin/cos形式）

Author: BINDRAE Team
Date: 2025-10-28
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorsionHead(nn.Module):
    """
    扭转角预测头
    
    输入: [B, N, c_s] 节点表示
    输出: [B, N, n_angles, 2] 扭转角的(sin, cos)
    """
    
    def __init__(self,
                 c_s: int = 384,
                 c_hidden: int = 128,
                 dropout: float = 0.1,
                 n_angles: int = 7):
        """
        Args:
            c_s: 输入维度
            c_hidden: 隐藏层维度
            dropout: Dropout概率
            n_angles: 角度数量（默认7: phi/psi/omega/chi1-4）
        """
        super().__init__()
        
        self.c_s = c_s
        self.n_angles = n_angles
        
        # 预测网络
        self.net = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, self.n_angles * 2)  # n_angles × 2(sin, cos)
        )
        
        # 小随机初始化（不能用zeros，会导致预测恒为0）
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            # Start near zero-angle torsions: sin=0, cos=1.
            # This keeps the subsequent normalization away from the near-zero regime
            # where its backward can become numerically extreme.
            self.net[-1].bias[1::2].fill_(1.0)
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        预测扭转角
        
        Args:
            s: [B, N, c_s] 节点表示
            
        Returns:
            angles_sincos: [B, N, 7, 2] 扭转角的(sin, cos)
        """
        B, N, _ = s.shape
        
        # 预测
        out = self.net(s)  # [B, N, 14]
        
        # 重塑为 [B, N, n_angles, 2]
        angles_sincos = out.view(B, N, self.n_angles, 2)
        
        # L2归一化（确保sin²+cos²=1）
        angles_sincos = nn.functional.normalize(angles_sincos, p=2, dim=-1, eps=1e-4)
        
        return angles_sincos
    
    def sincos_to_angles(self, sincos: torch.Tensor) -> torch.Tensor:
        """
        将(sin, cos)转换为角度
        
        Args:
            sincos: [..., 2] (sin, cos)
            
        Returns:
            angles: [...] 角度（弧度），范围[-π, π]
        """
        sin_vals = sincos[..., 0]
        cos_vals = sincos[..., 1]
        angles = torch.atan2(sin_vals, cos_vals)
        return angles


class Chi1RotamerHead(nn.Module):
    def __init__(self, c_s: int = 384, c_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, 3)
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class CandidateChi1Scorer(nn.Module):
    """Candidate-aware χ1 rotamer scorer.

    Unlike a plain three-class posterior, this head gives each canonical
    rotamer candidate its own learned embedding and scores candidate-node
    pairs. That keeps the first experiment lightweight while forcing the
    posterior to behave like a candidate selector rather than another generic
    residue classifier.
    """

    def __init__(self, c_s: int = 384, c_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.candidate_embedding = nn.Parameter(torch.zeros(3, c_s))
        nn.init.normal_(self.candidate_embedding, mean=0.0, std=0.02)
        self.scorer = nn.Sequential(
            nn.LayerNorm(c_s * 2),
            nn.Linear(c_s * 2, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, 1),
        )
        nn.init.normal_(self.scorer[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        B, N, C = s.shape
        s_expanded = s.unsqueeze(2).expand(B, N, 3, C)
        candidate = self.candidate_embedding.view(1, 1, 3, C).expand(B, N, 3, C)
        pair = torch.cat([s_expanded, candidate], dim=-1)
        return self.scorer(pair).squeeze(-1)


class ContactHead(nn.Module):
    """Residue-level ligand-contact posterior head.

    The head predicts one logit per residue. The probability is kept as a
    derived value in the model output so BCEWithLogitsLoss can use the stable
    logit form during training.
    """

    def __init__(self, c_s: int = 384, c_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


# ============================================================================
# Geometry-Bypass Candidate Scorer
# ============================================================================

CHI1_BIN_CENTERS = [-math.pi / 3.0, math.pi / 3.0, math.pi]  # 0=g-, 1=g+, 2=t (matches label order)


def compute_candidate_geometries(
    fk_module,
    backbone_rigids,
    aatype: torch.Tensor,
    torsion_apo: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate idealized sidechain atom14 positions for each chi1 rotamer bin.

    Args:
        fk_module: OpenFoldFK instance
        backbone_rigids: [B, N] Rigid frames from backbone
        aatype: [B, N] residue type indices
        torsion_apo: [B, N, 7] apo torsion angles (phi,psi,omega,chi1-4)

    Returns:
        cand_atom14: [B, N, 3, 14, 3] atom positions per candidate
        atom14_mask: [B, N, 14] atom existence mask
        chi1_dep_mask: [B, N, 14] mask for chi1-dependent atoms (group >= 4)
    """
    from .fk_openfold import reorder_torsions_to_openfold

    B, N = aatype.shape
    device = aatype.device

    phi_psi_omega = torsion_apo[:, :, :3]
    phi_psi_omega_sc = torch.stack(
        [torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1
    )  # [B, N, 3, 2]

    chi234_apo = torsion_apo[:, :, 4:7]  # chi2-4
    chi234_sc = torch.stack(
        [torch.sin(chi234_apo), torch.cos(chi234_apo)], dim=-1
    )  # [B, N, 3, 2]

    cand_positions = []
    with torch.no_grad():
        for k, center in enumerate(CHI1_BIN_CENTERS):
            chi1_sc = torch.tensor(
                [[math.sin(center), math.cos(center)]],
                device=device, dtype=phi_psi_omega_sc.dtype
            ).expand(B, N, 1, 2)  # [B, N, 1, 2]
            torsions_sc = torch.cat([phi_psi_omega_sc, chi1_sc, chi234_sc], dim=2)  # [B, N, 7, 2]
            torsions_sc = reorder_torsions_to_openfold(torsions_sc)
            result = fk_module(torsions_sc, backbone_rigids, aatype)
            cand_positions.append(result['atom14_pos'])  # [B, N, 14, 3]

    cand_atom14 = torch.stack(cand_positions, dim=2)  # [B, N, 3, 14, 3]
    atom14_mask = result['atom14_mask']  # [B, N, 14] (same for all candidates)

    # chi1-dependent atoms: rigid group >= 4
    group_idx = fk_module.restype_atom14_to_group[aatype]  # [B, N, 14]
    chi1_dep_mask = (group_idx >= 4) & atom14_mask.bool()

    return cand_atom14, atom14_mask, chi1_dep_mask


def compute_candidate_ligand_features(
    cand_atom14: torch.Tensor,
    chi1_dep_mask: torch.Tensor,
    lig_points: torch.Tensor,
    lig_mask: torch.Tensor,
    rbf_encoder: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute RBF distance features between candidate sidechain atoms and ligand.

    Args:
        cand_atom14: [B, N, 3, 14, 3]
        chi1_dep_mask: [B, N, 14] bool
        lig_points: [B, M, 3]
        lig_mask: [B, M] bool

    Returns:
        rbf_pooled: [B, N, 3, num_rbf] pooled RBF features
        min_dist: [B, N, 3] min sidechain-ligand distance per candidate
        clash: [B, N, 3] clash indicator (any dist < 2.0A)
    """
    B, N, K, A, _ = cand_atom14.shape  # K=3, A=14
    M = lig_points.shape[1]
    num_rbf = rbf_encoder.num_rbf

    # Expand for pairwise distances: [B, N, 3, 14, M]
    sc = cand_atom14  # [B, N, 3, 14, 3]
    lig = lig_points[:, None, None, None, :, :]  # [B, 1, 1, 1, M, 3]
    dists = torch.cdist(
        sc.reshape(B * N * K, A, 3),
        lig_points[:, None, :, :].expand(B, N * K, M, 3).reshape(B * N * K, M, 3),
    ).reshape(B, N, K, A, M)  # [B, N, 3, 14, M]

    # Mask invalid atoms and ligand tokens
    atom_mask = chi1_dep_mask.unsqueeze(2).unsqueeze(-1)  # [B, N, 1, 14, 1]
    lig_m = lig_mask[:, None, None, None, :]  # [B, 1, 1, 1, M]
    valid_mask = atom_mask & lig_m  # [B, N, 3, 14, M] broadcast over K=3

    dists_masked = dists.clone()
    dists_masked[~valid_mask.expand_as(dists_masked)] = 1e6

    # Min distance per candidate
    min_dist_per_atom = dists_masked.min(dim=-1).values  # [B, N, 3, 14]
    min_dist_per_atom[~chi1_dep_mask.unsqueeze(2).expand(B, N, K, A)] = 1e6
    min_dist = min_dist_per_atom.min(dim=-1).values  # [B, N, 3]

    # Clash indicator
    clash = (min_dist < 2.0).float()

    # RBF features: encode min distance per (candidate, sc_atom) then pool
    # Use min over ligand atoms per sc atom for RBF input
    rbf_input = min_dist_per_atom.clamp(max=20.0)  # [B, N, 3, 14]
    rbf_features = rbf_encoder(rbf_input)  # [B, N, 3, 14, num_rbf]

    # Pool over sc atoms (mean of valid chi1-dep atoms)
    sc_valid = chi1_dep_mask.unsqueeze(2).expand(B, N, K, A).float()  # [B, N, 3, 14]
    sc_count = sc_valid.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, N, 3, 1]
    rbf_pooled = (rbf_features * sc_valid.unsqueeze(-1)).sum(dim=3) / sc_count  # [B, N, 3, num_rbf]

    return rbf_pooled, min_dist, clash


class GeometryCandidateScorer(nn.Module):
    """Geometry-bypass χ1 rotamer scorer with base prior + ligand residual.

    Decomposition:
        logits = base_prior(aatype, φ/ψ, apo_χ1, cand_χ1)
               + gate(aatype, φ/ψ, apo_χ1, s_geo) * lig_residual(geom_features, s_geo)

    When no ligand is provided, lig_residual is zeroed and only base_prior fires.
    This lets the model learn a strong Dunbrack-like base prior independently,
    while the ligand residual only captures induced-fit corrections.

    Phase-1 retraining mode (Phase 1 of GPT-5.5 Pro plan):
        - reset_residual_and_gate(): reinitialize residual/gate from scratch
        - freeze_base(): freeze base_mlp parameters
        - forward(base_detach=True): apply stop-grad on base path (residual training)
        - forward(gate_override=1.0): force gate fully open (warmup)
        - forward(beta=0.1): scale residual contribution
        - bounded_residual=True: tanh-bound residual logits
    """

    def __init__(self, c_hidden: int = 128, num_rbf: int = 16, dropout: float = 0.1,
                 use_sgeo: bool = False, c_s: int = 384, sgeo_proj_dim: int = 32,
                 bounded_residual: bool = False, residual_max: float = 5.0,
                 residual_tau: float = 2.0, gate_norm: bool = False,
                 gate_clamp: float = 6.0, gate_init_bias: float = 0.0,
                 use_typed_energy: bool = False, lig_type_dim: int = 20,
                 typed_pair_dim: int = 64, typed_cutoff: float = 6.0,
                 typed_init_scale: float = 0.1):
        super().__init__()
        self.use_sgeo = use_sgeo
        self.num_rbf = num_rbf
        self.bounded_residual = bool(bounded_residual)
        self.residual_max = float(residual_max)
        self.residual_tau = float(residual_tau)
        self.gate_norm_enabled = bool(gate_norm)
        self.gate_clamp = float(gate_clamp)
        self.gate_init_bias = float(gate_init_bias)
        self._c_hidden = int(c_hidden)
        self._dropout_p = float(dropout)
        self._sgeo_proj_dim = int(sgeo_proj_dim)
        self._c_s = int(c_s)
        self.use_typed_energy = bool(use_typed_energy)
        self.lig_type_dim = int(lig_type_dim)
        self.typed_pair_dim = int(typed_pair_dim)
        self.typed_cutoff = float(typed_cutoff)

        # --- Base prior: aatype(21) + phi/psi_sc(4) + apo_chi1_sc(2) + cand_chi1_sc(2) = 29 ---
        self.base_dim = 21 + 4 + 2 + 2
        self.base_mlp = nn.Sequential(
            nn.Linear(self.base_dim, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, 1),
        )
        nn.init.normal_(self.base_mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.base_mlp[-1].bias)

        # --- Ligand residual: rbf(num_rbf) + clash(1) + min_dist(1) + optional s_geo ---
        self.residual_dim = num_rbf + 1 + 1
        if use_sgeo:
            self.sgeo_proj = nn.Sequential(
                nn.LayerNorm(c_s),
                nn.Linear(c_s, sgeo_proj_dim),
                nn.GELU(),
            )
            self.residual_dim += sgeo_proj_dim

        self.residual_mlp = nn.Sequential(
            nn.Linear(self.residual_dim, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, 1),
        )
        # Phase-1: zero-init residual final layer ensures gating decisions start neutral
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        # --- Gate: controls how much ligand residual contributes per residue ---
        gate_input_dim = 21 + 4 + 2  # aatype + phi/psi + apo_chi1 (no cand, since gate is per-residue)
        if use_sgeo:
            gate_input_dim += sgeo_proj_dim
        self._gate_input_dim = int(gate_input_dim)
        self.gate_norm = nn.LayerNorm(gate_input_dim) if self.gate_norm_enabled else nn.Identity()
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, c_hidden // 2),
            nn.GELU(),
            nn.Linear(c_hidden // 2, 1),
        )
        # Phase-1: bias gate open (sigmoid(gate_init_bias)). With gate_init_bias=0, sigmoid=0.5
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, self.gate_init_bias)

        # --- RBF encoder ---
        from .ligand_condition import RBFDistanceEncoding
        self.rbf_encoder = RBFDistanceEncoding(num_rbf=num_rbf, d_min=0.0, d_max=20.0, trainable=True)

        if self.use_typed_energy:
            self.typed_atom_embedding = nn.Embedding(21 * 14, self.typed_pair_dim)
            self.typed_ligand_proj = nn.Sequential(
                nn.LayerNorm(self.lig_type_dim),
                nn.Linear(self.lig_type_dim, self.typed_pair_dim),
                nn.GELU(),
                nn.Linear(self.typed_pair_dim, self.typed_pair_dim),
            )
            self.typed_energy_scale = nn.Parameter(torch.tensor(float(typed_init_scale)))
            nn.init.normal_(self.typed_atom_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.typed_ligand_proj[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.typed_ligand_proj[-1].bias)

        # Fixed candidate chi1 sin/cos for g-, t, g+
        cand_sc = torch.tensor(
            [[math.sin(c), math.cos(c)] for c in CHI1_BIN_CENTERS],
            dtype=torch.float32
        )  # [3, 2]
        self.register_buffer('cand_chi1_sincos', cand_sc)

    def _compute_typed_energy(
        self,
        cand_atom14: torch.Tensor,
        chi1_dep_mask: torch.Tensor,
        aatype: torch.Tensor,
        lig_points: torch.Tensor,
        lig_types: torch.Tensor,
        lig_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Factorized typed sidechain-ligand energy for each chi1 candidate.

        This keeps the first chemistry-aware branch memory bounded: residue
        atom identity and ligand token type are embedded separately, then their
        dot product is distance-gated and pooled over valid local pairs.
        """
        B, N, K, A, _ = cand_atom14.shape
        M = lig_points.shape[1]
        if M == 0 or lig_types is None:
            return cand_atom14.new_zeros(B, N, K)

        dists = torch.cdist(
            cand_atom14.reshape(B * N * K, A, 3),
            lig_points[:, None, :, :].expand(B, N * K, M, 3).reshape(B * N * K, M, 3),
        ).reshape(B, N, K, A, M)

        atom_valid = chi1_dep_mask.unsqueeze(2).unsqueeze(-1)
        lig_valid = lig_mask[:, None, None, None, :].bool()
        local_valid = atom_valid & lig_valid & (dists <= self.typed_cutoff)
        if node_mask is not None:
            local_valid = local_valid & node_mask[:, :, None, None, None].bool()

        atom_idx = torch.arange(A, device=aatype.device).view(1, 1, A)
        atom_keys = aatype.clamp(0, 20).unsqueeze(-1) * A + atom_idx
        atom_embed = self.typed_atom_embedding(atom_keys)  # [B, N, A, D]
        lig_embed = self.typed_ligand_proj(lig_types.to(atom_embed.dtype))  # [B, M, D]
        pair_type_score = torch.einsum('bnad,bmd->bnam', atom_embed, lig_embed)
        pair_type_score = pair_type_score / math.sqrt(float(self.typed_pair_dim))

        cutoff = max(self.typed_cutoff, 1e-3)
        dist_weight = torch.exp(-0.5 * (dists / cutoff).pow(2))
        pair_score = pair_type_score.unsqueeze(2) * dist_weight

        valid_f = local_valid.to(pair_score.dtype)
        pair_count = valid_f.sum(dim=(-1, -2))
        denom = pair_count.clamp(min=1.0)
        typed_energy = (pair_score * valid_f).sum(dim=(-1, -2)) / denom
        typed_energy = self.typed_energy_scale.to(typed_energy.dtype) * typed_energy
        typed_energy = typed_energy * (pair_count > 0).to(typed_energy.dtype)
        typed_energy = typed_energy - typed_energy.mean(dim=-1, keepdim=True)
        return typed_energy

    def _compute_base_features(self, aatype, torsion_apo, B, N):
        """Compute base prior features: aatype + phi/psi + apo_chi1 + cand_chi1."""
        aa_onehot = F.one_hot(aatype.clamp(0, 20), 21).float()  # [B, N, 21]
        phi_sc = torch.stack([torch.sin(torsion_apo[:, :, 0]), torch.cos(torsion_apo[:, :, 0])], dim=-1)
        psi_sc = torch.stack([torch.sin(torsion_apo[:, :, 1]), torch.cos(torsion_apo[:, :, 1])], dim=-1)
        phi_psi = torch.cat([phi_sc, psi_sc], dim=-1)  # [B, N, 4]
        apo_chi1 = torsion_apo[:, :, 3]
        apo_chi1_sc = torch.stack([torch.sin(apo_chi1), torch.cos(apo_chi1)], dim=-1)  # [B, N, 2]

        base_feats = torch.cat([aa_onehot, phi_psi, apo_chi1_sc], dim=-1)  # [B, N, 27]
        base_feats = base_feats.unsqueeze(2).expand(B, N, 3, -1)  # [B, N, 3, 27]

        cand_sc = self.cand_chi1_sincos.view(1, 1, 3, 2).expand(B, N, 3, 2)  # [B, N, 3, 2]
        return torch.cat([base_feats, cand_sc], dim=-1)  # [B, N, 3, 29]

    def _compute_gate_features(self, aatype, torsion_apo, s_geo, B, N):
        """Compute gate input: per-residue features (no candidate dim)."""
        aa_onehot = F.one_hot(aatype.clamp(0, 20), 21).float()  # [B, N, 21]
        phi_sc = torch.stack([torch.sin(torsion_apo[:, :, 0]), torch.cos(torsion_apo[:, :, 0])], dim=-1)
        psi_sc = torch.stack([torch.sin(torsion_apo[:, :, 1]), torch.cos(torsion_apo[:, :, 1])], dim=-1)
        phi_psi = torch.cat([phi_sc, psi_sc], dim=-1)  # [B, N, 4]
        apo_chi1_sc = torch.stack([torch.sin(torsion_apo[:, :, 3]), torch.cos(torsion_apo[:, :, 3])], dim=-1)

        gate_feats = torch.cat([aa_onehot, phi_psi, apo_chi1_sc], dim=-1)  # [B, N, 27]
        if self.use_sgeo and s_geo is not None:
            sgeo_compressed = self.sgeo_proj(s_geo)  # [B, N, sgeo_proj_dim]
            gate_feats = torch.cat([gate_feats, sgeo_compressed], dim=-1)
        return gate_feats  # [B, N, gate_dim]

    def reset_residual_and_gate(self) -> None:
        """Reinitialize residual_mlp + gate_mlp from scratch (Phase-1 retraining).

        - residual_mlp: Kaiming for hidden layers, zero-init final layer
        - gate_mlp: Kaiming for hidden layers, weight=0 + bias=gate_init_bias on final layer
        - sgeo_proj (if present): keep (it is shared input projection used elsewhere)
        - gate_norm LayerNorm: reset to identity-like statistics
        """
        # Rebuild residual_mlp on the existing parameters' device/dtype.
        ref_param = next(self.residual_mlp.parameters(), None)
        device = ref_param.device if ref_param is not None else torch.device('cpu')
        dtype = ref_param.dtype if ref_param is not None else torch.float32

        self.residual_mlp = nn.Sequential(
            nn.Linear(self.residual_dim, self._c_hidden),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
            nn.Linear(self._c_hidden, self._c_hidden),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
            nn.Linear(self._c_hidden, 1),
        ).to(device=device, dtype=dtype)
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        if self.gate_norm_enabled:
            self.gate_norm = nn.LayerNorm(self._gate_input_dim).to(device=device, dtype=dtype)
        else:
            self.gate_norm = nn.Identity()

        self.gate_mlp = nn.Sequential(
            nn.Linear(self._gate_input_dim, self._c_hidden // 2),
            nn.GELU(),
            nn.Linear(self._c_hidden // 2, 1),
        ).to(device=device, dtype=dtype)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, self.gate_init_bias)

    def freeze_base(self, freeze: bool = True) -> None:
        """Freeze (or unfreeze) base_mlp parameters."""
        for param in self.base_mlp.parameters():
            param.requires_grad = not freeze

    def freeze_gate(self, freeze: bool = True) -> None:
        """Freeze (or unfreeze) gate_mlp + gate_norm parameters.

        Used by Phase-1 v2: gate is fixed=1.0 throughout, so its parameters
        must be frozen to avoid DDP unused-parameter errors.
        """
        for param in self.gate_mlp.parameters():
            param.requires_grad = not freeze
        if isinstance(self.gate_norm, nn.LayerNorm):
            for param in self.gate_norm.parameters():
                param.requires_grad = not freeze

    def forward(
        self,
        fk_module,
        backbone_rigids,
        aatype: torch.Tensor,
        torsion_apo: torch.Tensor,
        lig_points: torch.Tensor,
        lig_types: Optional[torch.Tensor],
        lig_mask: torch.Tensor,
        node_mask: torch.Tensor,
        s_geo: Optional[torch.Tensor] = None,
        return_decomposition: bool = False,
        base_detach: bool = False,
        gate_override: Optional[float] = None,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns:
            logits: [B, N, 3] rotamer candidate scores
            If return_decomposition=True, also returns (base_logits, residual_logits, gate)

        Phase-1 controls:
            base_detach: if True, base contribution to full logits uses .detach()
                         (residual gradients do not flow into base_mlp).
            gate_override: if not None, replace learned gate by this scalar (e.g. 1.0 for warmup).
            beta: residual scale (warmup from 0.1 to 1.0).
        """
        B, N = aatype.shape

        # --- Base prior (always computed) ---
        base_input = self._compute_base_features(aatype, torsion_apo, B, N)  # [B, N, 3, 29]
        base_logits = self.base_mlp(base_input).squeeze(-1)  # [B, N, 3]

        # --- Ligand residual (geometry features) ---
        has_ligand = lig_mask is not None and lig_mask.any()

        if has_ligand:
            # FK candidate geometry
            cand_atom14, atom14_mask, chi1_dep_mask = compute_candidate_geometries(
                fk_module, backbone_rigids, aatype, torsion_apo
            )

            # Candidate-ligand distance features
            rbf_pooled, min_dist, clash = compute_candidate_ligand_features(
                cand_atom14, chi1_dep_mask, lig_points, lig_mask, self.rbf_encoder
            )

            # Residual features
            residual_feats_list = [rbf_pooled, clash.unsqueeze(-1), min_dist.unsqueeze(-1)]
            if self.use_sgeo and s_geo is not None:
                sgeo_compressed = self.sgeo_proj(s_geo)  # [B, N, sgeo_proj_dim]
                residual_feats_list.append(sgeo_compressed.unsqueeze(2).expand(B, N, 3, -1))
            residual_input = torch.cat(residual_feats_list, dim=-1)  # [B, N, 3, residual_dim]
            residual_logits = self.residual_mlp(residual_input).squeeze(-1)  # [B, N, 3]
            if self.use_typed_energy and lig_types is not None:
                typed_energy = self._compute_typed_energy(
                    cand_atom14, chi1_dep_mask, aatype, lig_points, lig_types, lig_mask, node_mask
                )
            else:
                typed_energy = torch.zeros_like(residual_logits)

            # Bounded residual: remove class-wise mean (logits are class-wise relative)
            # then optionally tanh-bound for stability.
            residual_logits = residual_logits - residual_logits.mean(dim=-1, keepdim=True)
            residual_logits = residual_logits + typed_energy
            if self.bounded_residual:
                residual_logits = self.residual_max * torch.tanh(residual_logits / self.residual_tau)

            # Gate (per-residue, expanded to 3 candidates)
            if gate_override is not None:
                # Skip gate_mlp entirely: gate is fixed and receives no grad.
                # This is the v2 residual-only training mode.
                gate = torch.full(
                    (B, N, 3),
                    float(gate_override),
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )
            else:
                gate_feats = self._compute_gate_features(aatype, torsion_apo, s_geo, B, N)  # [B, N, gate_dim]
                gate_feats = self.gate_norm(gate_feats)
                raw_gate = self.gate_mlp(gate_feats)  # [B, N, 1]
                if self.gate_clamp > 0:
                    raw_gate = raw_gate.clamp(-self.gate_clamp, self.gate_clamp)
                gate = torch.sigmoid(raw_gate)
                gate = gate.expand(B, N, 3)  # [B, N, 3]
        else:
            # No ligand: residual is zero
            residual_logits = torch.zeros(B, N, 3, device=base_logits.device, dtype=base_logits.dtype)
            typed_energy = torch.zeros_like(residual_logits)
            gate = torch.zeros(B, N, 1, device=base_logits.device, dtype=base_logits.dtype)

        # --- Combined output ---
        base_for_combined = base_logits.detach() if base_detach else base_logits
        beta_scale = float(beta) if beta is not None else 1.0
        logits = base_for_combined + beta_scale * gate * residual_logits

        if return_decomposition:
            # Always expose the (un-detached) base_logits and the actual gate*residual contribution.
            return logits, base_logits, residual_logits * beta_scale, gate, typed_energy * beta_scale
        return logits


def create_torsion_head(c_s: int = 384,
                        c_hidden: int = 128,
                        dropout: float = 0.1,
                        n_angles: int = 7) -> TorsionHead:
    """
    创建扭转角预测头
    
    Args:
        c_s: 输入维度
        c_hidden: 隐藏层维度
        dropout: Dropout概率
        
    Returns:
        TorsionHead实例
    """
    return TorsionHead(c_s, c_hidden, dropout, n_angles=n_angles)
