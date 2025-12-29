"""Geometry utilities for Stage-2 losses."""

from typing import Optional

import torch

# Atom14 indices (consistent with residue_constants)
ATOM14_N = 0
ATOM14_CA = 1
ATOM14_C = 2


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def angle_between(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute angle between vectors."""
    v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)
    cos = (v1_norm * v2_norm).sum(dim=-1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    return torch.acos(cos)


def compute_peptide_loss(atom14_pos: torch.Tensor,
                          atom14_mask: torch.Tensor,
                          node_mask: torch.Tensor,
                          bond_len: float = 1.33,
                          angle_cacn: float = 2.035,
                          angle_cnca: float = 2.124,
                          angle_weight: float = 0.1,
                          eps: float = 1e-8) -> torch.Tensor:
    """
    Peptide geometry guard: C-N bond length + angles.
    """
    # [B, N-1, 3]
    C_i = atom14_pos[:, :-1, ATOM14_C]
    N_ip1 = atom14_pos[:, 1:, ATOM14_N]
    Ca_i = atom14_pos[:, :-1, ATOM14_CA]
    Ca_ip1 = atom14_pos[:, 1:, ATOM14_CA]

    mask_cn = (
        atom14_mask[:, :-1, ATOM14_C] &
        atom14_mask[:, 1:, ATOM14_N] &
        node_mask[:, :-1] &
        node_mask[:, 1:]
    )

    dist_cn = torch.norm(C_i - N_ip1, dim=-1)
    bond_loss = (dist_cn - bond_len) ** 2
    bond_loss = (bond_loss * mask_cn.float()).sum() / (mask_cn.float().sum() + eps)

    # Angles
    v1 = Ca_i - C_i
    v2 = N_ip1 - C_i
    v3 = C_i - N_ip1
    v4 = Ca_ip1 - N_ip1

    angle1 = angle_between(v1, v2, eps=eps)
    angle2 = angle_between(v3, v4, eps=eps)

    mask_angle = (
        atom14_mask[:, :-1, ATOM14_CA] &
        atom14_mask[:, :-1, ATOM14_C] &
        atom14_mask[:, 1:, ATOM14_N] &
        atom14_mask[:, 1:, ATOM14_CA] &
        node_mask[:, :-1] &
        node_mask[:, 1:]
    )

    angle_loss = ((angle1 - angle_cacn) ** 2 + (angle2 - angle_cnca) ** 2)
    angle_loss = (angle_loss * mask_angle.float()).sum() / (mask_angle.float().sum() + eps)

    return bond_loss + angle_weight * angle_loss


def residue_ligand_min_dist(atom14_pos: torch.Tensor,
                             atom14_mask: torch.Tensor,
                             lig_points: torch.Tensor,
                             lig_mask: torch.Tensor,
                             eps: float = 1e-8) -> torch.Tensor:
    """
    Min residue-ligand distance per residue.

    Returns:
        min_dist: [B, N]
    """
    # [B, N, 14, M, 3]
    diff = atom14_pos[:, :, :, None, :] - lig_points[:, None, None, :, :]
    dist = torch.norm(diff, dim=-1)

    mask = atom14_mask[:, :, :, None] & lig_mask[:, None, None, :]
    dist = dist.masked_fill(~mask, 1e6)

    min_dist = dist.min(dim=3).values.min(dim=2).values
    return min_dist


def soft_contact_from_dist(min_dist: torch.Tensor,
                            d_c: float = 6.0,
                            tau: float = 1.0) -> torch.Tensor:
    """Soft contact score from distances."""
    return torch.sigmoid((d_c - min_dist) / max(tau, 1e-6))


def compute_contact_score(atom14_pos: torch.Tensor,
                           atom14_mask: torch.Tensor,
                           lig_points: torch.Tensor,
                           lig_mask: torch.Tensor,
                           w_res: torch.Tensor,
                           pocket_threshold: float = 0.5,
                           d_c: float = 6.0,
                           tau: float = 1.0,
                           eps: float = 1e-8) -> torch.Tensor:
    """
    Compute global pocket contact score C(t) for a batch.

    Returns:
        C: [B]
    """
    min_dist = residue_ligand_min_dist(atom14_pos, atom14_mask, lig_points, lig_mask, eps=eps)
    contact = soft_contact_from_dist(min_dist, d_c=d_c, tau=tau)
    pocket_mask = (w_res > pocket_threshold).float()

    numerator = (contact * pocket_mask).sum(dim=-1)
    denom = pocket_mask.sum(dim=-1).clamp(min=eps)
    return numerator / denom


def compute_w_eff(w_res: torch.Tensor,
                  nma_features: Optional[torch.Tensor] = None,
                  nma_lambda: float = 1.0,
                  nma_time_decay: float = 0.0,
                  t: Optional[torch.Tensor] = None,
                  eps: float = 1e-8) -> torch.Tensor:
    """
    Combine pocket weights with NMA features (optional).
    """
    if nma_features is None:
        return w_res

    if nma_features.ndim == 2:
        nma_norm = nma_features
    else:
        nma_norm = torch.norm(nma_features, dim=-1)

    # Normalize to [0,1] per sample to stabilize
    nma_min = nma_norm.min(dim=-1, keepdim=True).values
    nma_max = nma_norm.max(dim=-1, keepdim=True).values
    nma_norm = (nma_norm - nma_min) / (nma_max - nma_min + eps)

    if t is not None and nma_time_decay > 0:
        decay = torch.exp(-nma_time_decay * t).unsqueeze(-1)
        nma_norm = nma_norm * decay

    return torch.maximum(w_res, nma_lambda * nma_norm)
