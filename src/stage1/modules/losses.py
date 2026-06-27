"""
损失函数模块

功能：
1. FAPE损失（局部帧对齐）
2. 扭转角损失（wrap cosine）
3. 距离损失（pair-wise）
4. 碰撞惩罚（soft penalty）

Author: BINDRAE Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


BOND_EXCLUSION_MASK = None


def _build_bond_exclusion_mask() -> torch.Tensor:
    """
    Build a per-residue-type exclusion mask [21, 14, 14].
    Atom pairs with ideal literature distance < 3.0Å are excluded
    from clash computation (covers 1-2, 1-3, 1-4 bonded neighbors).

    NOTE: Literature coordinates come from rigid_group_atom_positions,
    where atoms in different rigid groups are in their respective local
    frames. Cross-group distances are therefore approximate (not in a
    shared coordinate system), which causes over-exclusion — effectively
    all intra-residue pairs are excluded. This is conservative (no false
    clashes) but means clash_penalty only detects inter-residue clashes.
    A future improvement could use explicit bond-connectivity tables.
    """
    from src.stage1.data.residue_constants import (
        restype_1to3,
        restype_3to1,
        restype_order,
        restype_name_to_atom14_names,
        rigid_group_atom_positions,
    )

    mask = torch.zeros(21, 14, 14, dtype=torch.bool)

    for resname in restype_1to3.values():
        aa_idx = restype_order.get(restype_3to1.get(resname, 'A'), 0)
        atom_names = restype_name_to_atom14_names[resname]

        pos_dict = {}
        for aname, _, apos in rigid_group_atom_positions[resname]:
            pos_dict[aname] = apos

        valid = [(i, atom_names[i]) for i in range(14) if atom_names[i] and atom_names[i] in pos_dict]

        for i, ai in valid:
            for j, aj in valid:
                if i >= j:
                    continue
                pi = torch.tensor(pos_dict[ai], dtype=torch.float32)
                pj = torch.tensor(pos_dict[aj], dtype=torch.float32)
                ideal_dist = torch.norm(pi - pj).item()
                if ideal_dist < 3.0:
                    mask[aa_idx, i, j] = True
                    mask[aa_idx, j, i] = True

    return mask


def _build_aatype_exclusion_mask(
    aatype: torch.Tensor,
    total_atoms: int,
    device: torch.device,
    sampled_indices: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Build per-sample atom-pair exclusions for atom14 clash loss."""
    global BOND_EXCLUSION_MASK

    B, n_res = aatype.shape
    if total_atoms % n_res != 0:
        return None

    n_atoms_per_res = total_atoms // n_res
    if n_atoms_per_res != 14:
        return None

    if BOND_EXCLUSION_MASK is None or BOND_EXCLUSION_MASK.device != device:
        BOND_EXCLUSION_MASK = _build_bond_exclusion_mask().to(device)

    if sampled_indices is None:
        res_exclude = BOND_EXCLUSION_MASK[aatype]  # [B, N_res, 14, 14]
        exclude = torch.zeros(B, total_atoms, total_atoms, dtype=torch.bool, device=device)
        for i in range(n_res):
            start = i * n_atoms_per_res
            end = start + n_atoms_per_res
            exclude[:, start:end, start:end] = res_exclude[:, i]
        return exclude

    sampled_indices = sampled_indices.long()
    sample_size = sampled_indices.shape[0]
    sampled_res_idx = sampled_indices // n_atoms_per_res
    sampled_atom_idx = sampled_indices % n_atoms_per_res

    aa_per_atom = aatype.gather(1, sampled_res_idx.unsqueeze(0).expand(B, -1))
    aa_grid = aa_per_atom.unsqueeze(-1).expand(-1, -1, sample_size)
    atom_i = sampled_atom_idx.view(1, sample_size, 1).expand(B, -1, sample_size)
    atom_j = sampled_atom_idx.view(1, 1, sample_size).expand(B, sample_size, -1)
    same_residue = (
        sampled_res_idx.view(1, sample_size, 1).expand(B, -1, sample_size)
        == sampled_res_idx.view(1, 1, sample_size).expand(B, sample_size, -1)
    )

    return BOND_EXCLUSION_MASK[aa_grid, atom_i, atom_j] & same_residue


# ============================================================================
# FAPE损失（Frame Aligned Point Error）
# ============================================================================

def fape_loss(pred_coords: torch.Tensor,
             true_coords: torch.Tensor,
             pred_frames: Tuple[torch.Tensor, torch.Tensor],
             true_frames: Tuple[torch.Tensor, torch.Tensor],
             w_res: Optional[torch.Tensor] = None,
             atom_mask: Optional[torch.Tensor] = None,
             clamp_distance: float = 10.0,
             eps: float = 1e-6) -> torch.Tensor:
    """
    FAPE损失（可微分版本，增强数值稳定性）
    
    公式：对每个残基i，将坐标变换到局部帧后计算L2误差
    
    Args:
        pred_coords: [B, N, 3] 或 [B, N, n_atoms, 3] 预测坐标
        true_coords: [B, N, 3] 或 [B, N, n_atoms, 3] 真实坐标
        pred_frames: (R[B,N,3,3], t[B,N,3]) 预测帧
        true_frames: (R[B,N,3,3], t[B,N,3]) 真实帧
        w_res: [B, N] 残基权重（口袋加权，可选）
        atom_mask: [B, N] 或 [B, N, n_atoms] 原子/点掩码（可选）
        clamp_distance: 裁剪距离（Å）
        eps: 数值稳定性
        
    Returns:
        loss: scalar tensor
    """
    pred_R, pred_t = pred_frames
    true_R, true_t = true_frames
    
    # NaN/Inf 输入检查
    if torch.isnan(pred_coords).any() or torch.isinf(pred_coords).any():
        return pred_coords.new_tensor(0.0)
    if torch.isnan(true_coords).any() or torch.isinf(true_coords).any():
        return pred_coords.new_tensor(0.0)
    
    # 如果是[B, N, n_atoms, 3]，重塑为[B, N*n_atoms, 3]
    if pred_coords.ndim == 4:
        B, N, n_atoms, _ = pred_coords.shape
        pred_coords = pred_coords.reshape(B, N * n_atoms, 3)
        true_coords = true_coords.reshape(B, N * n_atoms, 3)
        # 扩展帧
        pred_R = pred_R.unsqueeze(2).expand(-1, -1, n_atoms, -1, -1).reshape(B, N * n_atoms, 3, 3)
        pred_t = pred_t.unsqueeze(2).expand(-1, -1, n_atoms, -1).reshape(B, N * n_atoms, 3)
        true_R = true_R.unsqueeze(2).expand(-1, -1, n_atoms, -1, -1).reshape(B, N * n_atoms, 3, 3)
        true_t = true_t.unsqueeze(2).expand(-1, -1, n_atoms, -1).reshape(B, N * n_atoms, 3)
        # 扩展权重
        if w_res is not None:
            w_res = w_res.unsqueeze(2).expand(-1, -1, n_atoms).reshape(B, N * n_atoms)
        if atom_mask is not None:
            atom_mask = atom_mask.reshape(B, N * n_atoms)
    
    # 在 fp32 下计算以避免 fp16 溢出
    orig_dtype = pred_coords.dtype
    pred_coords = pred_coords.float()
    true_coords = true_coords.float()
    pred_R = pred_R.float()
    pred_t = pred_t.float()
    true_R = true_R.float()
    true_t = true_t.float()
    
    # 变换到局部坐标系
    # Design choice: use true_frames for BOTH pred and true coords.
    # This makes fape_loss equivalent to clamped RMSD (the true-frame
    # rotation cancels: ||R^T(a-b)|| == ||a-b||).  While a proper
    # cross-frame FAPE (each of N frames transforms ALL M atoms,
    # giving O(N*M) comparisons) would provide locality-aware backbone
    # supervision, clamped-RMSD is simpler and still provides gradient
    # signal for backbone frame prediction — crucial because backbone
    # frames have no other supervision in the current loss composition.
    # A per-residue FAPE (each frame transforms only its own atoms)
    # would be zero whenever FK internal geometry is correct, regardless
    # of backbone frame errors — leaving backbone updates unsupervised.
    pred_local = torch.einsum('bnik,bnk->bni', true_R.transpose(-2, -1), pred_coords - true_t)
    true_local = torch.einsum('bnik,bnk->bni', true_R.transpose(-2, -1), true_coords - true_t)
    
    # 计算误差（数值稳定：先 clamp 差值再求平方）
    coord_diff = pred_local - true_local
    coord_diff = torch.clamp(coord_diff, min=-clamp_distance * 10, max=clamp_distance * 10)  # 防止极端值
    diff_sq = torch.sum(coord_diff ** 2, dim=-1)  # [B, N]
    diff = torch.sqrt(diff_sq + eps)
    
    # Clamp 最终距离
    diff = torch.clamp(diff, max=clamp_distance)
    
    point_weight = None
    if atom_mask is not None:
        point_weight = atom_mask.float()
    if w_res is not None:
        w_res = w_res.float()
        point_weight = w_res if point_weight is None else point_weight * w_res

    # 应用权重 / 掩码
    if point_weight is not None:
        weight_sum = point_weight.sum()
        if weight_sum < eps:
            return pred_coords.new_tensor(0.0)
        loss = (diff * point_weight).sum() / (weight_sum + eps)
    else:
        loss = diff.mean()
    
    # 最终 NaN 检查
    if torch.isnan(loss) or torch.isinf(loss):
        return pred_coords.new_tensor(0.0)
    
    return loss.to(orig_dtype)


# ============================================================================
# 扭转角损失
# ============================================================================

def torsion_loss(pred_angles: torch.Tensor,
                 true_angles: torch.Tensor,
                 angle_mask: torch.Tensor,
                 w_res: Optional[torch.Tensor] = None,
                 eps: float = 1e-8) -> torch.Tensor:
    """
    扭转角损失（wrap cosine）
    
    公式: L = 1 - cos(pred - true)
    
    Args:
        pred_angles: [B, N, n_angles] 预测角度（弧度）
        true_angles: [B, N, n_angles] 真实角度（弧度）
        angle_mask: [B, N, n_angles] 有效角度掩码
        w_res: [B, N] 残基权重（可选）
        eps: 数值稳定性
        
    Returns:
        loss: scalar tensor
    """
    # 计算角度差的cosine
    diff = pred_angles - true_angles
    cosine_diff = torch.cos(diff)
    
    # 损失：1 - cos(diff)，范围[0, 2]
    angle_loss = 1.0 - cosine_diff
    
    # 应用掩码
    angle_loss = angle_loss * angle_mask.float()

    # 空 mask 保护：避免分母趋零导致梯度爆炸
    if angle_mask.float().sum() < 0.5:
        return pred_angles.new_tensor(0.0)

    # 应用残基权重
    if w_res is not None:
        # [B, N] → [B, N, 1] → broadcast
        w_res_expanded = w_res.unsqueeze(-1)
        loss = (angle_loss * w_res_expanded).sum() / (angle_mask.float() * w_res_expanded).sum().clamp(min=eps)
    else:
        loss = angle_loss.sum() / angle_mask.float().sum().clamp(min=eps)
    
    return loss


def torsion_sincos_loss(pred_sincos: torch.Tensor,
                        true_angles: torch.Tensor,
                        angle_mask: torch.Tensor,
                        w_res: Optional[torch.Tensor] = None,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable torsion loss computed directly in sin/cos space.

    For unit vectors, dot(pred, true) == cos(pred_angle - true_angle), so this
    is equivalent to the wrapped cosine loss without backpropagating through
    atan2().
    """
    orig_dtype = pred_sincos.dtype
    pred_sincos = pred_sincos.float()
    true_sincos = torch.stack([
        torch.sin(true_angles.float()),
        torch.cos(true_angles.float()),
    ], dim=-1)

    angle_loss = 1.0 - torch.sum(pred_sincos * true_sincos, dim=-1)
    angle_loss = angle_loss * angle_mask.float()

    if angle_mask.float().sum() < 0.5:
        return pred_sincos.new_tensor(0.0, dtype=orig_dtype)

    if w_res is not None:
        w_res_expanded = w_res.float().unsqueeze(-1)
        denom = (angle_mask.float() * w_res_expanded).sum().clamp(min=eps)
        loss = (angle_loss * w_res_expanded).sum() / denom
    else:
        loss = angle_loss.sum() / angle_mask.float().sum().clamp(min=eps)

    return loss.to(orig_dtype)


def chi1_rotamer_loss(logits: torch.Tensor,
                      true_chi1: torch.Tensor,
                      chi1_mask: torch.Tensor,
                      w_res: Optional[torch.Tensor] = None,
                      eps: float = 1e-8) -> torch.Tensor:
    """χ1 rotamer 三分类损失
    
    Args:
        logits: [B, N, 3] χ1 rotamer logits
        true_chi1: [B, N] 真实χ1角度（弧度，范围约[-π, π]）
        chi1_mask: [B, N] χ1是否有效的掩码
        w_res: [B, N] 残基权重（可选，通常为口袋权重warmup后结果）
        eps: 数值稳定性
    
    Returns:
        loss: scalar tensor
    """
    B, N, C = logits.shape
    assert C == 3, "chi1_rotamer_loss: logits 最后一维必须为3 (g-/t/g+)"

    # 将角度wrap到[-π, π)
    angle = ((true_chi1 + math.pi) % (2 * math.pi)) - math.pi

    # 构造离散标签：g- / g+ / t (circular nearest center)
    with torch.no_grad():
        centers = torch.tensor([-math.pi / 3, math.pi / 3, math.pi], device=angle.device, dtype=angle.dtype)
        diff = angle.unsqueeze(-1) - centers
        circ_dist = ((diff + math.pi) % (2 * math.pi) - math.pi).abs()
        labels = circ_dist.argmin(dim=-1)  # 0=g-, 1=g+, 2=t

    # 仅在有效χ1位置计算损失
    valid = chi1_mask.bool()
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    logits_flat = logits[valid]      # [K, 3]
    labels_flat = labels[valid]      # [K]

    if w_res is not None:
        weights_flat = w_res[valid]  # [K]
        loss_per = F.cross_entropy(logits_flat, labels_flat, reduction='none')  # [K]
        weighted = loss_per * weights_flat
        loss = weighted.sum() / (weights_flat.sum() + eps)
    else:
        loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')

    return loss


def chi1_rotamer_labels(true_chi1: torch.Tensor,
                        chi1_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return canonical χ1 rotamer labels and valid mask for g-/t/g+.

    Uses circular nearest-center assignment:
      g- center: -60° (-π/3)
      g+ center: +60° (+π/3)
      t  center: 180° (π)
    """
    angle = ((true_chi1 + math.pi) % (2 * math.pi)) - math.pi  # wrap to [-π, π)
    valid = chi1_mask.bool()

    # Circular distance to each center
    centers = torch.tensor([-math.pi / 3, math.pi / 3, math.pi], device=angle.device, dtype=angle.dtype)
    # diff shape: [..., 3]
    diff = angle.unsqueeze(-1) - centers
    circ_dist = (diff + math.pi) % (2 * math.pi) - math.pi
    circ_dist = circ_dist.abs()
    labels = circ_dist.argmin(dim=-1)  # 0=g-, 1=g+, 2=t

    return labels, valid


def ligand_contrastive_chi1_loss(correct_logits: torch.Tensor,
                                 decoy_logits: torch.Tensor,
                                 true_chi1: torch.Tensor,
                                 chi1_mask: torch.Tensor,
                                 w_res: Optional[torch.Tensor] = None,
                                 margin: float = 0.2,
                                 eps: float = 1e-8) -> torch.Tensor:
    """Margin loss that rewards correct-ligand confidence over decoy ligand confidence."""
    labels, valid = chi1_rotamer_labels(true_chi1, chi1_mask)
    if valid.sum() == 0:
        return correct_logits.new_tensor(0.0)

    correct_gold = correct_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    decoy_gold = decoy_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    loss_per = F.relu(float(margin) - (correct_gold - decoy_gold))

    if w_res is not None:
        weights = w_res.float() * valid.float()
        return (loss_per * weights).sum() / (weights.sum() + eps)
    return loss_per[valid].mean()


# ============================================================================
# Phase 2: Switch-aware and ligand-causal losses
# ============================================================================


def switch_bce_loss(
    logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    w_res: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """BCE loss for predicting apo→holo rotamer switch.

    Switch = apo rotamer bin != holo rotamer bin.
    Score = max(non-apo logit) - apo logit. Positive means predicting a switch.
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    switch_target = (apo_labels != holo_labels).float()  # [B, N]

    # Switch score: max non-apo logit minus apo logit
    apo_logit = logits.gather(-1, apo_labels.unsqueeze(-1)).squeeze(-1)  # [B, N]
    # Mask apo bin with -inf, then take max
    mask_for_max = torch.zeros_like(logits).scatter_(-1, apo_labels.unsqueeze(-1), -1e9)
    max_other_logit = (logits + mask_for_max).max(dim=-1).values  # [B, N]
    switch_score = max_other_logit - apo_logit  # [B, N]

    loss_per = F.binary_cross_entropy_with_logits(switch_score, switch_target, reduction='none')

    if w_res is not None:
        weights = w_res.float() * valid.float()
        return (loss_per * weights).sum() / (weights.sum() + eps)
    return loss_per[valid].mean()


def rescue_noharm_loss(
    logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    w_res: Optional[torch.Tensor] = None,
    margin: float = 0.5,
    noharm_weight: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Asymmetric margin loss: reward rescue, penalize harmful flips.

    Rescue (apo wrong, should predict holo): holo_logit - apo_logit > margin
    No-harm (apo correct, should keep): apo_logit - max_other_logit > margin
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    is_switch = (apo_labels != holo_labels) & valid  # rescue targets
    is_keep = (apo_labels == holo_labels) & valid     # no-harm targets

    holo_logit = logits.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
    apo_logit = logits.gather(-1, apo_labels.unsqueeze(-1)).squeeze(-1)

    # Rescue: holo_logit should exceed apo_logit by margin
    rescue_loss = F.relu(margin - (holo_logit - apo_logit))

    # No-harm: apo_logit should exceed max other logit by margin
    mask_for_max = torch.zeros_like(logits).scatter_(-1, apo_labels.unsqueeze(-1), -1e9)
    max_other_logit = (logits + mask_for_max).max(dim=-1).values
    noharm_loss = F.relu(margin - (apo_logit - max_other_logit))

    if w_res is not None:
        w = w_res.float()
        rescue_sum = (rescue_loss * w * is_switch.float()).sum()
        rescue_denom = (w * is_switch.float()).sum().clamp(min=eps)
        noharm_sum = (noharm_loss * w * is_keep.float()).sum()
        noharm_denom = (w * is_keep.float()).sum().clamp(min=eps)
    else:
        rescue_sum = rescue_loss[is_switch].sum()
        rescue_denom = is_switch.float().sum().clamp(min=eps)
        noharm_sum = noharm_loss[is_keep].sum()
        noharm_denom = is_keep.float().sum().clamp(min=eps)

    return rescue_sum / rescue_denom + noharm_weight * noharm_sum / noharm_denom


def _compute_g_i(full_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """G_i(k) = log_softmax(full)_k - log_softmax(base)_k.  [B, N, K] -> [B, N, K]"""
    return F.log_softmax(full_logits, dim=-1) - F.log_softmax(base_logits, dim=-1)


def g_lift_switch_loss(
    full_logits: torch.Tensor,
    base_logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    margin: float = 0.5,
    w_res: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Likelihood-ratio lift loss on switch residues.

    Switch residues = apo rotamer != holo rotamer (the only places where ligand
    evidence should rewrite the prior).  We require:

        G_i(holo) - G_i(apo) >= margin

    where G_i(k) = log_softmax(full)_k - log_softmax(base)_k.
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid & (apo_labels != holo_labels)
    if valid.sum() == 0:
        return full_logits.new_tensor(0.0)

    G = _compute_g_i(full_logits, base_logits)  # [B, N, K]
    G_holo = G.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
    G_apo = G.gather(-1, apo_labels.unsqueeze(-1)).squeeze(-1)

    loss_per = F.relu(margin - (G_holo - G_apo))  # [B, N]
    mask_f = valid.float()
    if w_res is not None:
        weights = w_res.float() * mask_f
    else:
        weights = mask_f
    return (loss_per * weights).sum() / (weights.sum() + eps)


def g_noharm_loss(
    full_logits: torch.Tensor,
    base_logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    margin: float = 0.5,
    w_res: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """No-harm: on apo-correct residues, ligand evidence should NOT push G_i(apo)
    below max_other.  We require G_i(apo) - max_{k!=apo} G_i(k) >= margin.
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid & (apo_labels == holo_labels)
    if valid.sum() == 0:
        return full_logits.new_tensor(0.0)

    G = _compute_g_i(full_logits, base_logits)  # [B, N, K]
    G_apo = G.gather(-1, apo_labels.unsqueeze(-1)).squeeze(-1)
    mask_for_max = torch.zeros_like(G).scatter_(-1, apo_labels.unsqueeze(-1), -1e9)
    G_max_other = (G + mask_for_max).max(dim=-1).values

    loss_per = F.relu(margin - (G_apo - G_max_other))
    mask_f = valid.float()
    if w_res is not None:
        weights = w_res.float() * mask_f
    else:
        weights = mask_f
    return (loss_per * weights).sum() / (weights.sum() + eps)


def g_zero_noncontact_loss(
    full_logits: torch.Tensor,
    base_logits: torch.Tensor,
    chi1_mask: torch.Tensor,
    non_contact_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """On non-contact residues, the ligand should not produce a likelihood
    lift in any direction.  We minimise mean ||G_i||^2.

    non_contact_mask: [B, N] bool, True where residue is far from ligand.
    """
    valid = chi1_mask.bool() & non_contact_mask.bool()
    if valid.sum() == 0:
        return full_logits.new_tensor(0.0)
    G = _compute_g_i(full_logits, base_logits)  # [B, N, K]
    sq = (G ** 2).sum(dim=-1)  # [B, N]
    valid_f = valid.float()
    return (sq * valid_f).sum() / (valid_f.sum() + eps)


# =====================================================================
# Phase-1 v2 losses (GPT-5.5 Pro plan): G-vector posterior learning
# Replaces pairwise g_lift_switch_loss / g_noharm_loss with a fuller
# objective that addresses (a) third-bin leakage, (b) over-strong noharm.
# =====================================================================

def g_switch_dir_loss(
    full_logits: torch.Tensor,
    base_logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """G-vector direction CE on switch (apo!=holo) residues.

    L = - log softmax(G / T)[holo_bin]

    where G = log_softmax(full) - log_softmax(base).
    Forces residual to push the *full 3-bin posterior* toward holo, instead of
    just lifting holo-vs-apo (which leaks into the third bin).

    contact_mask, if provided, restricts the loss to contact residues only.
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid & (apo_labels != holo_labels)
    if contact_mask is not None:
        valid = valid & contact_mask.bool()
    if valid.sum() == 0:
        return full_logits.new_tensor(0.0)

    G = _compute_g_i(full_logits, base_logits)  # [B, N, K]
    G_scaled = G / max(float(temperature), 1e-3)
    log_p_G = F.log_softmax(G_scaled, dim=-1)  # [B, N, K]
    nll_per = -log_p_G.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)  # [B, N]

    valid_f = valid.float()
    return (nll_per * valid_f).sum() / (valid_f.sum() + eps)


def g_switch_amp_loss(
    full_logits: torch.Tensor,
    base_logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    margin: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Hinge that ensures G_holo >= margin on switch residues.

    Without this, direction CE on G is scale-free and can collapse to small
    G everywhere.  We require the residual to actually *increase* the holo
    posterior by at least `margin` log-prob units.
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid & (apo_labels != holo_labels)
    if contact_mask is not None:
        valid = valid & contact_mask.bool()
    if valid.sum() == 0:
        return full_logits.new_tensor(0.0)

    G = _compute_g_i(full_logits, base_logits)
    G_holo = G.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)  # [B, N]

    loss_per = F.relu(margin - G_holo)
    valid_f = valid.float()
    return (loss_per * valid_f).sum() / (valid_f.sum() + eps)


def g_switch_rank_loss(
    full_logits: torch.Tensor,
    base_logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    margin: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Hinge that ensures G_holo - max_{k!=holo} G_k >= margin on switch residues.

    Companion to g_switch_dir_loss / g_switch_amp_loss: explicitly demands
    holo is the *top* bin in G_i (defeats the third-bin-leakage failure mode
    we observed in Phase-1 v1).
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid & (apo_labels != holo_labels)
    if contact_mask is not None:
        valid = valid & contact_mask.bool()
    if valid.sum() == 0:
        return full_logits.new_tensor(0.0)

    G = _compute_g_i(full_logits, base_logits)  # [B, N, K]
    G_holo = G.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
    mask_for_max = torch.zeros_like(G).scatter_(-1, holo_labels.unsqueeze(-1), -1e9)
    G_max_other = (G + mask_for_max).max(dim=-1).values

    loss_per = F.relu(margin - (G_holo - G_max_other))
    valid_f = valid.float()
    return (loss_per * valid_f).sum() / (valid_f.sum() + eps)


def g_antiharm_loss(
    full_logits: torch.Tensor,
    base_logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    tau: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Anti-harm regulariser on apo-correct residues.

    On residues where apo == holo (no switch needed), penalise any positive
    G on a non-apo bin: residual should *not* push the posterior away from
    the already-correct apo answer.

    L = relu(max_{k != apo} G_k - tau)

    Critically this does NOT require G_apo > other + margin (which v1's
    g_noharm did, forcing residual to support apo even without ligand
    evidence).  We only forbid harmful flips.
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid & (apo_labels == holo_labels)
    if contact_mask is not None:
        valid = valid & contact_mask.bool()
    if valid.sum() == 0:
        return full_logits.new_tensor(0.0)

    G = _compute_g_i(full_logits, base_logits)  # [B, N, K]
    mask_for_max = torch.zeros_like(G).scatter_(-1, apo_labels.unsqueeze(-1), -1e9)
    G_max_other = (G + mask_for_max).max(dim=-1).values  # [B, N]

    loss_per = F.relu(G_max_other - tau)
    valid_f = valid.float()
    return (loss_per * valid_f).sum() / (valid_f.sum() + eps)


def g_decoy_contrastive_loss(
    correct_full_logits: torch.Tensor,
    decoy_full_logits: torch.Tensor,
    base_logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    margin: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Decoy contrastive: correct ligand should lift G_holo more than a decoy.

    L = relu( margin - (G_correct[holo] - G_decoy[holo]) )

    Applied on switch residues (and optionally contact residues).  The decoy
    is typically a translated-away or no-ligand forward pass.

    This is the strongest evidence of *ligand-causality*: if shuffling /
    translating the ligand still produces the same posterior lift, the
    residual is not actually using ligand identity.
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid & (apo_labels != holo_labels)
    if contact_mask is not None:
        valid = valid & contact_mask.bool()
    if valid.sum() == 0:
        return correct_full_logits.new_tensor(0.0)

    G_correct = _compute_g_i(correct_full_logits, base_logits)
    G_decoy = _compute_g_i(decoy_full_logits, base_logits)
    G_correct_holo = G_correct.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
    G_decoy_holo = G_decoy.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)

    loss_per = F.relu(margin - (G_correct_holo - G_decoy_holo))
    valid_f = valid.float()
    return (loss_per * valid_f).sum() / (valid_f.sum() + eps)


def typed_candidate_energy_loss(
    correct_energy: torch.Tensor,
    decoy_energy: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    non_contact_mask: Optional[torch.Tensor] = None,
    margin: float = 0.05,
    noharm_weight: float = 0.1,
    noncontact_zero_weight: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Typed candidate energy supervision for ligand-chemistry controls.

    The primary term asks the correct ligand's typed energy on the holo
    rotamer bin to exceed the decoy ligand's typed energy on switch/contact
    residues.  Auxiliary terms keep the typed branch quiet on apo-correct and
    non-contact residues so it cannot become an unconditional rotamer prior.
    """
    # Keep the loss connected to the typed branch even for batches without
    # usable switch/contact supervision. Otherwise DDP can classify the whole
    # batch loss as detached and skip unrelated trainable updates.
    grad_anchor = (correct_energy.sum() + decoy_energy.sum()) * 0.0

    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid
    switch_valid = valid & (apo_labels != holo_labels)
    if contact_mask is not None:
        switch_valid = switch_valid & contact_mask.bool()

    if switch_valid.any():
        correct_holo = correct_energy.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
        decoy_holo = decoy_energy.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
        lift_loss = F.relu(float(margin) - (correct_holo - decoy_holo))
        lift = (lift_loss * switch_valid.float()).sum() / (switch_valid.float().sum() + eps)
    else:
        lift = grad_anchor

    keep_valid = valid & (apo_labels == holo_labels)
    if contact_mask is not None:
        keep_valid = keep_valid & contact_mask.bool()
    if keep_valid.any() and noharm_weight > 0.0:
        apo_energy = correct_energy.gather(-1, apo_labels.unsqueeze(-1)).squeeze(-1)
        mask_for_max = torch.zeros_like(correct_energy).scatter_(-1, apo_labels.unsqueeze(-1), -1e9)
        max_other = (correct_energy + mask_for_max).max(dim=-1).values
        noharm_loss = F.relu(max_other - apo_energy)
        noharm = (noharm_loss * keep_valid.float()).sum() / (keep_valid.float().sum() + eps)
    else:
        noharm = grad_anchor

    if non_contact_mask is not None and noncontact_zero_weight > 0.0:
        zero_valid = valid & non_contact_mask.bool()
        if zero_valid.any():
            zero_loss = (correct_energy ** 2).sum(dim=-1)
            zero = (zero_loss * zero_valid.float()).sum() / (zero_valid.float().sum() + eps)
        else:
            zero = grad_anchor
    else:
        zero = grad_anchor

    return lift + float(noharm_weight) * noharm + float(noncontact_zero_weight) * zero


def candidate_decoy_rerank_loss(
    correct_logits: torch.Tensor,
    decoy_logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    w_res: Optional[torch.Tensor] = None,
    decoy_margin: float = 0.1,
    rank_margin: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Explicit ligand-causal reranking loss on switch χ1 candidates.

    Unlike the G-vector contrastive loss, this objective acts directly on the
    candidate posterior.  On apo→holo switch residues, the correct ligand must
    assign the holo rotamer higher log-probability than a decoy/no-ligand pass;
    optionally, the holo rotamer must also outrank the other two candidates in
    the correct-ligand posterior.
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid & (apo_labels != holo_labels)
    if contact_mask is not None:
        valid = valid & contact_mask.bool()
    if valid.sum() == 0:
        return correct_logits.new_tensor(0.0)

    correct_logp = F.log_softmax(correct_logits, dim=-1)
    decoy_logp = F.log_softmax(decoy_logits, dim=-1)
    correct_holo = correct_logp.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
    decoy_holo = decoy_logp.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)

    loss_per = F.relu(float(decoy_margin) - (correct_holo - decoy_holo))

    if rank_margin > 0.0:
        mask_for_max = torch.zeros_like(correct_logp).scatter_(-1, holo_labels.unsqueeze(-1), -1e9)
        correct_max_other = (correct_logp + mask_for_max).max(dim=-1).values
        loss_per = loss_per + F.relu(float(rank_margin) - (correct_holo - correct_max_other))

    weights = valid.float()
    if w_res is not None:
        weights = weights * w_res.float()
    return (loss_per * weights).sum() / (weights.sum() + eps)


def candidate_decoy_contrastive_loss(
    correct_logits: torch.Tensor,
    decoy_logits: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    contact_mask: Optional[torch.Tensor] = None,
    decoy_margin: float = 0.1,
    repulsion_margin: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Strong contrastive loss that actively pushes decoy predictions away from holo.

    On switch residues:
    1. correct_holo > decoy_holo + margin (correct ligand prefers holo rotamer)
    2. decoy_max_other > decoy_holo + repulsion (decoy avoids holo rotamer)
    """
    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    valid = apo_valid & holo_valid & (apo_labels != holo_labels)
    if contact_mask is not None:
        valid = valid & contact_mask.bool()
    grad_anchor = (correct_logits.sum() + decoy_logits.sum()) * 0.0
    if valid.sum() == 0:
        return grad_anchor

    correct_logp = F.log_softmax(correct_logits, dim=-1)
    decoy_logp = F.log_softmax(decoy_logits, dim=-1)
    correct_holo = correct_logp.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
    decoy_holo = decoy_logp.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)

    margin_loss = F.relu(float(decoy_margin) - (correct_holo - decoy_holo))

    mask_for_max = torch.zeros_like(decoy_logp).scatter_(-1, holo_labels.unsqueeze(-1), -1e9)
    decoy_max_other = (decoy_logp + mask_for_max).max(dim=-1).values
    repulsion_loss = F.relu(float(repulsion_margin) - (decoy_max_other - decoy_holo))

    loss_per = margin_loss + repulsion_loss
    return (loss_per * valid.float()).sum() / (valid.float().sum() + eps)


def ligand_internal_guidance_loss(
    logits_with_ligand: torch.Tensor,
    logits_without_ligand: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    switch_margin: float = 0.1,
    nonswitch_weight: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Ligand internal guidance loss inspired by RAEv2.

    guidance = logits_with_ligand - logits_without_ligand

    On switch residues (apo != holo): guidance magnitude should be large.
    On non-switch residues: guidance magnitude should be small.

    Analogous to RAEv2 internal guidance: x_guided = x_full + w*(x_full - x_repa).
    """
    valid = chi1_mask.bool()
    if node_mask is not None:
        valid = valid & node_mask.bool()
    grad_anchor = (logits_with_ligand.sum() + logits_without_ligand.sum()) * 0.0
    if valid.sum() == 0:
        return grad_anchor

    guidance = logits_with_ligand - logits_without_ligand
    guidance_mag = guidance.norm(dim=-1)

    apo_labels, apo_valid = chi1_rotamer_labels(apo_chi1, chi1_mask)
    holo_labels, holo_valid = chi1_rotamer_labels(holo_chi1, chi1_mask)
    label_valid = apo_valid & holo_valid & valid

    switch_mask = label_valid & (apo_labels != holo_labels)
    nonswitch_mask = label_valid & (apo_labels == holo_labels)

    switch_loss = torch.tensor(0.0, device=logits_with_ligand.device)
    if switch_mask.any():
        switch_mag = guidance_mag[switch_mask]
        switch_loss = F.relu(float(switch_margin) - switch_mag).mean()

    nonswitch_loss = torch.tensor(0.0, device=logits_with_ligand.device)
    if nonswitch_mask.any() and nonswitch_weight > 0.0:
        nonswitch_mag = guidance_mag[nonswitch_mask]
        nonswitch_loss = nonswitch_mag.mean()

    return switch_loss + float(nonswitch_weight) * nonswitch_loss + grad_anchor


def ligand_discriminator_loss(
    correct_score: torch.Tensor,
    decoy_score: torch.Tensor,
    apo_chi1: torch.Tensor,
    holo_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    margin: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Contrastive loss for independent ligand discriminator.

    On all residues (not just switch): correct ligand binding score should
    exceed decoy binding score by margin. This trains the discriminator to
    distinguish correct from decoy ligands independent of rotamer prediction.
    """
    valid = chi1_mask.bool()
    if node_mask is not None:
        valid = valid & node_mask.bool()
    grad_anchor = (correct_score.sum() + decoy_score.sum()) * 0.0
    if valid.sum() == 0:
        return grad_anchor

    loss_per = F.relu(float(margin) - (correct_score - decoy_score))
    return (loss_per * valid.float()).sum() / (valid.float().sum() + eps)


def protein_ligand_contrastive_loss(
    protein_repr_correct: torch.Tensor,
    protein_repr_decoy: torch.Tensor,
    ligand_repr_correct: torch.Tensor,
    ligand_repr_decoy: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Protein-ligand contrastive representation loss (DrugCLIP/ConPLex-inspired).

    Learns to distinguish (protein, correct_ligand) from (protein, decoy_ligand)
    at the representation level, independent of chi1 classification.

    Uses InfoNCE-style contrastive loss:
    - Positive pairs: (protein_repr_correct, ligand_repr_correct)
    - Negative pairs: (protein_repr_correct, ligand_repr_decoy)

    The loss encourages the protein representation to be more similar to the
    correct ligand representation than to decoy ligand representations.

    Args:
        protein_repr_correct: [B, N, D_prot] protein representation with correct ligand
        protein_repr_decoy: [B, N, D_prot] protein representation with decoy ligand
        ligand_repr_correct: [B, D_lig] correct ligand representation (pooled)
        ligand_repr_decoy: [B, D_lig] decoy ligand representation (pooled)
        node_mask: [B, N] valid residue mask
        temperature: temperature for contrastive loss
        eps: numerical stability

    Returns:
        loss: scalar tensor
    """
    B, N, D_prot = protein_repr_correct.shape
    D_lig = ligand_repr_correct.shape[-1]

    # Pool protein representations over residues (mean pooling with mask)
    if node_mask is not None:
        mask_f = node_mask.float().unsqueeze(-1)
        protein_pooled_correct = (protein_repr_correct * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + eps)
        protein_pooled_decoy = (protein_repr_decoy * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + eps)
    else:
        protein_pooled_correct = protein_repr_correct.mean(dim=1)
        protein_pooled_decoy = protein_repr_decoy.mean(dim=1)

    # Do not instantiate projection layers inside a loss function: those
    # parameters would be random on every call and would never be optimized.
    # Use the shared leading subspace as a deterministic projection.  If a
    # learned projection is needed, it should live on the model as a real module.
    projection_dim = min(D_prot, D_lig)
    if projection_dim <= 0:
        return protein_repr_correct.new_tensor(0.0)
    protein_pooled_correct = protein_pooled_correct[..., :projection_dim]
    protein_pooled_decoy = protein_pooled_decoy[..., :projection_dim]
    ligand_repr_correct = ligand_repr_correct[..., :projection_dim]
    ligand_repr_decoy = ligand_repr_decoy[..., :projection_dim]

    # Normalize representations
    protein_pooled_correct = F.normalize(protein_pooled_correct, dim=-1)
    protein_pooled_decoy = F.normalize(protein_pooled_decoy, dim=-1)
    ligand_repr_correct = F.normalize(ligand_repr_correct, dim=-1)
    ligand_repr_decoy = F.normalize(ligand_repr_decoy, dim=-1)

    # Compute similarities: positive pair vs two stable negatives.
    sim_pos = (protein_pooled_correct * ligand_repr_correct).sum(dim=-1) / temperature
    sim_neg_ligand = (protein_pooled_correct * ligand_repr_decoy).sum(dim=-1) / temperature
    sim_neg_protein = (protein_pooled_decoy * ligand_repr_correct).sum(dim=-1) / temperature

    # InfoNCE loss: -log(exp(sim_pos) / sum(exp(sim_*)))
    logits = torch.stack([sim_pos, sim_neg_ligand, sim_neg_protein], dim=-1)
    targets = torch.zeros(B, dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, targets, reduction='mean')

    return loss


def ligand_residual_chi1_loss(
    correct_logits: torch.Tensor,
    nolig_logits: torch.Tensor,
    true_chi1: torch.Tensor,
    chi1_mask: torch.Tensor,
    w_res: Optional[torch.Tensor] = None,
    margin: float = 0.3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Maximize log-prob lift of correct ligand over no-ligand on gold rotamer.

    For pocket/contact residues (high w_res): correct ligand should increase
    the gold rotamer logit relative to no-ligand.
    """
    labels, valid = chi1_rotamer_labels(true_chi1, chi1_mask)
    if valid.sum() == 0:
        return correct_logits.new_tensor(0.0)

    correct_gold = correct_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    nolig_gold = nolig_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    loss_per = F.relu(margin - (correct_gold - nolig_gold))

    if w_res is not None:
        weights = w_res.float() * valid.float()
        return (loss_per * weights).sum() / (weights.sum() + eps)
    return loss_per[valid].mean()


def binary_contact_loss(logits: torch.Tensor,
                        contact_mask: torch.Tensor,
                        residue_mask: Optional[torch.Tensor] = None,
                        pos_weight: Optional[torch.Tensor] = None,
                        eps: float = 1e-8) -> torch.Tensor:
    """Residue-level binary contact classification loss.

    Args:
        logits: [B, N] predicted residue-contact logits.
        contact_mask: [B, N] binary residue-ligand contact labels.
        residue_mask: [B, N] valid residue mask. If omitted, all residues count.
        pos_weight: Optional scalar tensor for positive-class reweighting.
        eps: numerical stability.
    """
    if residue_mask is None:
        residue_mask = torch.ones_like(contact_mask, dtype=torch.bool)
    valid = residue_mask.bool()
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    targets = contact_mask.float()
    loss_per = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction='none',
        pos_weight=pos_weight,
    )
    loss = (loss_per * valid.float()).sum() / (valid.float().sum() + eps)
    return loss


# ============================================================================
# 距离损失
# ============================================================================

def distance_loss(pred_coords: torch.Tensor,
                 true_coords: torch.Tensor,
                 w_res: Optional[torch.Tensor] = None,
                 eps: float = 1e-8) -> torch.Tensor:
    """
    成对距离损失
    
    公式: L = (|pred_ij| - |true_ij|)²
    
    Args:
        pred_coords: [B, N, 3] 预测Cα坐标
        true_coords: [B, N, 3] 真实Cα坐标
        w_res: [B, N] 残基权重（可选）
        eps: 数值稳定性
        
    Returns:
        loss: scalar tensor
    """
    # 计算成对距离
    pred_diff = pred_coords.unsqueeze(2) - pred_coords.unsqueeze(1)  # [B, N, N, 3]
    true_diff = true_coords.unsqueeze(2) - true_coords.unsqueeze(1)
    
    pred_dist = torch.sqrt(torch.sum(pred_diff ** 2, dim=-1) + eps)  # [B, N, N]
    true_dist = torch.sqrt(torch.sum(true_diff ** 2, dim=-1) + eps)
    
    # 距离差的平方
    dist_diff = (pred_dist - true_dist) ** 2
    
    # 应用权重：max(w_i, w_j)
    if w_res is not None:
        w_pair = torch.maximum(w_res.unsqueeze(2), w_res.unsqueeze(1))  # [B, N, N]
        loss = (dist_diff * w_pair).sum() / (w_pair.sum() + eps)
    else:
        # 只计算上三角（避免重复）
        B, N = pred_coords.shape[:2]
        triu_mask = torch.triu(torch.ones(N, N, device=pred_coords.device, dtype=torch.bool), diagonal=1)
        loss = dist_diff[:, triu_mask].mean()
    
    return loss


# ============================================================================
# 碰撞惩罚
# ============================================================================

def clash_penalty(coords: torch.Tensor,
                 clash_threshold: float = 2.0,
                 bond_graph: Optional[torch.Tensor] = None,
                 aatype: Optional[torch.Tensor] = None,
                 atom_mask: Optional[torch.Tensor] = None,
                 eps: float = 1e-8,
                 sample_size: int = 512) -> torch.Tensor:
    """
    碰撞惩罚（随机采样，实验验证最优，增强数值稳定性）

    自动排除 1-2/1-3/1-4 共价键近邻（通过 aatype 查表或 bond_graph）。

    Args:
        coords: [B, N, 3] 原子坐标
        clash_threshold: 碰撞阈值（Å）
        bond_graph: [B, N, N] 成键关系（可选，优先级高于 aatype）
        aatype: [B, N_res] 残基类型索引，用于查表排除共价键近邻
        atom_mask: [B, N] 原子有效掩码（可选）
        eps: 数值稳定性
        sample_size: 采样大小

    Returns:
        loss: scalar tensor
    """
    global BOND_EXCLUSION_MASK

    if torch.isnan(coords).any() or torch.isinf(coords).any():
        return coords.new_tensor(0.0)

    B, N, _ = coords.shape
    sampled_indices = None
    total_atoms = N
    if atom_mask is not None:
        atom_mask = atom_mask.bool()

    if N > 1000:
        sampled_indices = torch.randperm(N, device=coords.device)[:sample_size]
        coords = coords[:, sampled_indices, :]
        if atom_mask is not None:
            atom_mask = atom_mask[:, sampled_indices]
        if bond_graph is not None:
            bond_graph = bond_graph[:, sampled_indices][:, :, sampled_indices]
        N = sample_size

    orig_dtype = coords.dtype
    coords = coords.float()

    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    diff = torch.clamp(diff, min=-1000.0, max=1000.0)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)

    pair_mask = torch.triu(torch.ones(N, N, device=coords.device, dtype=torch.bool), diagonal=1)
    pair_mask = pair_mask.unsqueeze(0).expand(B, -1, -1)
    if atom_mask is not None:
        pair_mask = pair_mask & atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)

    if bond_graph is not None:
        bonded = bond_graph.bool()
        bonded_13 = torch.matmul(bonded.float(), bonded.float()) > 0
        exclude_mask = bonded | bonded_13
        pair_mask = pair_mask & (~exclude_mask)
    elif aatype is not None:
        exclude_mask = _build_aatype_exclusion_mask(
            aatype,
            total_atoms=total_atoms,
            device=coords.device,
            sampled_indices=sampled_indices,
        )
        if exclude_mask is not None:
            pair_mask = pair_mask & (~exclude_mask)

    penetration = clash_threshold - dist
    penalty = torch.clamp(penetration, min=0.0) ** 2

    mask_sum = pair_mask.float().sum()
    if mask_sum < eps:
        return coords.new_tensor(0.0)

    loss = (penalty * pair_mask.float()).sum() / (mask_sum + eps)

    if torch.isnan(loss) or torch.isinf(loss):
        return coords.new_tensor(0.0)

    return loss.to(orig_dtype)


def latent_change_prediction_loss(
    delta_z_pred: torch.Tensor,
    delta_z_true: torch.Tensor,
    node_mask: torch.Tensor,
    reduction: str = 'mean',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Latent change prediction loss.
    
    Computes MSE between predicted and true delta_z in latent space.
    
    Args:
        delta_z_pred: [B, N, D] predicted change in latent space
        delta_z_true: [B, N, D] true change (z_holo - z_apo)
        node_mask: [B, N] valid residue mask
        reduction: 'mean' or 'sum'
        eps: small value for numerical stability
    
    Returns:
        Scalar loss value
    """
    diff = delta_z_pred - delta_z_true
    loss_per_residue = (diff ** 2).sum(dim=-1)
    
    mask_sum = node_mask.float().sum()
    if mask_sum < eps:
        return delta_z_pred.new_tensor(0.0)
    
    if reduction == 'mean':
        loss = (loss_per_residue * node_mask.float()).sum() / (mask_sum + eps)
    elif reduction == 'sum':
        loss = (loss_per_residue * node_mask.float()).sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    
    if torch.isnan(loss) or torch.isinf(loss):
        return delta_z_pred.new_tensor(0.0)
    
    return loss
