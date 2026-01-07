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
from typing import Tuple, Optional
import math


# ============================================================================
# FAPE损失（Frame Aligned Point Error）
# ============================================================================

def fape_loss(pred_coords: torch.Tensor,
             true_coords: torch.Tensor,
             pred_frames: Tuple[torch.Tensor, torch.Tensor],
             true_frames: Tuple[torch.Tensor, torch.Tensor],
             w_res: Optional[torch.Tensor] = None,
             clamp_distance: float = 10.0,
             eps: float = 1e-8) -> torch.Tensor:
    """
    FAPE损失（可微分版本，增强数值稳定性）
    
    公式：对每个残基i，将坐标变换到局部帧后计算L2误差
    
    Args:
        pred_coords: [B, N, 3] 或 [B, N, n_atoms, 3] 预测坐标
        true_coords: [B, N, 3] 或 [B, N, n_atoms, 3] 真实坐标
        pred_frames: (R[B,N,3,3], t[B,N,3]) 预测帧
        true_frames: (R[B,N,3,3], t[B,N,3]) 真实帧
        w_res: [B, N] 残基权重（口袋加权，可选）
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
    
    # 在 fp32 下计算以避免 fp16 溢出
    orig_dtype = pred_coords.dtype
    pred_coords = pred_coords.float()
    true_coords = true_coords.float()
    pred_R = pred_R.float()
    pred_t = pred_t.float()
    true_R = true_R.float()
    true_t = true_t.float()
    
    # 变换到真实帧的局部坐标系
    # x_local = R_true^T @ (x - t_true)
    pred_local = torch.einsum('bnik,bnk->bni', true_R.transpose(-2, -1), pred_coords - true_t)
    true_local = torch.einsum('bnik,bnk->bni', true_R.transpose(-2, -1), true_coords - true_t)
    
    # 计算误差（数值稳定：先 clamp 差值再求平方）
    coord_diff = pred_local - true_local
    coord_diff = torch.clamp(coord_diff, min=-clamp_distance * 10, max=clamp_distance * 10)  # 防止极端值
    diff_sq = torch.sum(coord_diff ** 2, dim=-1)  # [B, N]
    diff = torch.sqrt(diff_sq + eps)
    
    # Clamp 最终距离
    diff = torch.clamp(diff, max=clamp_distance)
    
    # 应用权重
    if w_res is not None:
        w_res = w_res.float()
        loss = (diff * w_res).sum() / (w_res.sum() + eps)
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
    
    # 应用残基权重
    if w_res is not None:
        # [B, N] → [B, N, 1] → broadcast
        w_res_expanded = w_res.unsqueeze(-1)
        loss = (angle_loss * w_res_expanded).sum() / (angle_mask.float() * w_res_expanded).sum().clamp(min=eps)
    else:
        loss = angle_loss.sum() / angle_mask.float().sum().clamp(min=eps)
    
    return loss


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

    # 构造离散标签：g- / t / g+
    with torch.no_grad():
        labels = torch.zeros_like(angle, dtype=torch.long)
        mask_valid = chi1_mask.bool()

        # g-: angle < -π/3
        labels[mask_valid & (angle < -math.pi / 3)] = 0
        # t: [-π/3, π/3)
        t_region = mask_valid & (angle >= -math.pi / 3) & (angle < math.pi / 3)
        labels[t_region] = 1
        # g+: angle >= π/3
        labels[mask_valid & (angle >= math.pi / 3)] = 2

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
                 eps: float = 1e-8,
                 sample_size: int = 512) -> torch.Tensor:
    """
    碰撞惩罚（随机采样，实验验证最优，增强数值稳定性）
    
    方法：
    - 当N>1000时，随机采样sample_size个原子
    - 在采样子集上计算clash
    
    科学依据（见docs/CLASH_ABLATION.md）：
    - 统计学：样本量512足够代表总体
    - 实验验证：χ1=71%, Val=0.21（vs完整近邻的χ1=70%, Val=0.41）
    - 隐式正则化：防止过拟合PDB的个体差异
    
    Args:
        coords: [B, N, 3] 原子坐标
        clash_threshold: 碰撞阈值（Å）
        bond_graph: [B, N, N] 成键关系（可选）
        eps: 数值稳定性
        sample_size: 采样大小
        
    Returns:
        loss: scalar tensor
    """
    # NaN/Inf 输入检查
    if torch.isnan(coords).any() or torch.isinf(coords).any():
        return coords.new_tensor(0.0)
    
    B, N, _ = coords.shape
    
    # 如果N太大，随机采样（实验验证的最优方法）
    if N > 1000:
        indices = torch.randperm(N, device=coords.device)[:sample_size]
        coords = coords[:, indices, :]
        N = sample_size
    
    # 在 fp32 下计算以避免 fp16 溢出
    orig_dtype = coords.dtype
    coords = coords.float()
    
    # 计算成对距离
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 3]
    # 先 clamp 差值，防止极端坐标导致溢出
    diff = torch.clamp(diff, min=-1000.0, max=1000.0)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)  # [B, N, N]
    
    # 创建上三角掩码
    triu_mask = torch.triu(torch.ones(N, N, device=coords.device, dtype=torch.bool), diagonal=1)
    
    # 排除成键原子（如果提供）
    if bond_graph is not None:
        bonded = bond_graph.bool()
        bonded_13 = torch.matmul(bonded.float(), bonded.float()) > 0
        exclude_mask = bonded | bonded_13
        triu_mask = triu_mask.unsqueeze(0) & (~exclude_mask)
    else:
        triu_mask = triu_mask.unsqueeze(0).expand(B, -1, -1)
    
    # 计算碰撞惩罚
    penetration = clash_threshold - dist  # [B, N, N]
    penalty = torch.clamp(penetration, min=0.0) ** 2
    
    # 只计算有效对
    mask_sum = triu_mask.float().sum()
    if mask_sum < eps:
        return coords.new_tensor(0.0)
    
    loss = (penalty * triu_mask.float()).sum() / (mask_sum + eps)
    
    # 最终 NaN 检查
    if torch.isnan(loss) or torch.isinf(loss):
        return coords.new_tensor(0.0)
    
    return loss.to(orig_dtype)


