"""
评估指标

功能：
1. 口袋iRMSD（Kabsch对齐）
2. χ1命中率（侧链扭转角准确度）
3. Clash百分比（原子碰撞检测）
4. FAPE（局部帧对齐误差）

Author: BINDRAE Team
Date: 2025-10-28
"""

import numpy as np
import torch
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as ScipyRotation


# ============================================================================
# Kabsch对齐
# ============================================================================

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kabsch算法：找到最优旋转矩阵使P对齐到Q
    
    Args:
        P: [N, 3] 待对齐的点集
        Q: [N, 3] 参考点集
        
    Returns:
        R: [3, 3] 旋转矩阵
        t: [3] 平移向量
        使得 P_aligned = P @ R.T + t 最接近 Q
    """
    # 中心化
    P_center = P.mean(axis=0)
    Q_center = Q.mean(axis=0)
    
    P_centered = P - P_center
    Q_centered = Q - Q_center
    
    # SVD分解
    H = P_centered.T @ Q_centered  # [3, 3]
    U, S, Vt = np.linalg.svd(H)
    
    # 旋转矩阵
    R = Vt.T @ U.T
    
    # 处理镜像情况
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 平移向量
    t = Q_center - P_center @ R.T
    
    return R, t


def compute_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    计算两个点集的RMSD
    
    Args:
        P: [N, 3]
        Q: [N, 3]
        
    Returns:
        rmsd: float
    """
    diff = P - Q
    msd = np.mean(np.sum(diff ** 2, axis=-1))
    rmsd = np.sqrt(msd)
    return float(rmsd)


# ============================================================================
# 口袋iRMSD
# ============================================================================

def compute_pocket_irmsd(pred_coords: np.ndarray,
                        true_coords: np.ndarray,
                        pocket_mask: np.ndarray,
                        eps: float = 1e-8) -> float:
    """
    计算口袋残基的iRMSD（interface RMSD）
    
    流程：
    1. 只用口袋残基做Kabsch对齐
    2. 计算口袋残基的RMSD
    
    Args:
        pred_coords: [N, 3] 或 [N, n_atoms, 3] 预测坐标
        true_coords: [N, 3] 或 [N, n_atoms, 3] 真实坐标
        pocket_mask: [N] 口袋残基掩码（bool或0/1）
        eps: 数值稳定性
        
    Returns:
        irmsd: float（Å）
    """
    # 确保是numpy
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(true_coords, torch.Tensor):
        true_coords = true_coords.detach().cpu().numpy()
    if isinstance(pocket_mask, torch.Tensor):
        pocket_mask = pocket_mask.detach().cpu().numpy()
    
    # 转换为bool
    pocket_mask = pocket_mask.astype(bool)
    
    # 如果没有口袋残基，返回NaN
    if not pocket_mask.any():
        return float('nan')
    
    # 展平为[N_atoms, 3]
    if pred_coords.ndim == 3:
        pred_coords = pred_coords.reshape(-1, 3)
        true_coords = true_coords.reshape(-1, 3)
        # 扩展pocket_mask
        n_atoms_per_res = pred_coords.shape[0] // len(pocket_mask)
        pocket_mask = np.repeat(pocket_mask, n_atoms_per_res)
    
    # 提取口袋坐标
    pred_pocket = pred_coords[pocket_mask]
    true_pocket = true_coords[pocket_mask]
    
    # Kabsch对齐
    R, t = kabsch_align(pred_pocket, true_pocket)
    pred_aligned = pred_pocket @ R.T + t
    
    # 计算RMSD
    irmsd = compute_rmsd(pred_aligned, true_pocket)
    
    return irmsd


# ============================================================================
# χ1 命中率
# ============================================================================

def wrap_angle_diff(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    计算角度差（考虑周期性）
    
    Args:
        angle1, angle2: [...] 角度（弧度）
        
    Returns:
        diff: [...] 角度差，范围[-π, π]
    """
    diff = angle1 - angle2
    # 映射到[-π, π]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return diff


def compute_chi1_accuracy(pred_angles: np.ndarray,
                         true_angles: np.ndarray,
                         angle_mask: np.ndarray,
                         threshold_deg: float = 20.0) -> float:
    """
    计算χ1扭转角命中率
    
    Args:
        pred_angles: [N] 预测的χ1角度（弧度）
        true_angles: [N] 真实的χ1角度（弧度）
        angle_mask: [N] 有效角度掩码（bool）
        threshold_deg: 命中阈值（度）
        
    Returns:
        accuracy: float 命中率（0-1）
    """
    # 确保是numpy
    if isinstance(pred_angles, torch.Tensor):
        pred_angles = pred_angles.detach().cpu().numpy()
    if isinstance(true_angles, torch.Tensor):
        true_angles = true_angles.detach().cpu().numpy()
    if isinstance(angle_mask, torch.Tensor):
        angle_mask = angle_mask.detach().cpu().numpy()
    
    # 转换为bool
    angle_mask = angle_mask.astype(bool)
    
    # 如果没有有效角度
    if not angle_mask.any():
        return float('nan')
    
    # 提取有效角度
    pred_valid = pred_angles[angle_mask]
    true_valid = true_angles[angle_mask]
    
    # 计算角度差（wrap）
    diff = wrap_angle_diff(pred_valid, true_valid)
    
    # 转换为度
    diff_deg = np.abs(diff) * 180.0 / np.pi
    
    # 计算命中率
    hits = diff_deg < threshold_deg
    accuracy = hits.mean()
    
    return float(accuracy)


# ============================================================================
# Clash检测
# ============================================================================

def compute_clash_percentage(coords: np.ndarray,
                            bond_graph: Optional[np.ndarray] = None,
                            clash_threshold: float = 2.0,
                            eps: float = 1e-8) -> float:
    """
    计算碰撞原子对的百分比
    
    Args:
        coords: [N, 3] 原子坐标
        bond_graph: [N, N] 成键关系（可选，1表示成键）
        clash_threshold: 碰撞阈值（Å）
        eps: 数值稳定性
        
    Returns:
        clash_pct: float 碰撞百分比（0-1）
    """
    # 确保是numpy
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    
    N = len(coords)
    
    # 计算成对距离
    diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 3]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + eps)  # [N, N]
    
    # 创建上三角掩码（避免重复计数和自身）
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # 排除成键原子对（1-2, 1-3邻接）
    if bond_graph is not None:
        # 排除直接成键（1-2）
        bonded_mask = bond_graph.astype(bool)
        # 排除1-3邻接（成键的成键）
        bonded_13 = (bond_graph @ bond_graph) > 0
        exclude_mask = bonded_mask | bonded_13
        triu_mask = triu_mask & (~exclude_mask)
    
    # 统计碰撞
    valid_pairs = triu_mask.sum()
    if valid_pairs == 0:
        return 0.0
    
    clash_pairs = (dist[triu_mask] < clash_threshold).sum()
    clash_pct = float(clash_pairs) / float(valid_pairs)
    
    return clash_pct


# ============================================================================
# FAPE（Frame Aligned Point Error）
# ============================================================================

def compute_fape(pred_coords: np.ndarray,
                true_coords: np.ndarray,
                pred_frames: Tuple[np.ndarray, np.ndarray],
                true_frames: Tuple[np.ndarray, np.ndarray],
                w_res: Optional[np.ndarray] = None,
                clamp_distance: float = 10.0,
                eps: float = 1e-8) -> float:
    """
    计算FAPE（Frame Aligned Point Error）
    
    流程：
    1. 对每个残基i，将坐标变换到其局部帧
    2. 计算局部坐标的误差
    3. 用口袋权重加权
    
    Args:
        pred_coords: [N, 3] 或 [N, n_atoms, 3] 预测坐标
        true_coords: [N, 3] 或 [N, n_atoms, 3] 真实坐标
        pred_frames: (R[N,3,3], t[N,3]) 预测帧
        true_frames: (R[N,3,3], t[N,3]) 真实帧
        w_res: [N] 残基权重（可选）
        clamp_distance: 裁剪距离（Å）
        eps: 数值稳定性
        
    Returns:
        fape: float（Å）
    """
    # 确保是numpy
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(true_coords, torch.Tensor):
        true_coords = true_coords.detach().cpu().numpy()
    
    pred_R, pred_t = pred_frames
    true_R, true_t = true_frames
    
    if isinstance(pred_R, torch.Tensor):
        pred_R = pred_R.detach().cpu().numpy()
        pred_t = pred_t.detach().cpu().numpy()
        true_R = true_R.detach().cpu().numpy()
        true_t = true_t.detach().cpu().numpy()
    
    # 展平为[N_atoms, 3]
    if pred_coords.ndim == 3:
        batch_shape = pred_coords.shape[:2]
        pred_coords_flat = pred_coords.reshape(-1, 3)
        true_coords_flat = true_coords.reshape(-1, 3)
        n_atoms_per_res = pred_coords.shape[1]
    else:
        pred_coords_flat = pred_coords
        true_coords_flat = true_coords
        n_atoms_per_res = 1
    
    N_res = len(pred_R)
    
    # 对每个残基计算局部误差
    errors = []
    weights = []
    
    for i in range(N_res):
        # 获取该残基对应的原子
        atom_start = i * n_atoms_per_res
        atom_end = atom_start + n_atoms_per_res
        
        pred_atoms = pred_coords_flat[atom_start:atom_end]
        true_atoms = true_coords_flat[atom_start:atom_end]
        
        # 变换到局部帧
        # 预测：x_local = R_true^T @ (x_pred - t_true)
        pred_local = (pred_atoms - true_t[i]) @ true_R[i]
        
        # 真实：x_local = R_true^T @ (x_true - t_true)
        true_local = (true_atoms - true_t[i]) @ true_R[i]
        
        # 计算误差
        diff = np.sqrt(np.sum((pred_local - true_local) ** 2, axis=-1) + eps)
        
        # Clamp
        diff = np.minimum(diff, clamp_distance)
        
        errors.append(diff.mean())
        
        # 权重
        if w_res is not None:
            weights.append(w_res[i])
        else:
            weights.append(1.0)
    
    errors = np.array(errors)
    weights = np.array(weights)
    
    # 加权平均
    fape = np.average(errors, weights=weights)
    
    return float(fape)

