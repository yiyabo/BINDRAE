"""
Forward Kinematics (FK) 模块

功能：
从扭转角重建全原子3D坐标（可微分）

理论依据：
docs/理论/理论与参考.md 第49-90行
- NeRF式原子放置
- Z-matrix迭代构建
- 端到端可导

Author: BINDRAE Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


# ============================================================================
# NeRF式原子放置（核心算法）
# ============================================================================

def place_atom_nerf(p1: torch.Tensor,
                   p2: torch.Tensor,
                   p3: torch.Tensor,
                   bond_length: float,
                   bond_angle: float,
                   torsion: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """
    NeRF式原子放置（从3个原子+内坐标放置第4个原子）
    
    理论：docs/理论/理论与参考.md 第69-84行
    
    给定：
    - 最近3个原子 p1, p2, p3
    - 新键的参数 (r, θ, φ)
    
    计算：
    - 新原子p4的3D坐标
    
    Args:
        p1: [..., 3] 第1个原子（最远）
        p2: [..., 3] 第2个原子
        p3: [..., 3] 第3个原子（最近，作为连接点）
        bond_length: 标准键长 r (Å)
        bond_angle: 标准键角 θ (弧度)
        torsion: [...] 扭转角 φ (弧度)
        eps: 数值稳定性
        
    Returns:
        p4: [..., 3] 新原子坐标
        
    数学推导：
    1. 建局部正交基 (e1, e2, e3)
    2. 球坐标 → 笛卡尔坐标
    3. 局部坐标 → 全局坐标
    """
    # 1. 构建局部正交基（以p3为原点）
    # e1: p2 → p3 方向
    v21 = p3 - p2  # [..., 3]
    e1 = v21 / (torch.norm(v21, dim=-1, keepdim=True) + eps)
    
    # e2: 垂直于e1和(p1-p3)平面
    v31 = p3 - p1
    v31_proj = v31 - torch.sum(v31 * e1, dim=-1, keepdim=True) * e1  # Gram-Schmidt
    e2 = v31_proj / (torch.norm(v31_proj, dim=-1, keepdim=True) + eps)
    
    # e3: e1 × e2
    e3 = torch.cross(e1, e2, dim=-1)
    
    # 2. 新原子在局部坐标系下的位置
    # 使用球坐标 (r, θ, φ)
    cos_theta = math.cos(math.pi - bond_angle)  # 键角的补角
    sin_theta = math.sin(math.pi - bond_angle)
    
    cos_phi = torch.cos(torsion)  # [..., 1] 如果是标量会broadcast
    sin_phi = torch.sin(torsion)
    
    # 笛卡尔坐标（局部系）
    # x = r * sin(θ) * cos(φ)
    # y = r * sin(θ) * sin(φ)  
    # z = r * cos(θ)
    x_local = bond_length * sin_theta * cos_phi
    y_local = bond_length * sin_theta * sin_phi
    z_local = bond_length * cos_theta
    
    # 3. 转换到全局坐标系
    # p4 = p3 + x*e1 + y*e2 + z*e3
    if torsion.dim() == 0:  # 标量torsion
        p4 = p3 + x_local * e1 + y_local * e2 + z_local * e3
    else:  # 张量torsion [..., ]
        x_local = x_local.unsqueeze(-1)  # [..., 1]
        y_local = y_local.unsqueeze(-1)
        z_local = z_local if z_local.dim() > e1.dim() - 1 else torch.tensor(z_local).expand_as(e1[..., :1])
        
        p4 = p3 + x_local * e1 + y_local * e2 + z_local * e3
    
    return p4


# ============================================================================
# FK模板加载
# ============================================================================

class FKTemplate:
    """FK模板（键长/键角/拓扑）"""
    
    def __init__(self, template_file: str):
        """
        Args:
            template_file: fk_template.pkl路径
        """
        with open(template_file, 'rb') as f:
            template = pickle.load(f)
        
        self.bond_lengths = template['bond_lengths']
        self.bond_angles = template['bond_angles']
        self.residue_topology = template['residue_topology']
        self.metadata = template.get('metadata', {})
        
        print(f"✓ FK模板加载:")
        print(f"  - 键长数: {len(self.bond_lengths)}")
        print(f"  - 键角数: {len(self.bond_angles)}")
        print(f"  - 残基类型: {len(self.residue_topology)}")
    
    def get_bond_length(self, atom1: str, atom2: str, default: float = 1.5) -> float:
        """获取键长"""
        key1 = f"{atom1}-{atom2}"
        key2 = f"{atom2}-{atom1}"
        return self.bond_lengths.get(key1, self.bond_lengths.get(key2, default))
    
    def get_bond_angle(self, atom1: str, atom2: str, atom3: str, default: float = 109.5) -> float:
        """获取键角（返回弧度）"""
        key = f"{atom1}-{atom2}-{atom3}"
        angle_deg = self.bond_angles.get(key, default)
        return angle_deg * math.pi / 180.0
    
    def get_sidechain_topology(self, residue_name: str) -> List:
        """获取侧链拓扑"""
        return self.residue_topology.get(residue_name, [])


# ============================================================================
# 主链重建
# ============================================================================

class BackboneBuilder(nn.Module):
    """
    主链重建器
    
    从phi/psi/omega扭转角重建N-CA-C-O主链
    """
    
    def __init__(self, fk_template: FKTemplate):
        """
        Args:
            fk_template: FK模板
        """
        super().__init__()
        
        self.template = fk_template
        
        # 注册标准键长（作为buffer，可在GPU上使用）
        self.register_buffer('N_CA_length', torch.tensor(fk_template.get_bond_length('N', 'CA')))
        self.register_buffer('CA_C_length', torch.tensor(fk_template.get_bond_length('CA', 'C')))
        self.register_buffer('C_O_length', torch.tensor(fk_template.get_bond_length('C', 'O')))
        self.register_buffer('C_N_length', torch.tensor(fk_template.get_bond_length('C', 'N')))
        
        # 注册标准键角
        self.register_buffer('N_CA_C_angle', torch.tensor(fk_template.get_bond_angle('N', 'CA', 'C')))
        self.register_buffer('CA_C_N_angle', torch.tensor(fk_template.get_bond_angle('CA', 'C', 'N')))
        self.register_buffer('CA_C_O_angle', torch.tensor(fk_template.get_bond_angle('CA', 'C', 'O')))
        self.register_buffer('C_N_CA_angle', torch.tensor(fk_template.get_bond_angle('C', 'N', 'CA')))
    
    def forward(self,
                phi: torch.Tensor,
                psi: torch.Tensor,
                omega: torch.Tensor,
                initial_N: Optional[torch.Tensor] = None,
                initial_CA: Optional[torch.Tensor] = None,
                initial_C: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        重建主链原子
        
        Args:
            phi: [B, N] phi扭转角
            psi: [B, N] psi扭转角
            omega: [B, N] omega扭转角
            initial_N/CA/C: [B, 3] 初始3个原子（第1个残基）
            
        Returns:
            {
                'N': [B, N, 3],
                'CA': [B, N, 3],
                'C': [B, N, 3],
                'O': [B, N, 3]
            }
        """
        B, N = phi.shape
        device = phi.device
        
        # 初始化坐标张量
        N_coords = torch.zeros(B, N, 3, device=device)
        CA_coords = torch.zeros(B, N, 3, device=device)
        C_coords = torch.zeros(B, N, 3, device=device)
        O_coords = torch.zeros(B, N, 3, device=device)
        
        # 第1个残基：使用初始坐标或默认值
        if initial_N is not None:
            N_coords[:, 0] = initial_N
            CA_coords[:, 0] = initial_CA
            C_coords[:, 0] = initial_C
        else:
            # 默认起始坐标（沿x轴）
            N_coords[:, 0] = torch.tensor([0.0, 0.0, 0.0], device=device)
            CA_coords[:, 0] = torch.tensor([self.N_CA_length, 0.0, 0.0], device=device)
            C_coords[:, 0] = CA_coords[:, 0] + torch.tensor([self.CA_C_length, 0.0, 0.0], device=device)
        
        # O: 从CA, C, psi放置
        O_coords[:, 0] = place_atom_nerf(
            CA_coords[:, 0],
            CA_coords[:, 0],  # 占位（O不需要前面第3个原子）
            C_coords[:, 0],
            self.C_O_length,
            self.CA_C_O_angle,
            psi[:, 0] + math.pi  # O在C-CA-N平面的反向
        )
        
        # 迭代构建后续残基
        for i in range(1, N):
            # 当前残基的3个扭转角
            phi_i = phi[:, i]
            psi_i = psi[:, i]
            omega_i = omega[:, i]
            
            # 从前一个残基的C构建当前残基的N（用omega）
            N_coords[:, i] = place_atom_nerf(
                CA_coords[:, i-1],
                C_coords[:, i-1],
                N_coords[:, i-1],  # 前一个N（占位，实际不影响）
                self.C_N_length,
                self.CA_C_N_angle,
                omega_i
            )
            
            # 从N构建CA（用phi）
            CA_coords[:, i] = place_atom_nerf(
                C_coords[:, i-1],
                N_coords[:, i],
                CA_coords[:, i-1],  # 占位
                self.N_CA_length,
                self.C_N_CA_angle,
                phi_i
            )
            
            # 从CA构建C（用psi）
            C_coords[:, i] = place_atom_nerf(
                N_coords[:, i],
                CA_coords[:, i],
                C_coords[:, i-1],  # 占位
                self.CA_C_length,
                self.N_CA_C_angle,
                psi_i
            )
            
            # 从C构建O
            O_coords[:, i] = place_atom_nerf(
                CA_coords[:, i],
                CA_coords[:, i],  # 占位
                C_coords[:, i],
                self.C_O_length,
                self.CA_C_O_angle,
                psi_i + math.pi
            )
        
        return {
            'N': N_coords,
            'CA': CA_coords,
            'C': C_coords,
            'O': O_coords
        }


# ============================================================================
# 侧链重建
# ============================================================================

class SidechainBuilder(nn.Module):
    """
    侧链重建器
    
    递归遍历拓扑树，从chi角度重建侧链原子
    """
    
    def __init__(self, fk_template: FKTemplate):
        """
        Args:
            fk_template: FK模板
        """
        super().__init__()
        self.template = fk_template
    
    def build_sidechain_recursive(self,
                                  topology_node: Tuple,
                                  coords_dict: Dict[str, torch.Tensor],
                                  chi_angles: torch.Tensor,
                                  chi_idx: int) -> int:
        """
        递归构建侧链原子
        
        Args:
            topology_node: (parent, current, children) 拓扑节点
            coords_dict: {atom_name: [B, 3]} 已构建的原子坐标字典
            chi_angles: [B, 4] chi1-chi4扭转角
            chi_idx: 当前使用的chi索引
            
        Returns:
            next_chi_idx: 下一个chi索引
        """
        parent_name, current_name, children = topology_node
        
        # 获取键长和键角
        bond_length = self.template.get_bond_length(parent_name, current_name)
        
        # 需要3个原子来放置新原子
        # 简化：暂时用已有原子的组合
        # TODO: 完整版需要更精确的3个原子选择
        
        # 构建当前原子（使用chi角）
        if chi_idx < chi_angles.shape[1]:
            torsion = chi_angles[:, chi_idx]
            chi_idx += 1
        else:
            # 没有chi角了，使用默认值
            torsion = torch.zeros(chi_angles.shape[0], device=chi_angles.device)
        
        # 放置原子（简化版，需要改进）
        # TODO: 完整实现
        coords_dict[current_name] = coords_dict.get(parent_name, torch.zeros(chi_angles.shape[0], 3))
        
        # 递归处理子节点
        for child in children:
            chi_idx = self.build_sidechain_recursive(child, coords_dict, chi_angles, chi_idx)
        
        return chi_idx
    
    def forward(self,
                residue_types: List[str],
                chi_angles: torch.Tensor,
                backbone_coords: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        重建所有残基的侧链
        
        Args:
            residue_types: [B, N] 残基类型列表（3字母代码）
            chi_angles: [B, N, 4] chi1-chi4扭转角
            backbone_coords: {'N': [B,N,3], 'CA': [B,N,3], 'C': [B,N,3]}
            
        Returns:
            sidechain_coords: {atom_type: [B, N, 3]} 侧链原子坐标
        """
        # TODO: 完整实现
        # 当前返回空字典（先让主链FK工作）
        return {}


# ============================================================================
# 完整FK模块
# ============================================================================

class ProteinFK(nn.Module):
    """
    完整蛋白质FK模块
    
    输入：扭转角(sin, cos)
    输出：全原子3D坐标
    """
    
    def __init__(self, template_file: str = 'data/casf2016/processed/fk_template.pkl'):
        """
        Args:
            template_file: FK模板文件路径
        """
        super().__init__()
        
        # 加载模板
        self.template = FKTemplate(template_file)
        
        # 主链重建器
        self.backbone_builder = BackboneBuilder(self.template)
        
        # 侧链重建器（暂时不用）
        # self.sidechain_builder = SidechainBuilder(self.template)
    
    def forward(self,
                torsions_sincos: torch.Tensor,
                initial_coords: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        从扭转角重建全原子坐标
        
        Args:
            torsions_sincos: [B, N, 7, 2] 扭转角的(sin, cos)
                            索引：0=phi, 1=psi, 2=omega, 3-6=chi1-4
            initial_coords: 初始坐标（可选）
                          {'N': [B, 3], 'CA': [B, 3], 'C': [B, 3]}
            
        Returns:
            coords: {
                'N': [B, N, 3],
                'CA': [B, N, 3],
                'C': [B, N, 3],
                'O': [B, N, 3],
                # 'sidechain': {...}  # TODO: 侧链
            }
        """
        B, N = torsions_sincos.shape[:2]
        
        # 提取扭转角
        phi = torch.atan2(torsions_sincos[:, :, 0, 0], torsions_sincos[:, :, 0, 1])  # [B, N]
        psi = torch.atan2(torsions_sincos[:, :, 1, 0], torsions_sincos[:, :, 1, 1])
        omega = torch.atan2(torsions_sincos[:, :, 2, 0], torsions_sincos[:, :, 2, 1])
        # chi = torsions_sincos[:, :, 3:, :]  # [B, N, 4, 2] TODO: 侧链用
        
        # 重建主链
        backbone_coords = self.backbone_builder(
            phi, psi, omega,
            initial_N=initial_coords.get('N') if initial_coords else None,
            initial_CA=initial_coords.get('CA') if initial_coords else None,
            initial_C=initial_coords.get('C') if initial_coords else None
        )
        
        # TODO: 重建侧链
        # sidechain_coords = self.sidechain_builder(...)
        
        return backbone_coords


def create_protein_fk(template_file: str = 'data/casf2016/processed/fk_template.pkl') -> ProteinFK:
    """
    创建蛋白质FK模块
    
    Args:
        template_file: FK模板路径
        
    Returns:
        ProteinFK实例
    """
    return ProteinFK(template_file)

