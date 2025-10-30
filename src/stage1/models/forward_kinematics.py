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
    标准Z-matrix原子放置
    
    从3个已知原子 p1-p2-p3 和内坐标 (r, θ, φ) 放置第4个原子 p4
    使得：
    - |p3-p4| = r (键长)
    - ∠(p2-p3-p4) = θ (键角)
    - 二面角(p1-p2-p3-p4) = φ (扭转角)
    
    Args:
        p1, p2, p3: [..., 3] 参考原子
        bond_length: r
        bond_angle: θ (弧度)
        torsion: φ (弧度，可以是张量)
        
    Returns:
        p4: [..., 3]
    """
    # 1. 向量
    bc = p3 - p2  # [..., 3]
    bc_norm = torch.norm(bc, dim=-1, keepdim=True) + eps
    bc = bc / bc_norm
    
    # 2. 平面法向量
    n = torch.cross(p2 - p1, bc, dim=-1)
    n_norm = torch.norm(n, dim=-1, keepdim=True) + eps
    n = n / n_norm
    
    # 3. 第三个基向量
    m = torch.cross(n, bc, dim=-1)
    
    # 4. 局部坐标（p3为原点）
    # 新原子相对于p3的位置
    cos_theta = torch.cos(torch.tensor(bond_angle)) if isinstance(bond_angle, float) else torch.cos(bond_angle)
    sin_theta = torch.sin(torch.tensor(bond_angle)) if isinstance(bond_angle, float) else torch.sin(bond_angle)
    
    # 处理torsion
    if isinstance(torsion, (int, float)):
        cos_phi = math.cos(torsion)
        sin_phi = math.sin(torsion)
    else:
        cos_phi = torch.cos(torsion)
        sin_phi = torch.sin(torsion)
        # 添加维度匹配
        if cos_phi.dim() < bc.dim():
            cos_phi = cos_phi.unsqueeze(-1)
            sin_phi = sin_phi.unsqueeze(-1)
    
    # d = r * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    # 在局部坐标系 (m, n, -bc) 中
    d = bond_length * (
        sin_theta * cos_phi * m + 
        sin_theta * sin_phi * n - 
        cos_theta * bc
    )
    
    p4 = p3 + d
    
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
        # 标准蛋白质Z-matrix构建顺序
        for i in range(1, N):
            phi_i = phi[:, i]
            psi_i = psi[:, i]
            omega_i = omega[:, i]
            
            # 1. N[i]: 从C[i-1]延伸
            # 参考原子: (CA[i-1], C[i-1]) 
            # 二面角omega定义: CA[i-1]-C[i-1]-N[i]-CA[i]
            if i == 1:
                # 第一次特殊处理
                N_coords[:, i] = C_coords[:, i-1] + torch.tensor([self.C_N_length, 0, 0], device=device)
            else:
                N_coords[:, i] = place_atom_nerf(
                    CA_coords[:, i-1],  # p1
                    C_coords[:, i-1],   # p2（连接点）
                    N_coords[:, i-1],   # p3（用于定义二面角）
                    self.C_N_length,
                    self.CA_C_N_angle,
                    omega[:, i-1]  # 注意：omega[i-1]定义的是C[i-1]-N[i]键
                )
            
            # 2. CA[i]: 从N[i]延伸
            # 二面角phi定义: C[i-1]-N[i]-CA[i]-C[i]
            CA_coords[:, i] = place_atom_nerf(
                C_coords[:, i-1],   # p1
                N_coords[:, i],     # p2（连接点）
                C_coords[:, i-1] if i == 1 else CA_coords[:, i-1],  # p3（定义平面）
                self.N_CA_length,
                self.C_N_CA_angle,
                phi_i
            )
            
            # 3. C[i]: 从CA[i]延伸  
            # 二面角psi定义: N[i]-CA[i]-C[i]-N[i+1]
            C_coords[:, i] = place_atom_nerf(
                N_coords[:, i],     # p1
                CA_coords[:, i],    # p2（连接点）
                N_coords[:, i],     # p3（定义平面）
                self.CA_C_length,
                self.N_CA_C_angle,
                psi_i
            )
            
            # 4. O[i]: 从C[i]延伸（固定位置）
            O_coords[:, i] = place_atom_nerf(
                N_coords[:, i],      # p1
                CA_coords[:, i],     # p2
                C_coords[:, i],      # p3（连接点）
                self.C_O_length,
                self.CA_C_O_angle,
                torch.tensor(0.0, device=device)  # 固定在平面上
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

