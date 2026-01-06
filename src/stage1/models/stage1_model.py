"""
Stage-1 完整模型

架构：
ESM-2(冻结) → Adapter → IPA → LigandCond → ChiHead

Author: BINDRAE Team
Date: 2025-10-28
"""

import sys
import os
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn

# FlashIPA路径 (项目内 vendor 目录)
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
flash_ipa_path = str(_project_root / 'vendor' / 'flash_ipa' / 'src')
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.rigid import Rigid, Rotation

from .adapter import ESMAdapter
from .ipa import FlashIPAModule, FlashIPAModuleConfig
from .ligand_condition import LigandConditioner, LigandConditionerConfig
from .torsion_head import TorsionHead
from .fk_openfold import OpenFoldFK, create_openfold_fk
from ..modules.edge_embed import EdgeEmbedderAdapter, ProjectEdgeConfig
from ..data.residue_constants import restype_order


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class Stage1ModelConfig:
    """Stage-1模型配置"""
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
    
    # Chi Head
    torsion_hidden: int = 128
    chi_angles: int = 4
    
    # 通用
    dropout: float = 0.1


# ============================================================================
# Stage-1 模型
# ============================================================================

class Stage1Model(nn.Module):
    """
    Stage-1 完整模型
    
    流程：
    ESM → Adapter → EdgeEmbed → IPA → LigandCond → ChiHead
    """
    
    def __init__(self, config: Stage1ModelConfig):
        """
        Args:
            config: Stage-1模型配置
        """
        super().__init__()
        
        self.config = config
        
        # 1. ESM Adapter
        self.esm_adapter = ESMAdapter(
            esm_dim=config.esm_dim,
            output_dim=config.c_s,
            dropout=config.dropout
        )
        
        # 2. EdgeEmbedder
        edge_config = ProjectEdgeConfig(
            c_s=config.c_s,
            c_p=config.c_p,
            z_factor_rank=config.z_factor_rank,
            num_rbf=config.num_rbf,
        )
        self.edge_embedder = EdgeEmbedderAdapter(edge_config)
        
        # 3. FlashIPA
        ipa_config = FlashIPAModuleConfig(
            c_s=config.c_s,
            c_z=config.c_p,
            c_hidden=config.c_hidden,
            no_heads=config.no_heads,
            depth=config.depth,
            no_qk_points=config.no_qk_points,
            no_v_points=config.no_v_points,
            z_factor_rank=config.z_factor_rank,
            dropout=config.dropout,
        )
        self.ipa_module = FlashIPAModule(ipa_config)
        
        # 4. LigandConditioner
        ligand_config = LigandConditionerConfig(
            c_s=config.c_s,
            d_lig=config.d_lig,
            num_heads=config.num_heads_cross,
            dropout=config.dropout,
            warmup_steps=config.warmup_steps,
        )
        self.ligand_conditioner = LigandConditioner(ligand_config)
        
        # 5. Chi Head (only chi1-4)
        self.chi_head = TorsionHead(
            c_s=config.c_s,
            c_hidden=config.torsion_hidden,
            dropout=config.dropout,
            n_angles=config.chi_angles
        )
        
        # 6. FK模块（扭转角→全原子坐标）
        self.fk_module = create_openfold_fk()
        
        print(f"✓ Stage1Model 初始化完成")
        print(f"  - 总参数量: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  - 包含FK模块（OpenFold式）")
    
    def _sequence_to_aatype(self, sequences: List[str], max_len: int, device) -> torch.Tensor:
        """
        将氨基酸序列转换为aatype索引
        
        Args:
            sequences: List[氨基酸序列（单字母）]
            max_len: 最大长度（用于padding）
            device: 设备
            
        Returns:
            aatype: [B, N] 残基类型索引(0-19, 20=UNK)
        """
        B = len(sequences)
        aatype = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if j >= max_len:
                    break
                aatype[i, j] = restype_order.get(aa, 20)  # 20=UNK
        
        return aatype
    
    def forward(self,
                batch: 'Stage1Batch',
                current_step: int = 0) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: Stage1Batch数据
            current_step: 当前训练步数（用于warmup）
            
        Returns:
            {
                'pred_chi': [B, N, 4, 2] 预测的chi(sin, cos)
                's_final': [B, N, c_s] 最终节点表示
                'rigids_final': Rigid对象
            }
        """
        B, N = batch.esm.shape[:2]
        device = batch.esm.device
        
        # 1. ESM Adapter
        s = self.esm_adapter(batch.esm)  # [B, N, 384]
        
        # 2. 创建初始Rigid帧（从apo的N, CA, C）
        rigids = self._build_rigids_from_backbone(
            batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
        )
        
        # 3. LigandConditioner（在IPA前注入，符合理论）
        # 理论：配体信息应该影响IPA的几何建模（docs/理论 第15-26行）
        s_with_ligand = self.ligand_conditioner(
            s,  # 在Adapter后立即注入
            batch.lig_points,
            batch.lig_types,
            batch.node_mask,
            batch.lig_mask,
            current_step=current_step
        )
        
        # 4. EdgeEmbedder
        edge_outputs = self.edge_embedder(s_with_ligand, batch.Ca_apo, batch.node_mask)
        z_f1 = edge_outputs['z_f1']
        z_f2 = edge_outputs['z_f2']
        
        # 5. FlashIPA（几何分支，已包含配体信息，并在每层后进行配体条件化）
        s_geo, rigids_updated = self.ipa_module(
            s_with_ligand,
            rigids,
            z_f1,
            z_f2,
            batch.node_mask,
            ligand_conditioner=self.ligand_conditioner,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            protein_mask=batch.node_mask,
            ligand_mask=batch.lig_mask,
            current_step=current_step,
        )
        
        # 6. TorsionHead（使用IPA输出）
        pred_chi = self.chi_head(s_geo)  # [B, N, 4, 2]
        
        # 7. FK重建全原子坐标
        # 将序列转换为aatype索引
        aatype = self._sequence_to_aatype(batch.sequences, N, device)

        # 组装完整torsions: 使用apo的phi/psi/omega，预测chi
        torsion_apo = batch.torsion_apo  # [B, N, 7] (phi/psi/omega/chi1-4)
        torsion_apo = torsion_apo.to(device)
        phi_psi_omega = torsion_apo[:, :, :3]
        phi_psi_omega_sincos = torch.stack(
            [torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1
        )  # [B, N, 3, 2]
        torsions_sincos = torch.cat([phi_psi_omega_sincos, pred_chi], dim=2)  # [B,N,7,2]

        atom14_result = self.fk_module(torsions_sincos, rigids_updated, aatype)
        
        return {
            'pred_chi': pred_chi,
            's_final': s_geo,  # IPA输出（已含配体信息）
            'rigids_final': rigids_updated,
            'atom14_pos': atom14_result['atom14_pos'],      # [B, N, 14, 3]
            'atom14_mask': atom14_result['atom14_mask'],    # [B, N, 14]
        }

    def _build_rigids_from_backbone(self,
                                    N: torch.Tensor,
                                    Ca: torch.Tensor,
                                    C: torch.Tensor,
                                    mask: torch.Tensor,
                                    eps: float = 1e-6) -> Rigid:
        """Build per-residue backbone frames from N/CA/C.
        
        添加了 NaN 保护：当向量接近零时使用单位向量。
        """
        # e1: CA -> C
        e1 = C - Ca
        e1_norm = torch.norm(e1, dim=-1, keepdim=True)
        # 保护：如果 norm 太小，使用默认方向 [1,0,0]
        e1_safe = torch.where(e1_norm > eps, e1, torch.tensor([1.0, 0.0, 0.0], device=e1.device))
        e1_norm_safe = torch.clamp(e1_norm, min=eps)
        e1 = e1_safe / e1_norm_safe

        # u: CA -> N
        u = N - Ca
        # e2: u orthogonalized to e1
        proj = (u * e1).sum(dim=-1, keepdim=True) * e1
        e2 = u - proj
        e2_norm = torch.norm(e2, dim=-1, keepdim=True)
        # 保护：如果 norm 太小，使用默认方向 [0,1,0]
        e2_safe = torch.where(e2_norm > eps, e2, torch.tensor([0.0, 1.0, 0.0], device=e2.device))
        e2_norm_safe = torch.clamp(e2_norm, min=eps)
        e2 = e2_safe / e2_norm_safe

        # e3: right-handed
        e3 = torch.cross(e1, e2, dim=-1)
        # 保护：确保 e3 是单位向量
        e3_norm = torch.norm(e3, dim=-1, keepdim=True)
        e3 = e3 / torch.clamp(e3_norm, min=eps)

        R = torch.stack([e1, e2, e3], dim=-1)  # [B, N, 3, 3]
        t = Ca

        # For padded residues, set identity rotation and zero translation
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(3, device=R.device).view(1, 1, 3, 3)
            R = torch.where(mask_expanded, R, eye)
            t = torch.where(mask.unsqueeze(-1), t, torch.zeros_like(t))

        # 最终检查：替换任何残留的 NaN
        R = torch.where(torch.isnan(R), torch.eye(3, device=R.device).view(1, 1, 3, 3).expand_as(R), R)
        t = torch.where(torch.isnan(t), torch.zeros_like(t), t)

        rotation = Rotation(rot_mats=R)
        return Rigid(rots=rotation, trans=t)


def create_stage1_model(config: Optional[Stage1ModelConfig] = None) -> Stage1Model:
    """
    创建Stage-1模型
    
    Args:
        config: 模型配置（可选，使用默认配置）
        
    Returns:
        Stage1Model实例
    """
    if config is None:
        config = Stage1ModelConfig()
    
    return Stage1Model(config)
