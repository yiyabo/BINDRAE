"""
Stage-1 完整模型

架构：
ESM-2(冻结) → Adapter → IPA → LigandCond → TorsionHead

Author: BINDRAE Team
Date: 2025-10-28
"""

import sys
import os
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

# FlashIPA路径
flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.rigid import Rigid, Rotation

from .adapter import ESMAdapter
from .ipa import FlashIPAModule, FlashIPAModuleConfig
from .ligand_condition import LigandConditioner, LigandConditionerConfig
from .torsion_head import TorsionHead
from .fk_openfold import OpenFoldFK, create_openfold_fk
from ..modules.edge_embed import EdgeEmbedderAdapter, ProjectEdgeConfig


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
    
    # TorsionHead
    torsion_hidden: int = 128
    
    # 通用
    dropout: float = 0.1


# ============================================================================
# Stage-1 模型
# ============================================================================

class Stage1Model(nn.Module):
    """
    Stage-1 完整模型
    
    流程：
    ESM → Adapter → EdgeEmbed → IPA → LigandCond → TorsionHead
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
        
        # 5. TorsionHead
        self.torsion_head = TorsionHead(
            c_s=config.c_s,
            c_hidden=config.torsion_hidden,
            dropout=config.dropout
        )
        
        # 6. FK模块（扭转角→全原子坐标）
        self.fk_module = create_openfold_fk()
        
        print(f"✓ Stage1Model 初始化完成")
        print(f"  - 总参数量: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  - 包含FK模块（OpenFold式）")
    
    def forward(self,
                batch: 'IPABatch',
                current_step: int = 0) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: IPABatch数据
            current_step: 当前训练步数（用于warmup）
            
        Returns:
            {
                'pred_torsions': [B, N, 7, 2] 预测的扭转角(sin,cos)
                's_final': [B, N, c_s] 最终节点表示
                'rigids_final': Rigid对象
            }
        """
        B, N = batch.esm.shape[:2]
        device = batch.esm.device
        
        # 1. ESM Adapter
        s = self.esm_adapter(batch.esm)  # [B, N, 384]
        
        # 2. 创建初始Rigid帧（从N, Ca, C）
        # 简化：用单位旋转 + Ca作为平移
        rot_identity = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        rotation = Rotation(rot_mats=rot_identity)
        rigids = Rigid(rots=rotation, trans=batch.Ca)
        
        # 3. EdgeEmbedder
        edge_outputs = self.edge_embedder(s, batch.Ca, batch.node_mask)
        z_f1 = edge_outputs['z_f1']
        z_f2 = edge_outputs['z_f2']
        
        # 4. FlashIPA
        s_geo, rigids_updated = self.ipa_module(s, rigids, z_f1, z_f2, batch.node_mask)
        
        # 5. LigandConditioner
        s_cond = self.ligand_conditioner(
            s_geo,
            batch.lig_points,
            batch.lig_types,
            batch.node_mask,
            batch.lig_mask,
            current_step=current_step
        )
        
        # 6. TorsionHead
        pred_torsions = self.torsion_head(s_cond)  # [B, N, 7, 2]
        
        # 7. FK重建全原子坐标
        # 创建残基类型（暂时全部用ALA=0）
        # TODO: 从数据集获取真实残基类型
        aatype = torch.zeros(B, N, dtype=torch.long, device=device)
        
        atom14_result = self.fk_module(pred_torsions, rigids_updated, aatype)
        
        return {
            'pred_torsions': pred_torsions,
            's_final': s_cond,
            'rigids_final': rigids_updated,
            'atom14_pos': atom14_result['atom14_pos'],      # [B, N, 14, 3]
            'atom14_mask': atom14_result['atom14_mask'],    # [B, N, 14]
        }


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

