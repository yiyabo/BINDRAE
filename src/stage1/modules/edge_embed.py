"""
FlashIPA EdgeEmbedder 适配层

直接使用 flash_ipa.edge_embedder.EdgeEmbedder，
提供适配接口以匹配项目需求。

Author: BINDRAE Team
Date: 2025-10-28
"""

import sys
import os
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn

# 添加FlashIPA路径
flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.edge_embedder import EdgeEmbedder, EdgeEmbedderConfig


# ============================================================================
# 配置适配
# ============================================================================

@dataclass
class ProjectEdgeConfig:
    """
    项目的边嵌入配置（适配FlashIPA）
    
    Args:
        c_s: 节点单表示维度 (从Adapter输出，默认384)
        c_p: 节点对表示维度 (边嵌入维度，默认128)
        z_factor_rank: 因子化秩 (默认16)
        num_rbf: RBF核数量 (默认16)
        mode: 边嵌入模式 (默认'flash_1d_bias')
        feat_dim: 特征维度 (默认64)
        relpos_k: 相对位置编码维度 (默认64)
    """
    c_s: int = 384
    c_p: int = 128
    z_factor_rank: int = 16
    num_rbf: int = 16
    mode: str = 'flash_1d_bias'
    feat_dim: int = 64
    relpos_k: int = 64
    
    def to_flashipa_config(self) -> EdgeEmbedderConfig:
        """转换为FlashIPA的配置"""
        return EdgeEmbedderConfig(
            c_s=self.c_s,
            c_p=self.c_p,
            z_factor_rank=self.z_factor_rank,
            num_rbf=self.num_rbf,
            mode=self.mode,
            feat_dim=self.feat_dim,
            relpos_k=self.relpos_k,
            use_rbf=True,  # 启用RBF
            self_condition=True,  # 启用自条件
        )


# ============================================================================
# EdgeEmbedder 适配器
# ============================================================================

class EdgeEmbedderAdapter(nn.Module):
    """
    FlashIPA EdgeEmbedder 适配器
    
    功能：
    1. 封装FlashIPA原生EdgeEmbedder
    2. 提供简化的接口：forward(S, t, node_mask)
    3. 自动处理侧链平移（trans_sc）
    
    注意：
    - FlashIPA的forward需要6个参数，我们简化为3个
    - trans_sc（侧链平移）暂时用主链平移代替
    - 输出适配为字典格式
    """
    
    def __init__(self, config: ProjectEdgeConfig):
        """
        Args:
            config: 项目边嵌入配置
        """
        super().__init__()
        
        self.config = config
        
        # 创建FlashIPA EdgeEmbedder
        flashipa_config = config.to_flashipa_config()
        self.edge_embedder = EdgeEmbedder(flashipa_config)
        
        print(f"✓ EdgeEmbedder初始化:")
        print(f"  - c_s: {config.c_s}")
        print(f"  - c_p: {config.c_p}")
        print(f"  - z_factor_rank: {config.z_factor_rank}")
        print(f"  - num_rbf: {config.num_rbf}")
        print(f"  - mode: {config.mode}")
    
    def forward(self,
                node_embed: torch.Tensor,
                translations: torch.Tensor,
                node_mask: torch.Tensor,
                trans_sc: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播（简化接口）
        
        Args:
            node_embed: (B, N, c_s) 节点嵌入
            translations: (B, N, 3) 主链坐标（Cα位置）
            node_mask: (B, N) 节点掩码
            trans_sc: (B, N, 3) 侧链坐标（可选，默认用主链代替）
            
        Returns:
            {
                'z_f1': (B, N, z_factor_rank, c_p) 边因子1
                'z_f2': (B, N, z_factor_rank, c_p) 边因子2
                'edge_mask': (B, N, N) 边掩码
                'raw_output': tuple - FlashIPA原始输出
            }
        """
        # 如果没有侧链坐标，用主链代替
        if trans_sc is None:
            trans_sc = translations
        
        # 调用FlashIPA原生forward
        # forward(node_embed, translations, trans_sc, node_mask, edge_embed, edge_mask)
        # edge_embed和edge_mask设为None（不预先提供边信息）
        outputs = self.edge_embedder(
            node_embed=node_embed,
            translations=translations,
            trans_sc=trans_sc,
            node_mask=node_mask,
            edge_embed=None,
            edge_mask=None
        )
        
        # FlashIPA返回tuple: (None, z_f1, z_f2, None)
        # z_f1, z_f2 的形状: [B, N, z_factor_rank, c_p]
        none1, z_f1, z_f2, none2 = outputs
        
        # 生成边掩码
        B, N = node_mask.shape
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # (B, N, N)
        
        return {
            'z_f1': z_f1,  # [B, N, z_factor_rank, c_p]
            'z_f2': z_f2,  # [B, N, z_factor_rank, c_p]
            'edge_mask': edge_mask,  # [B, N, N]
            'raw_output': outputs
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_edge_embedder(c_s: int = 384,
                        c_p: int = 128,
                        z_rank: int = 16,
                        num_rbf: int = 16,
                        mode: str = 'flash_1d_bias',
                        **kwargs) -> EdgeEmbedderAdapter:
    """
    创建边嵌入器（使用FlashIPA）
    
    Args:
        c_s: 节点单表示维度
        c_p: 节点对表示维度
        z_rank: 因子化秩
        num_rbf: RBF核数
        mode: 边嵌入模式
        **kwargs: 其他配置参数
        
    Returns:
        EdgeEmbedderAdapter 实例
    
    Example:
        >>> embedder = create_edge_embedder(c_s=384, c_p=128, z_rank=16)
        >>> outputs = embedder(node_embed, translations, node_mask)
        >>> z_f1, z_f2 = outputs['z_f1'], outputs['z_f2']
    """
    config = ProjectEdgeConfig(
        c_s=c_s,
        c_p=c_p,
        z_factor_rank=z_rank,
        num_rbf=num_rbf,
        mode=mode,
        **kwargs
    )
    return EdgeEmbedderAdapter(config)


# ============================================================================
# 单元测试
# ============================================================================

def _test_flashipa_adapter():
    """测试FlashIPA适配器"""
    print("=" * 80)
    print("FlashIPA EdgeEmbedder 适配器测试")
    print("=" * 80)
    
    # 配置
    B, N = 2, 10
    c_s = 384
    
    # 创建测试数据
    node_embed = torch.randn(B, N, c_s)
    translations = torch.randn(B, N, 3)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask[:, -2:] = False
    
    print(f"\n输入:")
    print(f"  node_embed: {node_embed.shape}")
    print(f"  translations: {translations.shape}")
    print(f"  node_mask: {node_mask.shape}")
    
    # 创建适配器
    print(f"\n创建EdgeEmbedder适配器...")
    embedder = create_edge_embedder(c_s=c_s, c_p=128, z_rank=16, num_rbf=16)
    
    # 前向传播（无梯度）
    print(f"\n前向传播（no_grad）...")
    with torch.no_grad():
        outputs = embedder(node_embed, translations, node_mask)
    
    print(f"\n输出:")
    print(f"  z_f1: {outputs['z_f1'].shape if outputs['z_f1'] is not None else None}")
    print(f"  z_f2: {outputs['z_f2'].shape if outputs['z_f2'] is not None else None}")
    print(f"  edge_mask: {outputs['edge_mask'].shape if outputs['edge_mask'] is not None else None}")
    
    # 数值验证
    print(f"\n数值验证:")
    if outputs['z_f1'] is not None:
        print(f"  z_f1 范围: [{outputs['z_f1'].min():.3f}, {outputs['z_f1'].max():.3f}]")
    if outputs['z_f2'] is not None:
        print(f"  z_f2 范围: [{outputs['z_f2'].min():.3f}, {outputs['z_f2'].max():.3f}]")
    
    # 梯度测试（需要重新前向传播）
    print(f"\n梯度测试...")
    node_embed_grad = torch.randn(B, N, c_s, requires_grad=True)
    outputs_grad = embedder(node_embed_grad, translations, node_mask)
    if outputs_grad['z_f1'] is not None and outputs_grad['z_f2'] is not None:
        loss = outputs_grad['z_f1'].sum() + outputs_grad['z_f2'].sum()
        loss.backward()
        print(f"  ✓ 反向传播成功")
        print(f"  ✓ 梯度形状: {node_embed_grad.grad.shape}")
    
    print(f"\n✅ FlashIPA适配器测试通过！")
    print("=" * 80)


if __name__ == '__main__':
    _test_flashipa_adapter()

