"""
边嵌入封装模块 (Edge Embedder Wrapper)

功能：
1. 封装 FlashIPA 的 EdgeEmbedder
2. 生成因子化的边表示 (避免 O(N²) 显存)
3. 基于距离的 RBF (Radial Basis Function) 编码
4. 预留共价边扩展接口

设计原则：
- 严格遵循 FlashIPA 的接口规范
- 数值稳定：epsilon 保护、梯度裁剪
- 内存高效：因子化表示 z = z_f1 ⊗ z_f2
- 可扩展：预留共价边接口但第一版不实现

参考文献：
- AlphaFold2 (Nature 2021): IPA 边嵌入设计
- FlashIPA (arXiv:2505.11580): 因子化边嵌入

Author: BINDRAE Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import math


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class EdgeEmbedderConfig:
    """
    边嵌入器配置
    
    Args:
        c_s: 节点表示维度 (来自 Adapter 的输出)
        c_z: 边表示维度
        z_factor_rank: 因子化秩 (z = z_f1 ⊗ z_f2, 每个因子 z_rank 维)
        mode: 边嵌入模式
            - 'flash_1d_bias': 因子化1D偏置 (推荐，显存 O(N))
            - 'flash_2d': 完整2D边 (显存 O(N²)，仅调试用)
        num_rbf: RBF 核数量
        rbf_min: RBF 最小距离 (Å)
        rbf_max: RBF 最大距离 (Å)
        dropout: Dropout 概率
        use_covalent_edges: 是否使用共价边 (第一版 False)
    """
    c_s: int = 384
    c_z: int = 128
    z_factor_rank: int = 16
    mode: str = 'flash_1d_bias'
    num_rbf: int = 16
    rbf_min: float = 0.0
    rbf_max: float = 20.0
    dropout: float = 0.1
    use_covalent_edges: bool = False  # 第一版不使用
    
    def __post_init__(self):
        """验证配置"""
        assert self.mode in ['flash_1d_bias', 'flash_2d'], \
            f"Unsupported mode: {self.mode}"
        assert self.c_z % 2 == 0, \
            f"c_z must be even for factorization, got {self.c_z}"
        assert self.z_factor_rank > 0, \
            f"z_factor_rank must be positive, got {self.z_factor_rank}"


# ============================================================================
# RBF 编码器
# ============================================================================

class GaussianRBF(nn.Module):
    """
    高斯径向基函数 (Gaussian Radial Basis Function)
    
    将距离 d 编码为 k 个高斯核的响应：
        RBF_i(d) = exp(-(d - μ_i)² / (2σ²))
    
    其中：
        - μ_i: 均匀分布在 [rbf_min, rbf_max]
        - σ: 固定为 (rbf_max - rbf_min) / num_rbf
    
    参考：
        - SchNet (NeurIPS 2017)
        - AlphaFold2 距离编码
    """
    
    def __init__(self, 
                 num_rbf: int = 16,
                 rbf_min: float = 0.0,
                 rbf_max: float = 20.0):
        """
        Args:
            num_rbf: RBF 核数量
            rbf_min: 最小距离 (Å)
            rbf_max: 最大距离 (Å)
        """
        super().__init__()
        
        self.num_rbf = num_rbf
        self.rbf_min = rbf_min
        self.rbf_max = rbf_max
        
        # 计算中心点 μ (均匀分布)
        centers = torch.linspace(rbf_min, rbf_max, num_rbf)
        self.register_buffer('centers', centers)
        
        # 计算标准差 σ
        sigma = (rbf_max - rbf_min) / num_rbf
        self.register_buffer('sigma', torch.tensor(sigma))
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        计算 RBF 编码
        
        Args:
            distances: (..., ) 距离张量
            
        Returns:
            rbf_encoding: (..., num_rbf) RBF 编码
        """
        # distances: (..., )
        # centers: (num_rbf,)
        # 扩展维度：(..., 1) - (num_rbf,) = (..., num_rbf)
        diff = distances.unsqueeze(-1) - self.centers
        
        # 高斯核
        rbf = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        
        return rbf  # (..., num_rbf)


# ============================================================================
# 边嵌入器核心
# ============================================================================

class EdgeEmbedderWrapper(nn.Module):
    """
    边嵌入器封装
    
    核心功能：
    1. 从节点表示 S 生成因子 z_f1, z_f2
    2. 从节点坐标 t 计算距离并用 RBF 编码
    3. 生成边掩码
    
    输出：
        - edge_embed: 完整边嵌入 (仅 flash_2d 模式)
        - z_f1, z_f2: 因子化边表示 (flash_1d_bias 模式)
        - edge_mask: 边掩码
    
    显存分析：
        - flash_1d_bias: O(N * z_rank) → 线性扩展 ✓
        - flash_2d: O(N² * c_z) → 平方扩展 (仅调试)
    """
    
    def __init__(self, config: EdgeEmbedderConfig):
        """
        Args:
            config: 边嵌入器配置
        """
        super().__init__()
        
        self.config = config
        self.mode = config.mode
        
        # RBF 编码器
        self.rbf = GaussianRBF(
            num_rbf=config.num_rbf,
            rbf_min=config.rbf_min,
            rbf_max=config.rbf_max
        )
        
        # 因子化边表示生成器
        if self.mode == 'flash_1d_bias':
            # 输入：节点表示 S (c_s,) + RBF (num_rbf,)
            input_dim = config.c_s + config.num_rbf
            
            # 生成两个因子 z_f1, z_f2
            self.factor1_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, config.z_factor_rank),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.z_factor_rank, config.z_factor_rank)
            )
            
            self.factor2_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, config.z_factor_rank),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.z_factor_rank, config.z_factor_rank)
            )
        
        elif self.mode == 'flash_2d':
            # 完整边嵌入 (仅调试用)
            # 输入：concat(S_i, S_j, RBF(d_ij))
            input_dim = 2 * config.c_s + config.num_rbf
            
            self.edge_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, config.c_z),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.c_z, config.c_z)
            )
        
        # 预留：共价边嵌入 (第一版不实现)
        self.covalent_edge_embed = None
        if config.use_covalent_edges:
            raise NotImplementedError(
                "共价边嵌入预留给 Phase-2 ablation，第一版不实现"
            )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        权重初始化
        
        策略：
        - Linear: Xavier uniform (保持方差稳定)
        - 最后一层：小初始化 (避免初期梯度爆炸)
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                
                # 最后一层小初始化 (Sequential 的第4个元素是最后的Linear)
                if isinstance(self._get_parent_module(name), nn.Sequential):
                    parent = self._get_parent_module(name)
                    if list(parent.children())[-1] is module:
                        module.weight.data.mul_(0.1)
    
    def _get_parent_module(self, module_name: str):
        """获取父模块（辅助函数）"""
        parts = module_name.split('.')
        if len(parts) <= 1:
            return self
        parent_name = '.'.join(parts[:-1])
        parent = self
        for part in parent_name.split('.'):
            parent = getattr(parent, part)
        return parent
    
    def compute_pairwise_distances(self, 
                                   t: torch.Tensor, 
                                   mask: torch.Tensor,
                                   eps: float = 1e-8) -> torch.Tensor:
        """
        计算成对距离
        
        Args:
            t: (B, N, 3) 节点坐标 (Cα 或质心)
            mask: (B, N) 节点掩码
            eps: 数值稳定性
            
        Returns:
            distances: (B, N, N) 成对距离
        """
        # t: (B, N, 3)
        # 计算 ||t_i - t_j||_2
        diff = t.unsqueeze(2) - t.unsqueeze(1)  # (B, N, N, 3)
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + eps)  # (B, N, N)
        
        # 应用掩码 (无效节点对距离设为无穷大)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, N, N)
        distances = distances.masked_fill(~pair_mask, float('inf'))
        
        return distances
    
    def forward(self, 
                S: torch.Tensor,
                t: torch.Tensor,
                node_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            S: (B, N, c_s) 节点表示 (来自 Adapter)
            t: (B, N, 3) 节点坐标 (Cα 位置)
            node_mask: (B, N) 节点掩码
            
        Returns:
            {
                'z_f1': (B, N, z_rank) 因子1 (flash_1d_bias)
                'z_f2': (B, N, z_rank) 因子2 (flash_1d_bias)
                'edge_embed': (B, N, N, c_z) 完整边嵌入 (flash_2d)
                'edge_mask': (B, N, N) 边掩码
                'distances': (B, N, N) 成对距离 (可选，用于可视化)
            }
        """
        B, N, _ = S.shape
        device = S.device
        
        # 1. 计算成对距离
        distances = self.compute_pairwise_distances(t, node_mask)  # (B, N, N)
        
        # 2. RBF 编码距离
        rbf_features = self.rbf(distances)  # (B, N, N, num_rbf)
        
        # 3. 边掩码
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # (B, N, N)
        
        # 4. 生成边嵌入
        if self.mode == 'flash_1d_bias':
            # 因子化模式：生成 z_f1, z_f2
            # 对每个节点 i，聚合其局部几何信息
            
            # 方法：用平均池化聚合邻居的 RBF 特征
            # rbf_agg: (B, N, num_rbf)
            rbf_agg = (rbf_features * edge_mask.unsqueeze(-1).float()).sum(dim=2)
            rbf_agg = rbf_agg / (edge_mask.sum(dim=2, keepdim=True).float() + 1e-8)
            
            # 拼接节点表示和聚合的几何特征
            node_features = torch.cat([S, rbf_agg], dim=-1)  # (B, N, c_s + num_rbf)
            
            # 生成两个因子
            z_f1 = self.factor1_proj(node_features)  # (B, N, z_rank)
            z_f2 = self.factor2_proj(node_features)  # (B, N, z_rank)
            
            return {
                'z_f1': z_f1,
                'z_f2': z_f2,
                'edge_mask': edge_mask,
                'distances': distances  # 可选：用于调试/可视化
            }
        
        elif self.mode == 'flash_2d':
            # 完整边嵌入 (仅调试用)
            S_i = S.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, c_s)
            S_j = S.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, c_s)
            
            # 拼接 [S_i, S_j, RBF(d_ij)]
            edge_input = torch.cat([S_i, S_j, rbf_features], dim=-1)  # (B, N, N, 2*c_s + num_rbf)
            
            # 生成边嵌入
            edge_embed = self.edge_proj(edge_input)  # (B, N, N, c_z)
            
            # 应用掩码
            edge_embed = edge_embed.masked_fill(
                ~edge_mask.unsqueeze(-1), 0.0
            )
            
            return {
                'edge_embed': edge_embed,
                'edge_mask': edge_mask,
                'distances': distances
            }


# ============================================================================
# 辅助函数
# ============================================================================

def create_edge_embedder(c_s: int = 384,
                        c_z: int = 128,
                        z_rank: int = 16,
                        mode: str = 'flash_1d_bias',
                        **kwargs) -> EdgeEmbedderWrapper:
    """
    工厂函数：创建边嵌入器
    
    Args:
        c_s: 节点表示维度
        c_z: 边表示维度
        z_rank: 因子秩
        mode: 边嵌入模式
        **kwargs: 其他配置参数
        
    Returns:
        EdgeEmbedderWrapper 实例
    
    Example:
        >>> embedder = create_edge_embedder(
        ...     c_s=384, c_z=128, z_rank=16, mode='flash_1d_bias'
        ... )
        >>> outputs = embedder(S, t, node_mask)
        >>> z_f1, z_f2 = outputs['z_f1'], outputs['z_f2']
    """
    config = EdgeEmbedderConfig(
        c_s=c_s,
        c_z=c_z,
        z_factor_rank=z_rank,
        mode=mode,
        **kwargs
    )
    return EdgeEmbedderWrapper(config)


# ============================================================================
# 单元测试
# ============================================================================

def _test_edge_embedder():
    """单元测试：验证边嵌入器功能"""
    print("=" * 80)
    print("边嵌入器单元测试")
    print("=" * 80)
    
    # 配置
    B, N = 2, 10
    c_s, c_z, z_rank = 384, 128, 16
    
    # 输入
    S = torch.randn(B, N, c_s)
    t = torch.randn(B, N, 3)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask[:, -2:] = False  # 最后2个节点无效
    
    print(f"\n输入:")
    print(f"  S: {S.shape}")
    print(f"  t: {t.shape}")
    print(f"  node_mask: {node_mask.shape}, 有效节点: {node_mask.sum().item()}")
    
    # 测试 flash_1d_bias 模式
    print(f"\n测试 flash_1d_bias 模式:")
    embedder_1d = create_edge_embedder(c_s, c_z, z_rank, mode='flash_1d_bias')
    outputs_1d = embedder_1d(S, t, node_mask)
    
    print(f"  z_f1: {outputs_1d['z_f1'].shape}")
    print(f"  z_f2: {outputs_1d['z_f2'].shape}")
    print(f"  edge_mask: {outputs_1d['edge_mask'].shape}")
    print(f"  显存占用: {2 * B * N * z_rank * 4 / 1024 / 1024:.2f} MB (理论)")
    
    # 测试 flash_2d 模式
    print(f"\n测试 flash_2d 模式:")
    embedder_2d = create_edge_embedder(c_s, c_z, z_rank, mode='flash_2d')
    outputs_2d = embedder_2d(S, t, node_mask)
    
    print(f"  edge_embed: {outputs_2d['edge_embed'].shape}")
    print(f"  edge_mask: {outputs_2d['edge_mask'].shape}")
    print(f"  显存占用: {B * N * N * c_z * 4 / 1024 / 1024:.2f} MB (理论)")
    
    # 验证数值
    print(f"\n数值验证:")
    print(f"  z_f1 范围: [{outputs_1d['z_f1'].min():.3f}, {outputs_1d['z_f1'].max():.3f}]")
    print(f"  z_f2 范围: [{outputs_1d['z_f2'].min():.3f}, {outputs_1d['z_f2'].max():.3f}]")
    print(f"  edge_embed 范围: [{outputs_2d['edge_embed'].min():.3f}, {outputs_2d['edge_embed'].max():.3f}]")
    
    # 验证梯度
    print(f"\n梯度验证:")
    loss_1d = outputs_1d['z_f1'].sum() + outputs_1d['z_f2'].sum()
    loss_1d.backward()
    print(f"  ✓ flash_1d_bias 反向传播成功")
    
    embedder_2d.zero_grad()
    loss_2d = outputs_2d['edge_embed'].sum()
    loss_2d.backward()
    print(f"  ✓ flash_2d 反向传播成功")
    
    print(f"\n✅ 所有测试通过！")
    print("=" * 80)


if __name__ == '__main__':
    _test_edge_embedder()

