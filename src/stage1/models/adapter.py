"""
ESM-2 Adapter 模块

功能：
将ESM-2的表征从1280维降维到384维

Author: BINDRAE Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn


class ESMAdapter(nn.Module):
    """
    ESM-2 → 几何分支的降维适配器
    
    输入: [B, N, 1280] ESM-2 per-residue表征
    输出: [B, N, 384] 适配后的表征
    """
    
    def __init__(self, 
                 esm_dim: int = 1280,
                 output_dim: int = 384,
                 dropout: float = 0.1):
        """
        Args:
            esm_dim: ESM-2输出维度（650M模型=1280）
            output_dim: 输出维度（几何分支输入维度）
            dropout: Dropout概率
        """
        super().__init__()
        
        self.esm_dim = esm_dim
        self.output_dim = output_dim
        
        # 降维网络：简单的Linear + LayerNorm
        self.adapter = nn.Sequential(
            nn.Linear(esm_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, esm_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            esm_features: [B, N, esm_dim] ESM-2表征
            
        Returns:
            adapted: [B, N, output_dim] 降维后的表征
        """
        return self.adapter(esm_features)


# ============================================================================
# 工厂函数
# ============================================================================

def create_esm_adapter(esm_dim: int = 1280,
                      output_dim: int = 384,
                      dropout: float = 0.1) -> ESMAdapter:
    """
    创建ESM Adapter
    
    Args:
        esm_dim: ESM输入维度
        output_dim: 输出维度
        dropout: Dropout概率
        
    Returns:
        ESMAdapter实例
        
    Example:
        >>> adapter = create_esm_adapter(esm_dim=1280, output_dim=384)
        >>> esm = torch.randn(2, 50, 1280)
        >>> out = adapter(esm)  # [2, 50, 384]
    """
    return ESMAdapter(esm_dim, output_dim, dropout)

