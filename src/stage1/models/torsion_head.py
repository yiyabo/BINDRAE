"""
扭转角预测头

功能：
预测主链和侧链扭转角（输出sin/cos形式）

Author: BINDRAE Team  
Date: 2025-10-28
"""

import torch
import torch.nn as nn


class TorsionHead(nn.Module):
    """
    扭转角预测头
    
    输入: [B, N, c_s] 节点表示
    输出: [B, N, 7, 2] 扭转角的(sin, cos)
           7个角度: phi, psi, omega, chi1, chi2, chi3, chi4
    """
    
    def __init__(self, c_s: int = 384, c_hidden: int = 128, dropout: float = 0.1):
        """
        Args:
            c_s: 输入维度
            c_hidden: 隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.c_s = c_s
        self.n_angles = 7  # phi, psi, omega, chi1-4
        
        # 预测网络
        self.net = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, self.n_angles * 2)  # 7个角度 × 2(sin, cos)
        )
        
        # 小初始化（避免初期预测过大）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        预测扭转角
        
        Args:
            s: [B, N, c_s] 节点表示
            
        Returns:
            angles_sincos: [B, N, 7, 2] 扭转角的(sin, cos)
        """
        B, N, _ = s.shape
        
        # 预测
        out = self.net(s)  # [B, N, 14]
        
        # 重塑为 [B, N, 7, 2]
        angles_sincos = out.view(B, N, self.n_angles, 2)
        
        # L2归一化（确保sin²+cos²=1）
        angles_sincos = nn.functional.normalize(angles_sincos, p=2, dim=-1)
        
        return angles_sincos
    
    def sincos_to_angles(self, sincos: torch.Tensor) -> torch.Tensor:
        """
        将(sin, cos)转换为角度
        
        Args:
            sincos: [..., 2] (sin, cos)
            
        Returns:
            angles: [...] 角度（弧度），范围[-π, π]
        """
        sin_vals = sincos[..., 0]
        cos_vals = sincos[..., 1]
        angles = torch.atan2(sin_vals, cos_vals)
        return angles


def create_torsion_head(c_s: int = 384,
                       c_hidden: int = 128,
                       dropout: float = 0.1) -> TorsionHead:
    """
    创建扭转角预测头
    
    Args:
        c_s: 输入维度
        c_hidden: 隐藏层维度
        dropout: Dropout概率
        
    Returns:
        TorsionHead实例
    """
    return TorsionHead(c_s, c_hidden, dropout)

