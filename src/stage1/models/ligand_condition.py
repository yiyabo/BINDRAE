"""
配体条件化模块

功能：
1. 配体Token嵌入（坐标+类型 → 64维）
2. Cross-Attention（蛋白Q × 配体KV）
3. 残基级FiLM调制（gamma/beta）
4. 门控warmup（λ: 0→1）

Author: BINDRAE Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class LigandConditionerConfig:
    """
    配体条件化模块配置
    
    Args:
        c_s: 蛋白节点表示维度
        d_lig: 配体token嵌入维度
        num_heads: Cross-Attention头数
        dropout: Dropout概率
        warmup_steps: 门控λ的warmup步数
    """
    c_s: int = 384
    d_lig: int = 64
    num_heads: int = 8
    dropout: float = 0.1
    warmup_steps: int = 2000


# ============================================================================
# 配体Token嵌入
# ============================================================================

class LigandTokenEmbedding(nn.Module):
    """
    配体Token嵌入层
    
    输入: concat([xyz(3), types(12)]) = 15维
    输出: d_lig维嵌入
    """
    
    def __init__(self, d_lig: int = 64, dropout: float = 0.1):
        """
        Args:
            d_lig: 配体嵌入维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.d_lig = d_lig
        
        # 嵌入网络: 15维 → d_lig维
        self.embed = nn.Sequential(
            nn.Linear(15, d_lig),
            nn.LayerNorm(d_lig),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_lig, d_lig)
        )
    
    def forward(self, 
                lig_points: torch.Tensor,
                lig_types: torch.Tensor) -> torch.Tensor:
        """
        配体Token嵌入
        
        Args:
            lig_points: [B, M, 3] 配体坐标（重原子+探针）
            lig_types: [B, M, 12] 配体类型（原子类型编码）
            
        Returns:
            lig_embed: [B, M, d_lig] 配体嵌入
        """
        # 拼接坐标和类型
        lig_features = torch.cat([lig_points, lig_types], dim=-1)  # [B, M, 15]
        
        # 嵌入
        lig_embed = self.embed(lig_features)  # [B, M, d_lig]
        
        return lig_embed


# ============================================================================
# Cross-Attention
# ============================================================================

class ProteinLigandCrossAttention(nn.Module):
    """
    蛋白-配体 Cross-Attention
    
    Q: 蛋白节点 [B, N, c_s]
    K/V: 配体token [B, M, d_lig]
    """
    
    def __init__(self, c_s: int, d_lig: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            c_s: 蛋白节点维度
            d_lig: 配体嵌入维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.c_s = c_s
        self.d_lig = d_lig
        self.num_heads = num_heads
        
        # 投影层
        self.q_proj = nn.Linear(c_s, c_s)
        self.k_proj = nn.Linear(d_lig, c_s)
        self.v_proj = nn.Linear(d_lig, c_s)
        self.out_proj = nn.Linear(c_s, c_s)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = (c_s // num_heads) ** -0.5
    
    def forward(self,
                protein_features: torch.Tensor,
                ligand_features: torch.Tensor,
                protein_mask: torch.Tensor,
                ligand_mask: torch.Tensor) -> torch.Tensor:
        """
        Cross-Attention前向传播
        
        Args:
            protein_features: [B, N, c_s] 蛋白节点表示
            ligand_features: [B, M, d_lig] 配体嵌入
            protein_mask: [B, N] 蛋白掩码
            ligand_mask: [B, M] 配体掩码
            
        Returns:
            cross_features: [B, N, c_s] 交叉注意力特征
        """
        B, N, _ = protein_features.shape
        M = ligand_features.shape[1]
        
        # 投影Q, K, V
        Q = self.q_proj(protein_features)  # [B, N, c_s]
        K = self.k_proj(ligand_features)   # [B, M, c_s]
        V = self.v_proj(ligand_features)   # [B, M, c_s]
        
        # 重塑为多头
        head_dim = self.c_s // self.num_heads
        Q = Q.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # [B, H, N, d]
        K = K.view(B, M, self.num_heads, head_dim).transpose(1, 2)  # [B, H, M, d]
        V = V.view(B, M, self.num_heads, head_dim).transpose(1, 2)  # [B, H, M, d]
        
        # 计算注意力分数
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        
        # 应用配体掩码
        if ligand_mask is not None:
            # [B, M] → [B, 1, 1, M]
            lig_mask_expanded = ligand_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~lig_mask_expanded, float('-inf'))
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = torch.matmul(attn, V)  # [B, H, N, d]
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, N, self.c_s)  # [B, N, c_s]
        
        # 输出投影
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


# ============================================================================
# FiLM 调制层
# ============================================================================

class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)
    
    公式: S_out = (1 + λ·γ) ⊙ S + λ·β
    
    其中：
        - γ (gamma): 缩放参数
        - β (beta): 偏移参数  
        - λ (gate_lambda): 门控系数（warmup从0到1）
    """
    
    def __init__(self, c_s: int, c_hidden: int = 128, dropout: float = 0.1):
        """
        Args:
            c_s: 输入/输出维度
            c_hidden: 隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.c_s = c_s
        
        # Gamma MLP
        self.gamma_mlp = nn.Sequential(
            nn.Linear(c_s, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, c_s)
        )
        
        # Beta MLP
        self.beta_mlp = nn.Sequential(
            nn.Linear(c_s, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, c_s)
        )
        
        # 特殊初始化
        self._init_film_weights()
    
    def _init_film_weights(self):
        """
        FiLM特殊初始化
        
        关键：
        - gamma最后层: 权重×0.1, 偏置=1
        - beta最后层: 权重×0.1, 偏置=0
        """
        # Gamma最后一层
        gamma_last = self.gamma_mlp[-1]
        nn.init.normal_(gamma_last.weight, mean=0.0, std=0.01)
        nn.init.ones_(gamma_last.bias)  # 偏置=1
        
        # Beta最后一层
        beta_last = self.beta_mlp[-1]
        nn.init.normal_(beta_last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(beta_last.bias)  # 偏置=0
    
    def forward(self,
                features: torch.Tensor,
                cross_features: torch.Tensor,
                gate_lambda: float = 1.0) -> torch.Tensor:
        """
        FiLM调制
        
        Args:
            features: [B, N, c_s] 原始特征
            cross_features: [B, N, c_s] 交叉注意力特征
            gate_lambda: 门控系数（0→1 warmup）
            
        Returns:
            modulated: [B, N, c_s] 调制后的特征
        """
        # 预测gamma和beta
        gamma = self.gamma_mlp(cross_features)  # [B, N, c_s]
        beta = self.beta_mlp(cross_features)    # [B, N, c_s]
        
        # FiLM调制: (1 + λ·γ) ⊙ S + λ·β
        modulated = (1.0 + gate_lambda * gamma) * features + gate_lambda * beta
        
        return modulated


# ============================================================================
# 配体条件化主模块
# ============================================================================

class LigandConditioner(nn.Module):
    """
    配体条件化模块
    
    流程:
        配体Token嵌入 → Cross-Attention → FiLM调制
        
    支持门控warmup（训练初期λ=0，逐渐到λ=1）
    """
    
    def __init__(self, config: LigandConditionerConfig):
        """
        Args:
            config: 配体条件化配置
        """
        super().__init__()
        
        self.config = config
        
        # 1. 配体Token嵌入
        self.ligand_embed = LigandTokenEmbedding(
            d_lig=config.d_lig,
            dropout=config.dropout
        )
        
        # 2. Cross-Attention
        self.cross_attn = ProteinLigandCrossAttention(
            c_s=config.c_s,
            d_lig=config.d_lig,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # 3. FiLM调制
        self.film = FiLMModulation(
            c_s=config.c_s,
            c_hidden=128,
            dropout=config.dropout
        )
    
    def compute_gate_lambda(self, current_step: int) -> float:
        """
        计算门控系数λ (warmup)
        
        Args:
            current_step: 当前训练步数
            
        Returns:
            lambda: 门控系数，范围[0, 1]
        """
        if current_step >= self.config.warmup_steps:
            return 1.0
        else:
            return float(current_step) / self.config.warmup_steps
    
    def forward(self,
                protein_features: torch.Tensor,
                lig_points: torch.Tensor,
                lig_types: torch.Tensor,
                protein_mask: torch.Tensor,
                ligand_mask: torch.Tensor,
                gate_lambda: Optional[float] = None,
                current_step: Optional[int] = None) -> torch.Tensor:
        """
        配体条件化前向传播
        
        Args:
            protein_features: [B, N, c_s] 蛋白节点表示
            lig_points: [B, M, 3] 配体坐标
            lig_types: [B, M, 12] 配体类型
            protein_mask: [B, N] 蛋白掩码
            ligand_mask: [B, M] 配体掩码
            gate_lambda: 门控系数（可选，优先于current_step）
            current_step: 当前训练步数（用于自动计算lambda）
            
        Returns:
            conditioned_features: [B, N, c_s] 配体条件化后的特征
        """
        # 1. 配体Token嵌入
        lig_embed = self.ligand_embed(lig_points, lig_types)  # [B, M, d_lig]
        
        # 2. Cross-Attention
        cross_features = self.cross_attn(
            protein_features, lig_embed, protein_mask, ligand_mask
        )  # [B, N, c_s]
        
        # 3. 计算门控系数
        if gate_lambda is None:
            if current_step is not None:
                gate_lambda = self.compute_gate_lambda(current_step)
            else:
                gate_lambda = 1.0  # 默认全开
        
        # 4. FiLM调制
        conditioned = self.film(protein_features, cross_features, gate_lambda)
        
        return conditioned


# ============================================================================
# 工厂函数
# ============================================================================

def create_ligand_conditioner(c_s: int = 384,
                              d_lig: int = 64,
                              num_heads: int = 8,
                              warmup_steps: int = 2000,
                              **kwargs) -> LigandConditioner:
    """
    创建配体条件化模块
    
    Args:
        c_s: 蛋白节点维度
        d_lig: 配体嵌入维度
        num_heads: 注意力头数
        warmup_steps: 门控warmup步数
        **kwargs: 其他配置参数
        
    Returns:
        LigandConditioner实例
        
    Example:
        >>> conditioner = create_ligand_conditioner(c_s=384, d_lig=64)
        >>> s_cond = conditioner(s, lig_points, lig_types, p_mask, l_mask, 
        ...                      current_step=1000)
    """
    config = LigandConditionerConfig(
        c_s=c_s,
        d_lig=d_lig,
        num_heads=num_heads,
        warmup_steps=warmup_steps,
        **kwargs
    )
    return LigandConditioner(config)
