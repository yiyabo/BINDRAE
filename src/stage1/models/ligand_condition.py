"""
配体条件化模块

功能：
1. 配体Token嵌入（坐标+类型 → 64维）
2. Cross-Attention（蛋白Q × 配体KV）
3. 残基级FiLM调制（gamma/beta）
4. 门控warmup（λ: 0→1）
5. [NEW] 增强配体编码器（RBF距离 + 自注意力）

Author: BINDRAE Team
Date: 2025-10-28
Updated: 2026-01-18 - 添加 EnhancedLigandEncoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass, field

LIGAND_TYPE_DIM = 20

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
        use_enhanced_encoder: 是否使用增强配体编码器
        enhanced_num_layers: 增强编码器自注意力层数
        enhanced_num_heads: 增强编码器自注意力头数
        num_rbf: RBF距离编码核数量
    """
    c_s: int = 384
    d_lig: int = 64
    num_heads: int = 8
    dropout: float = 0.1
    warmup_steps: int = 2000
    # 增强编码器配置
    use_enhanced_encoder: bool = False
    enhanced_num_layers: int = 2
    enhanced_num_heads: int = 4
    num_rbf: int = 16


# ============================================================================
# 配体Token嵌入
# ============================================================================

class LigandTokenEmbedding(nn.Module):
    """
    配体Token嵌入层
    
    输入: concat([xyz(3), types(20)]) = 23维
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
        
        # 嵌入网络: 23维 (3+20) → d_lig维
        self.embed = nn.Sequential(
            nn.Linear(3 + LIGAND_TYPE_DIM, d_lig),
            nn.LayerNorm(d_lig),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_lig, d_lig)
        )
    
    def forward(self, 
                lig_points: torch.Tensor,
                lig_types: torch.Tensor,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        配体Token嵌入
        
        Args:
            lig_points: [B, M, 3] 配体坐标（重原子+探针）
            lig_types: [B, M, 20] 配体类型
            lig_mask: [B, M] 配体掩码（可选，兼容增强编码器接口）
            
        Returns:
            lig_embed: [B, M, d_lig] 配体嵌入
        """
        # 拼接坐标和类型
        lig_features = torch.cat([lig_points, lig_types], dim=-1)  # [B, M, 23] (3+20)
        
        # 嵌入
        lig_embed = self.embed(lig_features)  # [B, M, d_lig]
        
        return lig_embed


# ============================================================================
# RBF 距离编码
# ============================================================================

class RBFDistanceEncoding(nn.Module):
    """
    RBF (Radial Basis Function) 距离编码
    
    将标量距离编码为高斯核特征向量。
    
    公式: φ_k(d) = exp(-γ * (d - μ_k)²)
    
    其中 μ_k 是等间距的中心点，γ 控制宽度。
    """
    
    def __init__(self, 
                 num_rbf: int = 16, 
                 d_min: float = 0.0, 
                 d_max: float = 20.0,
                 trainable: bool = True):
        """
        Args:
            num_rbf: RBF核数量
            d_min: 距离最小值
            d_max: 距离最大值
            trainable: 中心和宽度是否可训练
        """
        super().__init__()
        
        self.num_rbf = num_rbf
        self.d_min = d_min
        self.d_max = d_max
        
        # 初始化等间距中心点
        centers = torch.linspace(d_min, d_max, num_rbf)
        
        # gamma = 1 / (2 * sigma²), 其中 sigma = (d_max - d_min) / (num_rbf - 1)
        sigma = (d_max - d_min) / (num_rbf - 1) if num_rbf > 1 else 1.0
        gamma = 1.0 / (2 * sigma ** 2)
        
        if trainable:
            self.centers = nn.Parameter(centers)
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('gamma', torch.tensor(gamma))
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        RBF编码
        
        Args:
            distances: [...] 任意形状的距离张量
            
        Returns:
            rbf_features: [..., num_rbf] RBF编码
        """
        # 扩展维度以广播
        d = distances.unsqueeze(-1)  # [..., 1]
        centers = self.centers.view(*([1] * (d.ndim - 1)), -1)  # [1, ..., num_rbf]
        
        # 高斯核: exp(-γ * (d - μ)²)
        rbf = torch.exp(-self.gamma * (d - centers) ** 2)
        
        return rbf


# ============================================================================
# 增强配体编码器
# ============================================================================

class EnhancedLigandEncoder(nn.Module):
    """
    增强版配体编码器
    
    改进点：
    1. RBF距离编码：捕捉配体内原子间距离关系
    2. 自注意力：让配体原子相互交互
    
    相比原始 LigandTokenEmbedding 的简单 MLP，
    该编码器能更好地理解配体的3D几何结构。
    """
    
    def __init__(self, 
                 d_lig: int = 128, 
                 num_heads: int = 4, 
                 num_layers: int = 2,
                 num_rbf: int = 16,
                 dropout: float = 0.1):
        """
        Args:
            d_lig: 配体嵌入维度
            num_heads: 自注意力头数
            num_layers: 自注意力层数
            num_rbf: RBF核数量
            dropout: Dropout概率
        """
        super().__init__()
        
        self.d_lig = d_lig
        self.num_rbf = num_rbf
        
        # 1. 原子特征嵌入 (坐标 + 类型)
        # 输入: xyz(3) + types(20) = 23维
        self.atom_embed = nn.Sequential(
            nn.Linear(3 + LIGAND_TYPE_DIM, d_lig),
            nn.LayerNorm(d_lig),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_lig, d_lig)
        )
        
        # 2. RBF 距离编码
        self.dist_rbf = RBFDistanceEncoding(
            num_rbf=num_rbf,
            d_min=0.0,
            d_max=20.0,  # 配体内原子距离通常 < 20Å
            trainable=True
        )
        
        # 3. 距离特征投影（加到原子特征上）
        self.dist_proj = nn.Sequential(
            nn.Linear(num_rbf, d_lig),
            nn.LayerNorm(d_lig),
            nn.GELU(),
        )
        
        # 4. 配体内自注意力（核心改进！）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_lig,
            nhead=num_heads,
            dim_feedforward=d_lig * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 5. 输出层归一化
        self.out_norm = nn.LayerNorm(d_lig)
        
        print(f"✓ EnhancedLigandEncoder 初始化完成")
        print(f"  - d_lig: {d_lig}, num_rbf: {num_rbf}")
        print(f"  - self_attn: {num_layers} layers, {num_heads} heads")
    
    def _compute_pairwise_distances(self, points: torch.Tensor) -> torch.Tensor:
        """
        计算配体原子间的成对距离
        
        Args:
            points: [B, M, 3] 配体坐标
            
        Returns:
            distances: [B, M, M] 成对距离矩阵
        """
        # 使用欧氏距离
        diff = points.unsqueeze(2) - points.unsqueeze(1)  # [B, M, M, 3]
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [B, M, M]
        return distances
    
    def forward(self,
                lig_points: torch.Tensor,
                lig_types: torch.Tensor,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        增强配体编码
        
        Args:
            lig_points: [B, M, 3] 配体坐标
            lig_types: [B, M, 20] 配体类型
            lig_mask: [B, M] 配体掩码
            
        Returns:
            lig_embed: [B, M, d_lig] 配体嵌入
        """
        B, M, _ = lig_points.shape
        
        # 1. 原子特征嵌入
        lig_features = torch.cat([lig_points, lig_types], dim=-1)  # [B, M, 23]
        atom_feat = self.atom_embed(lig_features)  # [B, M, d_lig]
        
        # 2. 计算成对距离并编码
        distances = self._compute_pairwise_distances(lig_points)  # [B, M, M]
        dist_rbf = self.dist_rbf(distances)  # [B, M, M, num_rbf]
        
        # 3. 聚合距离信息：对每个原子，求其与其他原子的平均距离特征
        if lig_mask is not None:
            # 扩展掩码 [B, M] -> [B, M, M]
            pair_mask = lig_mask.unsqueeze(1) & lig_mask.unsqueeze(2)  # [B, M, M]
            dist_rbf = dist_rbf * pair_mask.unsqueeze(-1).float()
            
            # 加权平均（避免除零）
            mask_sum = pair_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)  # [B, M, 1]
            dist_agg = dist_rbf.sum(dim=2) / mask_sum  # [B, M, num_rbf]
        else:
            dist_agg = dist_rbf.mean(dim=2)  # [B, M, num_rbf]
        
        # 4. 投影距离特征并融合
        dist_feat = self.dist_proj(dist_agg)  # [B, M, d_lig]
        atom_feat = atom_feat + dist_feat  # 残差融合
        
        # 5. 自注意力（配体原子交互）
        if lig_mask is not None:
            # TransformerEncoder期望 True 表示需要 mask 的位置
            src_key_padding_mask = ~lig_mask
        else:
            src_key_padding_mask = None
        
        atom_feat = self.self_attn(
            atom_feat, 
            src_key_padding_mask=src_key_padding_mask
        )  # [B, M, d_lig]
        
        # 6. 输出归一化
        lig_embed = self.out_norm(atom_feat)
        
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
    
    支持两种配体编码器:
        - LigandTokenEmbedding: 简单2层MLP（默认）
        - EnhancedLigandEncoder: RBF距离 + 自注意力（推荐）
    """
    
    def __init__(self, config: LigandConditionerConfig):
        """
        Args:
            config: 配体条件化配置
        """
        super().__init__()
        
        self.config = config
        
        # 1. 配体Token嵌入（支持增强版）
        if config.use_enhanced_encoder:
            self.ligand_embed = EnhancedLigandEncoder(
                d_lig=config.d_lig,
                num_heads=config.enhanced_num_heads,
                num_layers=config.enhanced_num_layers,
                num_rbf=config.num_rbf,
                dropout=config.dropout
            )
            print(f"  Using EnhancedLigandEncoder")
        else:
            self.ligand_embed = LigandTokenEmbedding(
                d_lig=config.d_lig,
                dropout=config.dropout
            )
            print(f"  Using LigandTokenEmbedding (basic)")
        
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
            lig_types: [B, M, 20] 配体类型
            protein_mask: [B, N] 蛋白掩码
            ligand_mask: [B, M] 配体掩码
            gate_lambda: 门控系数（可选，优先于current_step）
            current_step: 当前训练步数（用于自动计算lambda）
            
        Returns:
            conditioned_features: [B, N, c_s] 配体条件化后的特征
        """
        # 1. 配体Token嵌入（传递mask给增强编码器）
        lig_embed = self.ligand_embed(lig_points, lig_types, ligand_mask)  # [B, M, d_lig]
        
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
                              use_enhanced_encoder: bool = False,
                              enhanced_num_layers: int = 2,
                              enhanced_num_heads: int = 4,
                              num_rbf: int = 16,
                              **kwargs) -> LigandConditioner:
    """
    创建配体条件化模块
    
    Args:
        c_s: 蛋白节点维度
        d_lig: 配体嵌入维度
        num_heads: Cross-Attention注意力头数
        warmup_steps: 门控warmup步数
        use_enhanced_encoder: 是否使用增强编码器
        enhanced_num_layers: 增强编码器自注意力层数
        enhanced_num_heads: 增强编码器自注意力头数
        num_rbf: RBF核数量
        **kwargs: 其他配置参数
        
    Returns:
        LigandConditioner实例
        
    Example:
        >>> # 基础版
        >>> conditioner = create_ligand_conditioner(c_s=384, d_lig=64)
        
        >>> # 增强版
        >>> conditioner = create_ligand_conditioner(
        ...     c_s=384, d_lig=128, use_enhanced_encoder=True
        ... )
    """
    config = LigandConditionerConfig(
        c_s=c_s,
        d_lig=d_lig,
        num_heads=num_heads,
        warmup_steps=warmup_steps,
        use_enhanced_encoder=use_enhanced_encoder,
        enhanced_num_layers=enhanced_num_layers,
        enhanced_num_heads=enhanced_num_heads,
        num_rbf=num_rbf,
        **kwargs
    )
    return LigandConditioner(config)
