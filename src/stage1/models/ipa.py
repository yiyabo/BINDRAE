"""
FlashIPA 几何分支模块

功能：
1. 多层 IPA 堆叠
2. 帧更新预测头
3. 逐层裁剪与更新
4. FFN + LayerNorm + 残差

架构：
    每层: Self-IPA → 帧更新 → 裁剪 → compose → FFN → 残差

Author: BINDRAE Team
Date: 2025-10-28
"""

import sys
import os
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 添加FlashIPA路径
flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.ipa import InvariantPointAttention, IPAConfig
from flash_ipa.rigid import Rigid, Rotation


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class FlashIPAModuleConfig:
    """
    FlashIPA 几何分支配置
    
    Args:
        c_s: 节点单表示维度
        c_z: 边表示维度（传给EdgeEmbedder的c_p）
        c_hidden: IPA隐藏层维度
        no_heads: 注意力头数
        depth: IPA堆叠层数
        no_qk_points: query/key点数
        no_v_points: value点数
        z_factor_rank: 边因子化秩
        dropout: Dropout概率
        max_rot_update_deg: 最大旋转更新角度（度）
        max_trans_update: 最大平移更新距离（Å）
    """
    c_s: int = 384
    c_z: int = 128
    c_hidden: int = 128
    no_heads: int = 8
    depth: int = 3
    no_qk_points: int = 8
    no_v_points: int = 12
    z_factor_rank: int = 2  # 降低到2以满足headdim_eff≤256（公式：128+36+2*32=228）
    dropout: float = 0.1
    max_rot_update_deg: float = 15.0
    max_trans_update: float = 1.5
    
    def to_ipa_config(self) -> IPAConfig:
        """转换为FlashIPA的IPAConfig"""
        return IPAConfig(
            c_s=self.c_s,
            c_z=self.c_z,
            c_hidden=self.c_hidden,
            no_heads=self.no_heads,
            z_factor_rank=self.z_factor_rank,
            no_qk_points=self.no_qk_points,
            no_v_points=self.no_v_points,
            use_flash_attn=True,
            attn_dtype='fp16',  # headdim_eff=676 > 256, 必须用fp16
        )


# ============================================================================
# 帧更新预测头
# ============================================================================

class BackboneUpdateHead(nn.Module):
    """
    预测刚体帧的增量更新
    
    输入: [B, N, c_s] 节点表示
    输出: [B, N, 6] 帧增量 [ωx, ωy, ωz, tx, ty, tz]
    
    其中：
        - ω (3维): 旋转轴角表示
        - t (3维): 平移向量
    """
    
    def __init__(self, c_s: int, c_hidden: int = 128):
        """
        Args:
            c_s: 输入维度
            c_hidden: 隐藏层维度
        """
        super().__init__()
        
        self.c_s = c_s
        self.c_hidden = c_hidden
        
        # 预测网络: LayerNorm → Linear → GELU → Linear(6)
        self.net = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, 6)
        )
        
        # 小初始化（避免初期更新过大）
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        # 最后一层初始化（需要让rigids有合理幅度的更新）
        # std=0.1: 初期约0.1-1 Å的更新幅度，会被clip到≤1.5Å
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        预测帧增量
        
        Args:
            s: [B, N, c_s] 节点表示
            
        Returns:
            updates: [B, N, 6] 帧增量 [ωx, ωy, ωz, tx, ty, tz]
        """
        return self.net(s)  # [B, N, 6]


# ============================================================================
# 帧更新工具函数
# ============================================================================

def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor, 
                                  eps: float = 1e-8) -> torch.Tensor:
    """
    轴角表示 → 旋转矩阵 (Rodrigues公式)
    
    Args:
        axis_angle: [..., 3] 轴角向量
        eps: 数值稳定性
        
    Returns:
        R: [..., 3, 3] 旋转矩阵
    """
    # 计算角度
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # [..., 1]
    
    # 处理零角度情况
    angle_safe = torch.clamp(angle, min=eps)
    
    # 归一化轴
    axis = axis_angle / angle_safe  # [..., 3]
    
    # Rodrigues公式
    cos_a = torch.cos(angle)  # [..., 1]
    sin_a = torch.sin(angle)  # [..., 1]
    
    # 构建反对称矩阵 K
    K = torch.zeros(*axis_angle.shape[:-1], 3, 3, device=axis_angle.device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    
    # R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, device=axis_angle.device).expand_as(K)
    R = I + sin_a.unsqueeze(-1) * K + (1 - cos_a).unsqueeze(-1) * (K @ K)
    
    return R


def clip_frame_update(axis_angle: torch.Tensor,
                     translation: torch.Tensor,
                     max_angle_deg: float = 15.0,
                     max_trans: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    裁剪帧增量（防止数值爆炸）
    
    Args:
        axis_angle: [..., 3] 轴角向量
        translation: [..., 3] 平移向量
        max_angle_deg: 最大旋转角度（度）
        max_trans: 最大平移距离（Å）
        
    Returns:
        裁剪后的 (axis_angle, translation)
    """
    max_angle_rad = max_angle_deg * math.pi / 180.0
    
    # 裁剪旋转
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    scale = torch.clamp(angle / max_angle_rad, max=1.0)
    axis_angle_clipped = axis_angle * scale
    
    # 裁剪平移
    trans_norm = torch.norm(translation, dim=-1, keepdim=True)
    trans_scale = torch.clamp(trans_norm / max_trans, max=1.0)
    translation_clipped = translation * trans_scale
    
    return axis_angle_clipped, translation_clipped


def create_rigid_from_updates(axis_angle: torch.Tensor,
                              translation: torch.Tensor) -> Rigid:
    """
    从轴角+平移创建Rigid增量
    
    Args:
        axis_angle: [B, N, 3] 轴角向量
        translation: [B, N, 3] 平移向量
        
    Returns:
        delta_rigid: Rigid对象 [B, N]
    """
    # 轴角 → 旋转矩阵
    rot_mats = axis_angle_to_rotation_matrix(axis_angle)  # [B, N, 3, 3]
    
    # 创建Rotation对象
    rotation = Rotation(rot_mats=rot_mats)
    
    # 创建Rigid对象
    delta_rigid = Rigid(rots=rotation, trans=translation)
    
    return delta_rigid


# ============================================================================
# FFN (Feed-Forward Network)
# ============================================================================

class IPAFeedForward(nn.Module):
    """
    IPA后的前馈网络
    
    结构: LayerNorm → Linear → GELU → Dropout → Linear
    """
    
    def __init__(self, c_s: int, expansion: int = 4, dropout: float = 0.1):
        """
        Args:
            c_s: 输入/输出维度
            expansion: 隐藏层扩展倍数
            dropout: Dropout概率
        """
        super().__init__()
        
        c_hidden = c_s * expansion
        
        self.net = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_hidden, c_s)
        )
        
        # 最后一层小初始化
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [B, N, c_s] 节点表示
            
        Returns:
            s_out: [B, N, c_s] 更新后的表示
        """
        return self.net(s)


# ============================================================================
# 单层 IPA 块
# ============================================================================

class IPABlock(nn.Module):
    """
    单层 IPA 块
    
    流程:
        输入 → IPA → 帧更新预测 → 裁剪 → compose → FFN → 残差 → 输出
    """
    
    def __init__(self, config: FlashIPAModuleConfig):
        """
        Args:
            config: IPA模块配置
        """
        super().__init__()
        
        self.config = config
        
        # 1. IPA 注意力
        ipa_config = config.to_ipa_config()
        self.ipa = InvariantPointAttention(ipa_config)
        
        # 2. 帧更新预测头
        self.backbone_update = BackboneUpdateHead(
            c_s=config.c_s,
            c_hidden=config.c_hidden
        )
        
        # 3. FFN
        self.ffn = IPAFeedForward(
            c_s=config.c_s,
            expansion=4,
            dropout=config.dropout
        )
        
        # 4. Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,
                s: torch.Tensor,
                rigids: Rigid,
                z_factor_1: torch.Tensor,
                z_factor_2: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, Rigid]:
        """
        单层IPA前向传播
        
        Args:
            s: [B, N, c_s] 节点表示
            rigids: Rigid对象 [B, N] 当前帧
            z_factor_1: [B, N, z_rank, c_z] 边因子1
            z_factor_2: [B, N, z_rank, c_z] 边因子2
            mask: [B, N] 节点掩码
            
        Returns:
            s_updated: [B, N, c_s] 更新后的节点表示
            rigids_updated: Rigid对象 [B, N] 更新后的帧
        """
        # 1. IPA 注意力
        s_ipa = self.ipa(
            s=s,
            z=None,  # 不使用完整边表示（用因子化）
            z_factor_1=z_factor_1,
            z_factor_2=z_factor_2,
            r=rigids,
            mask=mask
        )
        
        # 残差连接
        s = s + self.dropout(s_ipa)
        
        # 2. 预测帧更新
        updates = self.backbone_update(s)  # [B, N, 6]
        axis_angle = updates[..., :3]      # [B, N, 3]
        translation = updates[..., 3:]     # [B, N, 3]
        
        # 3. 裁剪增量
        axis_angle, translation = clip_frame_update(
            axis_angle,
            translation,
            max_angle_deg=self.config.max_rot_update_deg,
            max_trans=self.config.max_trans_update
        )
        
        # 4. 创建增量Rigid并compose
        delta_rigid = create_rigid_from_updates(axis_angle, translation)
        rigids_updated = rigids.compose(delta_rigid)
        
        # 5. FFN
        s_ffn = self.ffn(s)
        s = s + self.dropout(s_ffn)
        
        return s, rigids_updated


# ============================================================================
# 多层 FlashIPA 模块
# ============================================================================

class FlashIPAModule(nn.Module):
    """
    FlashIPA 几何分支
    
    架构:
        输入 → [IPA块 × depth] → 输出
        
    每个IPA块:
        Self-IPA → 帧更新 → 裁剪 → compose → FFN → 残差
    """
    
    def __init__(self, config: FlashIPAModuleConfig):
        """
        Args:
            config: FlashIPA模块配置
        """
        super().__init__()
        
        self.config = config
        
        # 多层IPA块
        self.ipa_blocks = nn.ModuleList([
            IPABlock(config) for _ in range(config.depth)
        ])
        
        # 最终LayerNorm
        self.final_norm = nn.LayerNorm(config.c_s)
    
    def forward(self,
                s: torch.Tensor,
                rigids: Rigid,
                z_factor_1: torch.Tensor,
                z_factor_2: torch.Tensor,
                mask: torch.Tensor,
                ligand_conditioner: Optional[nn.Module] = None,
                lig_points: Optional[torch.Tensor] = None,
                lig_types: Optional[torch.Tensor] = None,
                protein_mask: Optional[torch.Tensor] = None,
                ligand_mask: Optional[torch.Tensor] = None,
                current_step: Optional[int] = None) -> Tuple[torch.Tensor, Rigid]:
        """
        多层IPA前向传播
        
        Args:
            s: [B, N, c_s] 节点表示
            rigids: Rigid对象 [B, N] 初始帧
            z_factor_1: [B, N, z_rank, c_z] 边因子1
            z_factor_2: [B, N, z_rank, c_z] 边因子2
            mask: [B, N] 节点掩码
            
        Returns:
            s_geo: [B, N, c_s] 几何增强的节点表示
            rigids_final: Rigid对象 [B, N] 最终帧
        """
        # 逐层更新
        for i, block in enumerate(self.ipa_blocks):
            s, rigids = block(s, rigids, z_factor_1, z_factor_2, mask)
            if ligand_conditioner is not None:
                s = ligand_conditioner(
                    s,
                    lig_points,
                    lig_types,
                    protein_mask,
                    ligand_mask,
                    current_step=current_step,
                )
        
        # 最终归一化
        s = self.final_norm(s)
        
        return s, rigids


# ============================================================================
# 工厂函数
# ============================================================================

def create_flashipa_module(c_s: int = 384,
                          c_z: int = 128,
                          depth: int = 3,
                          **kwargs) -> FlashIPAModule:
    """
    创建FlashIPA几何分支模块
    
    Args:
        c_s: 节点表示维度
        c_z: 边表示维度
        depth: IPA堆叠层数
        **kwargs: 其他配置参数
        
    Returns:
        FlashIPAModule实例
        
    Example:
        >>> ipa_module = create_flashipa_module(c_s=384, c_z=128, depth=3)
        >>> s_geo, rigids_final = ipa_module(s, rigids, z_f1, z_f2, mask)
    """
    config = FlashIPAModuleConfig(
        c_s=c_s,
        c_z=c_z,
        depth=depth,
        **kwargs
    )
    return FlashIPAModule(config)

