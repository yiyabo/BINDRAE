"""
ESM-2 Adapter 模块

功能：
将ESM-2的表征从1280维降维到384维

Author: BINDRAE Team
Date: 2025-10-28
"""

from typing import Optional

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
            esm_features: [B, N, esm_dim] ESM-2表征；如果传入
                [B, N, K, esm_dim]，默认使用最后一层以保持旧行为。
            
        Returns:
            adapted: [B, N, output_dim] 降维后的表征
        """
        if esm_features.ndim == 4:
            esm_features = esm_features[..., -1, :]
        if esm_features.ndim != 3:
            raise ValueError(
                f"ESMAdapter expects [B, N, D] or [B, N, K, D], got {tuple(esm_features.shape)}"
            )
        return self.adapter(esm_features)


class ESMLayerFusionAdapter(nn.Module):
    """
    ESM last-K layer fusion adapter.

    输入:
        - [B, N, esm_dim]: 单层 ESM，退化为旧 adapter 行为；
        - [B, N, K, esm_dim]: last-K ESM，先融合 K 层再投影。
    输出:
        - [B, N, output_dim]

    Fusion modes:
        - sum / mean / softmax_weighted: 先融合 K 层再统一投影
        - gated_residual: 每层独立投影, last layer 作为 base,
          earlier layers 作为 per-residue gated 残差补充.
          ``last_layer_weights`` 属性暴露 softmax(layer_logits) 供外部
          entropy 正则使用.
    """

    _VALID_MODES = {"sum", "mean", "softmax_weighted", "gated_residual"}

    def __init__(
        self,
        esm_dim: int = 1280,
        output_dim: int = 384,
        num_layers: int = 1,
        fusion_mode: str = "softmax_weighted",
        layer_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if fusion_mode not in self._VALID_MODES:
            raise ValueError(f"Unsupported ESM fusion mode: {fusion_mode}")
        if not 0.0 <= layer_dropout < 1.0:
            raise ValueError(f"layer_dropout must be in [0, 1), got {layer_dropout}")
        if fusion_mode == "gated_residual" and num_layers < 2:
            raise ValueError(
                "gated_residual fusion requires num_layers >= 2, "
                f"got {num_layers}"
            )

        self.esm_dim = esm_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.fusion_mode = fusion_mode
        self.layer_dropout = layer_dropout

        if fusion_mode == "gated_residual":
            self.layer_projs = nn.ModuleList([
                nn.Linear(esm_dim, output_dim) for _ in range(num_layers)
            ])
            n_earlier = num_layers - 1
            self.gate_proj = nn.Linear(esm_dim, n_earlier)
            # sigmoid(-3) ≈ 0.05 → start near single-layer behavior
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, -3.0)
            self.layer_logits = nn.Parameter(torch.zeros(n_earlier))
            self.layer_dropout_module = nn.Dropout(layer_dropout)
            self.last_layer_weights: Optional[torch.Tensor] = None
            self.last_layer_gates: Optional[torch.Tensor] = None
            self.post_fusion = nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self._ddp_marker = nn.Parameter(torch.zeros(1), requires_grad=False)
        else:
            self.layer_logits = nn.Parameter(torch.zeros(num_layers))
            self.layer_dropout_module = nn.Dropout(layer_dropout)
            self.adapter = ESMAdapter(esm_dim=esm_dim, output_dim=output_dim, dropout=dropout)

    def _layer_weights(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        logits = self.layer_logits.to(device=device, dtype=dtype)
        weights = torch.softmax(logits, dim=0)
        if self.training and self.layer_dropout > 0.0 and self.num_layers > 1:
            dropped = self.layer_dropout_module(weights)
            denom = dropped.sum()
            if denom > torch.finfo(weights.dtype).eps:
                weights = dropped / denom
        return weights

    def _forward_gated_residual(self, esm_features: torch.Tensor) -> torch.Tensor:
        B, N, K, D = esm_features.shape
        s_layers = [proj(esm_features[..., k, :]) for k, proj in enumerate(self.layer_projs)]
        s_base = s_layers[-1]

        n_earlier = K - 1
        weights = self._layer_weights(dtype=s_base.dtype, device=s_base.device)
        self.last_layer_weights = weights

        ctx = esm_features[..., -1, :]
        gates = torch.sigmoid(self.gate_proj(ctx))
        self.last_layer_gates = gates.detach()

        residual = s_base.new_zeros(B, N, self.output_dim)
        for k in range(n_earlier):
            res_k = s_layers[k] - s_base
            residual = residual + gates[..., k:k + 1] * weights[k] * res_k

        fused = s_base + residual
        return self.post_fusion(fused)

    def forward(self, esm_features: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "gated_residual":
            if esm_features.ndim == 3:
                raise ValueError(
                    "gated_residual fusion requires [B, N, K, D] input, "
                    f"got single-layer {tuple(esm_features.shape)}"
                )
            if esm_features.ndim != 4:
                raise ValueError(
                    f"ESMLayerFusionAdapter (gated_residual) expects [B, N, K, D], "
                    f"got {tuple(esm_features.shape)}"
                )
            if esm_features.shape[-2] != self.num_layers:
                raise ValueError(
                    f"ESM layer count mismatch: input K={esm_features.shape[-2]} "
                    f"but adapter num_layers={self.num_layers}"
                )
            return self._forward_gated_residual(esm_features)

        if esm_features.ndim == 3:
            if self.num_layers != 1:
                raise ValueError(
                    f"ESM fusion expects [B, N, K, D] when num_layers={self.num_layers}, "
                    f"got single-layer input {tuple(esm_features.shape)}"
                )
            return self.adapter(esm_features)
        if esm_features.ndim != 4:
            raise ValueError(
                f"ESMLayerFusionAdapter expects [B, N, D] or [B, N, K, D], "
                f"got {tuple(esm_features.shape)}"
            )
        if esm_features.shape[-2] != self.num_layers:
            raise ValueError(
                f"ESM layer count mismatch: input K={esm_features.shape[-2]} "
                f"but adapter num_layers={self.num_layers}"
            )
        if self.fusion_mode == "sum":
            fused = esm_features.sum(dim=-2)
        elif self.fusion_mode == "mean":
            fused = esm_features.mean(dim=-2)
        else:
            weights = self._layer_weights(dtype=esm_features.dtype, device=esm_features.device)
            fused = torch.einsum("bnkd,k->bnd", esm_features, weights)
        return self.adapter(fused)


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


def create_esm_layer_fusion_adapter(
    esm_dim: int = 1280,
    output_dim: int = 384,
    num_layers: int = 1,
    fusion_mode: str = "softmax_weighted",
    layer_dropout: float = 0.0,
    dropout: float = 0.1,
) -> ESMLayerFusionAdapter:
    return ESMLayerFusionAdapter(
        esm_dim=esm_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        fusion_mode=fusion_mode,
        layer_dropout=layer_dropout,
        dropout=dropout,
    )
