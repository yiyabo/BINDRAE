#!/usr/bin/env python3
"""调试FlashIPA的真实输出"""

import sys
import os

flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path):
    sys.path.insert(0, flash_ipa_path)

import torch
from flash_ipa.edge_embedder import EdgeEmbedder, EdgeEmbedderConfig

# 创建配置
config = EdgeEmbedderConfig(
    c_s=384,
    c_p=128,
    z_factor_rank=16,
    num_rbf=16,
    mode='flash_1d_bias'
)

# 创建模型
embedder = EdgeEmbedder(config)

# 创建测试数据
B, N = 2, 10
node_embed = torch.randn(B, N, 384)
translations = torch.randn(B, N, 3)
trans_sc = translations  # 侧链用主链代替
node_mask = torch.ones(B, N, dtype=torch.bool)

print("=" * 80)
print("FlashIPA EdgeEmbedder 输出调试")
print("=" * 80)

# 前向传播
with torch.no_grad():
    outputs = embedder(
        node_embed=node_embed,
        translations=translations,
        trans_sc=trans_sc,
        node_mask=node_mask,
        edge_embed=None,
        edge_mask=None
    )

print(f"\n输出类型: {type(outputs)}")
print(f"输出长度: {len(outputs) if isinstance(outputs, (tuple, list)) else 'N/A'}")

# 详细检查每个输出
if isinstance(outputs, (tuple, list)):
    for i, out in enumerate(outputs):
        print(f"\n输出[{i}]:")
        if out is None:
            print(f"  类型: None")
        elif isinstance(out, torch.Tensor):
            print(f"  类型: Tensor")
            print(f"  形状: {out.shape}")
            print(f"  dtype: {out.dtype}")
            print(f"  范围: [{out.min():.3f}, {out.max():.3f}]")
        else:
            print(f"  类型: {type(out)}")
            print(f"  值: {out}")

print("\n" + "=" * 80)

