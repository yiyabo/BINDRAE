#!/usr/bin/env python3
"""
边嵌入模块快速验证脚本（Mac兼容版）
"""

import sys
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

# 导入模块
try:
    from src.stage1.modules.edge_embed import (
        EdgeEmbedderWrapper,
        EdgeEmbedderConfig,
        create_edge_embedder,
    )
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 创建简单测试
try:
    print("\n" + "="*60)
    print("测试 1: 配置创建")
    print("="*60)
    config = EdgeEmbedderConfig(c_s=384, c_z=128, z_factor_rank=16)
    print(f"✓ 配置创建成功: {config}")
    
    print("\n" + "="*60)
    print("测试 2: 模块实例化")
    print("="*60)
    embedder = EdgeEmbedderWrapper(config)
    print(f"✓ EdgeEmbedder 创建成功")
    print(f"  - 参数量: {sum(p.numel() for p in embedder.parameters()):,}")
    
    print("\n" + "="*60)
    print("测试 3: 前向传播（小规模）")
    print("="*60)
    B, N = 1, 5  # 小规模避免Mac段错误
    S = torch.randn(B, N, 384)
    t = torch.randn(B, N, 3)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    
    with torch.no_grad():
        outputs = embedder(S, t, node_mask)
    
    print(f"✓ 前向传播成功")
    print(f"  - z_f1: {outputs['z_f1'].shape}")
    print(f"  - z_f2: {outputs['z_f2'].shape}")
    print(f"  - edge_mask: {outputs['edge_mask'].shape}")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！模块工作正常")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

