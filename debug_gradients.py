#!/usr/bin/env python3
"""检查梯度是否传播"""

import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
from src.stage1.datasets import create_ipa_dataloader
from src.stage1.models import create_stage1_model
from src.stage1.modules.losses import torsion_loss

# 加载数据
data_loader = create_ipa_dataloader('data/casf2016', split='train', batch_size=1, shuffle=False)
batch = next(iter(data_loader))

if torch.cuda.is_available():
    batch.esm = batch.esm.cuda()
    batch.N = batch.N.cuda()
    batch.Ca = batch.Ca.cuda()
    batch.C = batch.C.cuda()
    batch.node_mask = batch.node_mask.cuda()
    batch.lig_points = batch.lig_points.cuda()
    batch.lig_types = batch.lig_types.cuda()
    batch.lig_mask = batch.lig_mask.cuda()
    batch.torsion_angles = batch.torsion_angles.cuda()
    batch.torsion_mask = batch.torsion_mask.cuda()
    batch.w_res = batch.w_res.cuda()

# 创建模型
model = create_stage1_model()
if torch.cuda.is_available():
    model = model.cuda()

model.train()

# 前向传播
outputs = model(batch, current_step=0)

# 计算torsion loss
pred_torsions = torch.atan2(outputs['pred_torsions'][..., 0], outputs['pred_torsions'][..., 1])
loss = torsion_loss(pred_torsions, batch.torsion_angles, batch.torsion_mask, batch.w_res)

print("="*80)
print("梯度检查")
print("="*80)

print(f"\nTorsion Loss: {loss.item():.6f}")
print(f"Requires grad: {loss.requires_grad}")

# 反向传播
loss.backward()

# 检查关键模块的梯度
print("\n梯度统计:")

# TorsionHead
torsion_head_params = list(model.torsion_head.parameters())
last_layer = torsion_head_params[-1]  # 最后一层权重
print(f"\n1. TorsionHead最后一层:")
print(f"   权重形状: {last_layer.shape}")
if last_layer.grad is not None:
    print(f"   梯度范围: [{last_layer.grad.min():.6f}, {last_layer.grad.max():.6f}]")
    print(f"   梯度均值: {last_layer.grad.abs().mean():.6f}")
else:
    print(f"   ❌ 梯度为None！")

# ESM Adapter
adapter_params = list(model.esm_adapter.parameters())
first_layer = adapter_params[0]
print(f"\n2. ESM Adapter第一层:")
print(f"   权重形状: {first_layer.shape}")
if first_layer.grad is not None:
    print(f"   梯度范围: [{first_layer.grad.min():.6f}, {first_layer.grad.max():.6f}]")
    print(f"   梯度均值: {first_layer.grad.abs().mean():.6f}")
else:
    print(f"   ❌ 梯度为None！")

# IPA
ipa_params = list(model.ipa_module.parameters())
print(f"\n3. FlashIPA模块:")
print(f"   总参数数: {len(ipa_params)}")
has_grad = sum(1 for p in ipa_params if p.grad is not None)
print(f"   有梯度的参数: {has_grad}/{len(ipa_params)}")

if has_grad > 0:
    grads = [p.grad.abs().mean().item() for p in ipa_params if p.grad is not None]
    print(f"   平均梯度: {sum(grads)/len(grads):.6f}")

print("\n" + "="*80)

