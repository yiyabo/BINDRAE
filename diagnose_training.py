#!/usr/bin/env python3
"""全面诊断训练问题"""

import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.stage1.datasets import create_ipa_dataloader
from src.stage1.models import create_stage1_model

# 加载数据
data_loader = create_ipa_dataloader('data/casf2016', split='train', batch_size=2, shuffle=False)
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

print("="*80)
print("训练诊断")
print("="*80)

# 1. 检查真实扭转角
print("\n[1] 真实扭转角数据:")
print(f"  形状: {batch.torsion_angles.shape}")
print(f"  范围: [{batch.torsion_angles.min():.3f}, {batch.torsion_angles.max():.3f}]")
print(f"  均值: {batch.torsion_angles.mean():.3f}")
print(f"  是否有NaN: {torch.isnan(batch.torsion_angles).any()}")
print(f"\n  Mask统计:")
print(f"  总元素: {batch.torsion_mask.numel()}")
print(f"  True数量: {batch.torsion_mask.sum().item()}")
print(f"  True比例: {batch.torsion_mask.float().mean().item():.1%}")

# 按角度类型统计
angle_names = ['phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4']
for i, name in enumerate(angle_names):
    mask_count = batch.torsion_mask[:, :, i].sum().item()
    total = batch.torsion_mask.shape[0] * batch.torsion_mask.shape[1]
    print(f"    {name}: {mask_count}/{total} ({mask_count/total:.1%})")

# 2. 创建模型并预测
model = create_stage1_model()
if torch.cuda.is_available():
    model = model.cuda()

model.eval()

# 第1次前向
with torch.no_grad():
    outputs1 = model(batch, current_step=0)
    pred1 = torch.atan2(outputs1['pred_torsions'][..., 0], outputs1['pred_torsions'][..., 1])

# 第2次前向（检查是否确定性）
with torch.no_grad():
    outputs2 = model(batch, current_step=0)
    pred2 = torch.atan2(outputs2['pred_torsions'][..., 0], outputs2['pred_torsions'][..., 1])

print(f"\n[2] 预测扭转角:")
print(f"  范围: [{pred1.min():.3f}, {pred1.max():.3f}]")
print(f"  均值: {pred1.mean():.3f}")
print(f"  std: {pred1.std():.3f}")
print(f"\n  两次预测是否相同: {torch.allclose(pred1, pred2)}")

# 3. 计算角度差
diff = pred1 - batch.torsion_angles
diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))

print(f"\n[3] 角度差（预测-真实）:")
print(f"  范围: [{diff_wrapped.min():.3f}, {diff_wrapped.max():.3f}]")
print(f"  均值: {diff_wrapped.mean():.3f}")
print(f"  绝对值均值: {diff_wrapped.abs().mean():.3f} rad = {diff_wrapped.abs().mean()*180/3.14159:.1f}°")

# 4. 手动计算torsion loss
cosine_diff = torch.cos(diff)
loss_manual = (1.0 - cosine_diff) * batch.torsion_mask.float()
loss_manual_mean = loss_manual.sum() / batch.torsion_mask.float().sum()

print(f"\n[4] Torsion Loss手动计算:")
print(f"  1 - cos(diff)范围: [{(1-cosine_diff).min():.3f}, {(1-cosine_diff).max():.3f}]")
print(f"  加权平均: {loss_manual_mean.item():.6f}")
print(f"  期望值（如果完全随机）: ~1.0")
print(f"  实际值: {loss_manual_mean.item():.3f}")

if loss_manual_mean.item() > 0.9:
    print(f"\n  ⚠️  Loss接近1.0，说明预测和真实角度**几乎不相关**（随机）")

print("\n" + "="*80)

