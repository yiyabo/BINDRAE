#!/usr/bin/env python3
"""调试：检查rigids是否真的在更新"""

import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
from src.stage1.datasets import create_ipa_dataloader
from src.stage1.models import create_stage1_model

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

# 前向传播
with torch.no_grad():
    outputs = model(batch, current_step=0)

# 检查rigids
rigids_final = outputs['rigids_final']
pred_Ca = rigids_final.get_trans()
true_Ca = batch.Ca

print("调试Rigids更新:")
print(f"  Initial Ca (from batch): {true_Ca[0, :5]}")  # 前5个残基
print(f"  Predicted Ca (from rigids): {pred_Ca[0, :5]}")
print(f"  Difference: {(pred_Ca - true_Ca)[0, :5]}")
print(f"  Max diff: {(pred_Ca - true_Ca).abs().max().item():.6f} Å")
print(f"  Mean diff: {(pred_Ca - true_Ca).abs().mean().item():.6f} Å")

if (pred_Ca - true_Ca).abs().max().item() < 1e-6:
    print("\n❌ 问题：Rigids完全没有更新！pred_Ca = true_Ca")
    print("   原因：BackboneUpdateHead可能初始化为0")
else:
    print(f"\n✓ Rigids有更新（平均位移 {(pred_Ca - true_Ca).abs().mean().item():.3f} Å）")

