#!/usr/bin/env python3
"""调试：检查每个loss的实际数值"""

import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
from src.stage1.datasets import create_ipa_dataloader
from src.stage1.models import create_stage1_model
from src.stage1.modules.losses import fape_loss, torsion_loss, distance_loss, clash_penalty

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

# 创建模型
model = create_stage1_model()
if torch.cuda.is_available():
    model = model.cuda()

model.eval()

# 前向传播
with torch.no_grad():
    outputs = model(batch, current_step=0)

# 计算每个loss
print("="*80)
print("损失函数详细数值")
print("="*80)

# 扭转角损失
pred_torsions = torch.atan2(outputs['pred_torsions'][..., 0], outputs['pred_torsions'][..., 1])
loss_tor = torsion_loss(pred_torsions, batch.torsion_angles, batch.torsion_mask, batch.w_res)
print(f"\n1. Torsion Loss: {loss_tor.item():.6f}")

# Rigids
rigids_final = outputs['rigids_final']
pred_Ca = rigids_final.get_trans()
true_Ca = batch.Ca

print(f"\n2. Rigids更新:")
print(f"   Max displacement: {(pred_Ca - true_Ca).abs().max().item():.6f} Å")
print(f"   Mean displacement: {(pred_Ca - true_Ca).abs().mean().item():.6f} Å")

# 距离损失
loss_dist = distance_loss(pred_Ca, true_Ca, batch.w_res)
print(f"\n3. Distance Loss: {loss_dist.item():.6f}")

# FAPE损失
pred_R = rigids_final.get_rots().get_rot_mats()
pred_t = pred_Ca
B, N = batch.Ca.shape[:2]
true_R = torch.eye(3, device=batch.Ca.device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
true_t = true_Ca

loss_fape_val = fape_loss(pred_Ca, true_Ca, (pred_R, pred_t), (true_R, true_t), batch.w_res)
print(f"\n4. FAPE Loss: {loss_fape_val.item():.6f}")

# Clash
loss_clash_val = clash_penalty(pred_Ca, clash_threshold=3.8)
print(f"\n5. Clash Penalty: {loss_clash_val.item():.6f}")

# 总损失
total = 1.0 * loss_tor + 0.1 * loss_dist + 0.1 * loss_clash_val + 1.0 * loss_fape_val
print(f"\n总损失:")
print(f"  1.0 × Torsion: {loss_tor.item():.4f}")
print(f"  0.1 × Distance: {0.1 * loss_dist.item():.4f}")
print(f"  0.1 × Clash: {0.1 * loss_clash_val.item():.4f}")
print(f"  1.0 × FAPE: {loss_fape_val.item():.4f}")
print(f"  Total: {total.item():.4f}")

print("\n" + "="*80)

