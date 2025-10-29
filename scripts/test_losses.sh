#!/bin/bash
# 损失函数测试脚本

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              损失函数测试"
echo "============================================================================"
echo ""

echo -e "${GREEN}[1/1]${NC} 功能验证..."
python << 'EOF'
import torch
from src.stage1.modules.losses import (
    fape_loss,
    torsion_loss,
    distance_loss,
    clash_penalty,
)

print("\n测试配置:")
print(f"  - 批大小: 2")
print(f"  - 残基数: 30")

# 准备测试数据
B, N = 2, 30

pred_coords = torch.randn(B, N, 3, requires_grad=True)
true_coords = torch.randn(B, N, 3)
w_res = torch.rand(B, N)  # 口袋权重

if torch.cuda.is_available():
    pred_coords = pred_coords.cuda()
    true_coords = true_coords.cuda()
    w_res = w_res.cuda()
    print(f"  - 设备: CUDA")
else:
    print(f"  - 设备: CPU")

# 准备帧
pred_R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
pred_t = torch.randn(B, N, 3)
true_R = pred_R.clone()
true_t = true_coords.clone()

if torch.cuda.is_available():
    pred_R = pred_R.cuda()
    pred_t = pred_t.cuda()
    true_R = true_R.cuda()
    true_t = true_t.cuda()

# 测试1: FAPE损失
print(f"\n[测试1] FAPE损失...")
loss_fape = fape_loss(pred_coords, true_coords, (pred_R, pred_t), (true_R, true_t), w_res)
print(f"  ✓ FAPE loss = {loss_fape.item():.4f}")
print(f"  ✓ requires_grad = {loss_fape.requires_grad}")

# 测试2: 扭转角损失
print(f"\n[测试2] 扭转角损失...")
pred_angles = torch.randn(B, N, 7, requires_grad=True)
true_angles = torch.randn(B, N, 7)
angle_mask = torch.ones(B, N, 7, dtype=torch.bool)

if torch.cuda.is_available():
    pred_angles = pred_angles.cuda()
    true_angles = true_angles.cuda()
    angle_mask = angle_mask.cuda()

loss_torsion = torsion_loss(pred_angles, true_angles, angle_mask, w_res)
print(f"  ✓ Torsion loss = {loss_torsion.item():.4f}")
print(f"  ✓ requires_grad = {loss_torsion.requires_grad}")

# 测试3: 距离损失
print(f"\n[测试3] 距离损失...")
loss_dist = distance_loss(pred_coords, true_coords, w_res)
print(f"  ✓ Distance loss = {loss_dist.item():.4f}")
print(f"  ✓ requires_grad = {loss_dist.requires_grad}")

# 测试4: 碰撞惩罚
print(f"\n[测试4] 碰撞惩罚...")
loss_clash = clash_penalty(pred_coords, clash_threshold=2.0)
print(f"  ✓ Clash penalty = {loss_clash.item():.4f}")
print(f"  ✓ requires_grad = {loss_clash.requires_grad}")

# 测试5: 组合损失+反向传播
print(f"\n[测试5] 组合损失+反向传播...")
total_loss = 1.0 * loss_fape + 1.0 * loss_torsion + 0.1 * loss_dist + 0.1 * loss_clash
print(f"  ✓ Total loss = {total_loss.item():.4f}")

total_loss.backward()
print(f"  ✓ 反向传播成功")
if pred_coords.grad is not None:
    print(f"  ✓ pred_coords梯度: {pred_coords.grad.shape}")
if pred_angles.grad is not None:
    print(f"  ✓ pred_angles梯度: {pred_angles.grad.shape}")

print(f"\n✅ 所有损失函数工作正常！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 测试通过！${NC}"
echo "============================================================================"

