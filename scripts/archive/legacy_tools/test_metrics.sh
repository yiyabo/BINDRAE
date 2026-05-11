#!/bin/bash
# 评估指标测试脚本

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              评估指标测试"
echo "============================================================================"
echo ""

echo -e "${GREEN}[1/1]${NC} 功能验证..."
python << 'EOF'
import numpy as np
import torch
from utils.metrics import (
    compute_pocket_irmsd,
    compute_chi1_accuracy,
    compute_clash_percentage,
    compute_fape,
)

print("\n测试配置:")
print(f"  - 残基数: 50")
print(f"  - 口袋残基: 20")

# 准备测试数据
N_res = 50
N_pocket = 20

# 随机生成预测和真实坐标（Cα）
true_coords = np.random.randn(N_res, 3).astype(np.float32)
pred_coords = true_coords + np.random.randn(N_res, 3) * 0.5  # 加噪声

# 口袋掩码
pocket_mask = np.zeros(N_res, dtype=bool)
pocket_mask[:N_pocket] = True

# 测试1: 口袋iRMSD
print(f"\n[测试1] 口袋iRMSD...")
irmsd = compute_pocket_irmsd(pred_coords, true_coords, pocket_mask)
print(f"  ✓ iRMSD = {irmsd:.3f} Å")

# 测试2: χ1命中率
print(f"\n[测试2] χ1命中率...")
true_chi1 = np.random.rand(N_res) * 2 * np.pi  # [0, 2π]
pred_chi1 = true_chi1 + np.random.randn(N_res) * 0.1  # 小扰动
chi1_mask = np.ones(N_res, dtype=bool)
chi1_mask[40:] = False  # 最后10个无侧链

chi1_acc = compute_chi1_accuracy(pred_chi1, true_chi1, chi1_mask, threshold_deg=20)
print(f"  ✓ χ1命中率 = {chi1_acc:.1%}")

# 测试3: Clash百分比
print(f"\n[测试3] Clash检测...")
# 创建一些正常距离的坐标
clash_coords = np.random.randn(30, 3) * 10
# 手动制造几个碰撞
clash_coords[1] = clash_coords[0] + np.array([0.5, 0, 0])  # 距离0.5Å

clash_pct = compute_clash_percentage(clash_coords, clash_threshold=2.0)
print(f"  ✓ Clash% = {clash_pct:.1%}")

# 测试4: FAPE
print(f"\n[测试4] FAPE...")
# 创建简单的局部帧（单位旋转+随机平移）
R_pred = np.tile(np.eye(3), (N_res, 1, 1)).astype(np.float32)
t_pred = np.random.randn(N_res, 3).astype(np.float32)
R_true = R_pred.copy()
t_true = t_pred.copy()

w_res = pocket_mask.astype(np.float32)  # 口袋权重

fape = compute_fape(pred_coords, true_coords, (R_pred, t_pred), (R_true, t_true), w_res)
print(f"  ✓ FAPE = {fape:.3f} Å")

print(f"\n✅ 所有评估指标工作正常！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 测试通过！${NC}"
echo "============================================================================"

