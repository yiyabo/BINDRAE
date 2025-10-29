#!/bin/bash
# 配体条件化模块测试脚本（Linux服务器）

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              配体条件化模块测试"
echo "============================================================================"
echo ""

echo -e "${GREEN}[1/1]${NC} 功能验证..."
python << 'EOF'
import torch
from src.stage1.models.ligand_condition import create_ligand_conditioner

print("\n测试配置:")
print(f"  - 批大小: 2")
print(f"  - 蛋白残基数: 30")
print(f"  - 配体token数: 50")

# 准备数据
B, N, M = 2, 30, 50
c_s = 384

protein_features = torch.randn(B, N, c_s)
lig_points = torch.randn(B, M, 3)
lig_types = torch.randn(B, M, 12)
protein_mask = torch.ones(B, N, dtype=torch.bool)
ligand_mask = torch.ones(B, M, dtype=torch.bool)
ligand_mask[:, -10:] = False  # 最后10个token无效

if torch.cuda.is_available():
    protein_features = protein_features.cuda()
    lig_points = lig_points.cuda()
    lig_types = lig_types.cuda()
    protein_mask = protein_mask.cuda()
    ligand_mask = ligand_mask.cuda()
    print(f"  - 设备: CUDA")
else:
    print(f"  - 设备: CPU")

# 创建模块
print("\n创建LigandConditioner...")
conditioner = create_ligand_conditioner(c_s=384, d_lig=64, num_heads=8, warmup_steps=2000)
if torch.cuda.is_available():
    conditioner = conditioner.cuda()

print(f"✓ LigandConditioner创建成功")
print(f"  - 参数量: {sum(p.numel() for p in conditioner.parameters()):,}")

# 测试：不同warmup阶段
print("\n前向传播测试:")

# 阶段1: 初期（lambda=0）
with torch.no_grad():
    s_cond_0 = conditioner(protein_features, lig_points, lig_types, 
                           protein_mask, ligand_mask, current_step=0)
print(f"  ✓ Step 0 (λ=0.00): {s_cond_0.shape}")

# 阶段2: 中期（lambda=0.5）
with torch.no_grad():
    s_cond_1000 = conditioner(protein_features, lig_points, lig_types,
                              protein_mask, ligand_mask, current_step=1000)
print(f"  ✓ Step 1000 (λ=0.50): {s_cond_1000.shape}")

# 阶段3: 后期（lambda=1）
with torch.no_grad():
    s_cond_2000 = conditioner(protein_features, lig_points, lig_types,
                              protein_mask, ligand_mask, current_step=2000)
print(f"  ✓ Step 2000 (λ=1.00): {s_cond_2000.shape}")

# 梯度测试
print("\n梯度测试...")
protein_grad = torch.randn(B, N, c_s, requires_grad=True)
if torch.cuda.is_available():
    protein_grad = protein_grad.cuda()

s_cond_grad = conditioner(protein_grad, lig_points, lig_types,
                          protein_mask, ligand_mask, gate_lambda=0.5)
loss = s_cond_grad.sum()
loss.backward()
print(f"  ✓ 反向传播成功")
print(f"  ✓ 梯度形状: {protein_grad.grad.shape}")

# 显存检查
if torch.cuda.is_available():
    mem = torch.cuda.memory_allocated() / 1024**2
    print(f"\n显存占用: {mem:.2f} MB")

print(f"\n✅ 配体条件化模块工作正常！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 测试通过！${NC}"
echo "============================================================================"

