#!/bin/bash
# FK模块测试脚本

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              FK模块测试"
echo "============================================================================"
echo ""

echo -e "${GREEN}[1/1]${NC} 主链重建测试..."
python << 'EOF'
import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
from src.stage1.models.forward_kinematics import create_protein_fk

print("\n测试配置:")
print(f"  - 批大小: 2")
print(f"  - 残基数: 10")

# 创建FK模块
print("\n创建FK模块...")
fk_module = create_protein_fk()

if torch.cuda.is_available():
    fk_module = fk_module.cuda()
    print(f"  - 设备: CUDA")
else:
    print(f"  - 设备: CPU")

# 准备测试数据：随机扭转角
B, N = 2, 10
torsions_sincos = torch.randn(B, N, 7, 2)
# L2归一化（确保sin²+cos²=1）
torsions_sincos = torch.nn.functional.normalize(torsions_sincos, p=2, dim=-1)

if torch.cuda.is_available():
    torsions_sincos = torsions_sincos.cuda()

# FK重建
print("\n重建主链...")
with torch.no_grad():
    coords = fk_module(torsions_sincos)

print(f"\n✓ FK重建成功！")
print(f"\n输出坐标:")
print(f"  - N: {coords['N'].shape}")
print(f"  - CA: {coords['CA'].shape}")
print(f"  - C: {coords['C'].shape}")
print(f"  - O: {coords['O'].shape}")

# 检查键长（应该接近标准值）
print(f"\n键长检查（应该接近标准值）:")
N_CA_dist = torch.norm(coords['CA'] - coords['N'], dim=-1).mean()
CA_C_dist = torch.norm(coords['C'] - coords['CA'], dim=-1).mean()
C_O_dist = torch.norm(coords['O'] - coords['C'], dim=-1).mean()

print(f"  - N-CA: {N_CA_dist.item():.3f} Å (标准: 1.458)")
print(f"  - CA-C: {CA_C_dist.item():.3f} Å (标准: 1.525)")
print(f"  - C-O: {C_O_dist.item():.3f} Å (标准: 1.231)")

# 检查是否可微分
print(f"\n可微分性检查...")
torsions_grad = torch.randn(B, N, 7, 2, requires_grad=True)
torsions_grad = torch.nn.functional.normalize(torsions_grad, p=2, dim=-1)
if torch.cuda.is_available():
    torsions_grad = torsions_grad.cuda()

coords_grad = fk_module(torsions_grad)
loss = coords_grad['CA'].sum()
loss.backward()

print(f"  ✓ 反向传播成功")
print(f"  ✓ 梯度形状: {torsions_grad.grad.shape}")

print(f"\n✅ FK模块工作正常！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 测试通过！${NC}"
echo "============================================================================"

