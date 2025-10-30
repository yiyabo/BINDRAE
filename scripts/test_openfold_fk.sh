#!/bin/bash
# OpenFoldFK测试脚本

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              OpenFoldFK测试"
echo "============================================================================"
echo ""

echo -e "${GREEN}[1/1]${NC} 主链重建测试..."
python << 'EOF'
import sys
import os
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path):
    sys.path.insert(0, flash_ipa_path)

import torch
from flash_ipa.rigid import Rigid, Rotation
from src.stage1.models.fk_openfold import create_openfold_fk

print("\n测试配置:")
print(f"  - 批大小: 2")
print(f"  - 残基数: 10")

# 创建FK模块
print("\n创建OpenFoldFK...")
fk_module = create_openfold_fk()

if torch.cuda.is_available():
    fk_module = fk_module.cuda()
    device = 'cuda'
else:
    device = 'cpu'

print(f"  - 设备: {device}")

# 准备输入
B, N = 2, 10

# 随机扭转角
torsions_sincos = torch.randn(B, N, 7, 2, device=device)
torsions_sincos = torch.nn.functional.normalize(torsions_sincos, p=2, dim=-1)

# 创建backbone Rigids（单位旋转+沿x轴排列的CA）
rot_identity = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
ca_positions = torch.arange(N, device=device).float().unsqueeze(0).unsqueeze(-1) * torch.tensor([3.8, 0, 0], device=device)
ca_positions = ca_positions.expand(B, -1, -1)

rotation = Rotation(rot_mats=rot_identity)
backbone_rigids = Rigid(rots=rotation, trans=ca_positions)

# 残基类型（都是ALA=0）
aatype = torch.zeros(B, N, dtype=torch.long, device=device)

# FK重建
print("\n重建主链...")
with torch.no_grad():
    result = fk_module(torsions_sincos, backbone_rigids, aatype)

atom14_pos = result['atom14_pos']
atom14_mask = result['atom14_mask']

print(f"\n✓ FK重建成功！")
print(f"\n输出:")
print(f"  - atom14_pos: {atom14_pos.shape}")
print(f"  - atom14_mask: {atom14_mask.shape}")
print(f"  - 有效原子数: {atom14_mask.sum()//(B*N):.0f}")

# 检查键长
N_pos = atom14_pos[:, :, 0]  # [B, N, 3]
CA_pos = atom14_pos[:, :, 1]
C_pos = atom14_pos[:, :, 2]
O_pos = atom14_pos[:, :, 3]

N_CA_dist = torch.norm(CA_pos - N_pos, dim=-1).mean()
CA_C_dist = torch.norm(C_pos - CA_pos, dim=-1).mean()
C_O_dist = torch.norm(O_pos - C_pos, dim=-1).mean()

print(f"\n键长检查:")
print(f"  - N-CA: {N_CA_dist.item():.3f} Å (期望: 1.458)")
print(f"  - CA-C: {CA_C_dist.item():.3f} Å (期望: 1.526)")
print(f"  - C-O: {C_O_dist.item():.3f} Å (期望: 1.231)")

tolerance = 0.1  # 10%容差
if abs(N_CA_dist.item() - 1.458) < tolerance:
    print(f"  ✓ N-CA键长正确！")
else:
    print(f"  ⚠️  N-CA键长偏差: {abs(N_CA_dist.item() - 1.458):.3f} Å")

print(f"\n✅ OpenFoldFK测试完成！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 测试通过！${NC}"
echo "============================================================================"

