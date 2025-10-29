#!/bin/bash
# IPA数据加载器测试脚本

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              IPA 数据加载器测试"
echo "============================================================================"
echo ""

echo -e "${GREEN}[1/1]${NC} 功能验证..."
python << 'EOF'
import sys
from pathlib import Path

# 添加项目路径
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.stage1.datasets import create_ipa_dataloader

print("\n测试配置:")
print(f"  - 数据集: CASF-2016")
print(f"  - Split: train")
print(f"  - Batch size: 2")

# 创建DataLoader
data_dir = 'data/casf2016'
train_loader = create_ipa_dataloader(
    data_dir,
    split='train',
    batch_size=2,
    shuffle=False,  # 测试时不打乱
    num_workers=0
)

print(f"\n✓ DataLoader创建成功")
print(f"  - 样本数: {len(train_loader.dataset)}")
print(f"  - Batch数: {len(train_loader)}")

# 加载第一个batch
print(f"\n加载第一个batch...")
batch = next(iter(train_loader))

print(f"\n✓ Batch加载成功")
print(f"\n蛋白数据:")
print(f"  - esm: {batch.esm.shape}")
print(f"  - N: {batch.N.shape}")
print(f"  - Ca: {batch.Ca.shape}")
print(f"  - C: {batch.C.shape}")
print(f"  - node_mask: {batch.node_mask.shape}, 有效节点: {batch.node_mask.sum().item()}")

print(f"\n配体数据:")
print(f"  - lig_points: {batch.lig_points.shape}")
print(f"  - lig_types: {batch.lig_types.shape}")
print(f"  - lig_mask: {batch.lig_mask.shape}, 有效token: {batch.lig_mask.sum().item()}")

print(f"\nGround Truth:")
print(f"  - torsion_angles: {batch.torsion_angles.shape}")
print(f"  - torsion_mask: {batch.torsion_mask.shape}")

print(f"\n口袋:")
print(f"  - w_res: {batch.w_res.shape}")

print(f"\nMeta:")
print(f"  - pdb_ids: {batch.pdb_ids}")
print(f"  - n_residues: {batch.n_residues}")

print(f"\n✅ 数据加载器工作正常！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 测试通过！${NC}"
echo "============================================================================"

