#!/bin/bash
# Stage-1完整模型测试脚本

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              Stage-1 完整模型测试"
echo "============================================================================"
echo ""

echo -e "${GREEN}[1/1]${NC} 端到端前向传播..."
python << 'EOF'
import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
from src.stage1.datasets import create_ipa_dataloader
from src.stage1.models import create_stage1_model

print("\n测试配置:")
print(f"  - 数据集: CASF-2016 train")
print(f"  - Batch size: 1")

# 创建DataLoader
data_loader = create_ipa_dataloader(
    'data/casf2016',
    split='train',
    batch_size=1,
    shuffle=False,
)

print(f"✓ DataLoader创建成功")

# 加载一个batch
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
    print(f"✓ 数据已转移到CUDA")

print(f"\nBatch信息:")
print(f"  - PDB ID: {batch.pdb_ids[0]}")
print(f"  - 残基数: {batch.n_residues[0]}")
print(f"  - ESM: {batch.esm.shape}")

# 创建模型
print(f"\n创建Stage-1模型...")
model = create_stage1_model()

if torch.cuda.is_available():
    model = model.cuda()

# 前向传播
print(f"\n前向传播...")
with torch.no_grad():
    outputs = model(batch, current_step=1000)

print(f"\n✓ 前向传播成功！")
print(f"\n输出:")
print(f"  - pred_torsions: {outputs['pred_torsions'].shape}")
print(f"  - s_final: {outputs['s_final'].shape}")
print(f"  - rigids_final: Rigid对象")

# 显存检查
if torch.cuda.is_available():
    mem = torch.cuda.memory_allocated() / 1024**2
    print(f"\n显存占用: {mem:.2f} MB")

print(f"\n✅ Stage-1模型工作正常！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 测试通过！${NC}"
echo "============================================================================"

