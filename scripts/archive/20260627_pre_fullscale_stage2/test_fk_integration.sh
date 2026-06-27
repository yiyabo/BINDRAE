#!/bin/bash
# 测试FK集成后的完整模型（服务器运行）

set -e

echo "============================================================================"
echo "              FK集成测试（完整端到端）"
echo "============================================================================"
echo ""

python << 'EOF'
import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import torch
from src.stage1.datasets import create_ipa_dataloader
from src.stage1.models import create_stage1_model

print("测试配置:")
print(f"  - 加载真实数据")
print(f"  - Batch size: 1")

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

print(f"\n✓ 数据加载成功")
print(f"  PDB: {batch.pdb_ids[0]}")
print(f"  残基数: {batch.n_residues[0]}")

# 创建模型
print(f"\n创建Stage1Model（含FK）...")
model = create_stage1_model()
if torch.cuda.is_available():
    model = model.cuda()

# 前向传播
print(f"\n前向传播...")
with torch.no_grad():
    outputs = model(batch, current_step=0)

print(f"\n✓ 前向传播成功！")
print(f"\n输出:")
print(f"  - pred_torsions: {outputs['pred_torsions'].shape}")
print(f"  - atom14_pos: {outputs['atom14_pos'].shape}")
print(f"  - atom14_mask: {outputs['atom14_mask'].shape}")

# 检查FK重建的键长
atom14 = outputs['atom14_pos'][0]  # 第一个样本
N_CA_dist = torch.norm(atom14[:, 1] - atom14[:, 0], dim=-1).mean()
CA_C_dist = torch.norm(atom14[:, 2] - atom14[:, 1], dim=-1).mean()
C_O_dist = torch.norm(atom14[:, 3] - atom14[:, 2], dim=-1).mean()

print(f"\nFK重建键长:")
print(f"  - N-CA: {N_CA_dist.item():.3f} Å")
print(f"  - CA-C: {CA_C_dist.item():.3f} Å")
print(f"  - C-O: {C_O_dist.item():.3f} Å")

print(f"\n✅ FK集成成功！扭转角预测→FK重建→几何损失 流程打通！")
EOF

echo ""
echo "============================================================================"
echo "✅ 测试通过！"
echo "============================================================================"

