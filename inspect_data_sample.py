#!/usr/bin/env python3
"""检查单个样本的数据格式"""

import numpy as np
import torch
from pathlib import Path
import os

# 自动检测项目根目录
script_dir = Path(__file__).resolve().parent
base_dir = script_dir / "data" / "casf2016"

# 如果不存在，尝试其他可能的路径
if not base_dir.exists():
    # 尝试当前目录
    base_dir = Path.cwd() / "data" / "casf2016"

print(f"数据目录: {base_dir}")
print(f"存在: {base_dir.exists()}\n")

features_dir = base_dir / "processed" / "features"
pockets_dir = base_dir / "processed" / "pockets"
complexes_dir = base_dir / "complexes"

# 检查第一个样本
pdb_id = "1a30"

print("=" * 80)
print(f"检查样本: {pdb_id}")
print("=" * 80)

# 1. ESM特征
esm_file = features_dir / f"{pdb_id}_esm.pt"
if esm_file.exists():
    esm_data = torch.load(esm_file)
    print(f"\n✓ ESM特征:")
    for k, v in esm_data.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: {v.shape}, dtype={v.dtype}")
        else:
            print(f"  - {k}: {type(v).__name__} = {v if not isinstance(v, str) or len(v)<50 else v[:50]+'...'}")

# 2. 配体坐标
ligand_coords = features_dir / f"{pdb_id}_ligand_coords.npy"
if ligand_coords.exists():
    coords = np.load(ligand_coords)
    print(f"\n✓ 配体坐标: {coords.shape}, dtype={coords.dtype}")

# 3. 扭转角
torsions = features_dir / f"{pdb_id}_torsions.npz"
if torsions.exists():
    tor_data = np.load(torsions)
    print(f"\n✓ 扭转角:")
    for k in tor_data.files:
        print(f"  - {k}: {tor_data[k].shape}, dtype={tor_data[k].dtype}")

# 4. 口袋权重
w_res = pockets_dir / f"{pdb_id}_w_res.npy"
if w_res.exists():
    weights = np.load(w_res)
    print(f"\n✓ 口袋权重: {weights.shape}, dtype={weights.dtype}")
    print(f"  - 范围: [{weights.min():.3f}, {weights.max():.3f}]")

# 5. PDB文件（检查是否存在）
protein_pdb = complexes_dir / pdb_id / "protein.pdb"
print(f"\n✓ PDB文件: {protein_pdb.exists()}")

print("\n" + "=" * 80)

