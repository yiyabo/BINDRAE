#!/usr/bin/env python3
"""
Regenerate index.json and splits from existing samples/ directory.
Useful if prepare_ahojdb_triplets.py was run in batches or checked partially.
"""

import json
import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="dataset root (apo_holo_triplets)")
    parser.add_argument("--split", type=str, default="0.9,0.05,0.05", help="split ratios")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    samples_dir = data_dir / "samples"
    
    if not samples_dir.exists():
        print(f"Error: {samples_dir} does not exist.")
        return

    valid_ids = []
    missing_stats = {}  # 统计缺失文件
    total_dirs = 0
    
    print(f"Scanning {samples_dir}...")
    
    required_files = [
        "esm.pt",           # ESM-2 features (必需)
        "apo.pdb",          # Apo structure
        "holo.pdb",         # Holo structure
        "ligand.sdf",       # Ligand
        "ligand_coords.npy", # Ligand coordinates
        "torsion_apo.npz",  # Apo torsion angles
        "torsion_holo.npz", # Holo torsion angles
        "meta.json",        # Metadata
    ]
    
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        total_dirs += 1
        
        # Check all required files
        missing = [f for f in required_files if not (sample_dir / f).exists()]
        
        if not missing:
            valid_ids.append(sample_dir.name)
        else:
            for f in missing:
                missing_stats[f] = missing_stats.get(f, 0) + 1
    
    print(f"\n=== 数据统计 ===")
    print(f"总样本目录: {total_dirs}")
    print(f"有效样本: {len(valid_ids)}")
    print(f"无效样本: {total_dirs - len(valid_ids)}")
    if missing_stats:
        print(f"\n缺失文件统计:")
        for f, count in sorted(missing_stats.items(), key=lambda x: -x[1]):
            print(f"  {f}: {count} 个样本缺失")
    
    # Write index.json
    with (data_dir / "index.json").open("w") as f:
        json.dump(valid_ids, f, indent=2)
        
    # Splits
    splits = [float(x) for x in args.split.split(",")]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(valid_ids)).tolist()
    n_train = int(len(valid_ids) * splits[0])
    n_val = int(len(valid_ids) * splits[1])
    
    train_ids = [valid_ids[i] for i in perm[:n_train]]
    val_ids = [valid_ids[i] for i in perm[n_train:n_train + n_val]]
    test_ids = [valid_ids[i] for i in perm[n_train + n_val:]]
    
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    (splits_dir / "train.json").write_text(json.dumps(train_ids, indent=2))
    (splits_dir / "val.json").write_text(json.dumps(val_ids, indent=2))
    (splits_dir / "test.json").write_text(json.dumps(test_ids, indent=2))
    
    print("Regenerated index.json and splits.")

if __name__ == "__main__":
    main()
