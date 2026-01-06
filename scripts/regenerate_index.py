#!/usr/bin/env python3
"""
Regenerate index.json and splits from existing samples/ directory.
Useful if prepare_ahojdb_triplets.py was run in batches or checked partially.
"""

import json
import argparse
import numpy as np
from pathlib import Path


def validate_pdb(pdb_path: Path) -> bool:
    """Check if PDB file can be parsed and has valid backbone atoms."""
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
        
        n_residues = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] != ' ':
                        continue
                    try:
                        _ = residue['N'].get_coord()
                        _ = residue['CA'].get_coord()
                        _ = residue['C'].get_coord()
                        n_residues += 1
                    except KeyError:
                        continue
        
        return n_residues > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="dataset root (apo_holo_triplets)")
    parser.add_argument("--split", type=str, default="0.9,0.05,0.05", help="split ratios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-pdb", action="store_true", help="validate PDB files (slower)")
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
    
    pdb_invalid = 0
    
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        total_dirs += 1
        
        # Check all required files
        missing = [f for f in required_files if not (sample_dir / f).exists()]
        
        if missing:
            for f in missing:
                missing_stats[f] = missing_stats.get(f, 0) + 1
            continue
        
        # Optional: validate PDB files (can parse and have backbone atoms)
        if args.validate_pdb:
            apo_ok = validate_pdb(sample_dir / "apo.pdb")
            holo_ok = validate_pdb(sample_dir / "holo.pdb")
            if not (apo_ok and holo_ok):
                pdb_invalid += 1
                continue
        
        valid_ids.append(sample_dir.name)
    
    print(f"\n=== 数据统计 ===")
    print(f"总样本目录: {total_dirs}")
    print(f"有效样本: {len(valid_ids)}")
    print(f"无效样本: {total_dirs - len(valid_ids)}")
    if missing_stats:
        print(f"\n缺失文件统计:")
        for f, count in sorted(missing_stats.items(), key=lambda x: -x[1]):
            print(f"  {f}: {count} 个样本缺失")
    if args.validate_pdb and pdb_invalid > 0:
        print(f"\nPDB 解析失败: {pdb_invalid} 个样本")
    
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
