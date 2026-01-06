#!/usr/bin/env python3
"""
Regenerate index.json and splits from existing samples/ directory.
Useful if prepare_ahojdb_triplets.py was run in batches or checked partially.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


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


def check_sample(sample_dir: Path, required_files: list, do_validate_pdb: bool) -> dict:
    """
    Check a single sample directory.
    Returns: {'id': str, 'valid': bool, 'missing': list, 'pdb_invalid': bool}
    """
    result = {
        'id': sample_dir.name,
        'valid': True,
        'missing': [],
        'pdb_invalid': False
    }
    
    # Check all required files
    for f in required_files:
        if not (sample_dir / f).exists():
            result['missing'].append(f)
            result['valid'] = False
    
    # Only validate PDB if all files exist
    if result['valid'] and do_validate_pdb:
        apo_ok = validate_pdb(sample_dir / "apo.pdb")
        holo_ok = validate_pdb(sample_dir / "holo.pdb")
        if not (apo_ok and holo_ok):
            result['pdb_invalid'] = True
            result['valid'] = False
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="dataset root (apo_holo_triplets)")
    parser.add_argument("--split", type=str, default="0.9,0.05,0.05", help="split ratios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-pdb", action="store_true", help="validate PDB files (slower)")
    parser.add_argument("--workers", type=int, default=None, 
                        help="number of worker processes (default: auto, max 64)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    samples_dir = data_dir / "samples"
    
    if not samples_dir.exists():
        print(f"Error: {samples_dir} does not exist.")
        return

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
    
    print(f"Scanning {samples_dir}...")
    
    # 获取所有目录
    all_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    total_dirs = len(all_dirs)
    
    # 确定 worker 数量
    if args.workers is not None:
        n_workers = args.workers
    else:
        # 自动选择：CPU 核数，最多 64（避免过多进程开销）
        n_workers = min(cpu_count(), 64)
    
    print(f"使用 {n_workers} 个进程并行处理...")
    
    # 并行处理
    check_fn = partial(check_sample, 
                       required_files=required_files, 
                       do_validate_pdb=args.validate_pdb)
    
    valid_ids = []
    missing_stats = {}
    pdb_invalid = 0
    
    desc = "验证样本 (含PDB解析)" if args.validate_pdb else "检查文件"
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(check_fn, all_dirs, chunksize=100),
            total=total_dirs,
            desc=desc,
            ncols=100
        ))
    
    # 汇总结果
    for r in results:
        if r['valid']:
            valid_ids.append(r['id'])
        else:
            for f in r['missing']:
                missing_stats[f] = missing_stats.get(f, 0) + 1
            if r['pdb_invalid']:
                pdb_invalid += 1
    
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
