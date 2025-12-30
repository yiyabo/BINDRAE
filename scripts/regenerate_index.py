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
    
    print(f"Scanning {samples_dir}...")
    
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
            
        # Basic validation: check for critical files
        if (sample_dir / "apo.pdb").exists() and \
           (sample_dir / "holo.pdb").exists() and \
           (sample_dir / "ligand.sdf").exists() and \
           (sample_dir / "meta.json").exists():
            valid_ids.append(sample_dir.name)
            
    print(f"Found {len(valid_ids)} valid samples.")
    
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
