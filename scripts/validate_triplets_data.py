#!/usr/bin/env python3
"""
Validate Stage-1 triplet data and generate valid sample list.

This script scans all samples in the data directory and checks:
1. All required files exist (apo.pdb, holo.pdb, esm.pt, torsion_*.npz, ligand_coords.npy)
2. apo.pdb has complete backbone atoms (N, CA, C), not just CA
3. No NaN/Inf values in data files

Outputs:
- valid_samples.txt: list of sample IDs that passed all checks
- invalid_samples.txt: list of sample IDs that failed with reasons

Usage:
    python scripts/validate_triplets_data.py --data_dir data/apo_holo_triplets

Author: BINDRAE Team
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np

# Required files for each sample
REQUIRED_FILES = [
    'apo.pdb',
    'holo.pdb',
    'esm.pt',
    'torsion_apo.npz',
    'torsion_holo.npz',
    'ligand_coords.npy',
]


def check_pdb_has_full_backbone(pdb_path: Path) -> Tuple[bool, str]:
    """
    Check if PDB file has complete backbone atoms (N, CA, C).
    
    Returns:
        (is_valid, error_message)
    """
    try:
        atom_types = set()
        n_residues = 0
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    atom_types.add(atom_name)
                    if atom_name == 'CA':
                        n_residues += 1
        
        if n_residues == 0:
            return False, "No CA atoms found (empty structure)"
        
        # Check for complete backbone
        has_n = 'N' in atom_types
        has_ca = 'CA' in atom_types
        has_c = 'C' in atom_types
        
        if not has_n and has_ca:
            return False, "Only CA atoms (missing N/C backbone)"
        
        if not (has_n and has_ca and has_c):
            missing = []
            if not has_n: missing.append('N')
            if not has_ca: missing.append('CA')
            if not has_c: missing.append('C')
            return False, f"Missing backbone atoms: {', '.join(missing)}"
        
        return True, ""
        
    except Exception as e:
        return False, f"PDB read error: {str(e)[:50]}"


def check_numpy_file(path: Path) -> Tuple[bool, str]:
    """Check numpy file for NaN/Inf values."""
    try:
        if path.suffix == '.npz':
            data = np.load(path)
            for key in data.keys():
                arr = data[key]
                if np.isnan(arr).any():
                    return False, f"NaN in {key}"
                if np.isinf(arr).any():
                    return False, f"Inf in {key}"
        else:
            arr = np.load(path)
            if np.isnan(arr).any():
                return False, "Contains NaN"
            if np.isinf(arr).any():
                return False, "Contains Inf"
        return True, ""
    except Exception as e:
        return False, f"Load error: {str(e)[:50]}"


def check_esm_file(path: Path) -> Tuple[bool, str]:
    """Check ESM file for validity."""
    try:
        import torch
        data = torch.load(path, weights_only=False)
        if 'per_residue' not in data:
            return False, "Missing 'per_residue' key"
        emb = data['per_residue']
        if torch.isnan(emb).any():
            return False, "NaN in embeddings"
        if torch.isinf(emb).any():
            return False, "Inf in embeddings"
        if emb.shape[0] == 0:
            return False, "Empty embeddings"
        return True, ""
    except Exception as e:
        return False, f"Load error: {str(e)[:50]}"


def validate_sample(sample_dir: Path) -> Tuple[str, bool, str]:
    """
    Validate a single sample directory.
    
    Returns:
        (sample_id, is_valid, error_message)
    """
    sample_id = sample_dir.name
    
    # Check required files exist
    for fname in REQUIRED_FILES:
        fpath = sample_dir / fname
        if not fpath.exists():
            return sample_id, False, f"Missing {fname}"
    
    # Check apo.pdb has complete backbone
    apo_valid, apo_error = check_pdb_has_full_backbone(sample_dir / 'apo.pdb')
    if not apo_valid:
        return sample_id, False, f"apo.pdb: {apo_error}"
    
    # Check holo.pdb has complete backbone
    holo_valid, holo_error = check_pdb_has_full_backbone(sample_dir / 'holo.pdb')
    if not holo_valid:
        return sample_id, False, f"holo.pdb: {holo_error}"
    
    # Check ESM file
    esm_valid, esm_error = check_esm_file(sample_dir / 'esm.pt')
    if not esm_valid:
        return sample_id, False, f"esm.pt: {esm_error}"
    
    # Check torsion files
    for torsion_file in ['torsion_apo.npz', 'torsion_holo.npz']:
        valid, error = check_numpy_file(sample_dir / torsion_file)
        if not valid:
            return sample_id, False, f"{torsion_file}: {error}"
    
    # Check ligand coords
    lig_valid, lig_error = check_numpy_file(sample_dir / 'ligand_coords.npy')
    if not lig_valid:
        return sample_id, False, f"ligand_coords.npy: {lig_error}"
    
    return sample_id, True, ""


def main():
    parser = argparse.ArgumentParser(description='Validate Stage-1 triplet data')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to triplet data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_dir)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to check (for testing)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    samples_dir = data_dir / 'samples'
    
    if not samples_dir.exists():
        print(f"Error: samples directory not found: {samples_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all sample directories
    sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    
    if args.limit:
        sample_dirs = sample_dirs[:args.limit]
    
    print(f"Validating {len(sample_dirs)} samples from {samples_dir}")
    print(f"Using {args.workers} workers")
    
    valid_samples = []
    invalid_samples = []
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(validate_sample, d): d for d in sample_dirs}
        
        with tqdm(total=len(sample_dirs), desc="Validating") as pbar:
            for future in as_completed(futures):
                sample_id, is_valid, error = future.result()
                
                if is_valid:
                    valid_samples.append(sample_id)
                else:
                    invalid_samples.append((sample_id, error))
                
                pbar.update(1)
                pbar.set_postfix({
                    'valid': len(valid_samples),
                    'invalid': len(invalid_samples),
                })
    
    # Sort results
    valid_samples.sort()
    invalid_samples.sort(key=lambda x: x[0])
    
    # Write valid samples list
    valid_path = output_dir / 'valid_samples.txt'
    with open(valid_path, 'w') as f:
        for sample_id in valid_samples:
            f.write(f"{sample_id}\n")
    
    # Write invalid samples with reasons
    invalid_path = output_dir / 'invalid_samples.txt'
    with open(invalid_path, 'w') as f:
        for sample_id, error in invalid_samples:
            f.write(f"{sample_id}\t{error}\n")
    
    # Summary
    total = len(sample_dirs)
    n_valid = len(valid_samples)
    n_invalid = len(invalid_samples)
    
    print(f"\n{'='*60}")
    print(f"Validation Summary")
    print(f"{'='*60}")
    print(f"Total samples:   {total:,}")
    print(f"Valid samples:   {n_valid:,} ({100*n_valid/total:.1f}%)")
    print(f"Invalid samples: {n_invalid:,} ({100*n_invalid/total:.1f}%)")
    print(f"{'='*60}")
    print(f"Valid list:   {valid_path}")
    print(f"Invalid list: {invalid_path}")
    
    # Error distribution
    if invalid_samples:
        print(f"\nError Distribution:")
        error_counts = {}
        for _, error in invalid_samples:
            # Extract error type
            if ':' in error:
                error_type = error.split(':')[0]
            else:
                error_type = error
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
    
    # Create split files if splits directory exists
    splits_dir = data_dir / 'splits'
    if splits_dir.exists():
        valid_set = set(valid_samples)
        for split_file in splits_dir.glob('*.json'):
            split_name = split_file.stem
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            # Handle different formats
            if isinstance(split_data, dict) and 'pdb_ids' in split_data:
                original_ids = split_data['pdb_ids']
            elif isinstance(split_data, list):
                original_ids = split_data
            else:
                continue
            
            filtered_ids = [sid for sid in original_ids if sid in valid_set]
            n_orig = len(original_ids)
            n_filt = len(filtered_ids)
            
            # Write filtered split
            output_split = output_dir / 'splits' / f'{split_name}_valid.json'
            output_split.parent.mkdir(parents=True, exist_ok=True)
            with open(output_split, 'w') as f:
                json.dump(filtered_ids, f, indent=2)
            
            print(f"\n{split_name}: {n_filt}/{n_orig} samples valid -> {output_split}")


if __name__ == '__main__':
    main()

