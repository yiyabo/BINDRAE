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
- sample_metadata.json: per-sample metadata for training-time bucketing

Usage:
    python scripts/validate_triplets_data.py --data_dir data/apo_holo_triplets

Author: BINDRAE Team
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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


def check_esm_file(path: Path, require_esm_layers: int = 0) -> Tuple[bool, str, Optional[int]]:
    """Check ESM file for validity and return residue count."""
    try:
        import torch
        data = torch.load(path, weights_only=False)
        if 'per_residue' not in data:
            return False, "Missing 'per_residue' key", None
        emb = data['per_residue']
        if emb.ndim != 2:
            return False, f"per_residue must be 2D, got {tuple(emb.shape)}", None
        if torch.isnan(emb).any():
            return False, "NaN in embeddings", None
        if torch.isinf(emb).any():
            return False, "Inf in embeddings", None
        if emb.shape[0] == 0:
            return False, "Empty embeddings", None
        layers = data.get('per_residue_layers')
        if require_esm_layers > 0 and layers is None:
            return False, f"Missing per_residue_layers required K>={require_esm_layers}", None
        if layers is not None:
            if layers.ndim != 3:
                return False, f"per_residue_layers must be 3D, got {tuple(layers.shape)}", None
            if layers.shape[0] != emb.shape[0] or layers.shape[-1] != emb.shape[-1]:
                return False, (
                    f"per_residue_layers shape {tuple(layers.shape)} inconsistent "
                    f"with per_residue {tuple(emb.shape)}"
                ), None
            if require_esm_layers > 0 and layers.shape[1] < require_esm_layers:
                return False, (
                    f"per_residue_layers K={layers.shape[1]} < required {require_esm_layers}"
                ), None
            if torch.isnan(layers).any():
                return False, "NaN in per_residue_layers", None
            if torch.isinf(layers).any():
                return False, "Inf in per_residue_layers", None
        return True, "", int(emb.shape[0])
    except Exception as e:
        return False, f"Load error: {str(e)[:50]}", None


def _resolve_optional_path(path_str: Optional[str], base_dir: Path) -> Optional[Path]:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path_str


def load_sample_ids(path: Path) -> List[str]:
    """Load sample IDs from txt/json while preserving order."""
    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            if 'pdb_ids' in data:
                raw_ids = data['pdb_ids']
            elif 'ids' in data:
                raw_ids = data['ids']
            else:
                raise ValueError(
                    f"Unsupported JSON format in {path}; expected list or dict with 'pdb_ids'/'ids'"
                )
        elif isinstance(data, list):
            raw_ids = data
        else:
            raise ValueError(f"Unsupported JSON format in {path}")
    else:
        with open(path, 'r') as f:
            raw_ids = [line.strip() for line in f if line.strip()]

    sample_ids = []
    seen = set()
    for item in raw_ids:
        if isinstance(item, str):
            sample_id = item.strip()
        elif isinstance(item, dict) and 'id' in item:
            sample_id = str(item['id']).strip()
        else:
            raise ValueError(
                f"Unsupported sample ID entry in {path}: expected string or dict with 'id'"
            )

        if sample_id and sample_id not in seen:
            sample_ids.append(sample_id)
            seen.add(sample_id)

    return sample_ids


def validate_sample(sample_dir: Path,
                    metadata_only: bool = False,
                    require_esm_layers: int = 0) -> Tuple[str, bool, str, Optional[Dict[str, int]]]:
    """
    Validate a single sample directory.
    
    Returns:
        (sample_id, is_valid, error_message, metadata)
    """
    sample_id = sample_dir.name
    
    # Check required files exist
    for fname in REQUIRED_FILES:
        fpath = sample_dir / fname
        if not fpath.exists():
            return sample_id, False, f"Missing {fname}", None
    
    # Check ESM file
    esm_valid, esm_error, n_residues = check_esm_file(
        sample_dir / 'esm.pt',
        require_esm_layers=require_esm_layers,
    )
    if not esm_valid:
        return sample_id, False, f"esm.pt: {esm_error}", None

    # Check ligand coords
    lig_valid, lig_error = check_numpy_file(sample_dir / 'ligand_coords.npy')
    if not lig_valid:
        return sample_id, False, f"ligand_coords.npy: {lig_error}", None

    ligand_coords = np.load(sample_dir / 'ligand_coords.npy')
    metadata = {
        'n_residues': int(n_residues if n_residues is not None else 0),
        'ligand_atoms': int(len(ligand_coords)),
    }

    if metadata_only:
        return sample_id, True, "", metadata

    # Check apo.pdb has complete backbone
    apo_valid, apo_error = check_pdb_has_full_backbone(sample_dir / 'apo.pdb')
    if not apo_valid:
        return sample_id, False, f"apo.pdb: {apo_error}", None
    
    # Check holo.pdb has complete backbone
    holo_valid, holo_error = check_pdb_has_full_backbone(sample_dir / 'holo.pdb')
    if not holo_valid:
        return sample_id, False, f"holo.pdb: {holo_error}", None
    
    # Check torsion files
    for torsion_file in ['torsion_apo.npz', 'torsion_holo.npz']:
        valid, error = check_numpy_file(sample_dir / torsion_file)
        if not valid:
            return sample_id, False, f"{torsion_file}: {error}", None

    return sample_id, True, "", metadata


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
    parser.add_argument('--sample_ids_file', type=str, default=None,
                        help='Optional txt/json file listing sample IDs to validate')
    parser.add_argument('--metadata_only', action='store_true',
                        help='Only extract metadata for a prevalidated sample list')
    parser.add_argument('--require_esm_layers', type=int, default=0,
                        help='Require esm.pt to contain per_residue_layers with at least this K')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    samples_dir = data_dir / 'samples'
    
    if not samples_dir.exists():
        print(f"Error: samples directory not found: {samples_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metadata_only and args.sample_ids_file is None:
        parser.error('--metadata_only requires --sample_ids_file')

    sample_ids_path = _resolve_optional_path(args.sample_ids_file, data_dir)

    if sample_ids_path is not None:
        sample_ids = load_sample_ids(sample_ids_path)
        sample_dirs = [samples_dir / sample_id for sample_id in sample_ids]
        sample_source = sample_ids_path
    else:
        # Get all sample directories
        sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
        sample_source = samples_dir
    
    if args.limit:
        sample_dirs = sample_dirs[:args.limit]
    
    print(f"Validating {len(sample_dirs)} samples from {sample_source}")
    print(f"Using {args.workers} workers")
    if args.metadata_only:
        print("Mode: metadata_only (extract metadata for prevalidated sample IDs)")
        print(
            "[WARN] metadata_only does not perform the full apo/holo/torsion validation "
            "used for training manifests; use the default mode before launching training."
        )
    if args.require_esm_layers > 0:
        print(f"Requiring ESM last-K layer cache: K>={args.require_esm_layers}")
    
    valid_samples = []
    invalid_samples = []
    sample_metadata = {}
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(validate_sample, d, args.metadata_only, args.require_esm_layers): d
            for d in sample_dirs
        }
        
        with tqdm(total=len(sample_dirs), desc="Validating") as pbar:
            for future in as_completed(futures):
                sample_id, is_valid, error, metadata = future.result()
                
                if is_valid:
                    valid_samples.append(sample_id)
                    if metadata is not None:
                        sample_metadata[sample_id] = metadata
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

    metadata_path = output_dir / 'sample_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(sample_metadata, f, indent=2, sort_keys=True)
    
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
    print(f"Metadata:     {metadata_path}")
    
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
    
    # Create split files if splits directory exists.
    # Skip this in metadata_only mode so we do not imply the sample list has passed
    # the full training-time validation path.
    splits_dir = data_dir / 'splits'
    if splits_dir.exists() and not args.metadata_only:
        valid_set = set(valid_samples)
        for split_file in splits_dir.glob('*.json'):
            if split_file.stem.endswith('_valid'):
                continue
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

            output_split_txt = output_dir / f'{split_name}_valid.txt'
            with open(output_split_txt, 'w') as f:
                for sample_id in filtered_ids:
                    f.write(f"{sample_id}\n")
            
            print(
                f"\n{split_name}: {n_filt}/{n_orig} samples valid "
                f"-> {output_split_txt} (+ {output_split})"
            )


if __name__ == '__main__':
    main()
