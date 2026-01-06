#!/usr/bin/env python3
"""
Precompute apo/holo backbone coords into .npz for fast dataset loading.

Outputs per sample:
  - apo_backbone.npz (keys: N, Ca, C)
  - holo_backbone.npz (keys: N, Ca, C)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.datasets.dataset_stage1 import ApoHoloTripletDataset, extract_backbone_coords


def save_backbone_npz(path: Path, N: np.ndarray, Ca: np.ndarray, C: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, N=N, Ca=Ca, C=C)


def parse_args():
    parser = argparse.ArgumentParser(description="Cache backbone coords for apo/holo PDBs")
    parser.add_argument("--data_dir", type=str, default="data/apo_holo_triplets",
                        help="Dataset root directory")
    parser.add_argument("--split", type=str, default="train",
                        help="Split to process (train/val/test)")
    parser.add_argument("--index_file", type=str, default=None,
                        help="Optional index file path relative to data_dir")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing backbone caches")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = ApoHoloTripletDataset(
        args.data_dir,
        split=args.split,
        index_file=args.index_file,
        require_atom14=False,
    )

    total = len(dataset.samples)
    ok, skipped, failed = 0, 0, 0

    for sample in tqdm(dataset.samples, total=total, desc="Caching backbone"):
        sample_id = sample.get('id', 'unknown')

        apo_pdb = dataset._resolve_path(sample, 'apo_pdb', 'apo.pdb')
        holo_pdb = dataset._resolve_path(sample, 'holo_pdb', 'holo.pdb')
        if apo_pdb is None or holo_pdb is None:
            failed += 1
            continue

        apo_out = dataset._resolve_path(sample, 'apo_backbone', 'apo_backbone.npz')
        holo_out = dataset._resolve_path(sample, 'holo_backbone', 'holo_backbone.npz')
        if apo_out is None or holo_out is None:
            failed += 1
            continue

        if not args.overwrite and apo_out.exists() and holo_out.exists():
            skipped += 1
            continue

        try:
            N_apo, Ca_apo, C_apo, _ = extract_backbone_coords(apo_pdb)
            N_holo, Ca_holo, C_holo, _ = extract_backbone_coords(holo_pdb)

            if len(N_apo) == 0 or len(N_holo) == 0:
                failed += 1
                continue

            save_backbone_npz(apo_out, N_apo, Ca_apo, C_apo)
            save_backbone_npz(holo_out, N_holo, Ca_holo, C_holo)
            ok += 1
        except Exception:
            failed += 1
            continue

    print(f"\nDone. ok={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
