#!/usr/bin/env python3
"""
Precompute apo/holo backbone coords into .npz for fast dataset loading.

Outputs per sample:
  - apo_backbone.npz (keys: N, Ca, C)
  - holo_backbone.npz (keys: N, Ca, C)
"""

import os
import sys
import concurrent.futures
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.datasets.dataset_stage1 import ApoHoloTripletDataset, extract_backbone_coords
from src.data.residue_identity import RESIDUE_ALIGNMENT_VERSION, residue_keys_to_array


def save_backbone_npz(path: Path, N: np.ndarray, Ca: np.ndarray, C: np.ndarray, 
                       residue_keys: list = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {'N': N, 'Ca': Ca, 'C': C}
    if residue_keys is not None:
        arrays.update(
            residue_keys=residue_keys_to_array(residue_keys),
            residue_alignment_version=np.asarray(RESIDUE_ALIGNMENT_VERSION),
        )
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def backbone_cache_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if 'residue_keys' not in data or 'residue_alignment_version' not in data:
                return False
            return str(np.asarray(data['residue_alignment_version']).item()) == RESIDUE_ALIGNMENT_VERSION
    except Exception:
        return False


def cache_backbone_sample(task):
    sample_id, apo_pdb, holo_pdb, apo_out, holo_out, overwrite = task
    apo_pdb = Path(apo_pdb)
    holo_pdb = Path(holo_pdb)
    apo_out = Path(apo_out)
    holo_out = Path(holo_out)
    if (
        not overwrite
        and backbone_cache_is_current(apo_out)
        and backbone_cache_is_current(holo_out)
    ):
        return "skipped", sample_id, ""
    try:
        N_apo, Ca_apo, C_apo, _, apo_residue_keys = extract_backbone_coords(apo_pdb)
        N_holo, Ca_holo, C_holo, _, holo_residue_keys = extract_backbone_coords(holo_pdb)
        if len(N_apo) == 0 or len(N_holo) == 0:
            return "failed", sample_id, "empty backbone"
        save_backbone_npz(apo_out, N_apo, Ca_apo, C_apo, apo_residue_keys)
        save_backbone_npz(holo_out, N_holo, Ca_holo, C_holo, holo_residue_keys)
        return "ok", sample_id, ""
    except Exception as exc:
        return "failed", sample_id, str(exc)
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
    parser.add_argument("--sample-list", type=str, default=None,
                        help="Optional file with sample IDs to process")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel PDB parsing workers")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional cap after filtering; 0 means all")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = ApoHoloTripletDataset(
        args.data_dir,
        split=args.split,
        index_file=args.index_file,
        require_atom14=False,
    )

    samples = dataset.samples
    if args.sample_list:
        sample_list_path = Path(args.sample_list)
        if not sample_list_path.is_absolute() and not sample_list_path.exists():
            sample_list_path = Path(args.data_dir) / sample_list_path
        if not sample_list_path.exists():
            raise FileNotFoundError(sample_list_path)
        requested = {
            line.strip()
            for line in sample_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        samples = [sample for sample in samples if sample.get('id', '') in requested]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    tasks = []
    for sample in samples:
        sample_id = sample.get('id', 'unknown')
        apo_pdb = dataset._resolve_path(sample, 'apo_pdb', 'apo.pdb')
        holo_pdb = dataset._resolve_path(sample, 'holo_pdb', 'holo.pdb')
        if apo_pdb is None or holo_pdb is None:
            continue
        apo_out = dataset._resolve_path(sample, 'apo_backbone', 'apo_backbone.npz')
        holo_out = dataset._resolve_path(sample, 'holo_backbone', 'holo_backbone.npz')
        if apo_out is None or holo_out is None:
            continue
        tasks.append((sample_id, str(apo_pdb), str(holo_pdb), str(apo_out), str(holo_out), args.overwrite))

    ok = skipped = failed = 0
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = executor.map(cache_backbone_sample, tasks, chunksize=32)
        for status, sample_id, error in tqdm(results, total=len(tasks), desc="Caching backbone"):
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                if len(failures) < 20:
                    failures.append((sample_id, error))

    print(f"\nDone. ok={ok}, skipped={skipped}, failed={failed}")
    if failures:
        print(f"Failure examples: {failures}")


if __name__ == "__main__":
    main()
