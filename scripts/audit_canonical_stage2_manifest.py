#!/usr/bin/env python3
"""Build a Stage-2 manifest from current canonical structural caches."""

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.residue_identity import RESIDUE_ALIGNMENT_VERSION


BACKBONE_FILES = ("apo_backbone.npz", "holo_backbone.npz")
TORSION_FILES = ("torsion_apo.npz", "torsion_holo.npz")


def _read_sample_ids(path: Path) -> List[str]:
    sample_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return sample_ids


def _cache_version(data: np.lib.npyio.NpzFile) -> str:
    if "residue_alignment_version" not in data:
        return ""
    return str(np.asarray(data["residue_alignment_version"]).item())


def _validate_backbone(path: Path) -> None:
    if not path.is_file():
        raise ValueError("missing")
    with np.load(path, allow_pickle=False) as data:
        required = {"N", "Ca", "C", "residue_keys", "residue_alignment_version"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing keys {missing}")
        if _cache_version(data) != RESIDUE_ALIGNMENT_VERSION:
            raise ValueError(f"stale version {_cache_version(data)!r}")
        lengths = {len(np.asarray(data[name])) for name in ("N", "Ca", "C", "residue_keys")}
        if len(lengths) != 1 or next(iter(lengths), 0) <= 0:
            raise ValueError(f"invalid lengths {sorted(lengths)}")
        for name in ("N", "Ca", "C"):
            values = np.asarray(data[name])
            if values.ndim != 2 or values.shape[-1] != 3:
                raise ValueError(f"{name} shape {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")


def _validate_torsion(path: Path) -> None:
    if not path.is_file():
        raise ValueError("missing")
    with np.load(path, allow_pickle=False) as data:
        required = {"chi", "chi_mask", "residue_keys", "residue_alignment_version"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing keys {missing}")
        if _cache_version(data) != RESIDUE_ALIGNMENT_VERSION:
            raise ValueError(f"stale version {_cache_version(data)!r}")
        chi = np.asarray(data["chi"])
        chi_mask = np.asarray(data["chi_mask"])
        residue_keys = np.asarray(data["residue_keys"])
        if chi.ndim != 2 or chi.shape[-1] != 4:
            raise ValueError(f"chi shape {chi.shape}")
        if chi_mask.shape != chi.shape:
            raise ValueError(f"chi_mask shape {chi_mask.shape}")
        if len(residue_keys) != len(chi) or len(chi) <= 0:
            raise ValueError(
                f"length mismatch residue_keys={len(residue_keys)} chi={len(chi)}"
            )
        if not np.isfinite(chi).all():
            raise ValueError("chi contains non-finite values")


def _audit_one(task: Tuple[str, str]) -> Tuple[str, bool, List[Dict[str, str]]]:
    sample_id, samples_dir_str = task
    sample_dir = Path(samples_dir_str) / sample_id
    errors: List[Dict[str, str]] = []
    for filename in BACKBONE_FILES:
        try:
            _validate_backbone(sample_dir / filename)
        except Exception as exc:
            errors.append({"file": filename, "error": str(exc)})
    for filename in TORSION_FILES:
        try:
            _validate_torsion(sample_dir / filename)
        except Exception as exc:
            errors.append({"file": filename, "error": str(exc)})
    return sample_id, not errors, errors


def _atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="processed_data/triplets")
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--valid-output", required=True)
    parser.add_argument("--invalid-output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    sample_list = Path(args.sample_list)
    if not sample_list.is_absolute() and not sample_list.exists():
        sample_list = data_dir / sample_list
    sample_ids = _read_sample_ids(sample_list)
    samples_dir = data_dir / "samples"
    tasks = [(sample_id, str(samples_dir)) for sample_id in sample_ids]

    results: Dict[str, Tuple[bool, List[Dict[str, str]]]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sample_id, valid, errors in executor.map(_audit_one, tasks, chunksize=64):
            results[sample_id] = (valid, errors)

    valid_ids = [sample_id for sample_id in sample_ids if results[sample_id][0]]
    invalid_records = [
        json.dumps(
            {"sample_id": sample_id, "errors": results[sample_id][1]},
            sort_keys=True,
        )
        for sample_id in sample_ids
        if not results[sample_id][0]
    ]
    _atomic_write_lines(Path(args.valid_output), valid_ids)
    _atomic_write_lines(Path(args.invalid_output), invalid_records)

    print(
        json.dumps(
            {
                "alignment_version": RESIDUE_ALIGNMENT_VERSION,
                "requested": len(sample_ids),
                "valid": len(valid_ids),
                "invalid": len(invalid_records),
                "valid_output": str(Path(args.valid_output)),
                "invalid_output": str(Path(args.invalid_output)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
