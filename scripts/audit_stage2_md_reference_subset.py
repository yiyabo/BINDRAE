#!/usr/bin/env python3
"""Freeze a residue-identity-safe Stage-2 MD-reference evaluation subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2.datasets.dataset_stage2 import ApoHoloBridgeDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_file", required=True)
    parser.add_argument("--md_reference_cache_dir", required=True)
    parser.add_argument("--output_samples", required=True)
    parser.add_argument("--output_audit", required=True)
    parser.add_argument(
        "--require_full_node_mask",
        action="store_true",
        help="Reject systems with any missing production endpoint residue.",
    )
    return parser.parse_args()


def safe_sample_id(sample_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in sample_id)


def reference_paths(cache_dir: Path, sample_id: str):
    safe_id = safe_sample_id(sample_id)
    direct = cache_dir / f"{safe_id}.npz"
    paths = sorted(cache_dir.glob(f"{safe_id}__silver_r*.npz"))
    if direct.is_file():
        paths.insert(0, direct)
    return paths


def main() -> None:
    args = parse_args()
    dataset = ApoHoloBridgeDataset(
        args.data_dir,
        split=args.split,
        valid_samples_file=args.sample_file,
        trust_prechecked_samples=True,
        stage1v2_posterior_feature_mode="none",
        esm_num_layers=1,
    )
    cache_dir = Path(args.md_reference_cache_dir)
    valid_ids = []
    records = []
    for index, sample in enumerate(dataset.samples):
        item = dataset[index]
        sample_id = str(sample["id"])
        n_residues = int(item["n_residues"])
        valid_residues = int(np.asarray(item["node_mask"], dtype=bool).sum())
        expected_hash = str(item["residue_identity_hash"])
        paths = reference_paths(cache_dir, sample_id)
        reasons = []
        replica_masks = []
        if not paths:
            reasons.append("missing_md_reference")
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                if str(data["schema_version"].item()) != "md_phase_normal_v1":
                    reasons.append("schema_mismatch")
                if int(data["n_residues"].item()) != n_residues:
                    reasons.append("residue_count_mismatch")
                cached_hash = (
                    str(data["residue_identity_hash"].item())
                    if "residue_identity_hash" in data
                    else ""
                )
                if cached_hash != expected_hash:
                    reasons.append("residue_identity_mismatch")
                replica_masks.append(int(np.asarray(data["node_mask"], dtype=bool).sum()))
        if valid_residues <= 0:
            reasons.append("empty_production_node_mask")
        if args.require_full_node_mask and valid_residues != n_residues:
            reasons.append("incomplete_production_node_mask")
        reasons = sorted(set(reasons))
        passed = not reasons
        if passed:
            valid_ids.append(sample_id)
        records.append(
            {
                "sample_id": sample_id,
                "passed": passed,
                "reasons": reasons,
                "n_residues": n_residues,
                "production_valid_residues": valid_residues,
                "replicas": len(paths),
                "reference_valid_residues": sorted(set(replica_masks)),
                "residue_identity_hash": expected_hash,
            }
        )

    output_samples = Path(args.output_samples)
    output_samples.parent.mkdir(parents=True, exist_ok=True)
    output_samples.write_text("".join(f"{sample_id}\n" for sample_id in valid_ids))
    result = {
        "schema_version": "stage2_md_reference_subset_audit_v1",
        "source_sample_file": args.sample_file,
        "md_reference_cache_dir": args.md_reference_cache_dir,
        "require_full_node_mask": args.require_full_node_mask,
        "requested_systems": len(dataset),
        "passed_systems": len(valid_ids),
        "failed_systems": len(dataset) - len(valid_ids),
        "output_samples": str(output_samples),
        "records": records,
    }
    output_audit = Path(args.output_audit)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    output_audit.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
