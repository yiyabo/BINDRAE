#!/usr/bin/env python3
"""Assemble passed md_phase_normal_v1 targets into one immutable cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_target(directory: Path) -> Dict[str, Any]:
    audit_path = directory / "target_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing target audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed") or audit.get("status") != "md_phase_normal_targets_passed":
        raise ValueError(f"Target directory did not pass: {directory}")
    cache_paths = sorted(directory.glob("*.npz"))
    if len(cache_paths) != 1:
        raise ValueError(f"Expected one NPZ in {directory}, found {len(cache_paths)}")
    cache_path = cache_paths[0]
    with np.load(cache_path, allow_pickle=False) as data:
        schema = str(data["schema_version"].item())
        if schema != "md_phase_normal_v1":
            raise ValueError(f"{cache_path} schema_version={schema!r}")
        sample_id = str(data["sample_id"].item())
        transition_id = str(data["transition_id"].item())
        n_residues = int(data["n_residues"].item())
        n_frames = int(np.asarray(data["t_values"]).size)
        valid_points = int(np.asarray(data["residual_valid_mask"]).sum())
    return {
        "source_dir": str(directory),
        "source_path": cache_path,
        "sample_id": sample_id,
        "transition_id": transition_id,
        "n_residues": n_residues,
        "n_frames": n_frames,
        "valid_residual_points": valid_points,
        "sha256": sha256(cache_path),
        "audit_metrics": audit.get("metrics", {}),
    }


def main() -> None:
    args = parse_args()
    records = [inspect_target(directory) for directory in args.input_dir]
    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs: {sample_ids}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: List[Dict[str, Any]] = []
    for record in records:
        source_path = Path(record.pop("source_path"))
        destination = args.output_dir / source_path.name
        if destination.exists():
            if sha256(destination) != record["sha256"]:
                raise FileExistsError(
                    f"Refusing to overwrite different cache target: {destination}"
                )
        else:
            shutil.copy2(source_path, destination)
        manifest_records.append({**record, "relative_path": destination.name})

    (args.output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest_records)
    )
    (args.output_dir / "valid_samples.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in sample_ids)
    )
    summary = {
        "schema_version": "md_phase_normal_cache_collection_v1",
        "samples": len(records),
        "frames": sum(int(record["n_frames"]) for record in records),
        "residues": sum(int(record["n_residues"]) for record in records),
        "valid_residual_points": sum(
            int(record["valid_residual_points"]) for record in records
        ),
        "sample_ids": sample_ids,
    }
    active_points = sum(
        int(record["audit_metrics"].get("active_interior_points", 0))
        for record in records
    )
    confident_phase_points = sum(
        int(record["audit_metrics"].get("confident_phase_points", 0))
        for record in records
    )
    residual_candidate_points = sum(
        int(record["audit_metrics"].get("residual_candidate_points", 0))
        for record in records
    )
    summary.update(
        {
            "active_interior_points": active_points,
            "confident_phase_points": confident_phase_points,
            "phase_supervision_density": (
                confident_phase_points / active_points if active_points else 0.0
            ),
            "residual_candidate_points": residual_candidate_points,
            "residual_supervision_density": (
                summary["valid_residual_points"] / residual_candidate_points
                if residual_candidate_points else 0.0
            ),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
