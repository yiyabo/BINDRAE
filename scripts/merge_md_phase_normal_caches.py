#!/usr/bin/env python3
"""Merge validated md_phase_normal cache collections without silent conflicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def _load_manifest(cache_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    records = []
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        for key in ("sample_id", "transition_id", "relative_path", "sha256"):
            if not record.get(key):
                raise ValueError(f"{manifest_path}:{line_number} missing {key}")
        records.append(record)
    return records


def inspect_collection(cache_dir: Path) -> List[Dict[str, Any]]:
    summary_path = cache_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing cache summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema_version") != "md_phase_normal_cache_collection_v1":
        raise ValueError(
            f"{summary_path} schema_version={summary.get('schema_version')!r}"
        )

    records = _load_manifest(cache_dir)
    if int(summary.get("samples", -1)) != len(records):
        raise ValueError(
            f"{cache_dir} summary has {summary.get('samples')} samples, "
            f"manifest has {len(records)}"
        )

    inspected = []
    for record in records:
        source_path = cache_dir / record["relative_path"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing cache target: {source_path}")
        actual_sha = sha256(source_path)
        if actual_sha != record["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {source_path}: {actual_sha} != {record['sha256']}"
            )
        with np.load(source_path, allow_pickle=False) as data:
            schema = str(data["schema_version"].item())
            sample_id = str(data["sample_id"].item())
            transition_id = str(data["transition_id"].item())
            phase_target_mode = (
                str(data["phase_target_mode"].item())
                if "phase_target_mode" in data.files
                else "inferred"
            )
            residual_envelope = (
                str(data["residual_envelope"].item())
                if "residual_envelope" in data.files
                else "poly"
            )
            metric_scales = {
                "rotation": float(data["rotation_metric_scale"].item())
                if "rotation_metric_scale" in data.files
                else 1.0,
                "translation": float(data["translation_metric_scale"].item())
                if "translation_metric_scale" in data.files
                else 1.0,
                "chi": float(data["chi_metric_scale"].item())
                if "chi_metric_scale" in data.files
                else 1.0,
            }
        if schema != "md_phase_normal_v1":
            raise ValueError(f"{source_path} schema_version={schema!r}")
        if sample_id != record["sample_id"] or transition_id != record["transition_id"]:
            raise ValueError(f"Manifest identity mismatch for {source_path}")
        inspected.append(
            {
                **record,
                "phase_target_mode": phase_target_mode,
                "residual_envelope": residual_envelope,
                "metric_scales": metric_scales,
                "source_cache_dir": str(cache_dir),
                "source_path": source_path,
            }
        )
    return inspected


def deduplicate_records(
    records: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    unique: List[Dict[str, Any]] = []
    by_sample: Dict[str, Dict[str, Any]] = {}
    by_transition: Dict[str, Dict[str, Any]] = {}
    by_sha: Dict[str, Dict[str, Any]] = {}
    duplicates = 0

    for record in records:
        matches = {
            id(candidate): candidate
            for candidate in (
                by_sample.get(record["sample_id"]),
                by_transition.get(record["transition_id"]),
                by_sha.get(record["sha256"]),
            )
            if candidate is not None
        }
        if matches:
            if len(matches) != 1:
                raise ValueError(
                    f"Conflicting duplicate identity for {record['sample_id']}"
                )
            existing = next(iter(matches.values()))
            identity = ("sample_id", "transition_id", "sha256")
            if any(existing[key] != record[key] for key in identity):
                raise ValueError(
                    f"Conflicting duplicate for {record['sample_id']}: "
                    f"{existing['source_path']} vs {record['source_path']}"
                )
            duplicates += 1
            continue
        unique.append(record)
        by_sample[record["sample_id"]] = record
        by_transition[record["transition_id"]] = record
        by_sha[record["sha256"]] = record

    return unique, duplicates


def _sum_metric(records: Iterable[Dict[str, Any]], key: str) -> int:
    return sum(int(record.get("audit_metrics", {}).get(key, 0)) for record in records)


def merge_collections(input_dirs: Iterable[Path], output_dir: Path) -> Dict[str, Any]:
    all_records = [
        record
        for cache_dir in input_dirs
        for record in inspect_collection(cache_dir)
    ]
    records, duplicate_count = deduplicate_records(all_records)
    records.sort(key=lambda record: (record["sample_id"], record["transition_id"]))
    phase_target_modes = sorted(
        {str(record["phase_target_mode"]) for record in records}
    )
    if len(phase_target_modes) != 1:
        raise ValueError(f"Mixed phase target modes: {phase_target_modes}")
    residual_envelopes = sorted(
        {str(record["residual_envelope"]) for record in records}
    )
    if len(residual_envelopes) != 1:
        raise ValueError(f"Mixed residual envelopes: {residual_envelopes}")
    metric_contracts = {
        json.dumps(record["metric_scales"], sort_keys=True)
        for record in records
    }
    if len(metric_contracts) != 1:
        raise ValueError(f"Mixed metric scales: {sorted(metric_contracts)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    for record in records:
        source_path = Path(record["source_path"])
        destination = output_dir / source_path.name
        if destination.exists():
            if sha256(destination) != record["sha256"]:
                raise FileExistsError(
                    f"Refusing to overwrite different cache target: {destination}"
                )
        else:
            shutil.copy2(source_path, destination)
        manifest_records.append(
            {
                key: value
                for key, value in record.items()
                if key != "source_path"
            }
            | {"relative_path": destination.name}
        )

    sample_ids = [record["sample_id"] for record in manifest_records]
    active_points = _sum_metric(manifest_records, "active_interior_points")
    confident_phase_points = _sum_metric(manifest_records, "confident_phase_points")
    residual_candidate_points = _sum_metric(
        manifest_records, "residual_candidate_points"
    )
    valid_residual_points = sum(
        int(record["valid_residual_points"]) for record in manifest_records
    )
    base_sample_ids = sorted(
        {sample_id.split("__silver_r", 1)[0] for sample_id in sample_ids}
    )
    summary = {
        "schema_version": "md_phase_normal_cache_collection_v1",
        "samples": len(manifest_records),
        "base_systems": len(base_sample_ids),
        "duplicate_records_removed": duplicate_count,
        "frames": sum(int(record["n_frames"]) for record in manifest_records),
        "residues": sum(int(record["n_residues"]) for record in manifest_records),
        "valid_residual_points": valid_residual_points,
        "active_interior_points": active_points,
        "confident_phase_points": confident_phase_points,
        "phase_supervision_density": (
            confident_phase_points / active_points if active_points else 0.0
        ),
        "residual_candidate_points": residual_candidate_points,
        "residual_supervision_density": (
            valid_residual_points / residual_candidate_points
            if residual_candidate_points
            else 0.0
        ),
        "base_sample_ids": base_sample_ids,
        "sample_ids": sample_ids,
        "phase_target_mode": phase_target_modes[0],
        "residual_envelope": residual_envelopes[0],
        "metric_scales": json.loads(next(iter(metric_contracts))),
    }
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest_records)
    )
    (output_dir / "valid_samples.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in sample_ids)
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = merge_collections(args.input_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
