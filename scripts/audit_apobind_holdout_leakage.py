#!/usr/bin/env python3
"""Filter exported APObind candidates against frozen Stage-2 holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_stage2_leakage_clean_subset import (  # noqa: E402
    file_sha256,
    find_holdout_family,
    init_alignment_worker,
    ligand_scaffold,
    read_ids,
)
from src.data.md_transition_manifest import (  # noqa: E402
    audit_transition_manifest,
    load_transition_manifest,
)


SCHEMA_VERSION = "bindrae_apobind_holdout_leakage_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--holdout-sample-list", type=Path, action="append", required=True
    )
    parser.add_argument("--holdout-samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--sequence-identity", type=float, default=0.30)
    parser.add_argument("--sequence-coverage", type=float, default=0.80)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolved(path: str | Path, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base_dir / value


def _sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _endpoint_ids(metadata: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            str(metadata[key]).strip().upper()
            for key in ("apo_pdb", "holo_pdb")
            if metadata.get(key)
        }
    )


def extract_candidate_metadata(task: tuple[str, str, tuple[str, ...]]) -> dict[str, Any]:
    sample_id, sample_dir_raw, manifest_endpoint_ids = task
    sample_dir = Path(sample_dir_raw)
    try:
        metadata = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
        sequence = str(metadata.get("protein_sequence") or "").strip()
        if not sequence:
            raise ValueError(f"{sample_dir / 'meta.json'} has no protein_sequence")
        digest = _sequence_digest(sequence)
        recorded_digest = str(metadata.get("protein_sequence_sha256") or "").strip()
        if recorded_digest and recorded_digest != digest:
            raise ValueError(f"{sample_dir / 'meta.json'} has a stale sequence digest")
        endpoints = _endpoint_ids(metadata)
        if endpoints != sorted(manifest_endpoint_ids):
            raise ValueError(
                f"manifest/meta endpoint mismatch: {sorted(manifest_endpoint_ids)} != {endpoints}"
            )
        return {
            "sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "sequence": sequence,
            "sequence_sha256": digest,
            "scaffold": ligand_scaffold(sample_dir / "ligand.sdf"),
            "endpoint_pdb_ids": endpoints,
            "error": None,
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "sequence": None,
            "sequence_sha256": None,
            "scaffold": None,
            "endpoint_pdb_ids": list(manifest_endpoint_ids),
            "error": f"{type(exc).__name__}: {exc}",
        }


def extract_holdout_metadata(task: tuple[str, str]) -> dict[str, Any]:
    sample_id, samples_dir_raw = task
    sample_dir = Path(samples_dir_raw) / sample_id
    try:
        torsion_path = sample_dir / "torsion_apo.npz"
        with np.load(torsion_path, allow_pickle=False) as data:
            if "sequence_str" not in data:
                raise KeyError(f"{torsion_path} missing sequence_str")
            sequence = str(np.asarray(data["sequence_str"]).item()).strip()
            if "residue_keys" in data and len(data["residue_keys"]) != len(sequence):
                raise ValueError(
                    f"{torsion_path} residue keys {len(data['residue_keys'])} "
                    f"!= sequence length {len(sequence)}"
                )
        if not sequence:
            raise ValueError(f"{torsion_path} contains an empty sequence")
        metadata = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
        return {
            "sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "sequence": sequence,
            "sequence_sha256": _sequence_digest(sequence),
            "scaffold": ligand_scaffold(sample_dir / "ligand.sdf"),
            "endpoint_pdb_ids": _endpoint_ids(metadata),
            "error": None,
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "sequence": None,
            "sequence_sha256": None,
            "scaffold": None,
            "endpoint_pdb_ids": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _map_tasks(
    function: Callable[[Any], dict[str, Any]],
    tasks: Sequence[Any],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, tasks, chunksize=8))


def candidate_entries(
    records: Sequence[Mapping[str, Any]], *, base_dir: Path
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for record in records:
        endpoints = record.get("endpoints") or {}
        apo_path = _resolved(str(endpoints.get("apo_structure_path") or ""), base_dir)
        holo_path = _resolved(str(endpoints.get("holo_structure_path") or ""), base_dir)
        if not apo_path.name or not holo_path.name or apo_path.parent != holo_path.parent:
            raise ValueError(
                f"Candidate {record.get('transition_id')} has inconsistent endpoint paths"
            )
        sample_id = apo_path.parent.name
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate or empty candidate sample ID: {sample_id!r}")
        seen_sample_ids.add(sample_id)
        endpoint_ids = tuple(
            sorted(
                {
                    str(endpoints[key]).strip().upper()
                    for key in ("apo_pdb_id", "holo_pdb_id")
                    if endpoints.get(key)
                }
            )
        )
        if len(endpoint_ids) != 2:
            raise ValueError(
                f"Candidate {record.get('transition_id')} needs two endpoint PDB IDs"
            )
        entries.append(
            {
                "sample_id": sample_id,
                "sample_dir": str(apo_path.parent),
                "endpoint_pdb_ids": endpoint_ids,
                "record": dict(record),
            }
        )
    return entries


def family_matches(
    candidate_sequences: Sequence[str],
    holdout_sequences: Sequence[tuple[str, str]],
    *,
    identity_threshold: float,
    coverage_threshold: float,
    workers: int,
) -> dict[str, dict[str, Any] | None]:
    unique_sequences = sorted(set(candidate_sequences))
    if workers <= 1:
        init_alignment_worker(
            list(holdout_sequences), identity_threshold, coverage_threshold
        )
        rows = [find_holdout_family(sequence) for sequence in unique_sequences]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_alignment_worker,
            initargs=(
                list(holdout_sequences),
                identity_threshold,
                coverage_threshold,
            ),
        ) as executor:
            rows = list(executor.map(find_holdout_family, unique_sequences, chunksize=8))
    return {str(row["sequence"]): row["family_match"] for row in rows}


def filter_candidates(
    entries: Sequence[Mapping[str, Any]],
    candidate_metadata: Mapping[str, Mapping[str, Any]],
    holdout_metadata: Mapping[str, Mapping[str, Any]],
    *,
    identity_threshold: float,
    coverage_threshold: float,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    invalid_holdout = [
        dict(row) for row in holdout_metadata.values() if row.get("error") is not None
    ]
    if invalid_holdout:
        raise ValueError(f"Holdout metadata is invalid: {invalid_holdout[:3]}")

    holdout_sequences = sorted(
        (sample_id, str(row["sequence"]))
        for sample_id, row in holdout_metadata.items()
    )
    matches = family_matches(
        [
            str(row["sequence"])
            for row in candidate_metadata.values()
            if row.get("error") is None
        ],
        holdout_sequences,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
        workers=workers,
    )
    holdout_scaffolds = {
        str(row["scaffold"]): sample_id
        for sample_id, row in sorted(holdout_metadata.items())
    }
    endpoint_to_holdouts: dict[str, list[str]] = {}
    for sample_id, row in sorted(holdout_metadata.items()):
        for pdb_id in row.get("endpoint_pdb_ids") or []:
            endpoint_to_holdouts.setdefault(str(pdb_id), []).append(sample_id)

    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for entry in entries:
        sample_id = str(entry["sample_id"])
        metadata = candidate_metadata[sample_id]
        reasons: list[str] = []
        family_match = None
        scaffold_match = None
        endpoint_matches: dict[str, list[str]] = {}
        if metadata.get("error") is not None:
            reasons.append("invalid_candidate_metadata")
        else:
            family_match = matches.get(str(metadata["sequence"]))
            if family_match is not None:
                reasons.append("holdout_protein_family")
            scaffold = str(metadata["scaffold"])
            if scaffold in holdout_scaffolds:
                scaffold_match = {
                    "scaffold": scaffold,
                    "holdout_id": holdout_scaffolds[scaffold],
                }
                reasons.append("holdout_ligand_scaffold")
            endpoint_matches = {
                str(pdb_id): endpoint_to_holdouts[str(pdb_id)]
                for pdb_id in metadata.get("endpoint_pdb_ids") or []
                if str(pdb_id) in endpoint_to_holdouts
            }
            if endpoint_matches:
                reasons.append("holdout_endpoint_pdb")
        audit_row = {
            "sample_id": sample_id,
            "transition_id": entry["record"].get("transition_id"),
            "reasons": reasons,
            "family_match": family_match,
            "scaffold_match": scaffold_match,
            "endpoint_matches": endpoint_matches,
            "sequence_sha256": metadata.get("sequence_sha256"),
            "scaffold": metadata.get("scaffold"),
            "endpoint_pdb_ids": metadata.get("endpoint_pdb_ids"),
            "error": metadata.get("error"),
        }
        if reasons:
            excluded.append(audit_row)
        else:
            retained.append({**audit_row, "record": dict(entry["record"])})
    return retained, excluded


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.sequence_identity <= 1.0:
        raise ValueError("sequence-identity must be in (0, 1]")
    if not 0.0 < args.sequence_coverage <= 1.0:
        raise ValueError("sequence-coverage must be in (0, 1]")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    base_dir = args.base_dir.resolve()
    records, parse_issues = load_transition_manifest(args.candidate_manifest)
    manifest_issues, manifest_summary = audit_transition_manifest(
        records,
        initial_issues=parse_issues,
        base_dir=base_dir,
        check_files=True,
    )
    manifest_errors = [issue for issue in manifest_issues if issue.severity == "error"]
    if manifest_errors:
        raise ValueError(
            f"Candidate manifest has {len(manifest_errors)} validation errors: "
            f"{[issue.code for issue in manifest_errors[:8]]}"
        )
    entries = candidate_entries(records, base_dir=base_dir)
    candidate_tasks = [
        (
            str(entry["sample_id"]),
            str(entry["sample_dir"]),
            tuple(entry["endpoint_pdb_ids"]),
        )
        for entry in entries
    ]
    candidate_rows = _map_tasks(
        extract_candidate_metadata, candidate_tasks, workers=args.workers
    )
    candidate_metadata = {str(row["sample_id"]): row for row in candidate_rows}

    holdout_ids = read_ids(args.holdout_sample_list)
    holdout_rows = _map_tasks(
        extract_holdout_metadata,
        [(sample_id, str(args.holdout_samples_dir)) for sample_id in holdout_ids],
        workers=args.workers,
    )
    holdout_metadata = {str(row["sample_id"]): row for row in holdout_rows}
    retained, excluded = filter_candidates(
        entries,
        candidate_metadata,
        holdout_metadata,
        identity_threshold=args.sequence_identity,
        coverage_threshold=args.sequence_coverage,
        workers=args.workers,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    retained_manifest_path = args.output_dir / "leakage_clean_transition_manifest.jsonl"
    retained_ids_path = args.output_dir / "leakage_clean_sample_ids.txt"
    excluded_path = args.output_dir / "excluded_candidates.jsonl"
    candidate_cache_path = args.output_dir / "candidate_metadata.jsonl"
    holdout_cache_path = args.output_dir / "holdout_metadata.jsonl"
    write_jsonl(retained_manifest_path, [row["record"] for row in retained])
    retained_ids_path.write_text(
        "".join(f"{row['sample_id']}\n" for row in retained), encoding="utf-8"
    )
    write_jsonl(excluded_path, excluded)
    write_jsonl(candidate_cache_path, candidate_rows)
    write_jsonl(holdout_cache_path, holdout_rows)

    reason_counts = Counter(
        reason for row in excluded for reason in row.get("reasons") or []
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "inputs": {
            "candidate_manifest": str(args.candidate_manifest),
            "candidate_manifest_sha256": file_sha256(args.candidate_manifest),
            "holdout_sample_lists": [
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for path in args.holdout_sample_list
            ],
            "holdout_samples_dir": str(args.holdout_samples_dir),
        },
        "criteria": {
            "sequence_alignment": "global_blosum62_gap_open_-10_extend_-0.5",
            "sequence_identity": args.sequence_identity,
            "sequence_coverage": args.sequence_coverage,
            "ligand_scaffold": "canonical_nonisomeric_bemis_murcko",
            "acyclic_scaffold_fallback": "canonical_full_molecule",
            "exact_endpoint_pdb_overlap": "excluded",
        },
        "counts": {
            "candidate_manifest": len(records),
            "holdout": len(holdout_ids),
            "retained": len(retained),
            "excluded": len(excluded),
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "candidate_manifest_audit": manifest_summary,
        "outputs": {
            "retained_manifest": str(retained_manifest_path),
            "retained_manifest_sha256": file_sha256(retained_manifest_path),
            "retained_sample_ids": str(retained_ids_path),
            "retained_sample_ids_sha256": file_sha256(retained_ids_path),
            "excluded_candidates": str(excluded_path),
            "candidate_metadata": str(candidate_cache_path),
            "holdout_metadata": str(holdout_cache_path),
        },
        "audit": {
            "retained_with_reasons": [
                row["sample_id"] for row in retained if row.get("reasons")
            ],
            "passed": all(not row.get("reasons") for row in retained),
        },
        "claim_boundary": (
            "Retained records are endpoint candidates disjoint from the frozen "
            "holdout exclusion sets under the declared family, scaffold, and PDB "
            "criteria. They are not prepared systems or Path-3 supervision."
        ),
    }
    report_path = args.output_dir / "leakage_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
