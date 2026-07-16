#!/usr/bin/env python3
"""Build an immutable setup/NVT/NPT matrix for unprepared MD pilot systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple


SCHEMA_VERSION = "bindrae_md_context_matrix_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--existing-context-manifest", type=Path)
    parser.add_argument(
        "--exclude-sample-list",
        type=Path,
        help="Optional newline-delimited sample IDs that must not be prepared again.",
    )
    parser.add_argument(
        "--exclude-candidate-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional prior candidate manifest whose apo/holo endpoint pairs "
            "must not be prepared again. May be passed more than once."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=60715000)
    parser.add_argument("--protocol-tag", default="endpoint_context_fixed_v1")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def load_sample_ids(path: Path) -> Set[str]:
    sample_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not sample_ids:
        raise ValueError(f"No sample IDs in {path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return set(sample_ids)


def sample_id(record: Mapping[str, Any]) -> str:
    endpoints = dict(record.get("endpoints") or {})
    path = Path(str(endpoints.get("holo_structure_path") or ""))
    if not path.parent.name:
        raise ValueError(f"Record has no usable holo_structure_path: {record}")
    return path.parent.name


def endpoint_pair(record: Mapping[str, Any]) -> Tuple[str, str]:
    endpoints = dict(record.get("endpoints") or {})
    apo = str(endpoints.get("apo_pdb_id") or "").strip().upper()
    holo = str(endpoints.get("holo_pdb_id") or "").strip().upper()
    if not apo or not holo:
        raise ValueError(
            f"Record has no usable apo/holo PDB identifiers: {record.get('transition_id')}"
        )
    return apo, holo


def write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite different matrix artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def validate_candidate_files(record: Mapping[str, Any]) -> None:
    endpoints = dict(record.get("endpoints") or {})
    required = [
        Path(str(endpoints.get("apo_structure_path") or "")),
        Path(str(endpoints.get("holo_structure_path") or "")),
    ]
    required.append(required[1].parent / "ligand.sdf")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Candidate {record.get('transition_id')} is missing inputs: {missing}"
        )


def build_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    candidates = load_jsonl(args.candidate_manifest)
    existing_context_manifest = getattr(args, "existing_context_manifest", None)
    exclude_sample_list = getattr(args, "exclude_sample_list", None)
    exclude_candidate_manifests = list(
        getattr(args, "exclude_candidate_manifest", None) or []
    )
    contexts = load_jsonl(existing_context_manifest) if existing_context_manifest else []
    prepared: Set[str] = {sample_id(record) for record in contexts}
    excluded_requested = (
        load_sample_ids(exclude_sample_list) if exclude_sample_list else set()
    )
    excluded_endpoint_pairs = {
        endpoint_pair(record)
        for manifest in exclude_candidate_manifests
        for record in load_jsonl(manifest)
    }
    candidate_ids = [sample_id(record) for record in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate manifest contains duplicate sample IDs")
    candidate_id_set = set(candidate_ids)
    unknown = prepared - candidate_id_set
    if unknown:
        raise ValueError(f"Existing contexts are absent from candidate manifest: {sorted(unknown)}")
    excluded = excluded_requested & candidate_id_set
    excluded_not_in_candidates = excluded_requested - candidate_id_set
    excluded_by_endpoint = {
        sample_id(record)
        for record in candidates
        if endpoint_pair(record) in excluded_endpoint_pairs
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "tag": args.protocol_tag,
        "platform": "CPU",
        "setup": {
            "solvent_minimization_iterations": 1000,
            "max_minimization_iterations": 1000,
        },
        "nvt": {
            "heating_steps_per_stage": 250,
            "restrained_equilibration_steps": 1000,
            "unrestrained_nvt_steps": 1000,
        },
        "npt": {
            "restrained_equilibration_steps": 2500,
            "unrestrained_production_steps": 5000,
        },
    }
    matrix: List[Dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        system_sample_id = sample_id(candidate)
        if (
            system_sample_id in prepared
            or system_sample_id in excluded
            or system_sample_id in excluded_by_endpoint
        ):
            continue
        validate_candidate_files(candidate)
        system_root = args.output_dir / "systems" / system_sample_id
        matrix.append(
            {
                "schema_version": SCHEMA_VERSION,
                "matrix_index": len(matrix),
                "candidate_index": candidate_index,
                "candidate_manifest": str(args.candidate_manifest),
                "transition_id": str(candidate["transition_id"]),
                "system_sample_id": system_sample_id,
                "seed": args.seed_base + candidate_index,
                "setup_dir": str(system_root / "setup"),
                "nvt_dir": str(system_root / "nvt"),
                "npt_dir": str(system_root / "npt"),
                "context_record": str(system_root / "context.jsonl"),
                "protocol": protocol,
            }
        )

    if not matrix:
        raise ValueError("All selected candidates already have prepared contexts")
    matrix_path = args.output_dir / "context_matrix.jsonl"
    matrix_text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in matrix)
    write_immutable(matrix_path, matrix_text)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate_manifest": str(args.candidate_manifest),
        "existing_context_manifest": (
            str(existing_context_manifest) if existing_context_manifest else None
        ),
        "exclude_sample_list": str(exclude_sample_list) if exclude_sample_list else None,
        "exclude_candidate_manifests": [
            str(path) for path in exclude_candidate_manifests
        ],
        "selected_systems": len(candidates),
        "already_prepared_systems": len(prepared),
        "excluded_requested_systems": len(excluded_requested),
        "excluded_systems": len(excluded),
        "excluded_sample_ids": sorted(excluded),
        "excluded_not_in_candidates": sorted(excluded_not_in_candidates),
        "excluded_endpoint_pairs_requested": len(excluded_endpoint_pairs),
        "excluded_endpoint_pair_systems": len(excluded_by_endpoint),
        "excluded_endpoint_pair_sample_ids": sorted(excluded_by_endpoint),
        "planned_systems": len(matrix),
        "seed_base": args.seed_base,
        "protocol": protocol,
        "matrix": str(matrix_path),
        "planned_sample_ids": [record["system_sample_id"] for record in matrix],
    }
    write_immutable(
        args.output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build_matrix(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
