#!/usr/bin/env python3
"""Build an immutable, auditable matrix of independent MD pull replicas."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


SCHEMA_VERSION = "bindrae_md_replica_matrix_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replica-start", type=int, default=1)
    parser.add_argument("--replica-stop", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=60714000)
    parser.add_argument("--protocol-tag", default="global_ca_rmsd_fixed_v1")
    parser.add_argument("--pre-equilibration-steps", type=int, default=500)
    parser.add_argument("--pulling-steps", type=int, default=10000)
    parser.add_argument("--endpoint-hold-steps", type=int, default=2000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--rmsd-k-kj-mol-nm2", type=float, default=200000.0)
    parser.add_argument("--final-target-rmsd-nm", type=float, default=0.025)
    parser.add_argument("--min-mapping-fraction", type=float, default=0.95)
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


def sample_id(record: Mapping[str, Any]) -> str:
    endpoints = dict(record.get("endpoints") or {})
    holo_path = Path(str(endpoints.get("holo_structure_path") or ""))
    if not holo_path.parent.name:
        raise ValueError(f"Record has no usable holo_structure_path: {record}")
    return holo_path.parent.name


def write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite different matrix artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_transition_id(parent_transition_id: str, replica_index: int) -> str:
    stem = parent_transition_id.rsplit(":", 1)[0]
    return f"{stem}:silver-pull-r{replica_index:02d}"


def build_matrix(args: argparse.Namespace) -> Dict[str, Any]:
    if args.replica_start < 0 or args.replica_stop < args.replica_start:
        raise ValueError("Replica range must satisfy 0 <= start <= stop")
    min_mapping_fraction = float(getattr(args, "min_mapping_fraction", 0.95))
    if not 0.0 < min_mapping_fraction <= 1.0:
        raise ValueError("min_mapping_fraction must be in (0, 1]")
    candidates = load_jsonl(args.candidate_manifest)
    contexts = load_jsonl(args.context_manifest)
    candidate_by_sample = {sample_id(record): record for record in candidates}
    if len(candidate_by_sample) != len(candidates):
        raise ValueError("Candidate manifest contains duplicate sample IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    replica_manifest = args.output_dir / "replica_candidates.jsonl"
    matrix_path = args.output_dir / "replica_matrix.jsonl"
    matrix: List[Dict[str, Any]] = []
    replica_candidates: List[Dict[str, Any]] = []
    protocol = {
        "tag": args.protocol_tag,
        "pre_equilibration_steps": args.pre_equilibration_steps,
        "pulling_steps": args.pulling_steps,
        "endpoint_hold_steps": args.endpoint_hold_steps,
        "report_interval": args.report_interval,
        "rmsd_k_kj_mol_nm2": args.rmsd_k_kj_mol_nm2,
        "final_target_rmsd_nm": args.final_target_rmsd_nm,
        "min_mapping_fraction": min_mapping_fraction,
        "resample_initial_velocities": True,
    }

    for system_index, context in enumerate(sorted(contexts, key=sample_id)):
        system_sample_id = sample_id(context)
        if system_sample_id not in candidate_by_sample:
            raise ValueError(f"No selected candidate for prepared system {system_sample_id}")
        candidate = candidate_by_sample[system_sample_id]
        topology_path = Path(
            str((context.get("trajectory") or {}).get("topology_path") or "")
        )
        npt_dir = topology_path.parent
        npt_report_path = npt_dir / "npt_report.json"
        if not npt_report_path.is_file():
            raise FileNotFoundError(f"Missing NPT report: {npt_report_path}")
        npt_report = json.loads(npt_report_path.read_text())
        if npt_report.get("status") != "npt_smoke_passed":
            raise ValueError(f"NPT context did not pass for {system_sample_id}")
        preparation_report = Path(str(npt_report.get("system_dir") or "")) / "preparation_report.json"
        if not preparation_report.is_file():
            raise FileNotFoundError(f"Missing preparation report: {preparation_report}")

        parent_transition_id = str(candidate["transition_id"])
        for replica_index in range(args.replica_start, args.replica_stop + 1):
            transition_id = make_transition_id(parent_transition_id, replica_index)
            seed = args.seed_base + system_index * 100 + replica_index
            replica_sample_id = f"{system_sample_id}__silver_r{replica_index:02d}"
            pull_dir = args.output_dir / "pulls" / system_sample_id / f"replica_{replica_index:02d}"
            target_dir = args.output_dir / "targets" / system_sample_id / f"replica_{replica_index:02d}"

            replica_candidate = copy.deepcopy(candidate)
            replica_candidate["transition_id"] = transition_id
            replica_candidate["status"] = "planned"
            replica_candidate["source_metadata"] = {
                **dict(replica_candidate.get("source_metadata") or {}),
                "parent_transition_id": parent_transition_id,
                "replica_index": replica_index,
                "planned_seed": seed,
                "protocol_tag": args.protocol_tag,
            }
            replica_candidates.append(replica_candidate)
            matrix.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "matrix_index": len(matrix),
                    "system_index": system_index,
                    "sample_id": replica_sample_id,
                    "system_sample_id": system_sample_id,
                    "transition_id": transition_id,
                    "parent_transition_id": parent_transition_id,
                    "replica_index": replica_index,
                    "seed": seed,
                    "candidate_manifest": str(replica_manifest),
                    "npt_dir": str(npt_dir),
                    "preparation_report": str(preparation_report),
                    "pull_dir": str(pull_dir),
                    "target_dir": str(target_dir),
                    "protocol": protocol,
                }
            )

    candidate_text = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in replica_candidates
    )
    matrix_text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in matrix)
    write_immutable(replica_manifest, candidate_text)
    write_immutable(matrix_path, matrix_text)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate_manifest": str(args.candidate_manifest),
        "context_manifest": str(args.context_manifest),
        "systems": len(contexts),
        "new_replicas_per_system": args.replica_stop - args.replica_start + 1,
        "tasks": len(matrix),
        "replica_range": [args.replica_start, args.replica_stop],
        "seed_base": args.seed_base,
        "protocol": protocol,
        "matrix": str(matrix_path),
        "replica_candidates": str(replica_manifest),
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
