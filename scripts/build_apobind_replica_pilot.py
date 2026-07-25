#!/usr/bin/env python3
"""Register a frozen APObind panel and build its independent replica matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_md_replica_matrix import build_matrix  # noqa: E402
from scripts.freeze_apobind_prepared_panel import (  # noqa: E402
    file_sha256,
    load_json,
    load_jsonl,
    sample_id,
    write_immutable,
)
from scripts.register_md_context_replica import (  # noqa: E402
    build_context_record,
    load_npt_metrics,
)
from src.data.md_transition_manifest import validate_transition_record  # noqa: E402


SCHEMA_VERSION = "bindrae_apobind_replica_pilot_plan_v1"
PREPARED_STATE_SCHEMA = "bindrae_apobind_prepared_panel_v1"
DYNAMICS_STATE_SCHEMA = "bindrae_apobind_prepared_panel_dynamics_state_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-transition-manifest", type=Path, required=True)
    parser.add_argument("--prepared-panel", type=Path, required=True)
    parser.add_argument("--prepared-panel-state", type=Path, required=True)
    parser.add_argument("--dynamics-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--expected-systems", type=int, default=8)
    parser.add_argument("--replica-start", type=int, default=1)
    parser.add_argument("--replica-stop", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=2026072600)
    parser.add_argument("--protocol-tag", default="apobind_global_ca_rmsd_fixed_v1")
    parser.add_argument("--pre-equilibration-steps", type=int, default=500)
    parser.add_argument("--pulling-steps", type=int, default=10000)
    parser.add_argument("--endpoint-hold-steps", type=int, default=2000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--rmsd-k-kj-mol-nm2", type=float, default=200000.0)
    parser.add_argument("--final-target-rmsd-nm", type=float, default=0.025)
    parser.add_argument("--min-mapping-fraction", type=float, default=0.95)
    return parser.parse_args()


def index_unique(
    rows: Sequence[Mapping[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in indexed:
            raise ValueError(f"{label} has a missing or duplicate {key}: {value!r}")
        indexed[value] = dict(row)
    return indexed


def ordered_contract_rows(
    candidates: Sequence[Mapping[str, Any]],
    prepared_rows: Sequence[Mapping[str, Any]],
    dynamics_rows: Sequence[Mapping[str, Any]],
    *,
    expected_systems: int,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    if expected_systems < 1:
        raise ValueError("expected-systems must be positive")
    if len(candidates) != expected_systems or len(prepared_rows) != expected_systems:
        raise ValueError(
            f"Expected {expected_systems} candidates and prepared rows, got "
            f"{len(candidates)} and {len(prepared_rows)}"
        )
    candidate_rows = [{**row, "sample_id": sample_id(row)} for row in candidates]
    candidate_by_id = index_unique(candidate_rows, "sample_id", "candidate manifest")
    prepared_by_id = index_unique(prepared_rows, "sample_id", "prepared panel")
    dynamics_by_id = index_unique(dynamics_rows, "sample_id", "dynamics state")
    expected_ids = [str(row["sample_id"]) for row in prepared_rows]
    if set(expected_ids) != set(candidate_by_id):
        raise ValueError("Prepared panel systems do not match the candidate manifest")
    if set(expected_ids) != set(dynamics_by_id):
        raise ValueError("Dynamics systems do not match the prepared panel")

    ordered = []
    for index, system_id in enumerate(expected_ids):
        prepared = prepared_by_id[system_id]
        dynamics = dynamics_by_id[system_id]
        if int(prepared.get("panel_index", -1)) != index:
            raise ValueError(f"Prepared panel index mismatch for {system_id}")
        if int(dynamics.get("panel_index", -1)) != index:
            raise ValueError(f"Dynamics panel index mismatch for {system_id}")
        if int(dynamics.get("nvt_exit_code", -1)) != 0 or dynamics.get(
            "nvt_status"
        ) != "nvt_smoke_passed":
            raise ValueError(f"NVT did not pass for {system_id}")
        if int(dynamics.get("npt_exit_code", -1)) != 0 or dynamics.get(
            "npt_status"
        ) != "npt_smoke_passed":
            raise ValueError(f"NPT did not pass for {system_id}")
        ordered.append((candidate_by_id[system_id], prepared, dynamics))
    return ordered


def validate_hash_contract(
    *,
    prepared_manifest: Path,
    prepared_panel: Path,
    prepared_state_path: Path,
    prepared_state: Mapping[str, Any],
    dynamics_state: Mapping[str, Any],
) -> None:
    if prepared_state.get("schema_version") != PREPARED_STATE_SCHEMA:
        raise ValueError(
            f"Unexpected prepared-panel schema: {prepared_state.get('schema_version')}"
        )
    if prepared_state.get("status") != "ready_for_dynamics_smoke":
        raise ValueError(f"Prepared panel is not ready: {prepared_state.get('status')}")
    outputs = prepared_state.get("outputs") or {}
    if outputs.get("prepared_transition_manifest_sha256") != file_sha256(
        prepared_manifest
    ):
        raise ValueError("Prepared transition manifest SHA256 mismatch")
    if outputs.get("prepared_panel_sha256") != file_sha256(prepared_panel):
        raise ValueError("Prepared panel SHA256 mismatch")
    if dynamics_state.get("schema_version") != DYNAMICS_STATE_SCHEMA:
        raise ValueError(
            f"Unexpected dynamics schema: {dynamics_state.get('schema_version')}"
        )
    if dynamics_state.get("status") != "complete":
        raise ValueError(f"Dynamics panel is not complete: {dynamics_state.get('status')}")
    if dynamics_state.get("prepared_panel_sha256") != file_sha256(prepared_panel):
        raise ValueError("Dynamics prepared-panel SHA256 mismatch")
    if dynamics_state.get("prepared_panel_state_sha256") != file_sha256(
        prepared_state_path
    ):
        raise ValueError("Dynamics prepared-panel-state SHA256 mismatch")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.replica_start < 0 or args.replica_stop < args.replica_start:
        raise ValueError("Replica range must satisfy 0 <= start <= stop")
    base_dir = args.base_dir.resolve()
    candidates = load_jsonl(args.prepared_transition_manifest)
    prepared_rows = load_jsonl(args.prepared_panel)
    prepared_state = load_json(args.prepared_panel_state)
    dynamics_state = load_json(args.dynamics_state)
    validate_hash_contract(
        prepared_manifest=args.prepared_transition_manifest,
        prepared_panel=args.prepared_panel,
        prepared_state_path=args.prepared_panel_state,
        prepared_state=prepared_state,
        dynamics_state=dynamics_state,
    )
    counts = dynamics_state.get("counts") or {}
    for key in ("attempted", "nvt_passed", "npt_passed"):
        if int(counts.get(key, -1)) != args.expected_systems:
            raise ValueError(f"Dynamics count {key} is not {args.expected_systems}")
    ordered = ordered_contract_rows(
        candidates,
        prepared_rows,
        dynamics_state.get("systems") or [],
        expected_systems=args.expected_systems,
    )

    context_records: list[dict[str, Any]] = []
    context_summaries: list[dict[str, Any]] = []
    for candidate, prepared, dynamics in ordered:
        npt_report_path = Path(str(dynamics.get("npt_report") or ""))
        resolved_report_path = (
            npt_report_path
            if npt_report_path.is_absolute()
            else base_dir / npt_report_path
        )
        if not resolved_report_path.is_file():
            raise FileNotFoundError(f"Missing NPT report: {resolved_report_path}")
        npt_dir = npt_report_path.parent
        resolved_npt_dir = resolved_report_path.parent
        metrics_path = resolved_npt_dir / "npt_metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing NPT metrics: {metrics_path}")
        npt_report = load_json(resolved_report_path)
        metrics = load_npt_metrics(metrics_path)
        context = build_context_record(
            candidate,
            npt_dir=npt_dir,
            npt_report=npt_report,
            metrics=metrics,
            replica_index=0,
        )
        issues = validate_transition_record(context, base_dir=base_dir, check_files=True)
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            detail = "; ".join(
                f"{issue.code}: {issue.message}" for issue in errors[:8]
            )
            raise ValueError(f"Context validation failed for {prepared['sample_id']}: {detail}")
        context_records.append(context)
        context_summaries.append(
            {
                "panel_index": prepared["panel_index"],
                "sample_id": prepared["sample_id"],
                "transition_id": context["transition_id"],
                "npt_report": str(npt_report_path),
                "n_frames": context["trajectory"]["n_frames"],
                "frame_interval_ps": context["trajectory"]["frame_interval_ps"],
                "warnings": [
                    issue.to_dict() for issue in issues if issue.severity == "warning"
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_manifest = args.output_dir / "context_manifest.jsonl"
    write_immutable(
        context_manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in context_records),
    )
    matrix_args = argparse.Namespace(
        candidate_manifest=args.prepared_transition_manifest,
        context_manifest=context_manifest,
        output_dir=args.output_dir,
        replica_start=args.replica_start,
        replica_stop=args.replica_stop,
        seed_base=args.seed_base,
        protocol_tag=args.protocol_tag,
        pre_equilibration_steps=args.pre_equilibration_steps,
        pulling_steps=args.pulling_steps,
        endpoint_hold_steps=args.endpoint_hold_steps,
        report_interval=args.report_interval,
        rmsd_k_kj_mol_nm2=args.rmsd_k_kj_mol_nm2,
        final_target_rmsd_nm=args.final_target_rmsd_nm,
        min_mapping_fraction=args.min_mapping_fraction,
    )
    matrix_summary = build_matrix(matrix_args)
    expected_tasks = args.expected_systems * (
        args.replica_stop - args.replica_start + 1
    )
    if matrix_summary["systems"] != args.expected_systems or matrix_summary[
        "tasks"
    ] != expected_tasks:
        raise ValueError(f"Unexpected replica matrix summary: {matrix_summary}")

    matrix_path = Path(str(matrix_summary["matrix"]))
    replica_candidates = Path(str(matrix_summary["replica_candidates"]))
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_to_launch",
        "inputs": {
            "prepared_transition_manifest": str(args.prepared_transition_manifest),
            "prepared_transition_manifest_sha256": file_sha256(
                args.prepared_transition_manifest
            ),
            "prepared_panel": str(args.prepared_panel),
            "prepared_panel_sha256": file_sha256(args.prepared_panel),
            "prepared_panel_state": str(args.prepared_panel_state),
            "prepared_panel_state_sha256": file_sha256(args.prepared_panel_state),
            "dynamics_state": str(args.dynamics_state),
            "dynamics_state_sha256": file_sha256(args.dynamics_state),
        },
        "counts": {
            "systems": args.expected_systems,
            "replicas_per_system": args.replica_stop - args.replica_start + 1,
            "tasks": expected_tasks,
        },
        "contexts": context_summaries,
        "protocol": matrix_summary["protocol"],
        "outputs": {
            "context_manifest": str(context_manifest),
            "context_manifest_sha256": file_sha256(context_manifest),
            "replica_matrix": str(matrix_path),
            "replica_matrix_sha256": file_sha256(matrix_path),
            "replica_candidates": str(replica_candidates),
            "replica_candidates_sha256": file_sha256(replica_candidates),
        },
        "claim_boundary": (
            "This plan creates independent biased silver-path attempts. Only replicas "
            "that pass pull, path, atomistic, and target-export gates can contribute "
            "to consensus Path-3 supervision; no kinetic interpretation is allowed."
        ),
    }
    write_immutable(
        args.output_dir / "pilot_plan.json",
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
    )
    return plan


def main() -> None:
    plan = run(parse_args())
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
