#!/usr/bin/env python3
"""Freeze a fully resolved APObind prepared-system panel for dynamics smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "bindrae_apobind_prepared_panel_v1"
PANEL_STATE_SCHEMA = "bindrae_apobind_preparation_panel_state_v1"
RESCUE_STATE_SCHEMA = "bindrae_apobind_preparation_rescue_state_v1"
RESCUE_PLAN_SCHEMA = "bindrae_apobind_preparation_rescue_plan_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--panel-state", type=Path, required=True)
    parser.add_argument("--rescue-manifest", type=Path, required=True)
    parser.add_argument("--rescue-plan", type=Path, required=True)
    parser.add_argument("--rescue-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--expected-count", type=int, default=8)
    parser.add_argument("--residue-force-threshold", type=float, default=500.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty JSON-object lines in {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    write_immutable(path, text)


def write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sample_id(record: Mapping[str, Any]) -> str:
    endpoints = record.get("endpoints") or {}
    apo = Path(str(endpoints.get("apo_structure_path") or ""))
    holo = Path(str(endpoints.get("holo_structure_path") or ""))
    if not apo.parent.name or apo.parent != holo.parent:
        raise ValueError(
            f"Inconsistent endpoint paths for {record.get('transition_id')}: {apo}, {holo}"
        )
    return apo.parent.name


def index_unique(rows: Iterable[Mapping[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in indexed:
            raise ValueError(f"{label} has missing or duplicate {key}: {value!r}")
        indexed[value] = dict(row)
    return indexed


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base_dir / value


def validate_state_manifest(
    state: Mapping[str, Any],
    *,
    schema: str,
    manifest: Path,
    hash_key: str,
    label: str,
) -> None:
    if state.get("schema_version") != schema:
        raise ValueError(f"Unexpected {label} schema: {state.get('schema_version')}")
    observed = str(state.get(hash_key) or "")
    expected = file_sha256(manifest)
    if observed != expected:
        raise ValueError(f"{label} manifest SHA256 mismatch: {observed} != {expected}")


def validate_preparation(
    state_row: Mapping[str, Any],
    *,
    base_dir: Path,
    force_threshold: float,
) -> dict[str, Any]:
    if int(state_row.get("exit_code", -1)) != 0:
        raise ValueError(f"Preparation process failed for {state_row.get('sample_id')}")
    if state_row.get("ready_for_dynamics_smoke") is not True:
        raise ValueError(f"Preparation state is not ready for {state_row.get('sample_id')}")
    report_path = resolve_path(str(state_row.get("preparation_report") or ""), base_dir)
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing preparation report: {report_path}")
    report = load_json(report_path)
    minimization = report.get("minimization") or {}
    maximum = float(minimization.get("maximum_residue_net_force_kj_mol_nm", math.inf))
    threshold = float(
        minimization.get("max_residue_net_force_threshold_kj_mol_nm", math.nan)
    )
    if report.get("status") != "minimized_ready_for_dynamics":
        raise ValueError(f"Preparation report is not ready: {report_path}")
    if minimization.get("ready_for_dynamics_smoke") is not True:
        raise ValueError(f"Preparation minimization gate is false: {report_path}")
    if not math.isclose(threshold, force_threshold, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Preparation threshold changed for {state_row.get('sample_id')}: {threshold}"
        )
    if not math.isfinite(maximum) or maximum > force_threshold:
        raise ValueError(
            f"Preparation force gate failed for {state_row.get('sample_id')}: {maximum}"
        )
    outputs = report.get("outputs") or {}
    required_outputs = ("minimized_pdb", "state_xml", "system_xml", "unsolvated_pdb")
    missing = [
        str(resolve_path(str(outputs.get(key) or ""), base_dir))
        for key in required_outputs
        if not resolve_path(str(outputs.get(key) or ""), base_dir).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Prepared system is missing outputs: {missing}")
    return {
        "preparation_report": str(state_row["preparation_report"]),
        "preparation_dir": str(Path(str(state_row["preparation_report"])).parent),
        "maximum_residue_net_force_kj_mol_nm": maximum,
        "residue_force_threshold_kj_mol_nm": threshold,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_count < 1:
        raise ValueError("expected-count must be positive")
    if args.residue_force_threshold <= 0.0:
        raise ValueError("residue-force-threshold must be positive")
    base_dir = args.base_dir.resolve()
    panel_records = load_jsonl(args.panel_manifest)
    rescue_records = load_jsonl(args.rescue_manifest)
    if len(panel_records) != args.expected_count:
        raise ValueError(
            f"Panel has {len(panel_records)} systems, expected {args.expected_count}"
        )
    panel_records_by_id = index_unique(
        ({**record, "sample_id": sample_id(record)} for record in panel_records),
        "sample_id",
        "panel manifest",
    )
    rescue_records_by_id = index_unique(
        ({**record, "sample_id": sample_id(record)} for record in rescue_records),
        "sample_id",
        "rescue manifest",
    )

    panel_state = load_json(args.panel_state)
    rescue_state = load_json(args.rescue_state)
    rescue_plan = load_json(args.rescue_plan)
    validate_state_manifest(
        panel_state,
        schema=PANEL_STATE_SCHEMA,
        manifest=args.panel_manifest,
        hash_key="candidate_manifest_sha256",
        label="panel state",
    )
    validate_state_manifest(
        rescue_state,
        schema=RESCUE_STATE_SCHEMA,
        manifest=args.rescue_manifest,
        hash_key="manifest_sha256",
        label="rescue state",
    )
    if rescue_plan.get("schema_version") != RESCUE_PLAN_SCHEMA:
        raise ValueError(f"Unexpected rescue plan schema: {rescue_plan.get('schema_version')}")
    if str(rescue_state.get("plan_sha256") or "") != file_sha256(args.rescue_plan):
        raise ValueError("Rescue state plan SHA256 mismatch")

    panel_results = index_unique(panel_state.get("systems") or [], "sample_id", "panel state")
    rescue_results = index_unique(
        rescue_state.get("systems") or [], "sample_id", "rescue state"
    )
    if set(panel_results) != set(panel_records_by_id):
        raise ValueError("Panel state systems do not match the panel manifest")
    if set(rescue_results) != set(rescue_records_by_id):
        raise ValueError("Rescue state systems do not match the rescue manifest")

    retry_by_sample: dict[str, dict[str, Any]] = {}
    replacement_by_failed: dict[str, dict[str, Any]] = {}
    for role in rescue_plan.get("roles") or []:
        role = dict(role)
        role_name = str(role.get("role") or "")
        role_sample = str(role.get("sample_id") or "")
        if role_sample not in rescue_records_by_id:
            raise ValueError(f"Rescue role sample is absent from manifest: {role_sample}")
        if role_name == "longer_minimization_rescue":
            if role_sample in retry_by_sample:
                raise ValueError(f"Duplicate retry role for {role_sample}")
            retry_by_sample[role_sample] = role
        elif role_name == "forcefield_unsupported_replacement":
            failed = str(role.get("replaces_sample_id") or "")
            if failed in replacement_by_failed or failed not in panel_records_by_id:
                raise ValueError(f"Invalid replacement target: {failed}")
            replacement_by_failed[failed] = role
        else:
            raise ValueError(f"Unsupported rescue role: {role_name}")

    accepted: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for panel_index, original_record in enumerate(panel_records):
        original_id = sample_id(original_record)
        result = panel_results[original_id]
        resolved_id = original_id
        resolved_record = original_record
        resolved_result = result
        resolution = "original_preparation"
        if result.get("ready_for_dynamics_smoke") is not True:
            if original_id in retry_by_sample:
                resolved_result = rescue_results[original_id]
                resolution = "longer_minimization_rescue"
            elif original_id in replacement_by_failed:
                role = replacement_by_failed[original_id]
                resolved_id = str(role["sample_id"])
                resolved_record = rescue_records_by_id[resolved_id]
                resolved_result = rescue_results[resolved_id]
                resolution = "forcefield_unsupported_replacement"
            else:
                unresolved.append(
                    {
                        "panel_index": panel_index,
                        "original_sample_id": original_id,
                        "reason": "no_rescue_role",
                    }
                )
                continue
        if resolved_result.get("ready_for_dynamics_smoke") is not True:
            unresolved.append(
                {
                    "panel_index": panel_index,
                    "original_sample_id": original_id,
                    "attempted_sample_id": resolved_id,
                    "resolution": resolution,
                    "reason": "rescue_not_ready",
                    "process_exit_code": resolved_result.get("exit_code"),
                    "status": resolved_result.get("status"),
                }
            )
            continue
        evidence = validate_preparation(
            resolved_result,
            base_dir=base_dir,
            force_threshold=float(args.residue_force_threshold),
        )
        accepted.append(
            {
                "panel_index": panel_index,
                "original_sample_id": original_id,
                "sample_id": resolved_id,
                "transition_id": resolved_record.get("transition_id"),
                "resolution": resolution,
                **evidence,
            }
        )
        accepted_records.append(dict(resolved_record))

    accepted_ids = [row["sample_id"] for row in accepted]
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ValueError(f"Resolved panel contains duplicate systems: {accepted_ids}")
    complete = len(accepted) == args.expected_count and not unresolved
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_manifest = args.output_dir / "prepared_transition_manifest.jsonl"
    prepared_panel = args.output_dir / "prepared_panel.jsonl"
    write_jsonl(prepared_manifest, accepted_records)
    write_jsonl(prepared_panel, accepted)
    state = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_dynamics_smoke" if complete else "incomplete",
        "inputs": {
            "panel_manifest": str(args.panel_manifest),
            "panel_manifest_sha256": file_sha256(args.panel_manifest),
            "panel_state": str(args.panel_state),
            "panel_state_sha256": file_sha256(args.panel_state),
            "rescue_manifest": str(args.rescue_manifest),
            "rescue_manifest_sha256": file_sha256(args.rescue_manifest),
            "rescue_plan": str(args.rescue_plan),
            "rescue_plan_sha256": file_sha256(args.rescue_plan),
            "rescue_state": str(args.rescue_state),
            "rescue_state_sha256": file_sha256(args.rescue_state),
        },
        "frozen_contract": {
            "expected_systems": args.expected_count,
            "residue_force_threshold_kj_mol_nm": args.residue_force_threshold,
            "threshold_changed": False,
        },
        "counts": {
            "original_panel": len(panel_records),
            "prepared_ready": len(accepted),
            "unresolved": len(unresolved),
            "retained_original_preparations": sum(
                row["resolution"] == "original_preparation" for row in accepted
            ),
            "longer_minimization_rescues": sum(
                row["resolution"] == "longer_minimization_rescue" for row in accepted
            ),
            "forcefield_replacements": sum(
                row["resolution"] == "forcefield_unsupported_replacement"
                for row in accepted
            ),
        },
        "systems": accepted,
        "unresolved_slots": unresolved,
        "outputs": {
            "prepared_transition_manifest": str(prepared_manifest),
            "prepared_transition_manifest_sha256": file_sha256(prepared_manifest),
            "prepared_panel": str(prepared_panel),
            "prepared_panel_sha256": file_sha256(prepared_panel),
        },
        "claim_boundary": (
            "This artifact freezes systems that passed the unchanged preparation "
            "gate for short NVT/NPT smoke. It contains no accepted MD replica or "
            "Path-3 supervision."
        ),
    }
    write_immutable(
        args.output_dir / "prepared_panel_state.json",
        json.dumps(state, indent=2, sort_keys=True) + "\n",
    )
    return state


def main() -> None:
    state = run(parse_args())
    print(json.dumps(state, indent=2, sort_keys=True))
    if state["status"] != "ready_for_dynamics_smoke":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
