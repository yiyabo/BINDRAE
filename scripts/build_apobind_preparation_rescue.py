#!/usr/bin/env python3
"""Build a bounded replacement plus minimization-rescue APObind panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.select_apobind_preparation_panel import (  # noqa: E402
    FEATURES,
    file_sha256,
    load_jsonl,
    normalized_features,
    sample_id_from_record,
    write_jsonl,
)
from src.data.md_transition_manifest import (  # noqa: E402
    audit_transition_manifest,
    load_transition_manifest,
)


SCHEMA_VERSION = "bindrae_apobind_preparation_rescue_plan_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leakage-clean-manifest", type=Path, required=True)
    parser.add_argument("--screening-results", type=Path, required=True)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--prior-panel-report", type=Path, required=True)
    parser.add_argument("--failed-sample-id", required=True)
    parser.add_argument("--rescue-sample-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--rescue-max-iterations", type=int, default=2500)
    parser.add_argument("--replacement-max-iterations", type=int, default=500)
    parser.add_argument("--residue-force-threshold", type=float, default=500.0)
    return parser.parse_args()


def choose_replacement(
    rows: Sequence[Mapping[str, Any]],
    *,
    prior_sample_ids: Sequence[str],
    failed_sample_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(row["sample_id"]): dict(row) for row in rows}
    if failed_sample_id not in prior_sample_ids or failed_sample_id not in by_id:
        raise ValueError(f"Failed sample is absent from the prior panel: {failed_sample_id}")
    survivor_ids = [
        sample_id for sample_id in prior_sample_ids if sample_id != failed_sample_id
    ]
    if not survivor_ids:
        raise ValueError("Replacement selection needs at least one surviving panel member")
    missing_survivors = sorted(set(survivor_ids) - set(by_id))
    if missing_survivors:
        raise ValueError(f"Prior panel survivors are missing screening rows: {missing_survivors}")

    failed_category = str(by_id[failed_sample_id]["motion_category"])
    features = normalized_features(list(by_id.values()))
    used_scaffolds = {str(by_id[sample_id]["ligand_scaffold"]) for sample_id in survivor_ids}
    used_endpoints = {
        str(pdb_id)
        for sample_id in survivor_ids
        for pdb_id in by_id[sample_id]["endpoint_pdb_ids"]
    }
    pool = [
        row
        for sample_id, row in by_id.items()
        if sample_id not in prior_sample_ids
        and row.get("eligible") is True
        and str(row["motion_category"]) == failed_category
        and not (set(str(value) for value in row["endpoint_pdb_ids"]) & used_endpoints)
    ]
    if not pool:
        raise ValueError(f"No replacement candidates remain in category {failed_category}")
    new_scaffold_pool = [
        row for row in pool if str(row["ligand_scaffold"]) not in used_scaffolds
    ]
    if new_scaffold_pool:
        pool = new_scaffold_pool

    ranked: list[dict[str, Any]] = []
    for row in pool:
        sample_id = str(row["sample_id"])
        distance = min(
            float(np.linalg.norm(features[sample_id] - features[survivor_id]))
            for survivor_id in survivor_ids
        )
        ranked.append(
            {
                "sample_id": sample_id,
                "motion_category": row["motion_category"],
                "maximin_distance": distance,
                "pilot_score": float(row.get("pilot_score", 0.0)),
                "ligand_scaffold": row["ligand_scaffold"],
                "endpoint_pdb_ids": row["endpoint_pdb_ids"],
                "features": {feature: row[feature] for feature in FEATURES},
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["maximin_distance"]),
            -float(row["pilot_score"]),
            str(row["sample_id"]),
        )
    )
    return ranked[0], ranked


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("rescue_max_iterations", "replacement_max_iterations"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if float(args.residue_force_threshold) <= 0.0:
        raise ValueError("residue-force-threshold must be positive")

    records, parse_issues = load_transition_manifest(args.leakage_clean_manifest)
    issues, manifest_summary = audit_transition_manifest(
        records,
        initial_issues=parse_issues,
        base_dir=args.base_dir.resolve(),
        check_files=True,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise ValueError(
            f"Leakage-clean manifest has {len(errors)} errors: "
            f"{[issue.code for issue in errors[:8]]}"
        )
    records_by_id = {sample_id_from_record(record): dict(record) for record in records}
    screening_by_id = {
        str(row["sample_id"]): row for row in load_jsonl(args.screening_results)
    }
    metadata_by_id = {
        str(row["sample_id"]): row for row in load_jsonl(args.candidate_metadata)
    }
    rows: list[dict[str, Any]] = []
    for sample_id, record in records_by_id.items():
        if sample_id not in screening_by_id or sample_id not in metadata_by_id:
            raise ValueError(f"Missing screening or metadata for {sample_id}")
        row = dict(screening_by_id[sample_id])
        row["ligand_scaffold"] = metadata_by_id[sample_id]["scaffold"]
        row["endpoint_pdb_ids"] = [
            str((record.get("endpoints") or {})[key]).upper()
            for key in ("apo_pdb_id", "holo_pdb_id")
        ]
        rows.append(row)

    prior_report = json.loads(args.prior_panel_report.read_text(encoding="utf-8"))
    prior_sample_ids = [str(row["sample_id"]) for row in prior_report["selected"]]
    if args.rescue_sample_id not in prior_sample_ids:
        raise ValueError("Rescue sample is absent from the prior panel")
    if args.rescue_sample_id == args.failed_sample_id:
        raise ValueError("Rescue and force-field-failed samples must differ")
    replacement, ranked = choose_replacement(
        rows,
        prior_sample_ids=prior_sample_ids,
        failed_sample_id=args.failed_sample_id,
    )
    replacement_id = str(replacement["sample_id"])

    roles = [
        {
            "candidate_index": 0,
            "sample_id": args.rescue_sample_id,
            "role": "longer_minimization_rescue",
            "max_minimization_iterations": int(args.rescue_max_iterations),
            "residue_force_threshold_kj_mol_nm": float(
                args.residue_force_threshold
            ),
        },
        {
            "candidate_index": 1,
            "sample_id": replacement_id,
            "role": "forcefield_unsupported_replacement",
            "replaces_sample_id": args.failed_sample_id,
            "replacement_reason": "openff_2.2.1_unassigned_boron_bonds",
            "max_minimization_iterations": int(args.replacement_max_iterations),
            "residue_force_threshold_kj_mol_nm": float(
                args.residue_force_threshold
            ),
        },
    ]
    manifest_rows = [records_by_id[row["sample_id"]] for row in roles]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "rescue_transition_manifest.jsonl"
    sample_ids_path = args.output_dir / "rescue_sample_ids.txt"
    write_jsonl(manifest_path, manifest_rows)
    sample_ids_path.write_text(
        "".join(f"{row['sample_id']}\n" for row in roles), encoding="utf-8"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_to_launch",
        "inputs": {
            "leakage_clean_manifest": str(args.leakage_clean_manifest),
            "leakage_clean_manifest_sha256": file_sha256(
                args.leakage_clean_manifest
            ),
            "prior_panel_report": str(args.prior_panel_report),
            "prior_panel_report_sha256": file_sha256(args.prior_panel_report),
        },
        "frozen_contract": {
            "residue_force_threshold_kj_mol_nm": float(
                args.residue_force_threshold
            ),
            "solvent_minimization_iterations": 1000,
            "threshold_changed": False,
        },
        "roles": roles,
        "replacement_selection": {
            "algorithm": "same_category_greedy_maximin_v1",
            "failed_sample_id": args.failed_sample_id,
            "selected": replacement,
            "ranking": ranked,
            "features": list(FEATURES),
        },
        "manifest_audit": manifest_summary,
        "outputs": {
            "rescue_manifest": str(manifest_path),
            "rescue_manifest_sha256": file_sha256(manifest_path),
            "rescue_sample_ids": str(sample_ids_path),
            "rescue_sample_ids_sha256": file_sha256(sample_ids_path),
        },
        "claim_boundary": (
            "This plan permits one longer minimization retry without changing the "
            "force threshold and one leakage-clean same-category replacement. It "
            "does not convert either endpoint candidate into Path-3 supervision."
        ),
    }
    report_path = args.output_dir / "rescue_plan.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
