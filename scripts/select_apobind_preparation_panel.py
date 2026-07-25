#!/usr/bin/env python3
"""Select a balanced, diverse APObind prepared-system smoke panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.md_transition_manifest import (  # noqa: E402
    audit_transition_manifest,
    load_transition_manifest,
)


SCHEMA_VERSION = "bindrae_apobind_preparation_panel_v1"
FEATURES = (
    "n_residues",
    "ligand_heavy_atoms",
    "ca_aligned_rmsd",
    "pocket_ca_rmsd",
    "max_ca_displacement",
    "contact_changes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leakage-clean-manifest", type=Path, required=True)
    parser.add_argument("--screening-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--count", type=int, default=8)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sample_id_from_record(record: Mapping[str, Any]) -> str:
    endpoints = record.get("endpoints") or {}
    apo_path = Path(str(endpoints.get("apo_structure_path") or ""))
    holo_path = Path(str(endpoints.get("holo_structure_path") or ""))
    if not apo_path.name or apo_path.parent != holo_path.parent:
        raise ValueError(
            f"Candidate {record.get('transition_id')} has inconsistent endpoint paths"
        )
    return apo_path.parent.name


def category_quotas(rows: Sequence[Mapping[str, Any]], count: int) -> dict[str, int]:
    capacity = Counter(str(row["motion_category"]) for row in rows)
    categories = sorted(capacity)
    if not categories:
        raise ValueError("No motion categories are available")
    quotas = {
        category: min(count // len(categories), capacity[category])
        for category in categories
    }
    remaining = count - sum(quotas.values())
    while remaining > 0:
        choices = [
            category
            for category in categories
            if quotas[category] < capacity[category]
        ]
        if not choices:
            raise ValueError(f"Only {count - remaining} candidates can satisfy the panel")
        choices.sort(key=lambda category: (quotas[category], category))
        quotas[choices[0]] += 1
        remaining -= 1
    return quotas


def normalized_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    matrix = np.asarray(
        [[float(row[feature]) for feature in FEATURES] for row in rows],
        dtype=np.float64,
    )
    minimum = matrix.min(axis=0)
    span = matrix.max(axis=0) - minimum
    span[span == 0.0] = 1.0
    return {
        str(row["sample_id"]): (values - minimum) / span
        for row, values in zip(rows, matrix)
    }


def select_panel(
    rows: Sequence[Mapping[str, Any]], count: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if count < 1:
        raise ValueError("count must be positive")
    eligible = [dict(row) for row in rows if row.get("eligible") is True]
    if len(eligible) < count:
        raise ValueError(f"Requested {count} candidates but only {len(eligible)} are eligible")
    sample_ids = [str(row["sample_id"]) for row in eligible]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Screening results contain duplicate sample IDs")
    for row in eligible:
        for feature in FEATURES:
            if row.get(feature) is None:
                raise ValueError(f"{row['sample_id']} is missing selection feature {feature}")

    quotas = category_quotas(eligible, count)
    features = normalized_features(eligible)
    by_category = {
        category: [
            row for row in eligible if str(row["motion_category"]) == category
        ]
        for category in quotas
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_scaffolds: set[str] = set()
    used_endpoint_ids: set[str] = set()
    category_counts: Counter[str] = Counter()

    def candidate_key(row: Mapping[str, Any], *, first_in_category: bool) -> tuple[Any, ...]:
        sample_id = str(row["sample_id"])
        if first_in_category or not selected:
            diversity = 0.0
        else:
            diversity = min(
                float(np.linalg.norm(features[sample_id] - features[chosen["sample_id"]]))
                for chosen in selected
            )
        return (-diversity, -float(row.get("pilot_score", 0.0)), sample_id)

    while len(selected) < count:
        progress = False
        for category in sorted(quotas):
            if category_counts[category] >= quotas[category]:
                continue
            pool = [
                row
                for row in by_category[category]
                if str(row["sample_id"]) not in selected_ids
                and not (
                    set(str(value) for value in row.get("endpoint_pdb_ids") or [])
                    & used_endpoint_ids
                )
            ]
            if not pool:
                raise ValueError(f"Cannot fill quota for motion category {category}")
            new_scaffold_pool = [
                row
                for row in pool
                if str(row.get("ligand_scaffold") or "") not in used_scaffolds
            ]
            if new_scaffold_pool:
                pool = new_scaffold_pool
            first_in_category = category_counts[category] == 0
            chosen = sorted(
                pool,
                key=lambda row: candidate_key(
                    row, first_in_category=first_in_category
                ),
            )[0]
            sample_id = str(chosen["sample_id"])
            selected.append(chosen)
            selected_ids.add(sample_id)
            category_counts[category] += 1
            used_scaffolds.add(str(chosen.get("ligand_scaffold") or ""))
            used_endpoint_ids.update(
                str(value) for value in chosen.get("endpoint_pdb_ids") or []
            )
            progress = True
        if not progress:
            raise RuntimeError("Panel selection made no progress")

    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    return selected, quotas


def run(args: argparse.Namespace) -> dict[str, Any]:
    records, parse_issues = load_transition_manifest(args.leakage_clean_manifest)
    manifest_issues, manifest_summary = audit_transition_manifest(
        records,
        initial_issues=parse_issues,
        base_dir=args.base_dir.resolve(),
        check_files=True,
    )
    errors = [issue for issue in manifest_issues if issue.severity == "error"]
    if errors:
        raise ValueError(
            f"Leakage-clean manifest has {len(errors)} errors: "
            f"{[issue.code for issue in errors[:8]]}"
        )
    records_by_id = {sample_id_from_record(record): dict(record) for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("Leakage-clean manifest contains duplicate sample IDs")

    screening_rows = load_jsonl(args.screening_results)
    screening_by_id = {str(row["sample_id"]): row for row in screening_rows}
    missing = sorted(set(records_by_id) - set(screening_by_id))
    if missing:
        raise ValueError(f"Screening results miss {len(missing)} clean candidates: {missing[:8]}")
    merged: list[dict[str, Any]] = []
    for sample_id, record in records_by_id.items():
        row = dict(screening_by_id[sample_id])
        row["endpoint_pdb_ids"] = [
            str((record.get("endpoints") or {})[key]).upper()
            for key in ("apo_pdb_id", "holo_pdb_id")
        ]
        row["ligand_scaffold"] = (record.get("split") or {}).get(
            "ligand_scaffold_group"
        )
        merged.append(row)

    selected, quotas = select_panel(merged, args.count)
    selected_ids = [str(row["sample_id"]) for row in selected]
    selected_records = [records_by_id[sample_id] for sample_id in selected_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "selected_transition_manifest.jsonl"
    sample_ids_path = args.output_dir / "selected_sample_ids.txt"
    write_jsonl(manifest_path, selected_records)
    sample_ids_path.write_text(
        "".join(f"{sample_id}\n" for sample_id in selected_ids), encoding="utf-8"
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "inputs": {
            "leakage_clean_manifest": str(args.leakage_clean_manifest),
            "leakage_clean_manifest_sha256": file_sha256(
                args.leakage_clean_manifest
            ),
            "screening_results": str(args.screening_results),
            "screening_results_sha256": file_sha256(args.screening_results),
        },
        "selection": {
            "count": len(selected),
            "motion_category_quotas": quotas,
            "algorithm": "category_balanced_greedy_maximin_v1",
            "features": list(FEATURES),
            "first_per_category": "highest_pilot_score",
            "ligand_scaffold_preference": "prefer_unseen_then_allow_repeat",
            "shared_endpoint_pdbs_allowed": False,
        },
        "selected": [
            {
                key: row.get(key)
                for key in (
                    "selection_rank",
                    "sample_id",
                    "motion_category",
                    "pilot_score",
                    "n_residues",
                    "ligand_heavy_atoms",
                    "ca_aligned_rmsd",
                    "pocket_ca_rmsd",
                    "max_ca_displacement",
                    "contact_changes",
                    "ligand_scaffold",
                    "endpoint_pdb_ids",
                )
            }
            for row in selected
        ],
        "manifest_audit": manifest_summary,
        "outputs": {
            "selected_manifest": str(manifest_path),
            "selected_manifest_sha256": file_sha256(manifest_path),
            "selected_sample_ids": str(sample_ids_path),
            "selected_sample_ids_sha256": file_sha256(sample_ids_path),
        },
        "claim_boundary": (
            "This category-balanced max-min panel is for prepared-system engineering "
            "coverage. It is not a representative acceptance-rate sample and contains "
            "no MD paths or Path-3 labels."
        ),
    }
    report_path = args.output_dir / "selection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
