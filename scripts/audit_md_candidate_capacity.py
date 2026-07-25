#!/usr/bin/env python3
"""Audit leakage-clean, previously unattempted capacity for MD acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "bindrae_md_candidate_capacity_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--leakage-clean-sample-list", type=Path, required=True)
    parser.add_argument("--leakage-report", type=Path, required=True)
    parser.add_argument(
        "--attempted-candidate-manifest",
        type=Path,
        action="append",
        required=True,
        help=(
            "Prior candidate manifest whose ordered apo/holo PDB pairs are excluded. "
            "May be passed more than once."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-planning-pool", type=int, default=3000)
    parser.add_argument("--mismatch-smoke-count", type=int, default=4)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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


def read_sample_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(record: Mapping[str, Any]) -> str:
    endpoints = dict(record.get("endpoints") or {})
    parents = {
        Path(str(endpoints.get(key) or "")).parent.name
        for key in ("apo_structure_path", "holo_structure_path")
    }
    parents.discard("")
    if len(parents) != 1:
        raise ValueError(
            f"Could not infer one sample ID for {record.get('transition_id')}: "
            f"{sorted(parents)}"
        )
    return next(iter(parents))


def endpoint_pair(record: Mapping[str, Any]) -> tuple[str, str]:
    endpoints = dict(record.get("endpoints") or {})
    apo = str(endpoints.get("apo_pdb_id") or "").strip().upper()
    holo = str(endpoints.get("holo_pdb_id") or "").strip().upper()
    if not apo or not holo:
        raise ValueError(
            f"Missing apo/holo PDB IDs for {record.get('transition_id')}"
        )
    return apo, holo


def write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite different audit artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    write_immutable(
        path,
        "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records),
    )


def _load_leakage_exclusions(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"Expected an object in leakage report {path}")
    reasons: dict[str, list[str]] = {}
    for row in report.get("excluded", []):
        if not isinstance(row, dict) or not row.get("sample_id"):
            raise ValueError(f"Malformed exclusion row in {path}: {row!r}")
        row_reasons = [str(value) for value in row.get("reasons", []) if str(value)]
        if not row_reasons:
            raise ValueError(f"Exclusion row has no reasons in {path}: {row!r}")
        reasons[str(row["sample_id"])] = row_reasons
    return reasons, report


def _validate_unique_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[tuple[str, str]]]:
    sample_ids = [sample_id(record) for record in candidates]
    endpoint_pairs = [endpoint_pair(record) for record in candidates]
    duplicate_samples = sorted(
        value for value, count in Counter(sample_ids).items() if count > 1
    )
    duplicate_pairs = sorted(
        value for value, count in Counter(endpoint_pairs).items() if count > 1
    )
    if duplicate_samples:
        raise ValueError(f"Candidate manifest repeats sample IDs: {duplicate_samples[:8]}")
    if duplicate_pairs:
        raise ValueError(f"Candidate manifest repeats endpoint pairs: {duplicate_pairs[:8]}")
    return sample_ids, endpoint_pairs


def _mismatch_values(record: Mapping[str, Any]) -> tuple[int, int, float] | None:
    screening = dict(record.get("screening") or {})
    apo_raw = screening.get("apo_n_residues")
    holo_raw = screening.get("holo_n_residues")
    mapping_raw = screening.get("residue_mapping_fraction")
    if apo_raw is None or holo_raw is None or mapping_raw is None:
        return None
    apo = int(apo_raw)
    holo = int(holo_raw)
    if apo == holo:
        return None
    return apo, holo, float(mapping_raw)


def _mismatch_sort_key(record: Mapping[str, Any]) -> tuple[float, int, int, str]:
    values = _mismatch_values(record)
    if values is None:
        raise ValueError("Mismatch sort requested for an equal-count candidate")
    apo, holo, mapping_fraction = values
    screening = dict(record.get("screening") or {})
    selection_rank = int(screening.get("selection_rank") or 10**9)
    return mapping_fraction, -abs(apo - holo), selection_rank, sample_id(record)


def audit_capacity(args: argparse.Namespace) -> dict[str, Any]:
    if args.minimum_planning_pool <= 0:
        raise ValueError("--minimum-planning-pool must be positive")
    if args.mismatch_smoke_count <= 0:
        raise ValueError("--mismatch-smoke-count must be positive")

    candidates = load_jsonl(args.candidate_manifest)
    candidate_ids, candidate_pairs = _validate_unique_candidates(candidates)
    candidate_id_set = set(candidate_ids)
    clean_ids = read_sample_ids(args.leakage_clean_sample_list)
    unknown_clean_ids = sorted(set(clean_ids) - candidate_id_set)
    if unknown_clean_ids:
        raise ValueError(
            "Leakage-clean list contains IDs absent from the candidate manifest: "
            f"{unknown_clean_ids[:8]}"
        )
    clean_id_set = set(clean_ids)
    leakage_reasons, leakage_report = _load_leakage_exclusions(args.leakage_report)
    reported_retained = (leakage_report.get("counts") or {}).get("retained")
    if reported_retained is not None and int(reported_retained) != len(clean_ids):
        raise ValueError(
            f"Leakage report retained={reported_retained}, but list has {len(clean_ids)} IDs"
        )
    survived_reported_exclusions = sorted(clean_id_set & set(leakage_reasons))
    if survived_reported_exclusions:
        raise ValueError(
            "Leakage-clean list contains report-excluded IDs: "
            f"{survived_reported_exclusions[:8]}"
        )

    attempted_records_by_path = {
        str(path): load_jsonl(path) for path in args.attempted_candidate_manifest
    }
    attempted_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path, records in attempted_records_by_path.items():
        for record in records:
            attempted_sources[endpoint_pair(record)].add(path)

    excluded: list[dict[str, Any]] = []
    novel: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for record, system_sample_id, pair in zip(candidates, candidate_ids, candidate_pairs):
        reasons: list[str] = []
        if system_sample_id not in clean_id_set:
            reasons.extend(
                leakage_reasons.get(system_sample_id, ["not_in_leakage_clean_set"])
            )
        if pair in attempted_sources:
            reasons.append("previously_attempted_endpoint_pair")
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            reason_counts.update(reasons)
            excluded.append(
                {
                    "sample_id": system_sample_id,
                    "transition_id": record.get("transition_id"),
                    "endpoint_pair": list(pair),
                    "reasons": reasons,
                    "attempted_candidate_manifests": sorted(
                        attempted_sources.get(pair, set())
                    ),
                }
            )
        else:
            novel.append(dict(record))

    novel_pairs = {endpoint_pair(record) for record in novel}
    if len(novel_pairs) != len(novel):
        raise RuntimeError("Novel candidate manifest is not endpoint-pair unique")

    mismatch_candidates = sorted(
        (record for record in novel if _mismatch_values(record) is not None),
        key=_mismatch_sort_key,
    )
    mismatch_smoke = mismatch_candidates[: args.mismatch_smoke_count]
    shortfall = max(args.minimum_planning_pool - len(novel), 0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    novel_manifest = args.output_dir / "novel_transition_manifest.jsonl"
    novel_sample_list = args.output_dir / "novel_sample_ids.txt"
    excluded_path = args.output_dir / "excluded_candidates.jsonl"
    mismatch_manifest = args.output_dir / "mismatched_residue_smoke_manifest.jsonl"
    mismatch_sample_list = args.output_dir / "mismatched_residue_smoke_sample_ids.txt"
    write_jsonl(novel_manifest, novel)
    write_immutable(
        novel_sample_list,
        "".join(f"{sample_id(record)}\n" for record in novel),
    )
    write_jsonl(excluded_path, excluded)
    write_jsonl(mismatch_manifest, mismatch_smoke)
    write_immutable(
        mismatch_sample_list,
        "".join(f"{sample_id(record)}\n" for record in mismatch_smoke),
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": file_sha256(args.candidate_manifest),
        "leakage_clean_sample_list": str(args.leakage_clean_sample_list),
        "leakage_clean_sample_list_sha256": file_sha256(
            args.leakage_clean_sample_list
        ),
        "leakage_report": str(args.leakage_report),
        "leakage_report_sha256": file_sha256(args.leakage_report),
        "attempted_candidate_manifests": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "records": len(attempted_records_by_path[str(path)]),
            }
            for path in args.attempted_candidate_manifest
        ],
        "criteria": {
            "endpoint_pair": "ordered_uppercase_apo_holo_pdb_ids",
            "minimum_planning_pool": args.minimum_planning_pool,
            "mismatch_smoke_count": args.mismatch_smoke_count,
            "mismatch_smoke_selection": (
                "lowest_mapping_fraction_then_largest_residue_count_delta"
            ),
        },
        "counts": {
            "input_candidates": len(candidates),
            "input_unique_endpoint_pairs": len(set(candidate_pairs)),
            "leakage_clean_candidates": len(clean_ids),
            "attempted_manifest_records": sum(
                len(records) for records in attempted_records_by_path.values()
            ),
            "attempted_unique_endpoint_pairs": len(attempted_sources),
            "excluded_candidates": len(excluded),
            "novel_candidates": len(novel),
            "novel_unique_endpoint_pairs": len(novel_pairs),
            "planning_pool_shortfall": shortfall,
            "mismatched_novel_candidates": len(mismatch_candidates),
            "mismatch_smoke_selected": len(mismatch_smoke),
            "mismatch_smoke_shortfall": max(
                args.mismatch_smoke_count - len(mismatch_smoke), 0
            ),
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "gate": {
            "minimum_planning_pool_passed": shortfall == 0,
            "mismatch_smoke_manifest_ready": len(mismatch_smoke)
            == args.mismatch_smoke_count,
        },
        "files": {
            "novel_transition_manifest": str(novel_manifest),
            "novel_sample_ids": str(novel_sample_list),
            "excluded_candidates": str(excluded_path),
            "mismatched_residue_smoke_manifest": str(mismatch_manifest),
            "mismatched_residue_smoke_sample_ids": str(mismatch_sample_list),
        },
    }
    write_immutable(
        args.output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main() -> None:
    summary = audit_capacity(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
