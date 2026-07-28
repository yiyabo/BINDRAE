#!/usr/bin/env python3
"""Validate a finalized replica run and build immutable system consensus targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_md_phase_normal_consensus_cache import (  # noqa: E402
    run as build_consensus_cache,
)

FINALIZATION_SCHEMA = "bindrae_md_replica_finalization_v1"
CONSENSUS_SCHEMA = "md_phase_normal_consensus_v1"
STATE_SCHEMA = "bindrae_md_corpus_consolidation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalization-dir", type=Path, required=True)
    parser.add_argument("--context-summary", type=Path)
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--min-replicas", type=int, default=2)
    parser.add_argument("--min-support-fraction", type=float, default=0.5)
    parser.add_argument("--reference-consensus-systems", type=int)
    parser.add_argument("--precheck-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_counts(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        count = int(raw_count)
        if count < 0 or count != raw_count:
            raise ValueError(f"{label}.{key} must be a non-negative integer")
        if count:
            counts[str(key)] = count
    return counts


def validate_finalization_summary(
    summary: Mapping[str, Any], *, min_replicas: int
) -> dict[str, Any]:
    if summary.get("schema_version") != FINALIZATION_SCHEMA:
        raise ValueError(
            f"Unexpected finalization schema: {summary.get('schema_version')}"
        )
    planned = int(summary.get("planned_replicas", 0))
    if planned <= 0:
        raise ValueError("planned_replicas must be positive")
    outcome_counts = _integer_counts(
        summary.get("outcome_counts"), label="outcome_counts"
    )
    incomplete = sorted(key for key in outcome_counts if key.startswith("incomplete_"))
    if incomplete:
        raise ValueError(f"Finalization contains incomplete outcomes: {incomplete}")
    if sum(outcome_counts.values()) != planned:
        raise ValueError(
            f"Outcome total {sum(outcome_counts.values())} does not match planned {planned}"
        )

    per_system_raw = summary.get("per_system")
    if not isinstance(per_system_raw, Mapping) or not per_system_raw:
        raise ValueError("per_system must be a non-empty object")
    aggregate: Counter[str] = Counter()
    eligible_systems: list[str] = []
    for system_id, raw_counts in per_system_raw.items():
        counts = _integer_counts(raw_counts, label=f"per_system.{system_id}")
        if not counts:
            raise ValueError(f"System has no recorded outcomes: {system_id}")
        aggregate.update(counts)
        if counts.get("target_passed", 0) >= min_replicas:
            eligible_systems.append(str(system_id))
    if dict(sorted(aggregate.items())) != dict(sorted(outcome_counts.items())):
        raise ValueError("Aggregated per-system outcomes do not match outcome_counts")

    passed_targets = int(summary.get("passed_targets", -1))
    if passed_targets != outcome_counts.get("target_passed", 0):
        raise ValueError("passed_targets does not match target_passed outcomes")
    return {
        "planned_replicas": planned,
        "passed_targets": passed_targets,
        "outcome_counts": outcome_counts,
        "systems": len(per_system_raw),
        "eligible_systems": sorted(eligible_systems),
    }


def validate_context_summary(
    summary: Mapping[str, Any], *, expected_systems: int
) -> dict[str, Any]:
    if summary.get("schema_version") != "bindrae_md_context_collection_v1":
        raise ValueError(f"Unexpected context schema: {summary.get('schema_version')}")
    admitted = int(summary.get("admitted_systems", -1))
    if admitted != expected_systems:
        raise ValueError(
            f"Context admitted_systems={admitted} but finalization has {expected_systems} systems"
        )
    return {
        "planned_systems": int(summary.get("planned_systems", 0)),
        "admitted_systems": admitted,
        "rejected_or_failed_systems": int(summary.get("rejected_or_failed_systems", 0)),
        "outcome_counts": summary.get("outcome_counts", {}),
    }


def validate_consensus_summary(
    summary_path: Path,
    *,
    input_cache: Path,
    output_cache: Path,
    expected_system_ids: set[str],
    min_replicas: int,
    min_support_fraction: float,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    if summary.get("schema_version") != CONSENSUS_SCHEMA:
        raise ValueError(
            f"Unexpected consensus schema: {summary.get('schema_version')}"
        )
    if Path(str(summary.get("input_cache"))).resolve() != input_cache.resolve():
        raise ValueError(
            "Consensus input_cache does not match finalized phase-normal cache"
        )
    if Path(str(summary.get("output_cache"))).resolve() != output_cache.resolve():
        raise ValueError("Consensus output_cache does not match requested output")
    if int(summary.get("min_replicas", -1)) != min_replicas:
        raise ValueError("Consensus min_replicas does not match requested contract")
    if (
        abs(float(summary.get("min_support_fraction", -1.0)) - min_support_fraction)
        > 1e-12
    ):
        raise ValueError(
            "Consensus min_support_fraction does not match requested contract"
        )
    consensus_systems = int(summary.get("consensus_systems", -1))
    if consensus_systems != len(expected_system_ids):
        raise ValueError(
            f"Consensus systems={consensus_systems}, expected eligible systems={len(expected_system_ids)}"
        )

    manifest_path = output_cache / "manifest.jsonl"
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    ]
    if len(manifest_rows) != consensus_systems:
        raise ValueError("Consensus manifest size does not match consensus_systems")
    seen: set[str] = set()
    for row in manifest_rows:
        sample_id = str(row.get("sample_id", ""))
        relative_path = str(row.get("relative_path", ""))
        expected_hash = str(row.get("sha256", ""))
        if not sample_id or sample_id in seen:
            raise ValueError(f"Invalid or duplicate consensus sample_id: {sample_id!r}")
        seen.add(sample_id)
        target = output_cache / relative_path
        if not target.is_file() or sha256(target) != expected_hash:
            raise ValueError(f"Consensus artifact hash mismatch: {target}")
    if seen != expected_system_ids:
        raise ValueError("Consensus manifest systems do not match eligible systems")
    return summary


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite different state: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return "written"


def consolidate(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_replicas < 2:
        raise ValueError("min_replicas must be at least 2")
    if not 0.0 < args.min_support_fraction <= 1.0:
        raise ValueError("min_support_fraction must be in (0, 1]")
    if (
        args.reference_consensus_systems is not None
        and args.reference_consensus_systems < 0
    ):
        raise ValueError("reference_consensus_systems must be non-negative")

    finalization_dir = args.finalization_dir
    finalization_path = finalization_dir / "summary.json"
    finalization = load_json(finalization_path)
    audit = validate_finalization_summary(finalization, min_replicas=args.min_replicas)
    context = None
    if args.context_summary is not None:
        context = validate_context_summary(
            load_json(args.context_summary), expected_systems=audit["systems"]
        )

    input_cache = finalization_dir / "phase_normal_cache"
    input_manifest = input_cache / "manifest.jsonl"
    input_rows = [
        line for line in input_manifest.read_text().splitlines() if line.strip()
    ]
    if len(input_rows) != audit["passed_targets"]:
        raise ValueError(
            "Finalized phase-normal manifest size does not match passed_targets"
        )
    eligible_count = len(audit["eligible_systems"])
    plan = {
        "planned_replicas": audit["planned_replicas"],
        "passed_targets": audit["passed_targets"],
        "systems": audit["systems"],
        "eligible_consensus_systems": eligible_count,
        "min_replicas": args.min_replicas,
        "min_support_fraction": args.min_support_fraction,
    }
    if args.precheck_only:
        return {"status": "precheck_passed", **plan}

    output_cache = finalization_dir / "consensus_cache"
    consensus_summary_path = output_cache / "summary.json"
    consensus = None
    if eligible_count:
        if not consensus_summary_path.is_file():
            build_consensus_cache(
                SimpleNamespace(
                    input_cache=input_cache,
                    output_cache=output_cache,
                    min_replicas=args.min_replicas,
                    min_support_fraction=args.min_support_fraction,
                    overwrite=False,
                )
            )
        consensus = validate_consensus_summary(
            consensus_summary_path,
            input_cache=input_cache,
            output_cache=output_cache,
            expected_system_ids=set(audit["eligible_systems"]),
            min_replicas=args.min_replicas,
            min_support_fraction=args.min_support_fraction,
        )

    matrix_path = Path(str(finalization.get("matrix", "")))
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing replica matrix: {matrix_path}")
    reference = None
    if args.reference_consensus_systems is not None:
        reference = {
            "consensus_systems": args.reference_consensus_systems,
            "delta_systems": eligible_count - args.reference_consensus_systems,
            "retained_fraction": (
                eligible_count / args.reference_consensus_systems
                if args.reference_consensus_systems
                else None
            ),
        }
    state = {
        "schema_version": STATE_SCHEMA,
        "status": "complete",
        "contract": {
            "min_replicas": args.min_replicas,
            "min_support_fraction": args.min_support_fraction,
            "replicas_are_repeated_observations": True,
        },
        "context": context,
        "replicas": {
            "planned": audit["planned_replicas"],
            "passed_targets": audit["passed_targets"],
            "outcome_counts": audit["outcome_counts"],
        },
        "consensus": (
            {
                "eligible_systems": eligible_count,
                "consensus_systems": int(consensus["consensus_systems"]),
                "mean_phase_agreement": consensus.get("mean_phase_agreement"),
                "mean_residual_agreement": consensus.get("mean_residual_agreement"),
            }
            if consensus is not None
            else {"eligible_systems": 0, "consensus_systems": 0}
        ),
        "reference_comparison": reference,
        "artifacts": {
            "replica_matrix": str(matrix_path),
            "replica_matrix_sha256": sha256(matrix_path),
            "finalization_summary": str(finalization_path),
            "finalization_summary_sha256": sha256(finalization_path),
            "phase_normal_cache": str(input_cache),
            "consensus_cache": str(output_cache) if consensus is not None else None,
            "consensus_summary_sha256": (
                sha256(consensus_summary_path) if consensus is not None else None
            ),
        },
        "claim_boundary": (
            "Endpoint-conditioned silver path supervision, not physical kinetics. "
            "Only system-level consensus targets are independent training examples."
        ),
    }
    disposition = write_immutable_json(args.state_output, state)
    return {
        "status": "complete",
        "state": str(args.state_output),
        "state_disposition": disposition,
        **plan,
    }


def main() -> None:
    result = consolidate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
