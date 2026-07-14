#!/usr/bin/env python3
"""Collect passed context-pipeline records without hiding failed systems."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.md_transition_manifest import validate_transition_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return record


def collect(matrix_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in matrix_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty context matrix: {matrix_path}")
    outcomes: List[Dict[str, Any]] = []
    contexts: List[Dict[str, Any]] = []
    for row in rows:
        context_path = Path(row["context_record"])
        pipeline_path = Path(row["setup_dir"]).parent / "pipeline_status.json"
        pipeline = load_json(pipeline_path)
        context = load_json(context_path)
        outcome = {
            "matrix_index": row["matrix_index"],
            "system_sample_id": row["system_sample_id"],
            "transition_id": row["transition_id"],
            "pipeline_status": (pipeline or {}).get("status", "missing"),
            "failed_stage": (pipeline or {}).get("failed_stage"),
            "context_record": str(context_path),
        }
        if context is None or context.get("status") != "prepared":
            outcome["admitted"] = False
            outcome["reason"] = "missing_or_unpassed_context"
            outcomes.append(outcome)
            continue
        issues = validate_transition_record(
            context, base_dir=Path.cwd(), check_files=True
        )
        errors = [issue.to_dict() for issue in issues if issue.severity == "error"]
        if errors:
            outcome["admitted"] = False
            outcome["reason"] = "manifest_validation_failed"
            outcome["errors"] = errors
            outcomes.append(outcome)
            continue
        outcome["admitted"] = True
        outcome["reason"] = "passed"
        outcomes.append(outcome)
        contexts.append(context)

    counts = Counter(
        "admitted" if outcome["admitted"] else str(outcome["failed_stage"] or outcome["reason"])
        for outcome in outcomes
    )
    summary = {
        "schema_version": "bindrae_md_context_collection_v1",
        "matrix": str(matrix_path),
        "planned_systems": len(rows),
        "admitted_systems": len(contexts),
        "rejected_or_failed_systems": len(rows) - len(contexts),
        "outcome_counts": dict(sorted(counts.items())),
        "outcomes": outcomes,
    }
    return contexts, summary


def main() -> None:
    args = parse_args()
    contexts, summary = collect(args.matrix)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in contexts)
    )
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
