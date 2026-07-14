#!/usr/bin/env python3
"""Summarize a replica matrix and assemble every passed target into one cache."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return record


def classify(row: Dict[str, Any]) -> Dict[str, Any]:
    pull_dir = Path(row["pull_dir"])
    target_dir = Path(row["target_dir"])
    pipeline = load_json(pull_dir / "pipeline_status.json") or {}
    pull_report = load_json(pull_dir / "rmsd_pull_report.json") or {}
    target_audit = load_json(target_dir / "target_audit.json") or {}
    pipeline_status = str(pipeline.get("status") or "missing")
    failed_stage = pipeline.get("failed_stage")
    target_passed = bool(target_audit.get("passed")) and (
        target_audit.get("status") == "md_phase_normal_targets_passed"
    )
    if target_passed and pipeline_status == "completed":
        outcome = "target_passed"
    elif failed_stage:
        outcome = f"failed_{failed_stage}"
    elif pipeline_status in {"running", "starting"}:
        outcome = "incomplete_running"
    else:
        outcome = f"incomplete_{pipeline_status}"
    return {
        "matrix_index": row["matrix_index"],
        "system_sample_id": row["system_sample_id"],
        "sample_id": row["sample_id"],
        "replica_index": row["replica_index"],
        "seed": row["seed"],
        "pipeline_status": pipeline_status,
        "failed_stage": failed_stage,
        "outcome": outcome,
        "pull_status": pull_report.get("status"),
        "target_passed": target_passed,
        "pull_dir": str(pull_dir),
        "target_dir": str(target_dir),
    }


def finalize(matrix_path: Path, output_dir: Path) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in matrix_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty replica matrix: {matrix_path}")
    outcomes = [classify(row) for row in rows]
    passed_dirs = [
        Path(outcome["target_dir"])
        for outcome in outcomes
        if outcome["target_passed"]
    ]
    per_system: Dict[str, Counter[str]] = defaultdict(Counter)
    for outcome in outcomes:
        per_system[outcome["system_sample_id"]][outcome["outcome"]] += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "outcomes.jsonl").write_text(
        "".join(json.dumps(outcome, sort_keys=True) + "\n" for outcome in outcomes)
    )
    (output_dir / "passed_target_dirs.txt").write_text(
        "".join(f"{directory}\n" for directory in passed_dirs)
    )
    cache_dir = output_dir / "phase_normal_cache"
    if passed_dirs:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "assemble_md_phase_normal_cache.py"),
        ]
        for directory in passed_dirs:
            command.extend(["--input-dir", str(directory)])
        command.extend(["--output-dir", str(cache_dir)])
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    summary = {
        "schema_version": "bindrae_md_replica_finalization_v1",
        "matrix": str(matrix_path),
        "planned_replicas": len(rows),
        "passed_targets": len(passed_dirs),
        "outcome_counts": dict(sorted(Counter(x["outcome"] for x in outcomes).items())),
        "per_system": {
            system: dict(sorted(counts.items()))
            for system, counts in sorted(per_system.items())
        },
        "cache_dir": str(cache_dir) if passed_dirs else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = finalize(args.matrix, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
