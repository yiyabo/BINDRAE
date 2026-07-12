#!/usr/bin/env python3
"""Audit a BINDRAE MD-transition JSONL manifest and report eligibility gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.md_transition_manifest import (  # noqa: E402
    audit_transition_manifest,
    load_transition_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Canonical JSONL manifest")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for relative local paths (default: manifest directory)",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Require every declared local endpoint/topology/trajectory file to exist",
    )
    parser.add_argument("--summary-output", default=None, help="Optional JSON report path")
    parser.add_argument("--max-issues", type=int, default=100)
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return a non-zero exit code when warnings are present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    base_dir = (
        Path(args.base_dir).expanduser().resolve()
        if args.base_dir is not None
        else manifest_path.parent
    )
    records, parse_issues = load_transition_manifest(manifest_path)
    issues, summary = audit_transition_manifest(
        records,
        initial_issues=parse_issues,
        base_dir=base_dir,
        check_files=bool(args.check_files),
    )
    report = {
        "manifest": str(manifest_path),
        "base_dir": str(base_dir),
        "check_files": bool(args.check_files),
        "summary": summary,
        "issues": [issue.to_dict() for issue in issues],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    for issue in issues[: max(int(args.max_issues), 0)]:
        location = f"line {issue.line_number}" if issue.line_number else "manifest"
        transition = f" [{issue.transition_id}]" if issue.transition_id else ""
        print(
            f"{issue.severity.upper()} {location}{transition} "
            f"{issue.code}: {issue.message}",
            file=sys.stderr,
        )
    if len(issues) > int(args.max_issues) >= 0:
        print(
            f"... {len(issues) - int(args.max_issues)} additional issues omitted",
            file=sys.stderr,
        )

    if args.summary_output:
        output_path = Path(args.summary_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if summary["num_errors"] > 0:
        return 1
    if args.strict_warnings and summary["num_warnings"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
