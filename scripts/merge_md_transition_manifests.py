#!/usr/bin/env python3
"""Merge canonical MD-transition JSONL files while rejecting duplicate IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.md_transition_manifest import load_transition_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merge_records(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: Dict[str, Path] = {}
    for path in paths:
        loaded, issues = load_transition_manifest(path)
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            raise ValueError(f"{path} contains {len(errors)} manifest parse errors")
        for loaded_record in loaded:
            record = dict(loaded_record)
            record.pop("_manifest_line_number", None)
            transition_id = str(record.get("transition_id") or "")
            if not transition_id:
                raise ValueError(f"{path} contains a record without transition_id")
            if transition_id in seen:
                raise ValueError(
                    f"Duplicate transition_id={transition_id!r} in {seen[transition_id]} and {path}"
                )
            seen[transition_id] = path
            records.append(record)
    records.sort(key=lambda record: str(record["transition_id"]))
    return records


def main() -> None:
    args = parse_args()
    records = merge_records(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
