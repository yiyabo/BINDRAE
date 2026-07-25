#!/usr/bin/env python3
"""Merge disjoint baseline manifests with duplicate and coverage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--sample_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    sample_ids = [
        line.strip() for line in Path(args.sample_file).read_text().splitlines()
        if line.strip()
    ]
    expected = set(sample_ids)
    rows = {}
    sources = {}
    for raw_path in args.input:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(
                    f"Duplicate sample_id {sample_id!r} in {path} and {sources[sample_id]}"
                )
            if sample_id not in expected:
                raise ValueError(f"Unexpected sample_id {sample_id!r} in {path}")
            rows[sample_id] = row
            sources[sample_id] = str(path)
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows]
    if missing:
        raise ValueError(f"Merged manifests miss {len(missing)} samples: {missing[:8]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            handle.write(json.dumps(rows[sample_id], sort_keys=True) + "\n")
    status_counts = {}
    for row in rows.values():
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema_version": "baseline_manifest_merge_v1",
        "sample_file": args.sample_file,
        "inputs": args.input,
        "output": str(output),
        "samples": len(rows),
        "status_counts": status_counts,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
