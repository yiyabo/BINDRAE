#!/usr/bin/env python3
"""Re-export one replica from a merged phase-normal collection manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--phase-target-mode",
        choices=("inferred", "identity"),
        default="identity",
    )
    parser.add_argument(
        "--residual-envelope",
        choices=("sin2", "poly"),
        default="sin2",
    )
    parser.add_argument(
        "--normal-projection-mode",
        choices=("product", "block"),
        default="product",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def infer_replica_matrix(source_dir: Path) -> Path:
    parts = source_dir.parts
    target_index = next(
        (index for index, part in enumerate(parts) if part.startswith("targets")),
        None,
    )
    if target_index is None:
        raise ValueError(f"Cannot infer collection root from source_dir={source_dir}")
    return Path(*parts[:target_index]) / "replica_matrix.jsonl"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_replica_record(collection_record: Dict[str, Any]) -> Dict[str, Any]:
    matrix_path = resolve_project_path(
        infer_replica_matrix(Path(collection_record["source_dir"]))
    )
    sample_id = str(collection_record["sample_id"])
    matches = [
        record
        for record in load_jsonl(matrix_path)
        if str(record.get("sample_id")) == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one matrix record for {sample_id!r} in {matrix_path}, "
            f"found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    records = load_jsonl(resolve_project_path(args.collection_manifest))
    if not 0 <= args.index < len(records):
        raise IndexError(f"index={args.index} outside [0, {len(records)})")
    collection_record = records[args.index]
    replica = find_replica_record(collection_record)
    system_id = str(replica["system_sample_id"])
    replica_index = int(replica["replica_index"])
    output_dir = resolve_project_path(args.output_root) / system_id / f"replica_{replica_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/export_md_phase_normal_targets.py"),
        "--pull-dir",
        str(resolve_project_path(replica["pull_dir"])),
        "--candidate-manifest",
        str(resolve_project_path(replica["candidate_manifest"])),
        "--transition-id",
        str(replica["transition_id"]),
        "--preparation-report",
        str(resolve_project_path(replica["preparation_report"])),
        "--output-dir",
        str(output_dir),
        "--sample-id",
        str(replica["sample_id"]),
        "--phase-target-mode",
        args.phase_target_mode,
        "--residual-envelope",
        args.residual_envelope,
        "--normal-projection-mode",
        args.normal_projection_mode,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(
        json.dumps(
            {
                "index": args.index,
                "sample_id": replica["sample_id"],
                "phase_target_mode": args.phase_target_mode,
                "residual_envelope": args.residual_envelope,
                "normal_projection_mode": args.normal_projection_mode,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
