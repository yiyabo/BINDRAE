#!/usr/bin/env python3
"""Create a leakage-free system split for an MD phase-normal cache."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--val-systems", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--available-samples-file",
        type=Path,
        default=None,
        help="Optional newline list used to verify that every endpoint feature exists.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def group_records(
    records: Sequence[Dict[str, Any]],
    base_sample_ids: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    base_ids = sorted(set(str(value) for value in base_sample_ids))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        matches = [
            base_id
            for base_id in base_ids
            if sample_id == base_id or sample_id.startswith(f"{base_id}__")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Could not map replica {sample_id!r} to exactly one base system: {matches}"
            )
        grouped[matches[0]].append(record)
    missing = sorted(set(base_ids) - set(grouped))
    if missing:
        raise ValueError(f"Base systems without replica records: {missing}")
    return dict(grouped)


def _record_metric(record: Dict[str, Any], name: str) -> float:
    return float(record.get("audit_metrics", {}).get(name, 0.0))


def split_score(
    selected: Iterable[str],
    grouped: Dict[str, List[Dict[str, Any]]],
) -> float:
    selected = set(selected)
    target_fraction = len(selected) / len(grouped)

    def fraction(value_fn) -> float:
        total = sum(value_fn(record) for rows in grouped.values() for record in rows)
        heldout = sum(
            value_fn(record)
            for system_id in selected
            for record in grouped[system_id]
        )
        return heldout / total if total > 0 else target_fraction

    fractions = (
        fraction(lambda _: 1.0),
        fraction(lambda row: _record_metric(row, "confident_phase_points")),
        fraction(lambda row: _record_metric(row, "valid_residual_points")),
        fraction(lambda row: _record_metric(row, "active_interior_points")),
    )
    return sum(abs(value - target_fraction) for value in fractions)


def choose_split(
    grouped: Dict[str, List[Dict[str, Any]]],
    val_systems: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    systems = sorted(grouped)
    if not 0 < val_systems < len(systems):
        raise ValueError(
            f"val_systems must be between 1 and {len(systems) - 1}, got {val_systems}"
        )
    rng = random.Random(seed)
    tie_break = {system_id: rng.random() for system_id in systems}

    def key(candidate: Tuple[str, ...]) -> Tuple[float, float, Tuple[str, ...]]:
        jitter = sum(tie_break[system_id] for system_id in candidate)
        return split_score(candidate, grouped), jitter, candidate

    val = list(min(itertools.combinations(systems, val_systems), key=key))
    val_set = set(val)
    train = [system_id for system_id in systems if system_id not in val_set]
    return train, val


def summarize_split(
    systems: Sequence[str],
    grouped: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    records = [record for system_id in systems for record in grouped[system_id]]
    return {
        "systems": len(systems),
        "replicas": len(records),
        "confident_phase_points": int(
            sum(_record_metric(record, "confident_phase_points") for record in records)
        ),
        "valid_residual_points": int(
            sum(_record_metric(record, "valid_residual_points") for record in records)
        ),
        "active_interior_points": int(
            sum(_record_metric(record, "active_interior_points") for record in records)
        ),
        "sample_ids": list(systems),
    }


def write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values))


def main() -> None:
    args = parse_args()
    summary = json.loads((args.cache_dir / "summary.json").read_text())
    records = load_jsonl(args.cache_dir / "manifest.jsonl")
    grouped = group_records(records, summary["base_sample_ids"])
    train, val = choose_split(grouped, args.val_systems, args.seed)

    if args.available_samples_file is not None:
        available = {
            line.strip()
            for line in args.available_samples_file.read_text().splitlines()
            if line.strip()
        }
        missing = sorted((set(train) | set(val)) - available)
        if missing:
            raise ValueError(f"Endpoint systems missing from feature cache: {missing}")

    if set(train) & set(val):
        raise RuntimeError("System leakage detected between train and validation")
    write_lines(args.train_out, train)
    write_lines(args.val_out, val)
    report = {
        "schema_version": "md_phase_normal_system_split_v1",
        "cache_dir": str(args.cache_dir),
        "seed": args.seed,
        "selection": "exhaustive_balance_over_replica_and_supervision_counts",
        "train": summarize_split(train, grouped),
        "val": summarize_split(val, grouped),
        "balance_score": split_score(val, grouped),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
