#!/usr/bin/env python3
"""Rank Stage-2 runs with the offline path critic.

Examples:
    python scripts/evaluate_stage2_path_critic.py logs/stage2/run_a/metrics.jsonl logs/stage2/run_b/metrics.jsonl
    python scripts/evaluate_stage2_path_critic.py logs/stage2 --best-per-run --top-k 20
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Mapping

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def _load_path_critic_module():
    module_path = project_root / "src" / "stage2" / "evaluation" / "path_critic.py"
    spec = importlib.util.spec_from_file_location("bindrae_stage2_path_critic", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load path critic module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_path_critic = _load_path_critic_module()
best_record_per_run = _path_critic.best_record_per_run
load_metric_records = _path_critic.load_metric_records
score_metric_records = _path_critic.score_metric_records


DISPLAY_KEYS = (
    "val_end",
    "val_contact_score_gain",
    "val_contact_score_holo_gap_abs",
    "val_interaction_prior_final",
    "val_clash",
    "val_contact_score_holo_delta",
    "val_contact_score_direction_acc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 metrics with an offline path critic")
    parser.add_argument(
        "metrics",
        nargs="+",
        help="metrics.jsonl files or directories containing metrics.jsonl files",
    )
    parser.add_argument("--top-k", type=int, default=10, help="number of rows to print")
    parser.add_argument(
        "--best-per-run",
        action="store_true",
        help="show only the best epoch/record from each run",
    )
    parser.add_argument("--json-out", type=str, default=None, help="optional JSON output path")
    return parser.parse_args()


def _fmt(value: object, width: int = 10) -> str:
    if isinstance(value, float):
        return f"{value:.4g}".rjust(width)
    if isinstance(value, int):
        return str(value).rjust(width)
    if value is None:
        return "NA".rjust(width)
    return str(value)[:width].rjust(width)


def _row_value(record: Mapping[str, object], key: str) -> object:
    return record.get(key, None)


def print_table(records, top_k: int) -> None:
    rows = records[: max(top_k, 0)]
    if not rows:
        print("No scorable metric records found.")
        return

    header = [
        "rank",
        "score",
        "run",
        "epoch",
        *DISPLAY_KEYS,
    ]
    print(" ".join(item.rjust(12) for item in header))
    for rank, record in enumerate(rows, start=1):
        values = [
            rank,
            record.get("path_critic_score"),
            record.get("_run_id", "unknown"),
            record.get("epoch", "NA"),
            *(_row_value(record, key) for key in DISPLAY_KEYS),
        ]
        print(" ".join(_fmt(value, width=12) for value in values))


def main() -> None:
    args = parse_args()
    records = load_metric_records(args.metrics)
    scored = score_metric_records(records)
    selected = best_record_per_run(scored) if args.best_per_run else scored

    print(f"Loaded {len(records)} records; scorable rows: {len(scored)}")
    print_table(selected, args.top_k)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(selected, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
