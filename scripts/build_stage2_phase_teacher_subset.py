#!/usr/bin/env python3
"""Select samples with usable phase or MD phase-normal supervision.

The resulting manifest is an enriched diagnostic subset. It must not replace
evaluation on the original validation distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument(
        "--candidate_list",
        default=None,
        help="Optional ordered sample list to intersect with usable cache entries.",
    )
    parser.add_argument(
        "--mask_mode",
        default="contact_event",
        choices=["contact_event", "formed_contact", "approach", "active", "pocket", "node"],
    )
    parser.add_argument("--min_confidence", type=float, default=0.05)
    parser.add_argument("--n_model_steps", type=int, default=8)
    parser.add_argument("--min_supervised_points", type=int, default=1)
    return parser.parse_args()


def selected_mask(data, mode: str) -> np.ndarray:
    if mode == "contact_event":
        return (
            np.asarray(data["approach_mask"]).astype(bool)
            | np.asarray(data["formed_contact_mask"]).astype(bool)
        )
    key = {
        "formed_contact": "formed_contact_mask",
        "approach": "approach_mask",
        "active": "active_mask",
        "pocket": "pocket_mask",
        "node": "node_mask",
    }[mode]
    return np.asarray(data[key]).astype(bool)


def interpolate_rows(
    source_t: np.ndarray,
    values: np.ndarray,
    target_t: np.ndarray,
) -> np.ndarray:
    rows = []
    for t_value in target_t:
        upper = int(np.searchsorted(source_t, float(t_value), side="left"))
        upper = min(max(upper, 1), source_t.size - 1)
        lower = upper - 1
        span = max(float(source_t[upper] - source_t[lower]), 1e-8)
        fraction = np.clip((float(t_value) - float(source_t[lower])) / span, 0.0, 1.0)
        rows.append(values[lower] * (1.0 - fraction) + values[upper] * fraction)
    return np.stack(rows, axis=0)


def summarize_cache_file(
    path: Path,
    mask_mode: str,
    min_confidence: float,
    n_model_steps: int,
) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema_version"].item())
        allowed_schemas = {"phase_teacher_v1", "md_phase_normal_v1"}
        if schema not in allowed_schemas:
            raise ValueError(
                f"{path} schema_version={schema!r}, expected one of "
                f"{sorted(allowed_schemas)}"
            )
        mask = selected_mask(data, mask_mode)
        target_t = np.linspace(0.0, 1.0, n_model_steps + 1, dtype=np.float32)[1:-1]
        confidence = interpolate_rows(
            np.asarray(data["t_values"], dtype=np.float32),
            np.asarray(data["phase_confidence"], dtype=np.float32),
            target_t,
        )
        supervised = (confidence >= float(min_confidence)) & mask[None]
        return {
            "sample_id": str(data["sample_id"].item()),
            "selected_residues": int(mask.sum()),
            "supervised_points": int(supervised.sum()),
            "max_confidence": float(confidence[:, mask].max()) if mask.any() else 0.0,
        }


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.min_confidence <= 1.0):
        raise ValueError("--min_confidence must be in [0, 1]")
    if args.n_model_steps < 2:
        raise ValueError("--n_model_steps must be >= 2")
    if args.min_supervised_points < 1:
        raise ValueError("--min_supervised_points must be >= 1")

    cache_dir = Path(args.cache_dir)
    records = [
        summarize_cache_file(
            path,
            args.mask_mode,
            args.min_confidence,
            args.n_model_steps,
        )
        for path in sorted(cache_dir.glob("*.npz"))
    ]
    candidate_ids = None
    missing_candidates = []
    if args.candidate_list:
        candidate_path = Path(args.candidate_list)
        candidate_ids = [
            line.strip()
            for line in candidate_path.read_text().splitlines()
            if line.strip()
        ]
        if not candidate_ids:
            raise ValueError(f"Candidate list is empty: {candidate_path}")
        by_sample_id = {str(record["sample_id"]): record for record in records}
        missing_candidates = [
            sample_id for sample_id in candidate_ids if sample_id not in by_sample_id
        ]
        records = [
            by_sample_id[sample_id]
            for sample_id in candidate_ids
            if sample_id in by_sample_id
        ]
    selected = [
        record for record in records
        if int(record["supervised_points"]) >= int(args.min_supervised_points)
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{record['sample_id']}\n" for record in selected))
    summary = {
        "cache_dir": str(cache_dir),
        "mask_mode": args.mask_mode,
        "min_confidence": float(args.min_confidence),
        "n_model_steps": int(args.n_model_steps),
        "min_supervised_points": int(args.min_supervised_points),
        "samples_scanned": len(records),
        "samples_selected": len(selected),
        "candidate_list": args.candidate_list,
        "candidates_requested": len(candidate_ids) if candidate_ids is not None else None,
        "candidates_missing_cache": missing_candidates,
        "selected_supervised_points": sum(int(r["supervised_points"]) for r in selected),
        "records": selected,
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
