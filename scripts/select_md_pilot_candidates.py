#!/usr/bin/env python3
"""Build an AHoJ endpoint index and select systems for the first MD pilot."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.md_pilot_selection import (
    MULTI_FRAGMENT_POLICIES,
    candidate_to_transition_record,
    read_sample_ids,
    screen_samples,
    select_diverse_candidates,
    summarize_screen,
)


DEFAULT_EXCLUDED_RESNAMES = ",".join(
    [
        "ADP",
        "AMP",
        "ATP",
        "CDP",
        "CMP",
        "COA",
        "CTP",
        "FAD",
        "FBP",
        "FMN",
        "GDP",
        "GLC",
        "GMP",
        "GTP",
        "HEM",
        "NAD",
        "NAG",
        "NAP",
        "SAH",
        "SAM",
        "UDP",
        "UMP",
        "UTP",
    ]
)

# Named ligand-exclusion presets. The blocklist is a convenience filter, not one of
# the frozen acquisition gates (0.95 residue mapping, holdout leakage, physical
# gates, and the consensus requirement are unchanged by any preset here).
#
# frozen23            the original 23-component blocklist; unchanged default.
# glycosylation_only  keep only N-acetylglucosamine excluded. Nucleotide and
#                     cofactor complexes (ATP/ADP/NAD/FAD/...) are classic
#                     induced-fit systems and are hard to justify excluding;
#                     metal-bearing components such as HEM are still refused by
#                     the unchanged metal gate.
# none                apply no resname blocklist at all.
EXCLUDED_RESNAME_PRESETS = {
    "frozen23": DEFAULT_EXCLUDED_RESNAMES,
    "glycosylation_only": "NAG",
    "none": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("processed_data/triplets"))
    parser.add_argument("--sample-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-limit", type=int, default=8000)
    parser.add_argument("--select-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--min-residues", type=int, default=80)
    parser.add_argument("--max-residues", type=int, default=500)
    parser.add_argument("--min-sequence-identity", type=float, default=0.95)
    parser.add_argument("--min-residue-mapping-fraction", type=float, default=0.95)
    parser.add_argument("--min-heavy-atoms", type=int, default=8)
    parser.add_argument("--max-heavy-atoms", type=int, default=70)
    parser.add_argument("--max-abs-charge", type=int, default=2)
    parser.add_argument("--exclude-resnames", default=DEFAULT_EXCLUDED_RESNAMES)
    parser.add_argument(
        "--exclude-resname-preset",
        choices=sorted(EXCLUDED_RESNAME_PRESETS),
        default=None,
        help=(
            "Named ligand-exclusion preset. Takes precedence over --exclude-resnames. "
            "Omit to keep the frozen 23-component blocklist."
        ),
    )
    parser.add_argument(
        "--multi-fragment-policy",
        choices=sorted(MULTI_FRAGMENT_POLICIES),
        default="reject",
        help=(
            "How to treat an SDF holding several distinct organic fragments. "
            "'reject' is the frozen behaviour; 'largest' keeps an unambiguously "
            "dominant fragment and discards co-exported additives."
        ),
    )
    parser.add_argument(
        "--fragment-dominance-ratio",
        type=float,
        default=2.0,
        help=(
            "Heavy-atom ratio the primary fragment must reach over the next "
            "fragment when --multi-fragment-policy=largest."
        ),
    )
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--contact-radius", type=float, default=8.0)
    parser.add_argument("--min-pocket-residues", type=int, default=5)
    parser.add_argument("--moving-threshold", type=float, default=1.0)
    parser.add_argument("--min-global-rmsd", type=float, default=0.35)
    parser.add_argument("--min-pocket-rmsd", type=float, default=0.60)
    parser.add_argument("--min-max-displacement", type=float, default=1.20)
    parser.add_argument("--max-global-rmsd", type=float, default=5.0)
    parser.add_argument("--max-pocket-rmsd", type=float, default=8.0)
    parser.add_argument("--max-max-displacement", type=float, default=20.0)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fieldnames = sorted({key for row in materialized for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.select_count <= 0:
        raise ValueError("--select-count must be positive")
    if not 0.0 < args.min_residue_mapping_fraction <= 1.0:
        raise ValueError("--min-residue-mapping-fraction must be in (0, 1]")
    if args.fragment_dominance_ratio < 1.0:
        raise ValueError("--fragment-dominance-ratio must be at least 1.0")
    if args.exclude_resname_preset is not None:
        args.exclude_resnames = EXCLUDED_RESNAME_PRESETS[args.exclude_resname_preset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = read_sample_ids(args.sample_list, scan_limit=args.scan_limit, seed=args.seed)
    config: Dict[str, Any] = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "sample_list",
            "output_dir",
            "scan_limit",
            "select_count",
            "seed",
            "workers",
            "exclude_resnames",
            "exclude_resname_preset",
        }
    }
    config["data_dir"] = str(args.data_dir)
    config["excluded_resnames"] = [
        value.strip().upper() for value in args.exclude_resnames.split(",") if value.strip()
    ]
    print(f"Screening {len(sample_ids)} samples with {args.workers} workers", flush=True)
    rows: List[Dict[str, Any]] = screen_samples(sample_ids, config, workers=args.workers)
    rows.sort(key=lambda row: str(row["sample_id"]))
    selected = select_diverse_candidates(rows, args.select_count)
    canonical_records = [candidate_to_transition_record(row) for row in selected]
    eligible = sorted(
        (row for row in rows if row.get("eligible") is True),
        key=lambda row: (-float(row.get("pilot_score", 0.0)), str(row["sample_id"])),
    )

    write_csv(args.output_dir / "endpoint_index.csv", rows)
    write_csv(args.output_dir / "eligible_candidates.csv", eligible)
    write_csv(args.output_dir / "selected_candidates.csv", selected)
    write_jsonl(args.output_dir / "selected_transition_manifest.jsonl", canonical_records)
    (args.output_dir / "selected_sample_ids.txt").write_text(
        "".join(f"{row['sample_id']}\n" for row in selected)
    )
    summary = {
        **summarize_screen(rows, selected),
        "requested_select_count": args.select_count,
        "selection_shortfall": max(args.select_count - len(selected), 0),
        "sample_list": str(args.sample_list),
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "scan_limit": args.scan_limit,
        "exclude_resname_preset": args.exclude_resname_preset or "frozen23_default",
        "excluded_resnames": config["excluded_resnames"],
        "multi_fragment_policy": args.multi_fragment_policy,
        "fragment_dominance_ratio": args.fragment_dominance_ratio,
        "thresholds": config,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
