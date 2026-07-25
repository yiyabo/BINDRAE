#!/usr/bin/env python3
"""Freeze a representative residue-mismatch smoke panel from MD candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "bindrae_md_mapping_smoke_panel_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=32)
    parser.add_argument("--minimum-mapping-fraction", type=float, default=0.95)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No records in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def sample_id(record: Mapping[str, Any]) -> str:
    endpoints = dict(record.get("endpoints") or {})
    holo = Path(str(endpoints.get("holo_structure_path") or ""))
    if not holo.parent.name:
        raise ValueError(f"Missing holo sample path: {record.get('transition_id')}")
    return holo.parent.name


def mismatch_fields(record: Mapping[str, Any]) -> tuple[int, int, float, str]:
    screening = dict(record.get("screening") or {})
    try:
        apo = int(screening["apo_n_residues"])
        holo = int(screening["holo_n_residues"])
        mapping = float(screening["residue_mapping_fraction"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Missing residue-mapping fields for {sample_id(record)}"
        ) from error
    category = str(screening.get("motion_category") or "unknown")
    return apo, holo, mapping, category


def write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def allocate_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    if total < 1:
        raise ValueError("panel size must be positive")
    available = {key: value for key, value in counts.items() if value > 0}
    if total > sum(available.values()):
        raise ValueError("panel size exceeds available mismatch candidates")
    quotas = {key: 0 for key in available}
    ordered = sorted(available)
    for key in ordered:
        if sum(quotas.values()) == total:
            break
        quotas[key] = 1
    while sum(quotas.values()) < total:
        key = max(
            [value for value in ordered if quotas[value] < available[value]],
            key=lambda value: (
                available[value] / (quotas[value] + 1),
                available[value],
                value,
            ),
        )
        quotas[key] += 1
    return quotas


def evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError("Selection count exceeds stratum size")
    if count == len(rows):
        return rows
    positions = [math.floor(index * len(rows) / count) for index in range(count)]
    return [rows[index] for index in positions]


def selection_key(record: Mapping[str, Any]) -> tuple[float, int, int, str]:
    apo, holo, mapping, _ = mismatch_fields(record)
    screening = dict(record.get("screening") or {})
    return (
        mapping,
        -abs(apo - holo),
        int(screening.get("selection_rank") or 10**9),
        sample_id(record),
    )


def build_panel(
    candidates: Iterable[Mapping[str, Any]], *, panel_size: int, minimum_mapping_fraction: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not 0.0 < minimum_mapping_fraction <= 1.0:
        raise ValueError("minimum mapping fraction must be in (0, 1]")
    by_category: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for raw in candidates:
        record = dict(raw)
        identifier = sample_id(record)
        if identifier in seen:
            raise ValueError(f"Duplicate sample ID: {identifier}")
        seen.add(identifier)
        apo, holo, mapping, category = mismatch_fields(record)
        if apo == holo or mapping < minimum_mapping_fraction:
            continue
        by_category.setdefault(category, []).append(record)
    counts = {category: len(rows) for category, rows in by_category.items()}
    quotas = allocate_quotas(counts, panel_size)
    selected: list[dict[str, Any]] = []
    for category in sorted(quotas):
        rows = sorted(by_category[category], key=selection_key)
        selected.extend(evenly_spaced(rows, quotas[category]))
    selected.sort(key=lambda record: (str((record.get("screening") or {}).get("motion_category")), selection_key(record)))
    if len(selected) != panel_size or len({sample_id(row) for row in selected}) != panel_size:
        raise RuntimeError("Panel selection did not produce unique requested systems")
    return selected, quotas


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.panel_size < 1:
        raise ValueError("--panel-size must be positive")
    candidates = load_jsonl(args.candidate_manifest)
    panel, quotas = build_panel(
        candidates,
        panel_size=args.panel_size,
        minimum_mapping_fraction=args.minimum_mapping_fraction,
    )
    manifest = args.output_dir / "mapping_smoke_panel_manifest.jsonl"
    sample_list = args.output_dir / "mapping_smoke_panel_sample_ids.txt"
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in panel)
    write_immutable(manifest, manifest_text)
    write_immutable(sample_list, "".join(f"{sample_id(row)}\n" for row in panel))
    category_counts = Counter(str((row.get("screening") or {}).get("motion_category")) for row in panel)
    mapping_values = [mismatch_fields(row)[2] for row in panel]
    residue_deltas = [abs(mismatch_fields(row)[0] - mismatch_fields(row)[1]) for row in panel]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_manifest_sha256": hashlib.sha256(args.candidate_manifest.read_bytes()).hexdigest(),
        "panel_size": len(panel),
        "minimum_mapping_fraction": args.minimum_mapping_fraction,
        "selection": "equal-category-quota_then_even_mapping_delta_coverage",
        "available_by_motion_category": dict(sorted(Counter(mismatch_fields(row)[3] for row in candidates if mismatch_fields(row)[0] != mismatch_fields(row)[1] and mismatch_fields(row)[2] >= args.minimum_mapping_fraction).items())),
        "selected_by_motion_category": dict(sorted(category_counts.items())),
        "category_quotas": quotas,
        "mapping_fraction_min": min(mapping_values),
        "mapping_fraction_max": max(mapping_values),
        "residue_count_delta_min": min(residue_deltas),
        "residue_count_delta_max": max(residue_deltas),
        "files": {"manifest": str(manifest), "sample_ids": str(sample_list)},
    }
    write_immutable(args.output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
