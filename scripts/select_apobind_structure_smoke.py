#!/usr/bin/env python3
"""Select a deterministic, stratified APObind structure-smoke cohort."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "bindrae_apobind_structure_smoke_selection_v1"
DEFAULT_SEED = "apobind_structure_smoke32_v1"
STRATUM_FIELDS = ("backbone_rmsd", "apo_resolution", "binding_site_size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def _valid_unit_interval(value: object) -> bool:
    return value is not None and 0.0 <= float(value) <= 1.0


def strict_proxy_rejection_reasons(record: Mapping[str, Any]) -> list[str]:
    """Return reasons a record is outside the frozen APObind metadata proxy."""

    reasons: list[str] = []
    if record.get("source") != "apobind":
        reasons.append("not_apobind")
        return reasons

    match_level = record.get("reference_pair_match", {}).get("match_level")
    if match_level != "none":
        reasons.append("reference_pdb_pair_match")

    endpoints = record.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        reasons.append("invalid_endpoint_count")
        return reasons
    if any(len(endpoint.get("chains", [])) != 1 for endpoint in endpoints):
        reasons.append("not_single_chain_endpoints")

    metadata = record.get("metadata", {})
    identity = metadata.get("sequence_identity")
    coverage = metadata.get("sequence_coverage")
    if not _valid_unit_interval(identity) or float(identity) < 0.95:
        reasons.append("sequence_identity_outside_0_95_to_1")
    if not _valid_unit_interval(coverage) or float(coverage) < 0.95:
        reasons.append("sequence_coverage_outside_0_95_to_1")

    rmsd = metadata.get("backbone_rmsd")
    if rmsd is None or not 0.35 <= float(rmsd) <= 5.0:
        reasons.append("backbone_rmsd_outside_0_35_to_5")
    tmscore = metadata.get("tmscore")
    if tmscore is None or not 0.5 <= float(tmscore) <= 1.0:
        reasons.append("tmscore_outside_0_5_to_1")
    resolution = metadata.get("apo_resolution")
    if resolution is None or float(resolution) <= 0.0:
        reasons.append("nonpositive_or_missing_apo_resolution")

    binding_site = metadata.get("binding_site", {})
    if not binding_site.get("apo_indices") or not binding_site.get("holo_indices"):
        reasons.append("empty_binding_site_signature")
    return reasons


def canonical_pair(record: Mapping[str, Any]) -> tuple[str, str]:
    endpoints = record["endpoints"]
    return tuple(sorted((str(endpoints[0]["pdb_id"]), str(endpoints[1]["pdb_id"]))))


def endpoint_ids(record: Mapping[str, Any]) -> tuple[str, str]:
    endpoints = record["endpoints"]
    return str(endpoints[0]["pdb_id"]), str(endpoints[1]["pdb_id"])


def binding_site_size(record: Mapping[str, Any]) -> int:
    binding_site = record["metadata"]["binding_site"]
    return min(
        len(binding_site["apo_indices"]),
        len(binding_site["holo_indices"]),
    )


def deduplicate_pairs(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    representatives: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_records = 0
    for record in sorted(records, key=lambda item: str(item["record_key"])):
        key = canonical_pair(record)
        if key in representatives:
            duplicate_records += 1
            continue
        representatives[key] = record
    return [dict(record) for record in representatives.values()], duplicate_records


def quartile_edges(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("Cannot calculate quartiles for an empty sequence")
    ordered = sorted(float(value) for value in values)
    last = len(ordered) - 1
    return [ordered[int(last * fraction)] for fraction in (0.25, 0.5, 0.75)]


def quartile_index(value: float, edges: Sequence[float]) -> int:
    return bisect.bisect_right(list(edges), float(value))


def _stable_key(record: Mapping[str, Any], seed: str) -> str:
    value = f"{seed}\0{record['record_key']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _metrics(record: Mapping[str, Any]) -> dict[str, float]:
    metadata = record["metadata"]
    return {
        "backbone_rmsd": float(metadata["backbone_rmsd"]),
        "apo_resolution": float(metadata["apo_resolution"]),
        "binding_site_size": float(binding_site_size(record)),
    }


def assign_strata(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, tuple[int, int, int]]]:
    edges = {
        field: quartile_edges([_metrics(record)[field] for record in records])
        for field in STRATUM_FIELDS
    }
    strata = {
        str(record["record_key"]): tuple(
            quartile_index(_metrics(record)[field], edges[field])
            for field in STRATUM_FIELDS
        )
        for record in records
    }
    return edges, strata


def select_balanced_records(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: str = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(records) < count:
        raise ValueError(f"Requested {count} records from only {len(records)} candidates")

    edges, strata = assign_strata(records)
    remaining = {str(record["record_key"]): dict(record) for record in records}
    selected: list[dict[str, Any]] = []
    used_pdbs: set[str] = set()
    cell_counts: Counter[tuple[int, int, int]] = Counter()
    marginal_counts = [Counter(), Counter(), Counter()]

    for allow_shared_endpoints in (False, True):
        while len(selected) < count:
            eligible: list[dict[str, Any]] = []
            for record in remaining.values():
                pdbs = set(endpoint_ids(record))
                if not allow_shared_endpoints and pdbs & used_pdbs:
                    continue
                eligible.append(record)
            if not eligible:
                break

            def score(record: Mapping[str, Any]) -> tuple[Any, ...]:
                stratum = strata[str(record["record_key"])]
                marginal = [marginal_counts[i][value] for i, value in enumerate(stratum)]
                return (
                    cell_counts[stratum],
                    sum(marginal),
                    max(marginal),
                    _stable_key(record, seed),
                    str(record["record_key"]),
                )

            chosen = min(eligible, key=score)
            record_key = str(chosen["record_key"])
            stratum = strata[record_key]
            shared = sorted(set(endpoint_ids(chosen)) & used_pdbs)
            annotated = dict(chosen)
            annotated["smoke_selection"] = {
                "schema_version": SCHEMA_VERSION,
                "selection_rank": len(selected) + 1,
                "stratum": dict(zip(STRATUM_FIELDS, stratum)),
                "metrics": _metrics(chosen),
                "shared_endpoint_pdb_ids": shared,
                "used_shared_endpoint_fallback": bool(shared),
                "stable_tiebreak_sha256": _stable_key(chosen, seed),
            }
            selected.append(annotated)
            del remaining[record_key]
            used_pdbs.update(endpoint_ids(chosen))
            cell_counts[stratum] += 1
            for index, value in enumerate(stratum):
                marginal_counts[index][value] += 1
    if len(selected) != count:
        raise RuntimeError(f"Selected {len(selected)} records; expected {count}")
    return selected, edges


def _counter_dict(values: Iterable[object]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def build_report(
    *,
    index_path: Path,
    all_records: Sequence[Mapping[str, Any]],
    strict_records: Sequence[Mapping[str, Any]],
    deduplicated_records: Sequence[Mapping[str, Any]],
    duplicate_pair_records: int,
    selected: Sequence[Mapping[str, Any]],
    edges: Mapping[str, Sequence[float]],
    seed: str,
) -> dict[str, Any]:
    selected_pdbs = [pdb for record in selected for pdb in endpoint_ids(record)]
    marginal = {
        field: _counter_dict(
            record["smoke_selection"]["stratum"][field] for record in selected
        )
        for field in STRATUM_FIELDS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "structure_preflight_pending",
        "input": {
            "path": str(index_path),
            "sha256": file_sha256(index_path),
            "records": len(all_records),
        },
        "selection_contract": {
            "seed": seed,
            "requested_count": len(selected),
            "strict_proxy": {
                "source": "apobind",
                "reference_pair_match": "none",
                "single_chain_endpoints": True,
                "sequence_identity": [0.95, 1.0],
                "sequence_coverage": [0.95, 1.0],
                "backbone_rmsd_angstrom": [0.35, 5.0],
                "tmscore": [0.5, 1.0],
                "positive_apo_resolution": True,
                "nonempty_binding_site_signature": True,
            },
            "stratification": "empirical_quartiles",
            "quartile_edges": {key: list(value) for key, value in edges.items()},
            "shared_endpoint_policy": "avoid_then_explicit_fallback",
        },
        "counts": {
            "all_index_records": len(all_records),
            "strict_proxy_records": len(strict_records),
            "strict_proxy_unique_undirected_pairs": len(deduplicated_records),
            "duplicate_pair_records_removed": duplicate_pair_records,
            "selected_systems": len(selected),
            "selected_unique_endpoint_pdbs": len(set(selected_pdbs)),
            "selected_shared_endpoint_fallbacks": sum(
                bool(record["smoke_selection"]["used_shared_endpoint_fallback"])
                for record in selected
            ),
        },
        "selected_marginal_strata": marginal,
        "selected_record_keys": [str(record["record_key"]) for record in selected],
        "claim_boundary": (
            "This cohort is a deterministic metadata-proxy structure smoke. "
            "Selection does not establish exact ligand/site identity, production "
            "mapping validity, leakage cleanliness, or accepted MD paths."
        ),
    }


def write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    jsonl = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in selected
    )
    write_immutable(output_dir / "selected_records.jsonl", jsonl)
    write_immutable(
        output_dir / "selection_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )

    fieldnames = [
        "selection_rank",
        "record_key",
        "source_row",
        "apo_pdb",
        "apo_chain",
        "holo_pdb",
        "holo_chain",
        "backbone_rmsd",
        "apo_resolution",
        "sequence_identity",
        "sequence_coverage",
        "tmscore",
        "apo_binding_site_size",
        "holo_binding_site_size",
        "rmsd_quartile",
        "resolution_quartile",
        "binding_site_size_quartile",
        "shared_endpoint_pdb_ids",
    ]
    rows: list[dict[str, object]] = []
    for record in selected:
        endpoints = record["endpoints"]
        metadata = record["metadata"]
        site = metadata["binding_site"]
        selection = record["smoke_selection"]
        rows.append(
            {
                "selection_rank": selection["selection_rank"],
                "record_key": record["record_key"],
                "source_row": record["source_rows"][0],
                "apo_pdb": endpoints[0]["pdb_id"],
                "apo_chain": endpoints[0]["chains"][0],
                "holo_pdb": endpoints[1]["pdb_id"],
                "holo_chain": endpoints[1]["chains"][0],
                "backbone_rmsd": metadata["backbone_rmsd"],
                "apo_resolution": metadata["apo_resolution"],
                "sequence_identity": metadata["sequence_identity"],
                "sequence_coverage": metadata["sequence_coverage"],
                "tmscore": metadata["tmscore"],
                "apo_binding_site_size": len(site["apo_indices"]),
                "holo_binding_site_size": len(site["holo_indices"]),
                "rmsd_quartile": selection["stratum"]["backbone_rmsd"],
                "resolution_quartile": selection["stratum"]["apo_resolution"],
                "binding_site_size_quartile": selection["stratum"][
                    "binding_site_size"
                ],
                "shared_endpoint_pdb_ids": ";".join(
                    selection["shared_endpoint_pdb_ids"]
                ),
            }
        )
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    write_immutable(output_dir / "selected_records.csv", handle.getvalue())


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.count <= 0:
        raise ValueError("--count must be positive")
    all_records = load_jsonl(args.index_jsonl)
    strict_records = [
        record for record in all_records if not strict_proxy_rejection_reasons(record)
    ]
    deduplicated, duplicate_pair_records = deduplicate_pairs(strict_records)
    selected, edges = select_balanced_records(
        deduplicated,
        count=args.count,
        seed=args.seed,
    )
    report = build_report(
        index_path=args.index_jsonl,
        all_records=all_records,
        strict_records=strict_records,
        deduplicated_records=deduplicated,
        duplicate_pair_records=duplicate_pair_records,
        selected=selected,
        edges=edges,
        seed=args.seed,
    )
    write_outputs(args.output_dir, selected, report)
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
