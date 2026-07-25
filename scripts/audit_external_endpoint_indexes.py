#!/usr/bin/env python3
"""Build a conservative index of external endpoint-pair annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "bindrae_external_endpoint_index_audit_v1"
PDB_ID_PATTERN = re.compile(r"^[0-9A-Za-z]{4}$")
PDB_CHAIN_PATTERN = re.compile(r"^([0-9A-Za-z]{4})(?:[_:-](.*))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pscdb-csv", type=Path)
    parser.add_argument("--apobind-csv", type=Path)
    parser.add_argument("--codnas-q-csv", type=Path)
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--reference-manifest",
        type=Path,
        help=(
            "Optional AHoJ-derived JSONL manifest. Matching remains PDB-pair-only "
            "unless both sources expose the same exact ligand/site identity."
        ),
    )
    reference_group.add_argument(
        "--reference-endpoint-csv",
        type=Path,
        help=(
            "Optional AHoJ endpoint_index.csv with apo_pdb, holo_pdb, and "
            "sample_id columns. Rows missing both endpoint IDs are counted and "
            "excluded from pair matching."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_pdb_id(value: object, *, context: str) -> str:
    pdb_id = str(value or "").strip().upper()
    if not PDB_ID_PATTERN.fullmatch(pdb_id):
        raise ValueError(f"Invalid four-character PDB ID for {context}: {value!r}")
    return pdb_id


def normalize_chains(value: object, *, compact: bool = False) -> list[str]:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"no/data", "none", "n/a"}:
        return []
    tokens = [token for token in re.split(r"[\s,;|/]+", raw) if token]
    if compact and len(tokens) == 1 and len(tokens[0]) > 1:
        tokens = list(tokens[0])
    return list(dict.fromkeys(token.upper() for token in tokens))


def parse_pdb_chain_value(
    value: object,
    *,
    context: str,
    compact_chains: bool = False,
) -> tuple[str, list[str]]:
    raw = str(value or "").strip()
    match = PDB_CHAIN_PATTERN.fullmatch(raw)
    if match is None:
        raise ValueError(f"Invalid PDB/chain value for {context}: {value!r}")
    return (
        normalize_pdb_id(match.group(1), context=context),
        normalize_chains(match.group(2), compact=compact_chains),
    )


def endpoint(
    *,
    role: str,
    pdb_id: str,
    chains: Sequence[str],
    source_value: object,
) -> dict[str, Any]:
    return {
        "role": role,
        "pdb_id": pdb_id,
        "chains": list(chains),
        "source_value": str(source_value or "").strip(),
    }


def pair_key(record: Mapping[str, Any]) -> tuple[str, str]:
    endpoints = list(record["endpoints"])
    if len(endpoints) != 2:
        raise ValueError(f"Expected two endpoints in {record.get('record_key')}")
    return str(endpoints[0]["pdb_id"]), str(endpoints[1]["pdb_id"])


def _read_csv(
    path: Path,
    *,
    delimiter: str,
    required_columns: Iterable[str],
) -> list[tuple[int, dict[str, str]]]:
    rows: list[tuple[int, dict[str, str]]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = set(reader.fieldnames or [])
        missing = sorted(set(required_columns) - columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            rows.append((reader.line_num, dict(row)))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def _float_or_none(value: object) -> float | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"no/data", "none", "n/a"}:
        return None
    return float(raw)


def _int_or_none(value: object) -> int | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"no/data", "none", "n/a"}:
        return None
    return int(raw)


def _tokens(value: object, *, delimiter: str | None = None) -> list[str]:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"no/data", "none", "n/a"}:
        return []
    if delimiter is not None:
        values = raw.split(delimiter)
    else:
        values = re.split(r"[\s,;|]+", raw)
    return [value.strip() for value in values if value.strip()]


def _site_identity(
    *,
    record_key: str,
    granularity: str,
    labels: Iterable[str],
) -> dict[str, Any]:
    return {
        "granularity": granularity,
        "labels": list(dict.fromkeys(str(label) for label in labels if str(label))),
        "exact_ligand_site_available": False,
        "record_identity_key": record_key,
    }


def load_pscdb(path: Path) -> list[dict[str, Any]]:
    required = {
        "PSCID",
        "FreeID",
        "BoundID",
        "Ligands",
        "Component No",
        "Type of motion",
    }
    rows = _read_csv(path, delimiter=",", required_columns=required)
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for source_row, row in rows:
        pscid = str(row["PSCID"] or "").strip()
        if not pscid:
            raise ValueError(f"Missing PSCID at {path}:{source_row}")
        grouped[pscid].append((source_row, row))

    records: list[dict[str, Any]] = []
    for pscid, component_rows in grouped.items():
        normalized_pairs: set[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = set()
        for source_row, row in component_rows:
            free_pdb, free_chains = parse_pdb_chain_value(
                row["FreeID"],
                context=f"{path}:{source_row} FreeID",
                compact_chains=True,
            )
            bound_pdb, bound_chains = parse_pdb_chain_value(
                row["BoundID"],
                context=f"{path}:{source_row} BoundID",
                compact_chains=True,
            )
            normalized_pairs.add(
                (free_pdb, tuple(free_chains), bound_pdb, tuple(bound_chains))
            )
        if len(normalized_pairs) != 1:
            raise ValueError(
                f"PSCID {pscid} has inconsistent endpoint definitions: "
                f"{sorted(normalized_pairs)}"
            )
        free_pdb, free_chains_raw, bound_pdb, bound_chains_raw = next(
            iter(normalized_pairs)
        )
        first = component_rows[0][1]
        record_key = f"pscdb:{pscid}"
        ligand_labels = list(
            dict.fromkeys(
                str(row.get("Ligands") or "").strip()
                for _, row in component_rows
                if str(row.get("Ligands") or "").strip()
            )
        )
        components = [
            {
                "source_row": source_row,
                "component_no": str(row.get("Component No") or "").strip() or None,
                "type_of_motion": str(row.get("Type of motion") or "").strip()
                or None,
                "ligand_binding": str(row.get("Ligand binding") or "").strip()
                or None,
                "rmsd": _float_or_none(row.get("RMSD")),
                "fixed_segment": str(row.get("Fixed segment") or "").strip()
                or None,
                "moving_segment": str(row.get("Moving segment") or "").strip()
                or None,
            }
            for source_row, row in component_rows
        ]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_key": record_key,
                "source": "pscdb",
                "source_rows": [source_row for source_row, _ in component_rows],
                "direction_semantics": "free_to_bound",
                "apo_holo_orientation_available": True,
                "pair_kind": "experimental_free_bound_endpoints",
                "endpoints": [
                    endpoint(
                        role="free",
                        pdb_id=free_pdb,
                        chains=free_chains_raw,
                        source_value=first["FreeID"],
                    ),
                    endpoint(
                        role="bound",
                        pdb_id=bound_pdb,
                        chains=bound_chains_raw,
                        source_value=first["BoundID"],
                    ),
                ],
                "site_identity": _site_identity(
                    record_key=record_key,
                    granularity="ligand_label_without_residue_site",
                    labels=ligand_labels,
                ),
                "metadata": {
                    "pscid": pscid,
                    "classification": str(first.get("Classification") or "").strip()
                    or None,
                    "protein_name": str(first.get("Protein name") or "").strip()
                    or None,
                    "ligands_raw": ligand_labels,
                    "distance": _float_or_none(first.get("Distance")),
                    "components": components,
                },
            }
        )
    return records


def load_apobind(path: Path) -> list[dict[str, Any]]:
    required = {"holo_id", "holo_chains", "apo_id", "apo_chains"}
    rows = _read_csv(path, delimiter=",", required_columns=required)
    records: list[dict[str, Any]] = []
    for source_row, row in rows:
        apo_pdb = normalize_pdb_id(
            row["apo_id"], context=f"{path}:{source_row} apo_id"
        )
        holo_pdb = normalize_pdb_id(
            row["holo_id"], context=f"{path}:{source_row} holo_id"
        )
        source_index = str(row.get("") or "").strip()
        record_key = f"apobind:{source_index or source_row}:{apo_pdb}:{holo_pdb}"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_key": record_key,
                "source": "apobind",
                "source_rows": [source_row],
                "direction_semantics": "apo_to_holo",
                "apo_holo_orientation_available": True,
                "pair_kind": "experimental_apo_holo_endpoints",
                "endpoints": [
                    endpoint(
                        role="apo",
                        pdb_id=apo_pdb,
                        chains=normalize_chains(row["apo_chains"]),
                        source_value=row["apo_id"],
                    ),
                    endpoint(
                        role="holo",
                        pdb_id=holo_pdb,
                        chains=normalize_chains(row["holo_chains"]),
                        source_value=row["holo_id"],
                    ),
                ],
                "site_identity": _site_identity(
                    record_key=record_key,
                    granularity="binding_residue_signature_without_ligand_identity",
                    labels=[],
                ),
                "metadata": {
                    "source_index": source_index or None,
                    "apo_resolution": _float_or_none(row.get("apo_resolution")),
                    "sequence_identity": _float_or_none(row.get("sequence_identity")),
                    "sequence_coverage": _float_or_none(row.get("sequence_coverage")),
                    "backbone_rmsd": _float_or_none(row.get("backbone_rmsd")),
                    "tmscore": _float_or_none(row.get("tmscore")),
                    "side_chain_rmsd": _float_or_none(row.get("side_chain_rmsd")),
                    "binding_site": {
                        "apo_residues": _tokens(row.get("apo_bind_res")),
                        "apo_indices": _tokens(row.get("apo_bind_indices")),
                        "holo_residues": _tokens(row.get("holo_bind_res")),
                        "holo_indices": _tokens(row.get("holo_bind_indices")),
                    },
                },
            }
        )
    return records


def load_codnas_q(path: Path) -> list[dict[str, Any]]:
    required = {
        "Cluster ID",
        "PDB_ID_query",
        "PDB_ID_target",
        "Query_Chain_ID",
        "Target_Chain_ID",
        "Query_ligands",
        "Target_ligands",
    }
    rows = _read_csv(path, delimiter=";", required_columns=required)
    records: list[dict[str, Any]] = []
    for source_row, row in rows:
        query_rep = normalize_pdb_id(
            row["PDB_ID_query"], context=f"{path}:{source_row} PDB_ID_query"
        )
        target_rep = normalize_pdb_id(
            row["PDB_ID_target"], context=f"{path}:{source_row} PDB_ID_target"
        )
        query_chain_raw = str(row.get("Query_Chain_ID") or "").strip()
        target_chain_raw = str(row.get("Target_Chain_ID") or "").strip()
        maximum_pair_available = (
            PDB_CHAIN_PATTERN.fullmatch(query_chain_raw) is not None
            and PDB_CHAIN_PATTERN.fullmatch(target_chain_raw) is not None
        )
        if maximum_pair_available:
            query_pdb, query_chains = parse_pdb_chain_value(
                query_chain_raw, context=f"{path}:{source_row} Query_Chain_ID"
            )
            target_pdb, target_chains = parse_pdb_chain_value(
                target_chain_raw, context=f"{path}:{source_row} Target_Chain_ID"
            )
            pair_kind = "maximum_tertiary_pair"
        else:
            query_pdb, query_chains = query_rep, []
            target_pdb, target_chains = target_rep, []
            pair_kind = "representative_alignment_pair_fallback"
        cluster_id = str(row["Cluster ID"] or "").strip()
        if not cluster_id:
            raise ValueError(f"Missing Cluster ID at {path}:{source_row}")
        record_key = f"codnas_q:{cluster_id}"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_key": record_key,
                "source": "codnas_q",
                "source_rows": [source_row],
                "direction_semantics": "query_to_target_not_apo_holo",
                "apo_holo_orientation_available": False,
                "pair_kind": pair_kind,
                "endpoints": [
                    endpoint(
                        role="query",
                        pdb_id=query_pdb,
                        chains=query_chains,
                        source_value=query_chain_raw if maximum_pair_available else query_rep,
                    ),
                    endpoint(
                        role="target",
                        pdb_id=target_pdb,
                        chains=target_chains,
                        source_value=target_chain_raw if maximum_pair_available else target_rep,
                    ),
                ],
                "site_identity": _site_identity(
                    record_key=record_key,
                    granularity="cluster_record_without_endpoint_site_identity",
                    labels=[],
                ),
                "metadata": {
                    "cluster_id": cluster_id,
                    "maximum_tertiary_pair_available": maximum_pair_available,
                    "maximum_tertiary_rmsd": _float_or_none(row.get("maxRMSD_T")),
                    "group": str(row.get("Group") or "").strip() or None,
                    "num_conformers": _int_or_none(row.get("num_of_conformers")),
                    "conformer_pdb_ids": _tokens(
                        row.get("conformers_ids"), delimiter="|"
                    ),
                    "representative_alignment_pair": {
                        "query_pdb_id": query_rep,
                        "target_pdb_id": target_rep,
                        "query_biological_assembly": str(
                            row.get("Biological_Assembly_query") or ""
                        ).strip()
                        or None,
                        "target_biological_assembly": str(
                            row.get("Biological_Assembly_target") or ""
                        ).strip()
                        or None,
                        "query_ligands_raw": str(row.get("Query_ligands") or "").strip()
                        or None,
                        "target_ligands_raw": str(row.get("Target_ligands") or "").strip()
                        or None,
                    },
                },
            }
        )
    return records


def load_reference_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            endpoints = dict(row.get("endpoints") or {})
            apo = normalize_pdb_id(
                endpoints.get("apo_pdb_id"),
                context=f"{path}:{line_number} endpoints.apo_pdb_id",
            )
            holo = normalize_pdb_id(
                endpoints.get("holo_pdb_id"),
                context=f"{path}:{line_number} endpoints.holo_pdb_id",
            )
            records.append(
                {
                    "source_row": line_number,
                    "reference_id": str(
                        row.get("transition_id") or row.get("sample_id") or line_number
                    ),
                    "pair": (apo, holo),
                    "ligand_comp_id": str(
                        (row.get("ligand") or {}).get("comp_id") or ""
                    ).strip()
                    or None,
                }
            )
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def load_reference_endpoint_csv(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = _read_csv(
        path,
        delimiter=",",
        required_columns={"apo_pdb", "holo_pdb", "sample_id"},
    )
    records: list[dict[str, Any]] = []
    missing_endpoint_rows = 0
    for source_row, row in rows:
        apo_raw = str(row.get("apo_pdb") or "").strip()
        holo_raw = str(row.get("holo_pdb") or "").strip()
        if not apo_raw and not holo_raw:
            missing_endpoint_rows += 1
            continue
        if not apo_raw or not holo_raw:
            raise ValueError(
                f"Only one endpoint PDB ID is present at {path}:{source_row}"
            )
        records.append(
            {
                "source_row": source_row,
                "reference_id": str(row["sample_id"] or source_row),
                "pair": (
                    normalize_pdb_id(
                        apo_raw, context=f"{path}:{source_row} apo_pdb"
                    ),
                    normalize_pdb_id(
                        holo_raw, context=f"{path}:{source_row} holo_pdb"
                    ),
                ),
                "ligand_comp_id": str(row.get("ligand_resname") or "").strip()
                or None,
            }
        )
    if not records:
        raise ValueError(f"No valid endpoint pairs in {path}")
    return records, {
        "source_rows": len(rows),
        "valid_endpoint_rows": len(records),
        "missing_endpoint_rows": missing_endpoint_rows,
    }


def _source_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    directed = Counter(pair_key(record) for record in records)
    undirected = Counter(tuple(sorted(pair)) for pair in directed.elements())
    directed_keys = set(directed)
    reciprocal_groups = {
        tuple(sorted(pair))
        for pair in directed_keys
        if pair[0] != pair[1] and (pair[1], pair[0]) in directed_keys
    }
    return {
        "records": len(records),
        "source_rows": sum(len(record["source_rows"]) for record in records),
        "unique_directed_pdb_pairs": len(directed),
        "unique_undirected_pdb_pairs": len(undirected),
        "records_beyond_unique_directed_pairs": len(records) - len(directed),
        "directed_pairs_with_multiple_source_records": sum(
            count > 1 for count in directed.values()
        ),
        "reciprocal_pair_groups": len(reciprocal_groups),
        "exact_ligand_site_records": sum(
            bool(record["site_identity"]["exact_ligand_site_available"])
            for record in records
        ),
        "site_identity_granularity": dict(
            sorted(
                Counter(
                    str(record["site_identity"]["granularity"])
                    for record in records
                ).items()
            )
        ),
    }


def _cross_source_pair_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    directed_pairs = {pair_key(record) for record in records}
    sources_by_undirected_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        pair = pair_key(record)
        sources_by_undirected_pair[tuple(sorted(pair))].add(str(record["source"]))
    source_combinations = Counter(
        "+".join(sorted(sources)) for sources in sources_by_undirected_pair.values()
    )
    return {
        "records": len(records),
        "unique_directed_pdb_pairs": len(directed_pairs),
        "unique_undirected_pdb_pairs": len(sources_by_undirected_pair),
        "undirected_pair_groups_with_multiple_sources": sum(
            len(sources) > 1 for sources in sources_by_undirected_pair.values()
        ),
        "source_combination_pair_groups": dict(sorted(source_combinations.items())),
    }


def match_reference_pairs(
    records: Sequence[dict[str, Any]],
    reference_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in reference_records:
        reference_by_pair[tuple(row["pair"])].append(str(row["reference_id"]))

    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_source_pairs: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    overall: Counter[str] = Counter()
    overall_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        pair = pair_key(record)
        reverse = (pair[1], pair[0])
        undirected_pair = tuple(sorted(pair))
        directed_ids = sorted(reference_by_pair.get(pair, []))
        reverse_ids = sorted(reference_by_pair.get(reverse, []))
        matched = bool(directed_ids or reverse_ids)
        record["reference_pair_match"] = {
            "match_level": "pdb_pair_only" if matched else "none",
            "directed_reference_ids": directed_ids,
            "reverse_reference_ids": reverse_ids,
            "exact_ligand_site_match": None,
        }
        flags = {
            "directed_pdb_pair_matches": bool(directed_ids),
            "reverse_pdb_pair_matches": bool(reverse_ids),
            "any_direction_pdb_pair_matches": matched,
            "pdb_pair_nonmatches": not matched,
        }
        for key, enabled in flags.items():
            if enabled:
                overall[key] += 1
                by_source[str(record["source"])][key] += 1
                overall_pairs[key].add(undirected_pair)
                by_source_pairs[str(record["source"])][key].add(undirected_pair)
        overall_pairs["all"].add(undirected_pair)
        by_source_pairs[str(record["source"])]["all"].add(undirected_pair)

    def complete(
        counter: Mapping[str, int],
        pair_sets: Mapping[str, set[tuple[str, str]]],
        total: int,
    ) -> dict[str, Any]:
        return {
            "external_records": total,
            "unique_undirected_pdb_pairs": len(pair_sets.get("all", set())),
            "directed_pdb_pair_matches": int(
                counter.get("directed_pdb_pair_matches", 0)
            ),
            "reverse_pdb_pair_matches": int(
                counter.get("reverse_pdb_pair_matches", 0)
            ),
            "any_direction_pdb_pair_matches": int(
                counter.get("any_direction_pdb_pair_matches", 0)
            ),
            "pdb_pair_nonmatches": int(counter.get("pdb_pair_nonmatches", 0)),
            "unique_undirected_pdb_pair_matches": len(
                pair_sets.get("any_direction_pdb_pair_matches", set())
            ),
            "unique_undirected_pdb_pair_nonmatches": len(
                pair_sets.get("pdb_pair_nonmatches", set())
            ),
            "exact_ligand_site_matches": None,
            "net_new_systems": None,
        }

    return {
        "reference_records": len(reference_records),
        "reference_unique_directed_pdb_pairs": len(reference_by_pair),
        "overall": complete(overall, overall_pairs, len(records)),
        "by_source": {
            source: complete(
                by_source[source],
                by_source_pairs[source],
                sum(record["source"] == source for record in records),
            )
            for source in sorted({str(record["source"]) for record in records})
        },
        "interpretation": (
            "PDB-pair matches are overlap candidates only. PDB-pair nonmatches are "
            "not net-new systems until ligand/site identity, sequence/mapping, and "
            "AHoJ provenance are resolved."
        ),
    }


def _unique_undirected_pair_count(records: Sequence[Mapping[str, Any]]) -> int:
    return len({tuple(sorted(pair_key(record))) for record in records})


def _apobind_metadata_funnel(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    apobind = [record for record in records if record["source"] == "apobind"]
    if not apobind or any("reference_pair_match" not in record for record in apobind):
        return None

    def valid_unit_interval(value: object) -> bool:
        return value is not None and 0.0 <= float(value) <= 1.0

    stages: list[tuple[str, list[Mapping[str, Any]]]] = []
    current = [
        record
        for record in apobind
        if record["reference_pair_match"]["match_level"] == "none"
    ]
    stages.append(("reference_pdb_pair_nonmatch", current))
    current = [
        record
        for record in current
        if len(record["endpoints"][0]["chains"]) == 1
        and len(record["endpoints"][1]["chains"]) == 1
    ]
    stages.append(("single_chain_endpoints", current))
    current = [
        record
        for record in current
        if valid_unit_interval(record["metadata"]["sequence_identity"])
        and valid_unit_interval(record["metadata"]["sequence_coverage"])
        and float(record["metadata"]["sequence_identity"]) >= 0.95
        and float(record["metadata"]["sequence_coverage"]) >= 0.95
    ]
    stages.append(("valid_identity_and_coverage_at_least_0_95", current))
    current = [
        record
        for record in current
        if record["metadata"]["backbone_rmsd"] is not None
        and 0.35 <= float(record["metadata"]["backbone_rmsd"]) <= 5.0
    ]
    stages.append(("backbone_rmsd_0_35_to_5_angstrom", current))
    current = [
        record
        for record in current
        if record["metadata"]["tmscore"] is not None
        and 0.5 <= float(record["metadata"]["tmscore"]) <= 1.0
    ]
    stages.append(("tmscore_0_5_to_1", current))
    current = [
        record
        for record in current
        if record["metadata"]["apo_resolution"] is not None
        and float(record["metadata"]["apo_resolution"]) > 0.0
    ]
    stages.append(("positive_apo_resolution", current))
    current = [
        record
        for record in current
        if record["metadata"]["binding_site"]["apo_indices"]
        and record["metadata"]["binding_site"]["holo_indices"]
    ]
    stages.append(("nonempty_binding_site_signature", current))

    sequence_identity_values = [
        record["metadata"]["sequence_identity"] for record in apobind
    ]
    sequence_coverage_values = [
        record["metadata"]["sequence_coverage"] for record in apobind
    ]
    return {
        "status": "metadata_proxy_only",
        "stages": [
            {
                "name": name,
                "records": len(stage_records),
                "unique_undirected_pdb_pairs": _unique_undirected_pair_count(
                    stage_records
                ),
            }
            for name, stage_records in stages
        ],
        "source_quality_flags": {
            "sequence_identity_above_1": sum(
                value is not None and float(value) > 1.0
                for value in sequence_identity_values
            ),
            "sequence_coverage_above_1": sum(
                value is not None and float(value) > 1.0
                for value in sequence_coverage_values
            ),
        },
        "proxy_records": len(current),
        "proxy_unique_undirected_pdb_pairs": _unique_undirected_pair_count(current),
        "eligible_net_new_systems": False,
        "missing_gates": [
            "exact_ligand_and_site_identity",
            "production_residue_mapping",
            "frozen_family_and_ligand_scaffold_leakage",
            "prepared_context_and_atomistic_path",
            "two_accepted_independent_md_replicas",
        ],
    }


def write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite different audit artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [
        ("pscdb", args.pscdb_csv, load_pscdb),
        ("apobind", args.apobind_csv, load_apobind),
        ("codnas_q", args.codnas_q_csv, load_codnas_q),
    ]
    selected = [(name, path, loader) for name, path, loader in inputs if path]
    if not selected:
        raise ValueError("At least one external CSV input is required")

    records: list[dict[str, Any]] = []
    input_reports: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    for name, path, loader in selected:
        source_records = loader(path)
        records.extend(source_records)
        source_summaries[name] = _source_summary(source_records)
        input_reports.append(
            {
                "source": name,
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    record_keys = [str(record["record_key"]) for record in records]
    duplicate_record_keys = sorted(
        key for key, count in Counter(record_keys).items() if count > 1
    )
    if duplicate_record_keys:
        raise ValueError(f"Duplicate source record keys: {duplicate_record_keys[:8]}")

    reference_manifest = getattr(args, "reference_manifest", None)
    reference_endpoint_csv = getattr(args, "reference_endpoint_csv", None)
    if reference_manifest is not None and reference_endpoint_csv is not None:
        raise ValueError(
            "Use only one of --reference-manifest and --reference-endpoint-csv"
        )
    matching = None
    reference_input = None
    if reference_manifest is not None:
        reference_records = load_reference_manifest(reference_manifest)
        matching = match_reference_pairs(records, reference_records)
        reference_input = {
            "kind": "transition_manifest_jsonl",
            "path": str(reference_manifest),
            "sha256": file_sha256(reference_manifest),
            "source_rows": len(reference_records),
            "valid_endpoint_rows": len(reference_records),
            "missing_endpoint_rows": 0,
        }
    elif reference_endpoint_csv is not None:
        reference_records, reference_counts = load_reference_endpoint_csv(
            reference_endpoint_csv
        )
        matching = match_reference_pairs(records, reference_records)
        reference_input = {
            "kind": "ahoj_endpoint_index_csv",
            "path": str(reference_endpoint_csv),
            "sha256": file_sha256(reference_endpoint_csv),
            **reference_counts,
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": input_reports,
        "reference_input": reference_input,
        "reference_manifest": (
            {
                "path": str(reference_manifest),
                "sha256": file_sha256(reference_manifest),
            }
            if reference_manifest is not None
            else None
        ),
        "counts": {
            "index_records": len(records),
            "source_rows": sum(
                len(record["source_rows"]) for record in records
            ),
        },
        "source_summaries": source_summaries,
        "cross_source_pdb_pair_summary": _cross_source_pair_summary(records),
        "source_metadata_funnels": {
            "apobind": _apobind_metadata_funnel(records),
        },
        "reference_matching": matching,
        "claim_boundary": {
            "contains_intermediate_paths": False,
            "pdb_pair_nonmatch_is_net_new_system": False,
            "deduplicate_distinct_source_records_by_pdb_pair": False,
            "exact_ligand_site_overlap_required_for_net_new_count": True,
            "codnas_query_target_is_apo_holo": False,
        },
        "files": {
            "index_jsonl": str(args.output_dir / "external_endpoint_index.jsonl"),
            "report_json": str(args.output_dir / "report.json"),
        },
    }
    write_immutable(
        args.output_dir / "external_endpoint_index.jsonl",
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )
    write_immutable(
        args.output_dir / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    return report


def main() -> None:
    report = run_audit(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
