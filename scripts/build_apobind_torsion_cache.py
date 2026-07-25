#!/usr/bin/env python3
"""Build and audit canonical torsion caches for a frozen APObind replica matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_ahojdb_torsions import TorsionExtractor  # noqa: E402
from src.data.residue_alignment import align_residue_names  # noqa: E402
from src.data.residue_identity import (  # noqa: E402
    RESIDUE_ALIGNMENT_VERSION,
    STANDARD_AA3_TO_1,
    load_residue_keys,
    residue_identity_hash,
    residue_names_to_sequence,
    serialize_residue_key,
)


SCHEMA_VERSION = "bindrae_apobind_torsion_cache_audit_v1"
MATRIX_SCHEMA_VERSION = "bindrae_md_replica_matrix_v1"
REQUIRED_ARRAY_FIELDS = (
    "phi",
    "psi",
    "omega",
    "chi",
    "bb_mask",
    "chi_mask",
    "omega_cis_trans",
    "residue_keys",
    "residue_names",
    "sequence_str",
    "residue_alignment_version",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="APObind triplet root containing samples/<sample_id>/{apo,holo}.pdb",
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-systems", type=int, default=8)
    parser.add_argument("--expected-replicas-per-system", type=int, default=2)
    parser.add_argument("--min-pair-mapping-fraction", type=float, default=0.95)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty replica matrix: {path}")
    return rows


def ordered_matrix_systems(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_systems: int,
    expected_replicas_per_system: int,
    min_pair_mapping_fraction: float,
) -> list[str]:
    if expected_systems < 1 or expected_replicas_per_system < 1:
        raise ValueError("Expected system and replica counts must be positive")
    if not 0.0 < min_pair_mapping_fraction <= 1.0:
        raise ValueError("min-pair-mapping-fraction must be in (0, 1]")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ordered: list[str] = []
    for index, row in enumerate(rows):
        if row.get("schema_version") != MATRIX_SCHEMA_VERSION:
            raise ValueError(
                f"Unexpected matrix schema at row {index}: {row.get('schema_version')}"
            )
        if int(row.get("matrix_index", -1)) != index:
            raise ValueError(f"Matrix row/index mismatch at {index}")
        system_id = str(row.get("system_sample_id") or "")
        if not system_id or Path(system_id).name != system_id:
            raise ValueError(f"Invalid system_sample_id at row {index}: {system_id!r}")
        replica_index = int(row.get("replica_index", -1))
        expected_sample_id = f"{system_id}__silver_r{replica_index:02d}"
        if replica_index < 0 or row.get("sample_id") != expected_sample_id:
            raise ValueError(f"Replica identity mismatch at row {index}")
        mapping_fraction = float(
            (row.get("protocol") or {}).get("min_mapping_fraction", -1.0)
        )
        if mapping_fraction != min_pair_mapping_fraction:
            raise ValueError(
                f"Matrix row {index} mapping fraction {mapping_fraction} does not "
                f"match frozen value {min_pair_mapping_fraction}"
            )
        if system_id not in grouped:
            ordered.append(system_id)
        grouped[system_id].append(row)

    if len(grouped) != expected_systems:
        raise ValueError(f"Matrix has {len(grouped)} systems, expected {expected_systems}")
    for system_id, system_rows in grouped.items():
        replicas = [int(row["replica_index"]) for row in system_rows]
        if len(replicas) != expected_replicas_per_system or len(set(replicas)) != len(
            replicas
        ):
            raise ValueError(
                f"{system_id} has replica indices {replicas}, expected "
                f"{expected_replicas_per_system} unique replicas"
            )
    return ordered


def validate_torsion_payload(
    payload: Mapping[str, Any], *, label: str
) -> tuple[list[tuple[str, int, str]], list[str]]:
    missing = [field for field in REQUIRED_ARRAY_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"{label} missing torsion fields: {missing}")

    version = str(np.asarray(payload["residue_alignment_version"]).item())
    if version != RESIDUE_ALIGNMENT_VERSION:
        raise ValueError(
            f"{label} residue_alignment_version={version!r}, "
            f"expected {RESIDUE_ALIGNMENT_VERSION!r}"
        )
    keys = load_residue_keys(payload)
    if not keys:
        raise ValueError(f"{label} has no canonical residue_keys")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} contains duplicate residue_keys")
    names = [str(value).strip().upper() for value in np.asarray(payload["residue_names"]).tolist()]
    n_residues = len(keys)
    if len(names) != n_residues:
        raise ValueError(
            f"{label} has {n_residues} residue keys but {len(names)} residue names"
        )
    invalid_names = sorted(set(names) - set(STANDARD_AA3_TO_1))
    if invalid_names:
        raise ValueError(f"{label} has non-standard residue names: {invalid_names}")
    if "n_residues" in payload and int(np.asarray(payload["n_residues"]).item()) != n_residues:
        raise ValueError(f"{label} n_residues does not match residue_keys")

    for field in ("phi", "psi", "omega", "omega_cis_trans"):
        values = np.asarray(payload[field])
        if values.shape != (n_residues,):
            raise ValueError(
                f"{label} {field} shape={values.shape}, expected {(n_residues,)}"
            )
    for field in ("chi", "chi_mask"):
        values = np.asarray(payload[field])
        if values.shape != (n_residues, 4):
            raise ValueError(
                f"{label} {field} shape={values.shape}, expected {(n_residues, 4)}"
            )
    bb_mask = np.asarray(payload["bb_mask"])
    if bb_mask.shape not in {(n_residues,), (n_residues, 3)}:
        raise ValueError(
            f"{label} bb_mask shape={bb_mask.shape}, expected {(n_residues,)} "
            f"or {(n_residues, 3)}"
        )
    for field in ("phi", "psi", "omega", "chi"):
        if not np.all(np.isfinite(np.asarray(payload[field], dtype=np.float64))):
            raise ValueError(f"{label} {field} contains non-finite values")

    sequence = str(np.asarray(payload["sequence_str"]).item())
    expected_sequence = residue_names_to_sequence(names)
    if sequence != expected_sequence:
        raise ValueError(f"{label} sequence_str does not match residue_names")
    return [tuple(key) for key in keys], names


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {field: np.asarray(data[field]).copy() for field in data.files}


def payloads_match(
    expected: Mapping[str, Any], existing: Mapping[str, Any]
) -> bool:
    return all(
        field in existing
        and np.array_equal(np.asarray(expected[field]), np.asarray(existing[field]))
        for field in expected
    )


def atomic_savez(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_endpoint_cache(
    *, extractor: TorsionExtractor, pdb_path: Path, cache_path: Path
) -> tuple[dict[str, Any], list[tuple[str, int, str]], list[str]]:
    payload = extractor.extract(pdb_path)
    if payload is None:
        raise ValueError(f"Torsion extraction failed for {pdb_path}")
    keys, names = validate_torsion_payload(payload, label=str(pdb_path))

    if cache_path.exists():
        existing = load_cache(cache_path)
        validate_torsion_payload(existing, label=str(cache_path))
        if not payloads_match(payload, existing):
            raise FileExistsError(
                f"Existing torsion cache differs from a fresh extraction: {cache_path}"
            )
        write_status = "reused_identical"
    else:
        atomic_savez(cache_path, payload)
        write_status = "created"

    persisted = load_cache(cache_path)
    persisted_keys, persisted_names = validate_torsion_payload(
        persisted, label=str(cache_path)
    )
    if persisted_keys != keys or persisted_names != names:
        raise RuntimeError(f"Persisted torsion identity differs from extraction: {cache_path}")
    serialized_keys = [serialize_residue_key(key) for key in keys]
    summary = {
        "pdb": str(pdb_path),
        "pdb_sha256": file_sha256(pdb_path),
        "cache": str(cache_path),
        "cache_sha256": file_sha256(cache_path),
        "write_status": write_status,
        "n_residues": len(keys),
        "residue_alignment_version": RESIDUE_ALIGNMENT_VERSION,
        "residue_identity_hash": residue_identity_hash(keys),
        "residue_keys_sha256": _text_sha256(serialized_keys),
        "residue_names_sha256": _text_sha256(names),
        "sequence_sha256": hashlib.sha256(
            residue_names_to_sequence(names).encode("ascii")
        ).hexdigest(),
        "first_residue_key": serialized_keys[0],
        "last_residue_key": serialized_keys[-1],
    }
    return summary, keys, names


def run(args: argparse.Namespace) -> dict[str, Any]:
    matrix_path = args.matrix.resolve()
    data_dir = args.data_dir.resolve()
    samples_dir = data_dir / "samples"
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    if not samples_dir.is_dir():
        raise FileNotFoundError(samples_dir)
    if args.output_report.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {args.output_report}")

    rows = load_jsonl(matrix_path)
    system_ids = ordered_matrix_systems(
        rows,
        expected_systems=args.expected_systems,
        expected_replicas_per_system=args.expected_replicas_per_system,
        min_pair_mapping_fraction=args.min_pair_mapping_fraction,
    )
    extractor = TorsionExtractor()
    systems = []
    for system_id in system_ids:
        sample_dir = samples_dir / system_id
        if sample_dir.parent != samples_dir:
            raise ValueError(f"Unsafe APObind sample path: {sample_dir}")
        endpoint_records: dict[str, dict[str, Any]] = {}
        endpoint_axes: dict[str, tuple[list[tuple[str, int, str]], list[str]]] = {}
        for endpoint in ("apo", "holo"):
            pdb_path = sample_dir / f"{endpoint}.pdb"
            cache_path = sample_dir / f"torsion_{endpoint}.npz"
            if not pdb_path.is_file():
                raise FileNotFoundError(pdb_path)
            summary, keys, names = build_endpoint_cache(
                extractor=extractor,
                pdb_path=pdb_path,
                cache_path=cache_path,
            )
            endpoint_records[endpoint] = summary
            endpoint_axes[endpoint] = (keys, names)

        apo_keys, apo_names = endpoint_axes["apo"]
        holo_keys, holo_names = endpoint_axes["holo"]
        alignment = align_residue_names(apo_names, holo_names)
        if alignment.symmetric_mapping_fraction < args.min_pair_mapping_fraction:
            raise ValueError(
                f"{system_id} apo/holo exact mapping fraction "
                f"{alignment.symmetric_mapping_fraction:.6f} is below "
                f"{args.min_pair_mapping_fraction:.6f}"
            )
        apo_by_key = dict(zip(apo_keys, apo_names))
        holo_by_key = dict(zip(holo_keys, holo_names))
        shared_keys = sorted(set(apo_by_key) & set(holo_by_key))
        systems.append(
            {
                "sample_id": system_id,
                "endpoints": endpoint_records,
                "pair_identity": {
                    "exact_sequence_matches": len(alignment.exact_pairs),
                    "aligned_pair_count": alignment.aligned_pair_count,
                    "sequence_identity": alignment.sequence_identity,
                    "apo_mapping_fraction": alignment.reference_mapping_fraction,
                    "holo_mapping_fraction": alignment.query_mapping_fraction,
                    "symmetric_mapping_fraction": alignment.symmetric_mapping_fraction,
                    "shared_residue_keys": len(shared_keys),
                    "shared_key_and_name_matches": sum(
                        apo_by_key[key] == holo_by_key[key] for key in shared_keys
                    ),
                },
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "matrix": str(matrix_path),
        "matrix_sha256": file_sha256(matrix_path),
        "data_dir": str(data_dir),
        "counts": {
            "systems": len(systems),
            "endpoint_caches": 2 * len(systems),
            "created": sum(
                endpoint["write_status"] == "created"
                for system in systems
                for endpoint in system["endpoints"].values()
            ),
            "reused_identical": sum(
                endpoint["write_status"] == "reused_identical"
                for system in systems
                for endpoint in system["endpoints"].values()
            ),
        },
        "contract": {
            "residue_alignment_version": RESIDUE_ALIGNMENT_VERSION,
            "min_pair_mapping_fraction": args.min_pair_mapping_fraction,
            "expected_replicas_per_system": args.expected_replicas_per_system,
        },
        "systems": systems,
        "claim_boundary": (
            "These caches restore the canonical residue-axis engineering input for "
            "target export. They do not change any pull, path, atomistic, or "
            "scientific acceptance gate and do not by themselves create a Path-3 label."
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_name(
        f".{args.output_report.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output_report)
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
