#!/usr/bin/env python3
"""Build an endpoint-pretraining subset disjoint from frozen holdout groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_HOLDOUT_SEQUENCES: list[tuple[str, str]] = []
_SEQUENCE_IDENTITY = 0.30
_SEQUENCE_COVERAGE = 0.80
_ALIGNER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sample-list", type=Path, required=True)
    parser.add_argument("--holdout-sample-list", type=Path, action="append", required=True)
    parser.add_argument("--samples-dir", type=Path, default=Path("processed_data/triplets/samples"))
    parser.add_argument("--output-sample-list", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--metadata-cache", type=Path, default=None)
    parser.add_argument("--sequence-identity", type=float, default=0.30)
    parser.add_argument("--sequence-coverage", type=float, default=0.80)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def read_ids(paths: Iterable[Path]) -> list[str]:
    values: list[str] = []
    for path in paths:
        values.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate sample IDs across lists: {[str(path) for path in paths]}")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ligand_scaffold(path: Path) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
    molecule = supplier[0] if len(supplier) else None
    if molecule is None:
        raise ValueError(f"RDKit could not parse {path}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    if scaffold_smiles:
        return scaffold_smiles
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=False)
    if not canonical:
        raise ValueError(f"RDKit produced an empty canonical molecule for {path}")
    return f"ACYCLIC:{canonical}"


def extract_metadata(task: tuple[str, str]) -> dict[str, Any]:
    sample_id, samples_dir_raw = task
    sample_dir = Path(samples_dir_raw) / sample_id
    try:
        torsion_path = sample_dir / "torsion_apo.npz"
        with np.load(torsion_path, allow_pickle=False) as data:
            if "sequence_str" not in data:
                raise KeyError(f"{torsion_path} missing sequence_str")
            sequence = str(np.asarray(data["sequence_str"]).item()).strip()
            if "residue_keys" in data and len(data["residue_keys"]) != len(sequence):
                raise ValueError(
                    f"{torsion_path} residue keys {len(data['residue_keys'])} "
                    f"!= sequence length {len(sequence)}"
                )
        if not sequence:
            raise ValueError(f"{torsion_path} contains an empty sequence")
        scaffold = ligand_scaffold(sample_dir / "ligand.sdf")
        return {
            "sample_id": sample_id,
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "scaffold": scaffold,
            "error": None,
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "sequence": None,
            "sequence_sha256": None,
            "scaffold": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_metadata_cache(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_metadata_cache(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def init_alignment_worker(
    holdout_sequences: list[tuple[str, str]],
    identity_threshold: float,
    coverage_threshold: float,
) -> None:
    global _HOLDOUT_SEQUENCES, _SEQUENCE_IDENTITY, _SEQUENCE_COVERAGE, _ALIGNER
    from Bio.Align import PairwiseAligner, substitution_matrices

    _HOLDOUT_SEQUENCES = holdout_sequences
    _SEQUENCE_IDENTITY = float(identity_threshold)
    _SEQUENCE_COVERAGE = float(coverage_threshold)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    _ALIGNER = aligner


def sequence_identity_coverage(left: str, right: str) -> tuple[float, float]:
    alignment = _ALIGNER.align(left, right)[0]
    aligned = 0
    matches = 0
    for left_block, right_block in zip(*alignment.aligned):
        left_start, left_end = (int(value) for value in left_block)
        right_start, right_end = (int(value) for value in right_block)
        block_length = min(left_end - left_start, right_end - right_start)
        aligned += block_length
        matches += sum(
            left[left_start + offset] == right[right_start + offset]
            for offset in range(block_length)
        )
    identity = matches / aligned if aligned else 0.0
    coverage = aligned / max(len(left), len(right))
    return identity, coverage


def find_holdout_family(sequence: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for holdout_id, holdout_sequence in _HOLDOUT_SEQUENCES:
        if min(len(sequence), len(holdout_sequence)) / max(
            len(sequence), len(holdout_sequence)
        ) < _SEQUENCE_COVERAGE:
            continue
        identity, coverage = sequence_identity_coverage(sequence, holdout_sequence)
        if identity >= _SEQUENCE_IDENTITY and coverage >= _SEQUENCE_COVERAGE:
            candidate = {
                "holdout_id": holdout_id,
                "identity": identity,
                "coverage": coverage,
            }
            if best is None or (identity, coverage, holdout_id) > (
                best["identity"],
                best["coverage"],
                best["holdout_id"],
            ):
                best = candidate
    return {"sequence": sequence, "family_match": best}


def count_reasons(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.sequence_identity <= 1.0:
        raise ValueError("sequence-identity must be in (0, 1]")
    if not 0.0 < args.sequence_coverage <= 1.0:
        raise ValueError("sequence-coverage must be in (0, 1]")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    input_ids = read_ids([args.input_sample_list])
    if args.max_samples > 0:
        input_ids = input_ids[: args.max_samples]
    holdout_ids = read_ids(args.holdout_sample_list)
    all_ids = list(dict.fromkeys([*input_ids, *holdout_ids]))

    metadata_rows: list[dict[str, Any]]
    if args.metadata_cache is not None and args.metadata_cache.is_file():
        cached = load_metadata_cache(args.metadata_cache)
        by_id = {str(row["sample_id"]): row for row in cached}
        missing = sorted(set(all_ids) - set(by_id))
        if missing:
            raise ValueError(
                f"Metadata cache {args.metadata_cache} misses {len(missing)} IDs: {missing[:8]}"
            )
        metadata_rows = [by_id[sample_id] for sample_id in all_ids]
    else:
        tasks = [(sample_id, str(args.samples_dir)) for sample_id in all_ids]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            metadata_rows = list(executor.map(extract_metadata, tasks, chunksize=8))
        if args.metadata_cache is not None:
            write_metadata_cache(args.metadata_cache, metadata_rows)

    metadata = {str(row["sample_id"]): row for row in metadata_rows}
    invalid_holdout = [metadata[sample_id] for sample_id in holdout_ids if metadata[sample_id]["error"]]
    if invalid_holdout:
        raise ValueError(f"Holdout metadata is invalid: {invalid_holdout[:3]}")

    holdout_sequences = sorted(
        {(sample_id, str(metadata[sample_id]["sequence"])) for sample_id in holdout_ids}
    )
    holdout_scaffolds = {str(metadata[sample_id]["scaffold"]) for sample_id in holdout_ids}
    valid_sequences = sorted(
        {
            str(metadata[sample_id]["sequence"])
            for sample_id in input_ids
            if metadata[sample_id]["error"] is None
        }
    )
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_alignment_worker,
        initargs=(holdout_sequences, args.sequence_identity, args.sequence_coverage),
    ) as executor:
        family_rows = list(executor.map(find_holdout_family, valid_sequences, chunksize=8))
    family_by_sequence = {
        str(row["sequence"]): row["family_match"] for row in family_rows
    }

    excluded = []
    retained = []
    holdout_set = set(holdout_ids)
    for sample_id in input_ids:
        row = metadata[sample_id]
        reasons = []
        if row["error"] is not None:
            reasons.append("invalid_metadata")
        if sample_id in holdout_set:
            reasons.append("exact_holdout_id")
        family_match = (
            family_by_sequence.get(str(row["sequence"]))
            if row["sequence"] is not None
            else None
        )
        if family_match is not None:
            reasons.append("holdout_protein_family")
        if row["scaffold"] is not None and str(row["scaffold"]) in holdout_scaffolds:
            reasons.append("holdout_ligand_scaffold")
        if reasons:
            excluded.append(
                {
                    "sample_id": sample_id,
                    "reasons": reasons,
                    "family_match": family_match,
                    "scaffold": row["scaffold"],
                    "error": row["error"],
                }
            )
        else:
            retained.append(sample_id)

    if set(retained) & holdout_set:
        raise RuntimeError("Exact holdout IDs survived leakage filtering")
    args.output_sample_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_sample_list.write_text("".join(f"{sample_id}\n" for sample_id in retained))
    report = {
        "schema_version": "stage2_leakage_clean_endpoint_subset_v1",
        "input_sample_list": str(args.input_sample_list),
        "input_sample_list_sha256": file_sha256(args.input_sample_list),
        "holdout_sample_lists": [str(path) for path in args.holdout_sample_list],
        "criteria": {
            "sequence_alignment": "global_blosum62_gap_open_-10_extend_-0.5",
            "sequence_identity": args.sequence_identity,
            "sequence_coverage": args.sequence_coverage,
            "ligand_scaffold": "canonical_nonisomeric_bemis_murcko",
            "acyclic_scaffold_fallback": "canonical_full_molecule",
        },
        "counts": {
            "input": len(input_ids),
            "holdout": len(holdout_ids),
            "unique_input_sequences": len(valid_sequences),
            "holdout_scaffolds": len(holdout_scaffolds),
            "retained": len(retained),
            "excluded": len(excluded),
        },
        "exclusion_reason_counts": count_reasons(excluded),
        "retained_sample_list": str(args.output_sample_list),
        "metadata_cache": str(args.metadata_cache) if args.metadata_cache else None,
        "excluded": excluded,
        "audit": {
            "exact_holdout_overlap": sorted(set(retained) & holdout_set),
            "passed": not bool(set(retained) & holdout_set),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report["counts"], **report["exclusion_reason_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
