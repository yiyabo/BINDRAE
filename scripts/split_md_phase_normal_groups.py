#!/usr/bin/env python3
"""Create family- and scaffold-disjoint splits for an MD phase-normal cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.split_md_phase_normal_systems import (  # noqa: E402
    group_records,
    load_jsonl,
    summarize_split,
    write_lines,
)
from src.data.md_pilot_selection import parse_ca_records  # noqa: E402


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}

SPLITS = ("train", "val", "test")
BALANCE_METRICS = (
    "systems",
    "replicas",
    "confident_phase_points",
    "valid_residual_points",
    "active_interior_points",
)
BALANCE_WEIGHTS = {
    "systems": 4.0,
    "replicas": 2.0,
    "confident_phase_points": 1.0,
    "valid_residual_points": 1.0,
    "active_interior_points": 1.0,
}


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in self.parent}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def components(self) -> List[List[str]]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for value in sorted(self.parent):
            grouped[self.find(value)].append(value)
        return sorted(grouped.values(), key=lambda members: (-len(members), members))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sequence-identity", type=float, default=0.30)
    parser.add_argument("--sequence-coverage", type=float, default=0.80)
    parser.add_argument(
        "--diagnostic-thresholds", type=float, nargs="+", default=(0.30, 0.40, 0.50)
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--available-cache-manifest",
        type=Path,
        default=None,
        help="Optional OracleMotion manifest whose sample IDs must cover the corpus.",
    )
    return parser.parse_args()


def _base_system_id(record: Mapping[str, Any]) -> str:
    endpoints = record.get("endpoints", {})
    parents = {
        Path(str(endpoints.get(name, ""))).parent.name
        for name in ("apo_structure_path", "holo_structure_path")
    }
    parents.discard("")
    if len(parents) != 1:
        raise ValueError(
            f"Could not infer one endpoint system from {record.get('transition_id')}: "
            f"{sorted(parents)}"
        )
    return next(iter(parents))


def load_candidate_records(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            system_id = _base_system_id(row)
            if system_id in records:
                old = records[system_id]
                old_signature = (
                    old["endpoints"]["apo_structure_path"],
                    old["endpoints"]["holo_structure_path"],
                    old["ligand"]["canonical_smiles"],
                )
                new_signature = (
                    row["endpoints"]["apo_structure_path"],
                    row["endpoints"]["holo_structure_path"],
                    row["ligand"]["canonical_smiles"],
                )
                if old_signature != new_signature:
                    raise ValueError(f"Conflicting candidate records for {system_id}")
                continue
            records[system_id] = row
    return records


def extract_sequence(record: Mapping[str, Any], project_root: Path) -> str:
    relative_path = Path(str(record["endpoints"]["apo_structure_path"]))
    path = relative_path if relative_path.is_absolute() else project_root / relative_path
    parsed = parse_ca_records(path)
    sequence = "".join(AA3_TO_1.get(name.upper(), "X") for name in parsed["residue_names"])
    if not sequence:
        raise ValueError(f"No CA sequence found in {path}")
    return sequence


def murcko_scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise RuntimeError("RDKit is required for ligand scaffold grouping") from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse ligand SMILES: {smiles!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    if scaffold_smiles:
        return scaffold_smiles
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=False)
    return f"ACYCLIC:{canonical}"


def pairwise_sequence_similarities(
    sequences: Mapping[str, str],
) -> List[Tuple[str, str, float, float]]:
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError as exc:
        raise RuntimeError("Biopython is required for sequence-family grouping") from exc

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    system_ids = sorted(sequences)
    similarities = []
    for left_index, left_id in enumerate(system_ids):
        left = sequences[left_id]
        for right_id in system_ids[left_index + 1 :]:
            right = sequences[right_id]
            alignment = aligner.align(left, right)[0]
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
            similarities.append((left_id, right_id, identity, coverage))
    return similarities


def components_from_links(
    systems: Sequence[str], links: Iterable[Tuple[str, str]]
) -> List[List[str]]:
    union_find = UnionFind(systems)
    for left, right in links:
        union_find.union(left, right)
    return union_find.components()


def group_labels(prefix: str, components: Sequence[Sequence[str]]) -> Dict[str, str]:
    labels = {}
    for index, members in enumerate(components):
        digest = hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:10]
        label = f"{prefix}_{index:04d}_{digest}"
        for system_id in members:
            labels[system_id] = label
    return labels


def joint_components(
    systems: Sequence[str],
    family_by_system: Mapping[str, str],
    scaffold_by_system: Mapping[str, str],
) -> List[List[str]]:
    union_find = UnionFind(systems)
    for grouping in (family_by_system, scaffold_by_system):
        members_by_value: Dict[str, List[str]] = defaultdict(list)
        for system_id in systems:
            members_by_value[grouping[system_id]].append(system_id)
        for members in members_by_value.values():
            for system_id in members[1:]:
                union_find.union(members[0], system_id)
    return union_find.components()


def _record_metric(record: Mapping[str, Any], name: str) -> float:
    return float(record.get("audit_metrics", {}).get(name, 0.0))


def component_statistics(
    members: Sequence[str], grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, float]:
    records = [record for system_id in members for record in grouped[system_id]]
    return {
        "systems": float(len(members)),
        "replicas": float(len(records)),
        "confident_phase_points": sum(
            _record_metric(record, "confident_phase_points") for record in records
        ),
        "valid_residual_points": sum(
            _record_metric(record, "valid_residual_points") for record in records
        ),
        "active_interior_points": sum(
            _record_metric(record, "active_interior_points") for record in records
        ),
    }


def _assignment_score(
    totals_by_split: Mapping[str, Mapping[str, float]],
    corpus_totals: Mapping[str, float],
    fractions: Mapping[str, float],
) -> float:
    score = 0.0
    for metric in BALANCE_METRICS:
        total = corpus_totals[metric]
        if total <= 0:
            continue
        for split in SPLITS:
            observed = totals_by_split[split][metric] / total
            score += BALANCE_WEIGHTS[metric] * (observed - fractions[split]) ** 2
    return score


def assign_components(
    components: Sequence[Sequence[str]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    fractions: Mapping[str, float],
    seed: int,
) -> Dict[str, List[str]]:
    if len(components) < len(SPLITS):
        raise ValueError(
            f"Need at least {len(SPLITS)} joint groups for a three-way split; "
            f"found {len(components)}"
        )
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-8):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    if any(fractions[split] <= 0 for split in SPLITS):
        raise ValueError(f"Every split fraction must be positive, got {fractions}")

    statistics = [component_statistics(members, grouped) for members in components]
    corpus_totals = {
        metric: sum(values[metric] for values in statistics) for metric in BALANCE_METRICS
    }
    totals: Dict[str, Dict[str, float]] = {
        split: {metric: 0.0 for metric in BALANCE_METRICS} for split in SPLITS
    }
    component_owner: Dict[int, str] = {}
    rng = random.Random(seed)
    jitter = {
        (index, split): rng.random() for index in range(len(components)) for split in SPLITS
    }
    order = sorted(
        range(len(components)),
        key=lambda index: (
            -statistics[index]["systems"],
            -statistics[index]["replicas"],
            tuple(components[index]),
        ),
    )

    for index in order:
        candidates = []
        for split in SPLITS:
            trial = {name: dict(values) for name, values in totals.items()}
            for metric in BALANCE_METRICS:
                trial[split][metric] += statistics[index][metric]
            candidates.append(
                (
                    _assignment_score(trial, corpus_totals, fractions),
                    jitter[(index, split)],
                    split,
                    trial,
                )
            )
        _, _, owner, totals = min(candidates, key=lambda item: item[:3])
        component_owner[index] = owner

    assignments = {split: [] for split in SPLITS}
    for index, members in enumerate(components):
        assignments[component_owner[index]].extend(members)
    if any(not assignments[split] for split in SPLITS):
        raise RuntimeError(f"Greedy assignment produced an empty split: {assignments}")
    return {split: sorted(values) for split, values in assignments.items()}


def _load_available_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Feature manifest has no records list: {path}")
    return {str(record["sample_id"]) for record in records}


def _overlap_report(
    assignments: Mapping[str, Sequence[str]], values: Mapping[str, str]
) -> Dict[str, List[str]]:
    value_sets = {
        split: {values[system_id] for system_id in assignments[split]}
        for split in SPLITS
    }
    report = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            report[f"{left}_vs_{right}"] = sorted(value_sets[left] & value_sets[right])
    return report


def _replica_ids(
    systems: Sequence[str], grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> List[str]:
    return sorted(
        str(record["sample_id"]) for system_id in systems for record in grouped[system_id]
    )


def _top_group_sizes(values: Mapping[str, str], limit: int = 10) -> List[int]:
    counts: Dict[str, int] = defaultdict(int)
    for value in values.values():
        counts[value] += 1
    return sorted(counts.values(), reverse=True)[:limit]


def main() -> None:
    args = parse_args()
    if not 0.0 < args.sequence_identity <= 1.0:
        raise ValueError("sequence_identity must be in (0, 1]")
    if not 0.0 < args.sequence_coverage <= 1.0:
        raise ValueError("sequence_coverage must be in (0, 1]")

    cache_summary = json.loads((args.cache_dir / "summary.json").read_text())
    cache_records = load_jsonl(args.cache_dir / "manifest.jsonl")
    base_systems = sorted(set(str(value) for value in cache_summary["base_sample_ids"]))
    grouped = group_records(cache_records, base_systems)

    candidate_records = load_candidate_records(args.candidate_manifest)
    missing_metadata = sorted(set(base_systems) - set(candidate_records))
    if missing_metadata:
        raise ValueError(f"Systems missing candidate metadata: {missing_metadata}")
    candidate_records = {system_id: candidate_records[system_id] for system_id in base_systems}

    sequences = {
        system_id: extract_sequence(candidate_records[system_id], args.project_root)
        for system_id in base_systems
    }
    scaffolds = {
        system_id: murcko_scaffold(
            str(candidate_records[system_id]["ligand"]["canonical_smiles"])
        )
        for system_id in base_systems
    }
    similarities = pairwise_sequence_similarities(sequences)

    diagnostics = {}
    selected_family_components: List[List[str]] | None = None
    selected_joint_components: List[List[str]] | None = None
    selected_family_labels: Dict[str, str] | None = None
    thresholds = sorted(set(args.diagnostic_thresholds) | {args.sequence_identity})
    scaffold_labels = {
        system_id: f"scaf_{hashlib.sha256(scaffold.encode()).hexdigest()[:12]}"
        for system_id, scaffold in scaffolds.items()
    }
    for threshold in thresholds:
        links = [
            (left, right)
            for left, right, identity, coverage in similarities
            if identity >= threshold and coverage >= args.sequence_coverage
        ]
        family_components = components_from_links(base_systems, links)
        family_labels = group_labels("fam", family_components)
        combined_components = joint_components(
            base_systems, family_labels, scaffold_labels
        )
        diagnostics[f"{threshold:.3f}"] = {
            "sequence_edges": len(links),
            "family_groups": len(family_components),
            "largest_family_groups": [len(values) for values in family_components[:10]],
            "joint_groups": len(combined_components),
            "largest_joint_groups": [len(values) for values in combined_components[:10]],
            "three_way_split_feasible": len(combined_components) >= 3,
        }
        if math.isclose(threshold, args.sequence_identity, abs_tol=1e-12):
            selected_family_components = family_components
            selected_joint_components = combined_components
            selected_family_labels = family_labels

    if (
        selected_family_components is None
        or selected_joint_components is None
        or selected_family_labels is None
    ):
        raise RuntimeError("Selected sequence threshold was not evaluated")

    fractions = {
        "train": args.train_fraction,
        "val": args.val_fraction,
        "test": args.test_fraction,
    }
    assignments = assign_components(
        selected_joint_components, grouped, fractions, args.seed
    )
    joint_labels = group_labels("joint", selected_joint_components)

    if args.available_cache_manifest is not None:
        available = _load_available_ids(args.available_cache_manifest)
        missing_features = sorted(set(base_systems) - available)
        if missing_features:
            raise ValueError(
                f"Endpoint systems missing from feature cache: {missing_features}"
            )

    system_owner = {
        system_id: split
        for split in SPLITS
        for system_id in assignments[split]
    }
    system_overlap = {
        f"{left}_vs_{right}": sorted(set(assignments[left]) & set(assignments[right]))
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    }
    family_overlap = _overlap_report(assignments, selected_family_labels)
    scaffold_overlap = _overlap_report(assignments, scaffold_labels)
    joint_overlap = _overlap_report(assignments, joint_labels)
    leakage_passed = not any(
        values
        for report in (system_overlap, family_overlap, scaffold_overlap, joint_overlap)
        for values in report.values()
    )
    if not leakage_passed:
        raise RuntimeError("Family/scaffold leakage audit failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(args.output_dir / "all_systems.txt", base_systems)
    for split in SPLITS:
        write_lines(args.output_dir / f"{split}_systems.txt", assignments[split])
        write_lines(
            args.output_dir / f"{split}_replicas.txt",
            _replica_ids(assignments[split], grouped),
        )

    with (args.output_dir / "groups.jsonl").open("w") as handle:
        for system_id in base_systems:
            row = {
                "system_id": system_id,
                "split": system_owner[system_id],
                "sequence_sha256": hashlib.sha256(
                    sequences[system_id].encode()
                ).hexdigest(),
                "sequence_length": len(sequences[system_id]),
                "sequence_unknown_residues": sequences[system_id].count("X"),
                "family_group": selected_family_labels[system_id],
                "ligand_scaffold_group": scaffold_labels[system_id],
                "ligand_scaffold_smiles": scaffolds[system_id],
                "joint_group": joint_labels[system_id],
                "replicas": len(grouped[system_id]),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with (args.output_dir / "pairwise_sequence_identity.tsv").open("w") as handle:
        handle.write("system_a\tsystem_b\tidentity\tcoverage\n")
        for left, right, identity, coverage in similarities:
            handle.write(f"{left}\t{right}\t{identity:.8f}\t{coverage:.8f}\n")

    split_summaries = {}
    for split in SPLITS:
        summary = summarize_split(assignments[split], grouped)
        summary["system_fraction"] = len(assignments[split]) / len(base_systems)
        summary["replica_ids"] = _replica_ids(assignments[split], grouped)
        summary["family_groups"] = len(
            {selected_family_labels[system_id] for system_id in assignments[split]}
        )
        summary["ligand_scaffold_groups"] = len(
            {scaffold_labels[system_id] for system_id in assignments[split]}
        )
        summary["joint_groups"] = len(
            {joint_labels[system_id] for system_id in assignments[split]}
        )
        split_summaries[split] = summary

    report = {
        "schema_version": "md_phase_normal_group_split_v1",
        "cache_dir": str(args.cache_dir),
        "candidate_manifests": [str(path) for path in args.candidate_manifest],
        "seed": args.seed,
        "selection": "joint_family_scaffold_components_balanced_greedy",
        "criteria": {
            "sequence_identity": args.sequence_identity,
            "sequence_coverage": args.sequence_coverage,
            "sequence_alignment": "global_blosum62_gap_open_-10_extend_-0.5",
            "family_linkage": "single_linkage_connected_components",
            "ligand_scaffold": "canonical_nonisomeric_bemis_murcko",
            "acyclic_scaffold_fallback": "canonical_full_molecule",
            "target_fractions": fractions,
        },
        "corpus": {
            "systems": len(base_systems),
            "replicas": len(cache_records),
            "family_groups": len(selected_family_components),
            "ligand_scaffold_groups": len(set(scaffold_labels.values())),
            "joint_groups": len(selected_joint_components),
            "largest_family_groups": [
                len(values) for values in selected_family_components[:10]
            ],
            "largest_ligand_scaffold_groups": _top_group_sizes(scaffold_labels),
            "largest_joint_groups": [
                len(values) for values in selected_joint_components[:10]
            ],
            "unknown_sequence_residues": sum(
                sequence.count("X") for sequence in sequences.values()
            ),
        },
        "threshold_diagnostics": diagnostics,
        "splits": split_summaries,
        "leakage_audit": {
            "passed": leakage_passed,
            "system_overlap": system_overlap,
            "family_overlap": family_overlap,
            "ligand_scaffold_overlap": scaffold_overlap,
            "joint_group_overlap": joint_overlap,
        },
        "files": {
            "groups": "groups.jsonl",
            "pairwise_sequence_identity": "pairwise_sequence_identity.tsv",
            "all_systems": "all_systems.txt",
            **{
                f"{split}_{kind}": f"{split}_{kind}.txt"
                for split in SPLITS
                for kind in ("systems", "replicas")
            },
        },
    }
    report_path = args.output_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
