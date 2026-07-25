"""Sequence-based residue alignment for endpoint and topology mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from src.data.residue_identity import STANDARD_AA3_TO_1


RESIDUE_ALIASES = {
    "ASH": "ASP",
    "CYM": "CYS",
    "CYX": "CYS",
    "GLH": "GLU",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "LYN": "LYS",
    "MSE": "MET",
}


def canonical_resname(name: object) -> str:
    value = str(name).strip().upper()
    return RESIDUE_ALIASES.get(value, value)


@dataclass(frozen=True)
class ExactResidueAlignment:
    """Exact residue matches retained from one global sequence alignment."""

    exact_pairs: Tuple[Tuple[int, int], ...]
    aligned_pair_count: int
    sequence_identity: float
    reference_mapping_fraction: float
    query_mapping_fraction: float
    symmetric_mapping_fraction: float


def align_residue_names(
    reference_names: Sequence[object], query_names: Sequence[object]
) -> ExactResidueAlignment:
    """Globally align two residue sequences and retain exact standard-AA matches."""

    if not reference_names or not query_names:
        raise ValueError("Residue sequences must both be non-empty")

    from Bio.Align import PairwiseAligner

    reference = [canonical_resname(name) for name in reference_names]
    query = [canonical_resname(name) for name in query_names]
    reference_sequence = "".join(STANDARD_AA3_TO_1.get(name, "X") for name in reference)
    query_sequence = "".join(STANDARD_AA3_TO_1.get(name, "X") for name in query)

    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(reference_sequence, query_sequence)[0]

    aligned_pairs = []
    for (reference_start, reference_end), (query_start, query_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        aligned_pairs.extend(
            zip(
                range(int(reference_start), int(reference_end)),
                range(int(query_start), int(query_end)),
            )
        )
    exact_pairs = tuple(
        (reference_index, query_index)
        for reference_index, query_index in aligned_pairs
        if reference[reference_index] == query[query_index]
        and reference[reference_index] in STANDARD_AA3_TO_1
    )
    exact_count = len(exact_pairs)
    aligned_count = len(aligned_pairs)
    reference_fraction = exact_count / len(reference)
    query_fraction = exact_count / len(query)
    return ExactResidueAlignment(
        exact_pairs=exact_pairs,
        aligned_pair_count=aligned_count,
        sequence_identity=(exact_count / aligned_count if aligned_count else 0.0),
        reference_mapping_fraction=reference_fraction,
        query_mapping_fraction=query_fraction,
        symmetric_mapping_fraction=min(reference_fraction, query_fraction),
    )


def triple_exact_residue_alignment(
    reference_names: Sequence[object],
    pivot_names: Sequence[object],
    query_names: Sequence[object],
) -> Tuple[Tuple[int, int, int], ...]:
    """Map reference and query residues through exact matches on a pivot axis."""

    reference_to_pivot = align_residue_names(reference_names, pivot_names).exact_pairs
    pivot_to_query = dict(align_residue_names(pivot_names, query_names).exact_pairs)
    return tuple(
        (reference_index, pivot_index, pivot_to_query[pivot_index])
        for reference_index, pivot_index in reference_to_pivot
        if pivot_index in pivot_to_query
    )
