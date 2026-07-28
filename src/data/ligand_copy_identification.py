"""Identify which copy in a written ligand SDF is the deposited ligand.

The corpus holds 41,599 samples whose ``ligand.sdf`` contains several copies of
the same chemical component, because ``extract_ligand_from_pdb`` filtered HETATM
records by residue *name* only.  ``src/data/ligand_residue_selection`` fixes the
generator; this module repairs what the generator already wrote.

Repairing in place is a choice, not a necessity.  The AHoJ source is on disk
(``ahojdb_v2c/`` with ``db_entries.json``, 561 data shards and 119,208 PDB files),
so full regeneration is available.  It is not worth it: regeneration rewrites
``apo.pdb``, ``holo.pdb`` and above all ``esm.pt``, which would mean re-running
ESM over 91,189 samples to fix two files per sample.  Repairing in place touches
only ``ligand.sdf`` and ``ligand_coords.npy`` and leaves everything else
byte-identical, which is both cheaper and far easier to audit.

That needs an answer to one question -- *which* of the N copies is the residue the
sample is named after -- and the written files do not record it.  ``SDWriter``
drops PDB residue numbers, and ``ligand_coords.npy`` is a bare array.  The
deposited residue itself is recoverable from the local ``ahojdb_v2c/pdb_files``
(or from RCSB when an entry is missing), but its coordinates are in the
deposition frame while the SDF holds them after ``apply_rt``.

Two rejected approaches, recorded so they are not retried:

* **Pocket contacts.** "The bound copy is the one touching the protein" fails
  because the query-to-apo rigid transform maps distant copies to arbitrary
  positions, including inside the apo protein, where they score *more* contacts
  than the true ligand.  Measured on ``2hdr-A-4A3-506``: the top two fragments
  scored 195 and 157 contacts, a ratio of 1.2, and on ``5nhd-C-XYP-506`` three
  fragments scored 84, 59 and 54.  Not discriminating.
* **Fragment order.** Atoms are emitted in chain file order, so fragment ``k`` is
  the k-th same-resname residue.  Recovering ``k`` still needs the source file.

The approach that works is exact rather than heuristic.  ``apply_rt`` is a single
rigid transform applied to every ligand atom, and a rigid transform preserves
intramolecular distances.  So the sorted vector of pairwise distances *within* a
copy is invariant under it, and can be compared against the same vector computed
from the deposited residue -- without ever knowing the transform, and therefore
without needing the alignment matrices.  Measured on three multi-copy samples the
true copy matched to ``0.0001 A``, the float32 and SDF-precision floor, while the
closest wrong copy was off by ``0.028``-``0.237 A``: a margin of 280x or better.

Once one copy is identified the transform itself follows by Kabsch fit, which
extends the identification to a ligand spanning several residues -- an
oligosaccharide -- where per-fragment matching alone would not work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

SCHEMA_VERSION = "bindrae_ligand_copy_identification_v1"

#: Maximum acceptable deviation, in angstroms, between the sorted internal
#: distance vectors of the deposited residue and the matched copy.  The floor is
#: set by storage precision, not by chemistry: ``ligand_coords.npy`` is float32
#: and SDF coordinates carry four decimals, so a genuine match still shows
#: ~1e-4 A.  1e-3 leaves an order of magnitude of headroom.
EXACT_MATCH_TOLERANCE_ANGSTROM = 1.0e-3

#: The runner-up must be this many times worse than the match.  An *absolute*
#: margin is the wrong test and was tried first: the true copy sits at 1e-4 A
#: while a wrong copy of the same component sits at 0.028 A, so the absolute gap
#: is small in angstroms yet the ratio is 280x.
MIN_RUNNER_UP_RATIO = 10.0

#: Tolerance for pairing a transformed deposited atom with an SDF atom once the
#: rigid transform is known.  Loose relative to the match tolerance because it
#: only has to beat the nearest *other* atom, which is at least a bond length
#: away.
ATOM_PAIRING_TOLERANCE_ANGSTROM = 0.05


class CopyIdentificationError(RuntimeError):
    """The deposited residue could not be located in the written SDF.

    Raised rather than returning a best guess.  A wrong copy is
    indistinguishable downstream from the defect being repaired, so an
    unidentifiable sample must be refused and left untouched.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class CopyMatch:
    """Which candidate matched, how well, and by what margin."""

    candidate_index: int
    deviation_angstrom: float
    runner_up_deviation_angstrom: float
    candidate_count: int

    @property
    def runner_up_ratio(self) -> float:
        if not np.isfinite(self.runner_up_deviation_angstrom):
            return float("inf")
        if self.deviation_angstrom <= 0.0:
            return float("inf")
        return self.runner_up_deviation_angstrom / self.deviation_angstrom


@dataclass(frozen=True)
class LigandAtomSelection:
    """The SDF atom indices that constitute the deposited ligand."""

    atom_indices: tuple[int, ...]
    match: CopyMatch
    fit_rmsd_angstrom: float
    reference_atom_count: int
    observed_atom_count: int

    def as_diagnostics(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "selected_atoms": len(self.atom_indices),
            "observed_atoms": self.observed_atom_count,
            "reference_atoms": self.reference_atom_count,
            "matched_candidate": self.match.candidate_index,
            "candidate_count": self.match.candidate_count,
            "match_deviation_angstrom": float(f"{self.match.deviation_angstrom:.6f}"),
            "runner_up_deviation_angstrom": (
                None
                if not np.isfinite(self.match.runner_up_deviation_angstrom)
                else float(f"{self.match.runner_up_deviation_angstrom:.6f}")
            ),
            "runner_up_ratio": (
                None
                if not np.isfinite(self.match.runner_up_ratio)
                else float(f"{self.match.runner_up_ratio:.1f}")
            ),
            "fit_rmsd_angstrom": float(f"{self.fit_rmsd_angstrom:.6f}"),
        }


def sorted_internal_distances(coords: np.ndarray) -> np.ndarray:
    """The sorted vector of pairwise distances within one atom set.

    Invariant under rotation, translation *and* atom reordering, which is what
    makes it usable against an SDF whose atom names were dropped.
    """
    xyz = np.asarray(coords, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"coords must be (n_atoms, 3), got {xyz.shape}")
    if xyz.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    return np.sort(distances[np.triu_indices(xyz.shape[0], k=1)])


def match_copy_by_internal_geometry(
    reference_coords: np.ndarray,
    candidates: Sequence[np.ndarray],
    *,
    tolerance: float = EXACT_MATCH_TOLERANCE_ANGSTROM,
    min_runner_up_ratio: float = MIN_RUNNER_UP_RATIO,
) -> CopyMatch:
    """Find the candidate whose internal geometry is the reference's.

    Candidates with a different atom count are scored as infinitely far rather
    than skipped, so they still participate as runner-up evidence.

    Raises:
        CopyIdentificationError: when nothing matches within ``tolerance``, or
            when the runner-up is too close to call.
    """
    if not candidates:
        raise CopyIdentificationError("no_candidates", "no candidate copies supplied")

    reference = sorted_internal_distances(reference_coords)
    n_reference = np.asarray(reference_coords).shape[0]

    deviations: list[float] = []
    for candidate in candidates:
        coords = np.asarray(candidate, dtype=np.float64)
        if coords.shape[0] != n_reference:
            deviations.append(float("inf"))
            continue
        deviations.append(
            float(np.max(np.abs(sorted_internal_distances(coords) - reference)))
            if reference.size
            else 0.0
        )

    order = np.argsort(deviations)
    best_index = int(order[0])
    best = deviations[best_index]
    runner_up = deviations[int(order[1])] if len(order) > 1 else float("inf")

    if not np.isfinite(best) or best > tolerance:
        raise CopyIdentificationError(
            "no_copy_matched_reference_geometry",
            f"closest copy deviates by {best:.6f} A, above the {tolerance:g} A "
            "storage-precision tolerance; the deposited residue is not in this file",
        )

    match = CopyMatch(
        candidate_index=best_index,
        deviation_angstrom=best,
        runner_up_deviation_angstrom=runner_up,
        candidate_count=len(candidates),
    )
    if match.runner_up_ratio < min_runner_up_ratio:
        raise CopyIdentificationError(
            "copy_match_ambiguous",
            f"copy {best_index} deviates by {best:.6f} A but the runner-up is "
            f"{runner_up:.6f} A, a ratio of {match.runner_up_ratio:.1f} below the "
            f"required {min_runner_up_ratio:g}; refusing to guess",
        )
    return match


def kabsch_transform(
    mobile: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares rigid transform taking ``mobile`` onto ``target``.

    Returns ``(rotation, translation, rmsd)`` with ``mobile @ rotation.T +
    translation`` approximating ``target``.  Rows must already correspond.
    """
    P = np.asarray(mobile, dtype=np.float64)
    Q = np.asarray(target, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch: {P.shape} vs {Q.shape}")
    if P.shape[0] == 0:
        raise ValueError("cannot fit an empty atom set")

    p_center = P.mean(axis=0)
    q_center = Q.mean(axis=0)
    covariance = (P - p_center).T @ (Q - q_center)
    u, _, vt = np.linalg.svd(covariance)
    reflection = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, reflection])
    rotation = vt.T @ correction @ u.T
    translation = q_center - rotation @ p_center
    fitted = P @ rotation.T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - Q) ** 2, axis=1))))
    return rotation, translation, rmsd


def _correspond_by_internal_distances(
    reference_coords: np.ndarray, observed_coords: np.ndarray
) -> np.ndarray:
    """Pair reference atoms with observed atoms of one matched copy.

    Both sets describe the same rigid body, so each atom's sorted vector of
    distances to the rest of its own set is a signature that survives the
    transform and any reordering.
    """
    reference = np.asarray(reference_coords, dtype=np.float64)
    observed = np.asarray(observed_coords, dtype=np.float64)

    def signatures(xyz: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        return np.sort(distances, axis=1)

    reference_signatures = signatures(reference)
    observed_signatures = signatures(observed)

    cost = np.linalg.norm(
        reference_signatures[:, None, :] - observed_signatures[None, :, :], axis=-1
    )
    # Greedy assignment: the signatures are near-exact, so the minimum is
    # unambiguous except for genuinely symmetric atoms, where either choice
    # yields the same rigid fit.
    pairing = np.full(reference.shape[0], -1, dtype=int)
    remaining = set(range(observed.shape[0]))
    for reference_index in np.argsort(cost.min(axis=1)):
        candidates = [i for i in np.argsort(cost[reference_index]) if i in remaining]
        if not candidates:
            raise CopyIdentificationError(
                "atom_correspondence_failed",
                "ran out of observed atoms while pairing the matched copy",
            )
        chosen = candidates[0]
        pairing[reference_index] = chosen
        remaining.discard(chosen)
    return pairing


def identify_ligand_atom_indices(
    reference_residue_coords: Sequence[np.ndarray],
    observed_coords: np.ndarray,
    candidate_atom_groups: Sequence[Sequence[int]],
    *,
    tolerance: float = EXACT_MATCH_TOLERANCE_ANGSTROM,
    min_runner_up_ratio: float = MIN_RUNNER_UP_RATIO,
    pairing_tolerance: float = ATOM_PAIRING_TOLERANCE_ANGSTROM,
) -> LigandAtomSelection:
    """Locate the deposited ligand's atoms inside a written SDF.

    Args:
        reference_residue_coords: heavy-atom coordinates of the deposited ligand,
            one array per residue.  A single-residue ligand supplies one array; an
            oligosaccharide supplies one per sugar unit, in any order.
        observed_coords: ``(n_atoms, 3)`` from the written SDF, in file order.
        candidate_atom_groups: index groups over ``observed_coords`` -- normally
            connected fragments -- each a candidate copy.

    The first reference residue is matched against the candidate groups to
    recover the rigid transform; the remaining residues are then located by
    transforming them and pairing atoms by position, which is what lets a
    multi-residue ligand be selected even when the SDF holds it as one fragment.

    Raises:
        CopyIdentificationError: on no match, an ambiguous match, or a reference
            atom that cannot be paired.  Never returns a partial selection.
    """
    if not reference_residue_coords:
        raise CopyIdentificationError("no_reference", "no deposited residue supplied")

    observed = np.asarray(observed_coords, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError(f"observed_coords must be (n_atoms, 3), got {observed.shape}")

    reference_order = sorted(
        range(len(reference_residue_coords)),
        key=lambda i: -np.asarray(reference_residue_coords[i]).shape[0],
    )
    candidate_coords = [observed[list(group)] for group in candidate_atom_groups]

    # Whether a multi-residue ligand occupies one SDF fragment or several depends
    # on whether proximity perception found the linking bonds, and both occur in
    # the corpus. Try the whole reference against the groups first -- that is the
    # one-fragment case -- then the largest single residue, which is the
    # one-fragment-per-residue case. A ligand split across *some but not all* of
    # its fragments matches neither and is refused rather than guessed at.
    whole = np.vstack([np.asarray(r, dtype=np.float64) for r in reference_residue_coords])
    attempts: list[tuple[np.ndarray, Exception]] = []
    match = None
    anchor = whole
    for probe in (whole, np.asarray(reference_residue_coords[reference_order[0]],
                                    dtype=np.float64)):
        if match is not None:
            break
        try:
            match = match_copy_by_internal_geometry(
                probe,
                candidate_coords,
                tolerance=tolerance,
                min_runner_up_ratio=min_runner_up_ratio,
            )
            anchor = probe
        except CopyIdentificationError as exc:
            attempts.append((probe, exc))
    if match is None:
        raise attempts[-1][1]

    matched_group = list(candidate_atom_groups[match.candidate_index])
    matched_coords = observed[matched_group]
    pairing = _correspond_by_internal_distances(anchor, matched_coords)
    rotation, translation, fit_rmsd = kabsch_transform(
        anchor, matched_coords[pairing]
    )

    selected: list[int] = []
    for reference_index in reference_order:
        residue = np.asarray(reference_residue_coords[reference_index], dtype=np.float64)
        transformed = residue @ rotation.T + translation
        for atom_index, position in enumerate(transformed):
            distances = np.linalg.norm(observed - position, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > pairing_tolerance:
                raise CopyIdentificationError(
                    "reference_atom_unpaired",
                    f"deposited residue {reference_index} atom {atom_index} has no "
                    f"SDF atom within {pairing_tolerance:g} A "
                    f"(nearest {distances[nearest]:.4f} A)",
                )
            if nearest in selected:
                raise CopyIdentificationError(
                    "duplicate_atom_pairing",
                    f"SDF atom {nearest} claimed by two deposited atoms",
                )
            selected.append(nearest)

    reference_atom_count = sum(
        int(np.asarray(residue).shape[0]) for residue in reference_residue_coords
    )
    return LigandAtomSelection(
        atom_indices=tuple(sorted(selected)),
        match=match,
        fit_rmsd_angstrom=fit_rmsd,
        reference_atom_count=reference_atom_count,
        observed_atom_count=int(observed.shape[0]),
    )
