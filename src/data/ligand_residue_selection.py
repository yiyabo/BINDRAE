"""Select the residues that actually constitute the ligand.

``extract_ligand_from_pdb`` in ``scripts/prepare_ahojdb_triplets.py`` selects
HETATM records by chain and residue *name* only:

    if residue.get_resname().strip() != ligand_resname:
        continue

A deposited chain routinely contains several copies of the same chemical
component -- a second ATP at a regulatory site, a string of crystallisation
phosphates, ten xylose units in a groove -- so every copy anywhere in the chain
was concatenated into one "ligand".  Measured on the corpus, 41,599 of 91,189
samples (45.6%) hold more than one copy, and a twenty-sample spot check found
fragment centroids a median of 72 A and a maximum of 195 A apart.  No pocket is
195 A across.

The defect is silent.  Nothing downstream validates that the ligand is one
molecule, so ``ligand_coords.npy`` carries every scattered copy and Stage-2
consumes all of them as its ligand conditioning input.

The sibling function ``extract_ligand_by_id`` already filters by residue number
correctly.  This module supplies the same discipline to the fallback path, plus
the part ``extract_ligand_by_id`` does not handle: a ligand may legitimately span
several residues when they are *covalently linked*.  An oligosaccharide is one
chemical entity written as one residue per sugar unit, and truncating it to the
seed residue would be a different corruption of the same input.

So selection is a covalent closure, not a single-residue filter:

    seed residue (resname + resnum, exact)
      -> transitively add residues bonded to the selection
      -> stop at anything not covalently reachable

Bonding is judged by summed covalent radii rather than a flat cutoff, because a
flat cutoff cannot separate a disulfide (2.05 A) from zinc-oxygen coordination
(2.0-2.1 A).  Metal-ion residues are excluded from traversal: a metal inside the
component (heme iron) shares the seed residue and needs no traversal, whereas a
separate site ion is not part of the ligand.

This module deliberately does **not** guess.  If the named residue is absent the
caller gets an exception, never a silent fallback to "collect everything" -- that
fallback is the defect this module exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "bindrae_ligand_residue_selection_v1"

#: Covalent radii in angstroms (Cordero et al., Dalton Trans. 2008).  Only the
#: elements that occur in deposited small-molecule ligands are listed; anything
#: else falls back to :data:`DEFAULT_COVALENT_RADIUS_ANGSTROM`.
COVALENT_RADII_ANGSTROM: Mapping[str, float] = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "SI": 1.11, "P": 1.07, "S": 1.05, "CL": 1.02,
    "AS": 1.19, "SE": 1.20, "BR": 1.20, "I": 1.39, "TE": 1.38,
}

#: Used when an element is missing from :data:`COVALENT_RADII_ANGSTROM`.  Chosen
#: small on purpose: under-linking drops a genuine partner and shows up as a
#: multi-fragment warning, while over-linking silently re-creates the defect.
DEFAULT_COVALENT_RADIUS_ANGSTROM = 0.80

#: Multiplies the summed radii to give the bond-distance ceiling.  1.15 admits
#: every organic single bond (C-C 1.54 against a 1.75 ceiling, S-S 2.05 against
#: 2.42) while staying below metal-ligand coordination for first-row metals.
COVALENT_TOLERANCE = 1.15

#: Elements treated as ions rather than ligand scaffold when they appear as a
#: residue of their own.  A metal that belongs to the component -- heme iron,
#: the cobalt in B12 -- is written inside the component's own residue and is
#: therefore never traversed *to*, so excluding these costs nothing there.
METAL_ELEMENTS: frozenset[str] = frozenset({
    "LI", "BE", "NA", "MG", "AL", "K", "CA", "SC", "TI", "V", "CR", "MN",
    "FE", "CO", "NI", "CU", "ZN", "GA", "RB", "SR", "Y", "ZR", "NB", "MO",
    "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "CS", "BA", "LA", "CE",
    "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
    "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB",
    "BI", "RA", "TH", "U", "PU",
})


class LigandResidueNotFound(LookupError):
    """The requested ``(resname, resnum)`` is absent from the supplied residues.

    Raised instead of degrading to a name-only match.  A name-only match is what
    produced the 195 A "ligands" this module exists to prevent, so the failure
    must stay visible to the caller.
    """


@dataclass(frozen=True)
class ResidueView:
    """A residue reduced to what selection needs: identity plus heavy atoms.

    Deliberately free of Biopython so the selection logic is testable from plain
    arrays.  ``coords`` is ``(n_atoms, 3)`` and ``elements`` is parallel to it,
    uppercase, hydrogens already dropped by the caller.
    """

    resname: str
    resnum: int
    coords: np.ndarray
    elements: tuple[str, ...]
    insertion_code: str = " "
    chain_id: str = ""

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"coords must be (n_atoms, 3), got {np.asarray(self.coords).shape}"
            )
        if len(self.elements) != coords.shape[0]:
            raise ValueError(
                f"elements has {len(self.elements)} entries for {coords.shape[0]} atoms"
            )
        object.__setattr__(self, "coords", coords)

    @property
    def centroid(self) -> np.ndarray:
        return self.coords.mean(axis=0)

    @property
    def is_metal_ion(self) -> bool:
        """A lone metal atom standing as its own residue."""
        return len(self.elements) == 1 and self.elements[0].upper() in METAL_ELEMENTS


@dataclass(frozen=True)
class LigandSelection:
    """Which residues form the ligand, and what was left out.

    ``discarded_same_resname`` is the headline diagnostic: it counts the copies
    the old name-only filter would have concatenated.  A non-zero value on a
    previously generated sample means that sample's ligand input was wrong.
    """

    selected_indices: tuple[int, ...]
    seed_index: int
    covalent_partner_indices: tuple[int, ...]
    discarded_same_resname: tuple[int, ...]
    discarded_max_centroid_distance_angstrom: float
    selected_atom_count: int
    seed_atom_count: int
    seed_ambiguous: bool = False
    unknown_elements: tuple[str, ...] = field(default_factory=tuple)

    @property
    def defect_present(self) -> bool:
        """Whether the name-only filter would have produced a different ligand."""
        return bool(self.discarded_same_resname)

    def as_diagnostics(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "selected_residues": len(self.selected_indices),
            "covalent_partners": len(self.covalent_partner_indices),
            "discarded_same_resname": len(self.discarded_same_resname),
            "discarded_max_centroid_distance_angstrom": round(
                self.discarded_max_centroid_distance_angstrom, 3
            ),
            "selected_atom_count": self.selected_atom_count,
            "seed_atom_count": self.seed_atom_count,
            "seed_ambiguous": self.seed_ambiguous,
            "unknown_elements": list(self.unknown_elements),
            "defect_present": self.defect_present,
        }


def _radius(element: str, unknown: set[str]) -> float:
    key = element.upper()
    try:
        return COVALENT_RADII_ANGSTROM[key]
    except KeyError:
        if key not in METAL_ELEMENTS:
            unknown.add(key)
        return DEFAULT_COVALENT_RADIUS_ANGSTROM


def residues_are_bonded(
    left: ResidueView,
    right: ResidueView,
    *,
    tolerance: float = COVALENT_TOLERANCE,
    unknown: set[str] | None = None,
) -> bool:
    """Whether any heavy-atom pair across the two residues is a covalent bond.

    Uses per-pair summed covalent radii rather than one distance, so a disulfide
    bridge and zinc coordination -- which are barely distinguishable by raw
    distance -- land on opposite sides.
    """
    sink: set[str] = unknown if unknown is not None else set()
    if left.coords.size == 0 or right.coords.size == 0:
        return False

    distances = np.linalg.norm(
        left.coords[:, None, :] - right.coords[None, :, :], axis=-1
    )
    left_radii = np.array([_radius(e, sink) for e in left.elements])
    right_radii = np.array([_radius(e, sink) for e in right.elements])
    ceilings = (left_radii[:, None] + right_radii[None, :]) * tolerance
    return bool(np.any(distances <= ceilings))


def select_ligand_residues(
    residues: Sequence[ResidueView],
    *,
    resname: str,
    resnum: int,
    insertion_code: str | None = None,
    tolerance: float = COVALENT_TOLERANCE,
    traverse_metal_ions: bool = False,
    covalent_closure: bool = True,
) -> LigandSelection:
    """Select the seed residue and everything covalently attached to it.

    ``residues`` should already exclude waters and standard polymer residues;
    this function does not re-derive that classification.

    ``covalent_closure=False`` restricts the result to the seed residue alone.
    The closure is right for an oligosaccharide written one unit per residue, but
    it can over-reach: on ``3hxy-A-MDN-443`` it grew a 9-atom diphosphonate into
    68 atoms.  Callers that can verify their result -- the extraction repair
    checks the selection against what was already written -- should try both and
    keep whichever reproduces the deposited ligand, rather than deciding here.

    Raises:
        LigandResidueNotFound: if no residue matches ``resname`` and ``resnum``.
            The caller must not fall back to a name-only match.
    """
    target = resname.strip().upper()
    seed_candidates = [
        index
        for index, residue in enumerate(residues)
        if residue.resname.strip().upper() == target and residue.resnum == resnum
        and (insertion_code is None or residue.insertion_code == insertion_code)
    ]
    if not seed_candidates:
        available = sorted({r.resname.strip().upper() for r in residues})
        raise LigandResidueNotFound(
            f"residue {target}/{resnum} not present; "
            f"resnames available: {available[:12]}"
        )

    seed_index = seed_candidates[0]
    unknown: set[str] = set()

    # Transitive covalent closure. Chains hold tens of hetero residues after the
    # caller drops waters, so the naive pairwise sweep is not worth optimising.
    selected = {seed_index}
    frontier = [seed_index] if covalent_closure else []
    while frontier:
        current = frontier.pop()
        for index, candidate in enumerate(residues):
            if index in selected:
                continue
            if candidate.is_metal_ion and not traverse_metal_ions:
                continue
            if residues_are_bonded(
                residues[current], candidate, tolerance=tolerance, unknown=unknown
            ):
                selected.add(index)
                frontier.append(index)

    discarded = tuple(
        index
        for index, residue in enumerate(residues)
        if index not in selected and residue.resname.strip().upper() == target
    )

    seed_centroid = residues[seed_index].centroid
    max_distance = max(
        (
            float(np.linalg.norm(residues[index].centroid - seed_centroid))
            for index in discarded
        ),
        default=0.0,
    )

    selected_sorted = tuple(sorted(selected))
    return LigandSelection(
        selected_indices=selected_sorted,
        seed_index=seed_index,
        covalent_partner_indices=tuple(i for i in selected_sorted if i != seed_index),
        discarded_same_resname=discarded,
        discarded_max_centroid_distance_angstrom=max_distance,
        selected_atom_count=sum(len(residues[i].elements) for i in selected_sorted),
        seed_atom_count=len(residues[seed_index].elements),
        seed_ambiguous=len(seed_candidates) > 1,
        unknown_elements=tuple(sorted(unknown)),
    )


def parse_residue_number(entry_key: str) -> int | None:
    """Recover the ligand residue number from an AHoJ entry key.

    Keys look like ``2hdr-A-4A3-506``: pdb, chain, component, residue number.
    Returns ``None`` when the trailing field is not an integer, so the caller can
    decide whether to fail or fall back -- this function does not guess.
    """
    parts = entry_key.strip().split("-")
    if len(parts) < 4:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None
