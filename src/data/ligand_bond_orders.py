"""CCD-templated bond-order and formal-charge reconstruction for observed ligands.

Ligand SDFs written from PDB ``HETATM`` records carry *connectivity only*: every
bond is emitted as order 1 and no ``M  CHG`` record is written, because the PDB
format does not store bond orders.  For most organic ligands RDKit still
perceives a legal -- but chemically wrong -- molecule; aromatic rings come back
fully saturated.  For polyphosphates the defect is fatal:

    a phosphorus with four single bonds and no formal charge has valence 4,
    which is illegal (P allows 3 or 5), so RDKit completes it to valence 5 by
    adding an implicit hydrogen.

The resulting ``P-H`` pseudo-phosphate is parameterised by OpenFF as a neutral
species and the solvated system reaches a non-finite initial energy, which is
where ``prepare_md_pilot_system.py`` aborts.

This module rebuilds the correct bond orders and formal charges by matching the
observed heavy-atom graph against the corresponding PDB Chemical Component
Dictionary (CCD) entry.  Observed atom order and observed coordinates are
preserved exactly, so ``ligand_coords.npy`` and any downstream index-aligned
tensor stay valid.

Matching is name-free: it uses the element-labelled connectivity graph, so it
works both at triplet-generation time (where PDB atom names exist) and when
repairing an already-written SDF (where ``SDWriter`` has dropped the names).
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "bindrae_ligand_bond_order_reconstruction_v1"

RCSB_CCD_URLS: tuple[str, ...] = (
    "https://files.rcsb.org/ligands/download/{resname}_ideal.sdf",
    "https://files.rcsb.org/ligands/download/{resname}_model.sdf",
)

#: Elements whose valence RDKit silently completes with implicit hydrogens when
#: bond orders are missing.  Membership here is *not* by itself a defect: a
#: glutathione thiol legitimately carries S-H, and the CCD template says so.
#: The defect signature is carrying *more* hydrogens than the matched template
#: atom -- see :func:`unexpected_hydrides`.
HYDRIDE_SENTINEL_ELEMENTS: frozenset[str] = frozenset({"P", "S", "B", "Si", "As", "Se"})

#: Policies for a bond the CCD declares between two observed atoms but which the
#: proximity-perceived observed graph does not contain.
MISSING_BOND_POLICIES: frozenset[str] = frozenset({"restore", "reject", "ignore"})


class LigandBondOrderError(RuntimeError):
    """Raised when a ligand cannot be reconstructed against its CCD template."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# CCD template acquisition
# --------------------------------------------------------------------------


def ccd_cache_path(resname: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{str(resname).strip().upper()}.sdf"


def download_ccd_sdf(
    resname: str,
    cache_dir: Path,
    *,
    timeout_seconds: float = 45.0,
    urls: Sequence[str] = RCSB_CCD_URLS,
    retries: int = 3,
    retry_backoff_seconds: float = 1.5,
) -> Path:
    """Fetch the CCD reference SDF for ``resname`` into ``cache_dir``.

    Compute nodes are usually offline, so this is expected to run on a login
    node (or locally) to pre-populate the cache; the reconstruction path itself
    never downloads unless explicitly allowed.

    RCSB intermittently drops TLS connections under concurrent fetches, so each
    URL is retried; a genuine ``404`` on the first URL still falls through to
    the next one.
    """

    import time

    resname = str(resname).strip().upper()
    destination = ccd_cache_path(resname, cache_dir)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for attempt in range(max(1, int(retries))):
        for template in urls:
            url = template.format(resname=resname)
            try:
                with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                    payload = response.read()
            except urllib.error.HTTPError as exc:
                errors.append(f"{url}: HTTP {exc.code}")
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                errors.append(f"{url} (attempt {attempt + 1}): {exc}")
                continue
            if not payload.strip():
                errors.append(f"{url}: empty payload")
                continue
            destination.write_bytes(payload)
            return destination
        if attempt + 1 < max(1, int(retries)):
            time.sleep(retry_backoff_seconds * (attempt + 1))
    raise LigandBondOrderError(
        "ccd_download_failed",
        f"Could not download CCD SDF for {resname}: " + "; ".join(errors[-6:]),
    )


def load_ccd_template(
    resname: str,
    *,
    ccd_dir: Path,
    allow_download: bool = False,
    timeout_seconds: float = 45.0,
) -> tuple[Any, dict[str, Any]]:
    """Load the CCD reference for ``resname`` as a sanitized heavy-atom molecule."""

    from rdkit import Chem

    resname = str(resname).strip().upper()
    path = ccd_cache_path(resname, ccd_dir)
    if not (path.is_file() and path.stat().st_size > 0):
        if not allow_download:
            raise LigandBondOrderError(
                "ccd_template_missing",
                f"CCD template for {resname} not present at {path} and downloads are disabled",
            )
        path = download_ccd_sdf(resname, ccd_dir, timeout_seconds=timeout_seconds)

    template = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if template is None:
        raise LigandBondOrderError(
            "ccd_template_parse_failed", f"RDKit could not parse CCD SDF {path}"
        )
    heavy_template = Chem.RemoveHs(template, sanitize=True)
    if heavy_template.GetNumAtoms() == 0:
        raise LigandBondOrderError(
            "ccd_template_empty", f"CCD template {path} has no heavy atoms"
        )
    provenance = {
        "resname": resname,
        "ccd_sdf": str(path),
        "ccd_sdf_sha256": file_sha256(path),
        "template_heavy_atoms": int(heavy_template.GetNumAtoms()),
        "template_formal_charge": int(Chem.GetFormalCharge(heavy_template)),
        "template_canonical_smiles": str(
            Chem.MolToSmiles(heavy_template, canonical=True, isomericSmiles=False)
        ),
    }
    return heavy_template, provenance


# --------------------------------------------------------------------------
# Graph matching
# --------------------------------------------------------------------------


def _kekulized_copy(molecule: Any) -> Any:
    """Return a copy with aromatic flags cleared, so matching is element-only."""

    from rdkit import Chem

    copy = Chem.Mol(molecule)
    Chem.Kekulize(copy, clearAromaticFlags=True)
    return copy


def _generic_bond_query(molecule: Any) -> Any:
    """Build a query in which every bond matches any order."""

    from rdkit import Chem

    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    return Chem.AdjustQueryProperties(Chem.Mol(molecule), params)


def match_observed_to_template(observed: Any, template: Any) -> tuple[int, ...]:
    """Map each observed atom index onto a template atom index.

    The observed graph must be a (possibly partial) subgraph of the template.
    Returns the lexicographically first match so repeated runs are deterministic
    under molecular symmetry; symmetric matches are chemically equivalent.
    """

    observed_flat = _kekulized_copy(observed)
    template_flat = _kekulized_copy(template)
    query = _generic_bond_query(observed_flat)
    matches = template_flat.GetSubstructMatches(
        query, uniquify=False, useChirality=False, maxMatches=1000
    )
    if not matches:
        raise LigandBondOrderError(
            "substructure_match_failed",
            "Observed heavy-atom graph is not a subgraph of the CCD template "
            f"(observed atoms={observed.GetNumAtoms()}, template atoms={template.GetNumAtoms()})",
        )
    return tuple(sorted(matches)[0])


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------


def _submol(molecule: Any, indices: Sequence[int]) -> Any:
    """Extract the induced sub-molecule on ``indices`` (ascending atom order)."""

    from rdkit import Chem

    keep = set(int(index) for index in indices)
    editable = Chem.RWMol(molecule)
    for index in range(molecule.GetNumAtoms() - 1, -1, -1):
        if index not in keep:
            editable.RemoveAtom(index)
    sub = editable.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception as exc:  # noqa: BLE001 - RDKit raises bare exceptions
        raise LigandBondOrderError(
            "observed_fragment_sanitize_failed",
            f"RDKit could not sanitize an observed component copy: {exc}",
        ) from exc
    return sub


def pdb_residue_groups(molecule: Any) -> list[tuple[int, ...]] | None:
    """Group atoms by their PDB residue, when the molecule carries that info.

    Oligosaccharides are the reason this exists: a cellotetraose arrives as four
    BGC residues joined by glycosidic bonds, so it is a *single* connected
    fragment four times the size of the component.  Splitting on connectivity
    cannot separate them, but the PDB residue numbers can.
    """

    groups: dict[tuple[Any, ...], list[int]] = {}
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            return None
        key = (
            str(info.GetChainId()),
            int(info.GetResidueNumber()),
            str(info.GetInsertionCode()),
            str(info.GetResidueName()).strip(),
        )
        groups.setdefault(key, []).append(int(atom.GetIdx()))
    return [tuple(sorted(indices)) for _, indices in sorted(groups.items())]


def reconstruct_bond_orders(
    observed: Any,
    template: Any,
    *,
    missing_bond_policy: str = "restore",
    atom_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Copy CCD bond orders and formal charges onto the observed molecule.

    The observed molecule is never rebuilt from the template: bond orders and
    formal charges are written *onto* it, so atom count, atom order, element
    sequence and coordinates are preserved by construction.
    """

    from rdkit import Chem

    if missing_bond_policy not in MISSING_BOND_POLICIES:
        raise ValueError(
            f"missing_bond_policy must be one of {sorted(MISSING_BOND_POLICIES)}, "
            f"got {missing_bond_policy!r}"
        )
    if any(atom.GetAtomicNum() == 1 for atom in observed.GetAtoms()):
        raise LigandBondOrderError(
            "observed_contains_hydrogen",
            "Observed ligand must be heavy-atom only before reconstruction",
        )

    # A single ligand.sdf can hold several copies of the component.
    # extract_ligand_from_pdb() collects every HETATM residue in the chain that
    # shares the resname, without a residue-number filter, so glycans and
    # repeated sugar units arrive as one file with N disconnected copies (a
    # 12-atom component observed as 144 atoms).  Each connected fragment is one
    # copy and is matched against the template independently.
    template_flat = _kekulized_copy(template)
    result = Chem.RWMol(observed)
    if atom_groups is None:
        atom_groups = [tuple(sorted(frag)) for frag in Chem.GetMolFrags(observed)]
    groups: list[tuple[int, ...]] = [
        tuple(sorted(int(index) for index in group)) for group in atom_groups
    ]

    bond_orders_changed = 0
    missing_bonds: list[tuple[int, int]] = []
    unobserved_neighbors: list[dict[str, Any]] = []
    matched_atoms = 0
    fragment_sizes: list[int] = []
    group_matches: list[dict[int, int]] = []

    for atom_indices in groups:
        fragment_sizes.append(len(atom_indices))
        fragment = _submol(observed, atom_indices)
        fragment_match = match_observed_to_template(fragment, template)
        matched_atoms += len(fragment_match)
        # sub-molecule atom j -> whole-molecule atom index -> template atom index
        match = {atom_indices[j]: t for j, t in enumerate(fragment_match)}
        group_matches.append(match)
        observed_position = {t: o for o, t in match.items()}

        for observed_index, template_index in match.items():
            observed_atom = result.GetAtomWithIdx(observed_index)
            template_atom = template_flat.GetAtomWithIdx(template_index)
            if observed_atom.GetAtomicNum() != template_atom.GetAtomicNum():
                raise LigandBondOrderError(
                    "element_mismatch_after_match",
                    f"Atom {observed_index} is {observed_atom.GetSymbol()} but matched "
                    f"template atom {template_index} is {template_atom.GetSymbol()}",
                )
            observed_atom.SetFormalCharge(int(template_atom.GetFormalCharge()))
            observed_atom.SetNoImplicit(False)
            observed_atom.SetNumExplicitHs(0)

        # Only bonds *inside* a residue come from the component.  A glycosidic
        # C-O-C linking two residues is a real single bond between components
        # and is left exactly as observed.
        for bond in fragment.GetBonds():
            begin = atom_indices[bond.GetBeginAtomIdx()]
            end = atom_indices[bond.GetEndAtomIdx()]
            template_bond = template_flat.GetBondBetweenAtoms(match[begin], match[end])
            if template_bond is None:
                raise LigandBondOrderError(
                    "observed_bond_absent_in_template",
                    f"Observed bond {begin}-{end} has no counterpart in the CCD template",
                )
            result_bond = result.GetBondBetweenAtoms(begin, end)
            if result_bond.GetBondType() != template_bond.GetBondType():
                bond_orders_changed += 1
            result_bond.SetBondType(template_bond.GetBondType())
            result_bond.SetIsAromatic(False)

        for template_bond in template_flat.GetBonds():
            begin_t, end_t = template_bond.GetBeginAtomIdx(), template_bond.GetEndAtomIdx()
            if begin_t not in observed_position or end_t not in observed_position:
                continue
            begin, end = observed_position[begin_t], observed_position[end_t]
            if result.GetBondBetweenAtoms(begin, end) is not None:
                continue
            missing_bonds.append((begin, end))
            if missing_bond_policy == "restore":
                result.AddBond(begin, end, template_bond.GetBondType())

        # A heavy atom whose CCD neighbours were not all observed carries a
        # dangling valence that RDKit fills with hydrogen.  On carbon that is a
        # harmless truncation (an alkyl becomes a methyl); on phosphorus it
        # recreates the P-H pseudo-phosphate this module exists to remove.
        for observed_index, template_index in match.items():
            absent = sum(
                1
                for neighbor in template_flat.GetAtomWithIdx(template_index).GetNeighbors()
                if neighbor.GetIdx() not in observed_position
            )
            if absent:
                unobserved_neighbors.append(
                    {
                        "atom_index": int(observed_index),
                        "element": result.GetAtomWithIdx(observed_index).GetSymbol(),
                        "unobserved_neighbor_count": int(absent),
                    }
                )

    if missing_bonds and missing_bond_policy == "reject":
        raise LigandBondOrderError(
            "template_bond_absent_in_observed",
            f"{len(missing_bonds)} CCD bond(s) between observed atoms are absent from "
            f"the perceived graph: {missing_bonds[:8]}",
        )

    sentinel_dangling = [
        row for row in unobserved_neighbors if row["element"] in HYDRIDE_SENTINEL_ELEMENTS
    ]
    if sentinel_dangling:
        raise LigandBondOrderError(
            "unobserved_neighbor_on_sentinel_element",
            "Partially observed ligand leaves a dangling valence on "
            + ", ".join(f"{row['element']}#{row['atom_index']}" for row in sentinel_dangling[:6])
            + "; the component is too incomplete for a chemically correct rebuild",
        )

    molecule = result.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:  # noqa: BLE001 - RDKit raises bare exceptions
        raise LigandBondOrderError(
            "sanitize_failed_after_reconstruction",
            f"RDKit could not sanitize the reconstructed ligand: {exc}",
        ) from exc

    expected_hydrogens = template_hydrogen_counts(molecule, template_flat, group_matches)
    unexpected_hydride_atoms = unexpected_hydrides(molecule, expected_hydrogens)
    if unexpected_hydride_atoms:
        raise LigandBondOrderError(
            "unexpected_hydride_vs_template",
            "Reconstructed ligand carries more hydrogens than the CCD component on "
            + ", ".join(
                f"{row['element']}#{row['atom_index']}"
                f"({row['hydrogens']}>{row['template_hydrogens']})"
                for row in unexpected_hydride_atoms[:6]
            ),
        )

    observed_atoms = int(observed.GetNumAtoms())
    template_atoms = int(template.GetNumAtoms())
    copies = len(groups)
    diagnostics = {
        "matched_template_atoms": matched_atoms,
        "template_heavy_atoms": template_atoms,
        "observed_heavy_atoms": observed_atoms,
        "component_copies": copies,
        "fragment_atom_counts": fragment_sizes,
        "partial_observation": bool(any(size < template_atoms for size in fragment_sizes)),
        # Per copy, so a 12-atom component seen 12 times still reads 1.0.
        "observed_heavy_atom_fraction": (
            (float(observed_atoms) / float(copies)) / float(template_atoms)
            if template_atoms and copies
            else 0.0
        ),
        "bond_orders_changed": int(bond_orders_changed),
        "missing_template_bonds": len(missing_bonds),
        "missing_bond_policy": missing_bond_policy,
        "atoms_with_unobserved_neighbors": len(unobserved_neighbors),
        "expected_hydrogen_counts": expected_hydrogens,
        # Observed atom index -> template atom index, valid across all copies.
        # Callers that need the correspondence must use this rather than
        # re-matching the whole molecule, which fails for multi-copy ligands.
        "atom_template_index": [
            match[index]
            for index in range(observed_atoms)
            for match in group_matches
            if index in match
        ],
    }
    return molecule, diagnostics


def template_hydrogen_counts(
    molecule: Any,
    template_flat: Any,
    group_matches: Sequence[Mapping[int, int]],
) -> list[int]:
    """Hydrogen count the CCD component assigns to each observed atom."""

    from rdkit import Chem

    template_hs = [
        int(atom.GetTotalNumHs()) for atom in Chem.RemoveHs(template_flat).GetAtoms()
    ]
    counts = [0] * molecule.GetNumAtoms()
    for match in group_matches:
        for observed_index, template_index in match.items():
            counts[observed_index] = template_hs[template_index]
    return counts


def unexpected_hydrides(
    molecule: Any, expected_hydrogens: Sequence[int]
) -> list[dict[str, Any]]:
    """Atoms carrying more hydrogens than their CCD counterpart.

    A fixed element blacklist cannot express this: a glutathione cysteine
    legitimately carries S-H and the component says so, while the same S-H on a
    thioester is the valence-completion artifact.  Only an excess over the
    template is evidence of the defect.
    """

    offenders: list[dict[str, Any]] = []
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() not in HYDRIDE_SENTINEL_ELEMENTS:
            continue
        index = atom.GetIdx()
        hydrogens = int(atom.GetTotalNumHs())
        expected = int(expected_hydrogens[index]) if index < len(expected_hydrogens) else 0
        if hydrogens > expected:
            offenders.append(
                {
                    "atom_index": int(index),
                    "element": atom.GetSymbol(),
                    "hydrogens": hydrogens,
                    "template_hydrogens": expected,
                    "formal_charge": int(atom.GetFormalCharge()),
                }
            )
    return offenders


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def bond_order_histogram(molecule: Any) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for bond in molecule.GetBonds():
        key = str(bond.GetBondType())
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items()))


def find_sentinel_hydrides(molecule: Any) -> list[dict[str, Any]]:
    """Return every hydrogen bonded to a hydride-sentinel element.

    ``molecule`` must already carry explicit hydrogens.  A non-empty result on a
    reconstructed ligand means the valence defect survived reconstruction.
    """

    offenders: list[dict[str, Any]] = []
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() not in HYDRIDE_SENTINEL_ELEMENTS:
            continue
        hydrogens = [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        ]
        if hydrogens:
            offenders.append(
                {
                    "atom_index": int(atom.GetIdx()),
                    "element": atom.GetSymbol(),
                    "formal_charge": int(atom.GetFormalCharge()),
                    # GetExplicitValence() is deprecated in newer RDKit builds and
                    # absent from older ones; degree is stable across both.
                    "degree": int(atom.GetDegree()),
                    "hydrogen_indices": [int(index) for index in hydrogens],
                }
            )
    return offenders


def describe_ligand(molecule: Any) -> dict[str, Any]:
    """Chemistry summary used both for validation and for provenance records."""

    from rdkit import Chem

    with_hydrogens = Chem.AddHs(Chem.Mol(molecule))
    return {
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "bonds": int(molecule.GetNumBonds()),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "bond_orders": bond_order_histogram(molecule),
        "aromatic_bonds": int(
            sum(1 for bond in molecule.GetBonds() if bond.GetIsAromatic())
        ),
        "canonical_smiles": str(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        ),
        "sentinel_hydrides": find_sentinel_hydrides(with_hydrogens),
    }


def validate_reconstruction(
    molecule: Any,
    *,
    reference_elements: Sequence[str],
    reference_coordinates: Any,
    template_formal_charge: int | None = None,
    expected_hydrogen_counts: Sequence[int] | None = None,
    coordinate_tolerance_angstrom: float = 1e-3,
) -> dict[str, Any]:
    """Assert the reconstruction preserved identity and removed the defect.

    ``expected_hydrogen_counts`` is the per-atom hydrogen count the CCD
    component assigns.  When supplied, an atom is a defect only if it carries
    *more* hydrogens than that -- which is what distinguishes a real P-H
    artifact from a legitimate glutathione thiol.  Without it the check only
    reports, because an absolute element blacklist gives false positives.
    """

    import numpy as np

    elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    if list(elements) != list(reference_elements):
        raise LigandBondOrderError(
            "element_sequence_changed",
            "Reconstruction changed the element sequence "
            f"({len(reference_elements)} -> {len(elements)} atoms)",
        )
    if molecule.GetNumConformers() == 0:
        raise LigandBondOrderError(
            "conformer_missing", "Reconstructed ligand lost its conformer"
        )
    positions = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
    reference = np.asarray(reference_coordinates, dtype=np.float64)
    if positions.shape != reference.shape:
        raise LigandBondOrderError(
            "coordinate_shape_changed",
            f"Coordinate shape {positions.shape} differs from reference {reference.shape}",
        )
    coordinate_error = float(np.max(np.abs(positions - reference))) if positions.size else 0.0
    if coordinate_error > coordinate_tolerance_angstrom:
        raise LigandBondOrderError(
            "coordinate_drift",
            f"Coordinate drift {coordinate_error:.6f} A exceeds "
            f"{coordinate_tolerance_angstrom:.6f} A",
        )

    summary = describe_ligand(molecule)
    if expected_hydrogen_counts is not None:
        offenders = unexpected_hydrides(molecule, expected_hydrogen_counts)
        summary["unexpected_hydrides"] = offenders
        if offenders:
            raise LigandBondOrderError(
                "unexpected_hydride_vs_template",
                "Reconstructed ligand carries more hydrogens than the CCD component on "
                + ", ".join(
                    f"{row['element']}#{row['atom_index']}"
                    f"({row['hydrogens']}>{row['template_hydrogens']})"
                    for row in offenders[:6]
                ),
            )
    summary["coordinate_max_abs_deviation_angstrom"] = coordinate_error
    if template_formal_charge is not None:
        summary["template_formal_charge"] = int(template_formal_charge)
        summary["formal_charge_matches_template"] = bool(
            int(summary["formal_charge"]) == int(template_formal_charge)
        )
    return summary


# --------------------------------------------------------------------------
# High-level SDF entry points
# --------------------------------------------------------------------------


def read_observed_ligand(sdf_path: Path) -> tuple[Any, list[str], Any]:
    """Read a connectivity-only ligand SDF as a heavy-atom molecule."""

    import numpy as np
    from rdkit import Chem

    molecule = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=False)
    if molecule is None:
        raise LigandBondOrderError(
            "observed_parse_failed", f"RDKit could not parse ligand SDF {sdf_path}"
        )
    molecule = Chem.RemoveAllHs(molecule, sanitize=False)
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:  # noqa: BLE001 - RDKit raises bare exceptions
        raise LigandBondOrderError(
            "observed_sanitize_failed",
            f"RDKit could not sanitize observed ligand {sdf_path}: {exc}",
        ) from exc
    if molecule.GetNumConformers() == 0:
        raise LigandBondOrderError(
            "observed_conformer_missing", f"Ligand SDF {sdf_path} has no conformer"
        )
    elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    coordinates = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=np.float64
    )
    return molecule, elements, coordinates


def write_ligand_sdf(molecule: Any, output_path: Path, *, name: str | None = None) -> None:
    from rdkit import Chem

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if name:
        molecule.SetProp("_Name", str(name).upper())
    writer = Chem.SDWriter(str(output_path))
    writer.write(molecule)
    writer.close()


def roundtrip_check(
    output_path: Path,
    *,
    reference_elements: Sequence[str],
    reference_coordinates: Any,
    expected_hydrogen_counts: Sequence[int] | None = None,
    coordinate_tolerance_angstrom: float = 1e-3,
) -> dict[str, Any]:
    """Re-read the written SDF and confirm it survives a strict parse."""

    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(output_path), removeHs=False, sanitize=True)
    roundtrip = next((value for value in supplier if value is not None), None)
    if roundtrip is None:
        raise LigandBondOrderError(
            "roundtrip_parse_failed",
            f"Written SDF {output_path} does not parse back under strict sanitization",
        )
    roundtrip = Chem.RemoveAllHs(roundtrip)
    return validate_reconstruction(
        roundtrip,
        reference_elements=reference_elements,
        reference_coordinates=reference_coordinates,
        expected_hydrogen_counts=expected_hydrogen_counts,
        coordinate_tolerance_angstrom=coordinate_tolerance_angstrom,
    )


def reconstruct_ligand_sdf(
    *,
    input_sdf: Path,
    resname: str,
    output_sdf: Path,
    ccd_dir: Path,
    allow_download: bool = False,
    missing_bond_policy: str = "restore",
    min_observed_heavy_atom_fraction: float = 0.0,
    atom_groups: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Repair one ligand SDF in place-equivalent form and return a full record.

    Raises :class:`LigandBondOrderError` with a machine-readable ``reason`` on
    every rejection path so callers can keep an honest reject ledger.
    """

    observed, elements, coordinates = read_observed_ligand(Path(input_sdf))
    before = describe_ligand(observed)
    template, template_provenance = load_ccd_template(
        resname, ccd_dir=Path(ccd_dir), allow_download=allow_download
    )
    reconstructed, diagnostics = reconstruct_bond_orders(
        observed, template,
        missing_bond_policy=missing_bond_policy,
        atom_groups=atom_groups,
    )
    fraction = float(diagnostics["observed_heavy_atom_fraction"])
    if fraction < float(min_observed_heavy_atom_fraction):
        raise LigandBondOrderError(
            "observed_heavy_atom_fraction_below_threshold",
            f"Observed heavy-atom fraction {fraction:.4f} is below "
            f"{float(min_observed_heavy_atom_fraction):.4f}",
        )
    expected_hydrogens = diagnostics.pop("expected_hydrogen_counts", None)
    diagnostics.pop("atom_template_index", None)
    after = validate_reconstruction(
        reconstructed,
        reference_elements=elements,
        reference_coordinates=coordinates,
        template_formal_charge=int(template_provenance["template_formal_charge"]),
        expected_hydrogen_counts=expected_hydrogens,
    )
    write_ligand_sdf(reconstructed, Path(output_sdf), name=resname)
    roundtrip = roundtrip_check(
        Path(output_sdf),
        reference_elements=elements,
        reference_coordinates=coordinates,
        expected_hydrogen_counts=expected_hydrogens,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "resname": str(resname).strip().upper(),
        "input_sdf": str(input_sdf),
        "output_sdf": str(output_sdf),
        "output_sdf_sha256": file_sha256(Path(output_sdf)),
        "ccd": template_provenance,
        "reconstruction": diagnostics,
        "before": before,
        "after": after,
        "roundtrip": roundtrip,
    }
