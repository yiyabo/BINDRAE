#!/usr/bin/env python3
"""Export preflight-valid APObind endpoints as aligned MD-pilot triplets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preflight_apobind_structure_smoke import (  # noqa: E402
    file_sha256,
    load_jsonl,
    load_structure_context,
    protein_residue_records,
)
from src.data.md_pilot_selection import screen_sample  # noqa: E402
from src.data.md_transition_manifest import SCHEMA_VERSION as MANIFEST_SCHEMA_VERSION  # noqa: E402
from src.data.residue_alignment import align_residue_names, canonical_resname  # noqa: E402
from src.data.residue_identity import STANDARD_AA3_TO_1  # noqa: E402


SCHEMA_VERSION = "bindrae_apobind_aligned_triplet_export_v1"
DEFAULT_SCREEN_CONFIG = {
    "min_residues": 80,
    "max_residues": 500,
    "min_sequence_identity": 0.95,
    "min_residue_mapping_fraction": 0.95,
    "min_heavy_atoms": 8,
    "max_heavy_atoms": 70,
    "max_abs_charge": 2,
    "excluded_resnames": [],
    "pocket_radius": 10.0,
    "contact_radius": 8.0,
    "min_pocket_residues": 5,
    "moving_threshold": 1.0,
    "min_global_rmsd": 0.35,
    "min_pocket_rmsd": 0.60,
    "min_max_displacement": 1.20,
    "max_global_rmsd": 5.0,
    "max_pocket_rmsd": 8.0,
    "max_max_displacement": 20.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-jsonl", type=Path, required=True)
    parser.add_argument("--preflight-results-jsonl", type=Path, required=True)
    parser.add_argument("--structures-dir", type=Path, required=True)
    parser.add_argument("--ccd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def kabsch_transform(
    mobile: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return row-vector rigid transform mapping mobile coordinates to target."""

    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Expected matching [N, 3] arrays, got {mobile.shape} and {target.shape}")
    if len(mobile) < 3:
        raise ValueError("At least three mapped coordinates are required")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd(
        (mobile - mobile_center).T @ (target - target_center)
    )
    correction = np.sign(np.linalg.det(left @ right))
    rotation = left @ np.diag([1.0, 1.0, correction]) @ right
    translation = target_center - mobile_center @ rotation
    aligned = mobile @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, translation, aligned, rmsd


def transform_model(model: Any, rotation: np.ndarray, translation: np.ndarray) -> None:
    for atom in model.get_atoms():
        coordinate = np.asarray(atom.get_coord(), dtype=np.float64)
        atom.set_coord(coordinate @ rotation + translation)


def write_protein_chain(model: Any, chain_id: str, output_path: Path) -> None:
    from Bio.PDB import PDBIO, Select

    class ProteinChainSelect(Select):
        def accept_chain(self, chain: Any) -> bool:
            return str(chain.id) == chain_id

        def accept_residue(self, residue: Any) -> bool:
            parent = residue.get_parent()
            selected = parent[residue.get_id()]
            if (
                selected.is_disordered() == 2
                and selected.selected_child.get_resname() != residue.get_resname()
            ):
                return False
            return canonical_resname(residue.get_resname()) in STANDARD_AA3_TO_1

        def accept_atom(self, atom: Any) -> bool:
            residue = atom.get_parent()
            selected = residue[atom.get_id()]
            if selected.is_disordered() != 2:
                return True
            if selected.selected_child.get_altloc() != atom.get_altloc():
                return False
            # The exported endpoint has one resolved conformer. Leaving a B/C
            # altloc marker would make downstream PDB readers silently skip it.
            atom.set_altloc(" ")
            return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    writer.set_structure(model)
    writer.save(str(output_path), ProteinChainSelect())
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"No protein atoms written to {output_path}")


def load_component_atom_rows(cif_path: Path, resname: str) -> list[dict[str, Any]]:
    import gzip

    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    with gzip.open(cif_path, "rt", encoding="utf-8", errors="replace") as handle:
        cif = MMCIF2Dict(handle)
    columns = {
        "comp_id": cif.get("_chem_comp_atom.comp_id", []),
        "atom_id": cif.get("_chem_comp_atom.atom_id", []),
        "element": cif.get("_chem_comp_atom.type_symbol", []),
        "ordinal": cif.get("_chem_comp_atom.pdbx_ordinal", []),
    }
    for key, value in list(columns.items()):
        if isinstance(value, str):
            columns[key] = [value]
    lengths = {len(value) for value in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent _chem_comp_atom columns in {cif_path}: {lengths}")
    rows = [
        {
            "atom_id": str(atom_id).strip(),
            "element": str(element).strip().upper(),
            "ordinal": int(ordinal),
        }
        for comp_id, atom_id, element, ordinal in zip(
            columns["comp_id"],
            columns["atom_id"],
            columns["element"],
            columns["ordinal"],
        )
        if str(comp_id).strip().upper() == resname.upper()
    ]
    rows.sort(key=lambda row: int(row["ordinal"]))
    if not rows:
        raise ValueError(f"No _chem_comp_atom rows for {resname} in {cif_path}")
    atom_ids = [str(row["atom_id"]) for row in rows]
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError(f"Duplicate CCD atom IDs for {resname} in {cif_path}")
    return rows


def find_ligand_residue(model: Any, ligand: Mapping[str, Any]) -> Any:
    chain_id = str(ligand["chain_id"])
    resname = str(ligand["resname"]).upper()
    sequence_id = int(ligand["author_sequence_id"])
    insertion_code = str(ligand.get("insertion_code") or "").strip()
    if chain_id not in model:
        raise KeyError(f"Ligand chain {chain_id!r} is missing")
    matches = [
        residue
        for residue in model[chain_id]
        if residue.get_resname().strip().upper() == resname
        and int(residue.get_id()[1]) == sequence_id
        and str(residue.get_id()[2]).strip() == insertion_code
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one ligand {chain_id}:{resname}:{sequence_id}{insertion_code}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _element(atom: Any) -> str:
    value = str(getattr(atom, "element", "") or "").strip().upper()
    if value:
        return value
    letters = "".join(character for character in atom.get_name() if character.isalpha())
    return letters[:1].upper()


def remove_component_hydrogens(
    molecule: Any, component_rows: Sequence[Mapping[str, Any]]
) -> Any:
    """Remove CCD-declared H/D atoms while preserving the heavy-atom graph."""

    from rdkit import Chem

    params = Chem.RemoveHsParameters()
    params.removeDegreeZero = True
    params.removeDefiningBondStereo = True
    params.removeHydrides = True
    params.removeWithWedgedBond = True
    heavy_molecule = Chem.RemoveHs(molecule, params, sanitize=True)
    expected_heavy_atoms = sum(
        str(row["element"]).strip().upper() not in {"H", "D"}
        for row in component_rows
    )
    if heavy_molecule.GetNumAtoms() != expected_heavy_atoms:
        raise ValueError(
            "RDKit/CCD heavy atom count mismatch after hydrogen filtering: "
            f"RDKit={heavy_molecule.GetNumAtoms()} CCD={expected_heavy_atoms}"
        )
    if any(atom.GetAtomicNum() == 1 for atom in heavy_molecule.GetAtoms()):
        raise ValueError("CCD hydrogen filtering left hydrogen atoms in the ligand graph")
    return heavy_molecule


def build_observed_ligand_sdf(
    *,
    ccd_sdf: Path,
    component_rows: Sequence[Mapping[str, Any]],
    ligand_residue: Any,
    output_sdf: Path,
    output_coords: Path,
) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    molecule = Chem.MolFromMolFile(str(ccd_sdf), removeHs=False, sanitize=True)
    if molecule is None:
        raise ValueError(f"RDKit could not parse CCD SDF {ccd_sdf}")
    if molecule.GetNumAtoms() != len(component_rows):
        raise ValueError(
            f"CCD atom count mismatch for {ccd_sdf.stem}: "
            f"SDF={molecule.GetNumAtoms()} mmCIF={len(component_rows)}"
        )
    for atom, row in zip(molecule.GetAtoms(), component_rows):
        if atom.GetSymbol().upper() != str(row["element"]).upper():
            raise ValueError(
                f"CCD element-order mismatch at {row['atom_id']}: "
                f"SDF={atom.GetSymbol()} mmCIF={row['element']}"
            )
        atom.SetProp("ccd_atom_id", str(row["atom_id"]))

    heavy_molecule = remove_component_hydrogens(molecule, component_rows)
    expected_names = [atom.GetProp("ccd_atom_id") for atom in heavy_molecule.GetAtoms()]
    observed_atoms = {
        str(atom.get_name()).strip(): atom
        for atom in ligand_residue.get_atoms()
        if _element(atom) not in {"H", "D"}
    }
    if set(observed_atoms) != set(expected_names):
        raise ValueError(
            f"Observed/CCD heavy atom names differ for {ccd_sdf.stem}: "
            f"missing={sorted(set(expected_names) - set(observed_atoms))}, "
            f"extra={sorted(set(observed_atoms) - set(expected_names))}"
        )
    coordinates = np.asarray(
        [observed_atoms[name].get_coord() for name in expected_names], dtype=np.float64
    )
    conformer = heavy_molecule.GetConformer()
    for index, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(index, coordinate.tolist())
    heavy_molecule.SetProp("_Name", ccd_sdf.stem.upper())
    Chem.SanitizeMol(heavy_molecule)

    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_sdf))
    writer.write(heavy_molecule)
    writer.close()
    np.save(output_coords, coordinates.astype(np.float32))

    supplier = Chem.SDMolSupplier(str(output_sdf), removeHs=False, sanitize=True)
    roundtrip = next((value for value in supplier if value is not None), None)
    if roundtrip is None or roundtrip.GetNumAtoms() != len(coordinates):
        raise ValueError(f"Round-trip validation failed for {output_sdf}")
    roundtrip_xyz = np.asarray(roundtrip.GetConformer().GetPositions(), dtype=np.float64)
    coordinate_error = float(np.max(np.abs(roundtrip_xyz - coordinates)))
    if coordinate_error > 1e-3:
        raise ValueError(
            f"SDF coordinate round-trip error {coordinate_error:.6f} A exceeds 1e-3"
        )

    scaffold = MurckoScaffold.GetScaffoldForMol(roundtrip)
    scaffold_smiles = str(Chem.MolToSmiles(scaffold, isomericSmiles=False))
    canonical_smiles = str(
        Chem.MolToSmiles(roundtrip, canonical=True, isomericSmiles=True)
    )
    try:
        inchikey = str(Chem.MolToInchiKey(roundtrip)) or None
    except Exception:
        inchikey = None
    return {
        "resname": ccd_sdf.stem.upper(),
        "heavy_atoms": int(roundtrip.GetNumHeavyAtoms()),
        "formal_charge": int(Chem.GetFormalCharge(roundtrip)),
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
        "murcko_scaffold": scaffold_smiles or f"ACYCLIC:{canonical_smiles}",
        "coordinate_roundtrip_max_abs_angstrom": coordinate_error,
        "ccd_sdf": str(ccd_sdf),
        "ccd_sdf_sha256": file_sha256(ccd_sdf),
    }


def sequence_sha256(residues: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    sequence = "".join(str(residue["one_letter"]) for residue in residues)
    return sequence, hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def sample_id_for(record: Mapping[str, Any], ligand: Mapping[str, Any]) -> str:
    source_index = str(record["metadata"].get("source_index") or record["source_rows"][0])
    apo, holo = record["endpoints"]
    return (
        f"apobind_{source_index}_{str(apo['pdb_id']).upper()}_"
        f"{str(holo['pdb_id']).upper()}_{str(ligand['resname']).upper()}"
    )


def make_transition_record(
    *,
    source_record: Mapping[str, Any],
    preflight: Mapping[str, Any],
    sample_dir: Path,
    sample_id: str,
    sequence_digest: str,
    ligand: Mapping[str, Any],
    screening: Mapping[str, Any],
) -> dict[str, Any]:
    apo, holo = source_record["endpoints"]
    mapping_fraction = float(preflight["mapping"]["symmetric_mapping_fraction"])
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "transition_id": f"apobind:{sample_id}:pilot",
        "ensemble_id": (
            f"apobind:{str(apo['pdb_id']).upper()}:{str(holo['pdb_id']).upper()}:"
            f"{ligand['resname']}"
        ),
        "status": "metadata_verified",
        "source": {
            "name": "APObind",
            "record_url": f"https://www.rcsb.org/structure/{str(holo['pdb_id']).upper()}",
            "license": "RCSB PDB source terms; APObind annotation redistribution unresolved",
            "record_key": source_record["record_key"],
        },
        "evidence": {
            "tier": "context_equilibrium",
            "contains_endpoint_transition": False,
            "biased_sampling": False,
            "physical_time_interpretable": False,
        },
        "protein": {
            "uniprot_id": None,
            "chain_ids": [apo["chains"][0], holo["chains"][0]],
            "sequence_sha256": sequence_digest,
        },
        "ligand": {
            "comp_id": ligand["resname"],
            "inchikey": ligand["inchikey"],
            "canonical_smiles": ligand["canonical_smiles"],
        },
        "endpoints": {
            "apo_pdb_id": str(apo["pdb_id"]).upper(),
            "holo_pdb_id": str(holo["pdb_id"]).upper(),
            "apo_structure_path": str(sample_dir / "apo.pdb"),
            "holo_structure_path": str(sample_dir / "holo.pdb"),
        },
        "trajectory": {
            "topology_path": None,
            "coordinate_paths": [],
            "n_frames": None,
            "frame_interval_ps": None,
        },
        "usage": {
            "phase_supervision": False,
            "heldout_benchmark": False,
            "kinetics_claims": False,
        },
        "split": {
            "name": "unassigned",
            "family_group": None,
            "ligand_scaffold_group": ligand["murcko_scaffold"],
        },
        "quality": {
            "endpoint_mapping_verified": mapping_fraction >= 0.95,
            "residue_mapping_fraction": mapping_fraction,
            "transition_verified": False,
            "notes": [
                "APObind endpoint pair exported from a local structure smoke; no MD path exists."
            ],
        },
        "screening": {
            key: screening.get(key)
            for key in (
                "eligible",
                "reason",
                "n_residues",
                "apo_n_residues",
                "holo_n_residues",
                "mapped_residues",
                "sequence_identity",
                "residue_mapping_fraction",
                "ca_aligned_rmsd",
                "pocket_ca_rmsd",
                "max_ca_displacement",
                "contact_changes",
                "motion_category",
                "motion_score",
                "pilot_score",
                "ligand_heavy_atoms",
                "ligand_formal_charge",
            )
        },
    }


def export_one(
    *,
    source_record: Mapping[str, Any],
    preflight: Mapping[str, Any],
    structures_dir: Path,
    ccd_dir: Path,
    samples_dir: Path,
) -> dict[str, Any]:
    apo_endpoint, holo_endpoint = source_record["endpoints"]
    apo_path = structures_dir / f"{str(apo_endpoint['pdb_id']).upper()}.cif.gz"
    holo_path = structures_dir / f"{str(holo_endpoint['pdb_id']).upper()}.cif.gz"
    apo_model, _ = load_structure_context(apo_path)
    holo_model, _ = load_structure_context(holo_path)
    apo_residues = protein_residue_records(apo_model, apo_endpoint["chains"][0])
    holo_residues = protein_residue_records(holo_model, holo_endpoint["chains"][0])
    alignment = align_residue_names(
        [residue["name"] for residue in apo_residues],
        [residue["name"] for residue in holo_residues],
    )
    apo_indices = [left for left, _ in alignment.exact_pairs]
    holo_indices = [right for _, right in alignment.exact_pairs]
    apo_ca = np.asarray([apo_residues[index]["ca"] for index in apo_indices])
    holo_ca = np.asarray([holo_residues[index]["ca"] for index in holo_indices])
    rotation, translation, _, aligned_rmsd = kabsch_transform(holo_ca, apo_ca)
    expected_rmsd = float(preflight["motion"]["global_ca_rmsd_angstrom"])
    if abs(aligned_rmsd - expected_rmsd) > 1e-5:
        raise ValueError(
            f"Transform RMSD {aligned_rmsd:.8f} differs from preflight {expected_rmsd:.8f}"
        )
    transform_model(holo_model, rotation, translation)

    ligand_identity = preflight["ligand_resolution"]["ligand"]
    sample_id = sample_id_for(source_record, ligand_identity)
    sample_dir = samples_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_protein_chain(apo_model, apo_endpoint["chains"][0], sample_dir / "apo.pdb")
    write_protein_chain(holo_model, holo_endpoint["chains"][0], sample_dir / "holo.pdb")

    ligand_residue = find_ligand_residue(holo_model, ligand_identity)
    resname = str(ligand_identity["resname"]).upper()
    ligand = build_observed_ligand_sdf(
        ccd_sdf=ccd_dir / f"{resname}.sdf",
        component_rows=load_component_atom_rows(holo_path, resname),
        ligand_residue=ligand_residue,
        output_sdf=sample_dir / "ligand.sdf",
        output_coords=sample_dir / "ligand_coords.npy",
    )
    sequence, sequence_digest = sequence_sha256(apo_residues)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "entry_key": sample_id,
        "source": "APObind",
        "source_record_key": source_record["record_key"],
        "source_rows": source_record["source_rows"],
        "selection_rank": preflight.get("selection_rank"),
        "apo_pdb": str(apo_endpoint["pdb_id"]).upper(),
        "apo_chain": apo_endpoint["chains"][0],
        "holo_pdb": str(holo_endpoint["pdb_id"]).upper(),
        "holo_chain": holo_endpoint["chains"][0],
        "ligand_resname": resname,
        "ligand_chain": ligand_identity["chain_id"],
        "ligand_author_sequence_id": ligand_identity["author_sequence_id"],
        "ligand_insertion_code": ligand_identity.get("insertion_code") or "",
        "ligand": ligand,
        "protein_sequence": sequence,
        "protein_sequence_sha256": sequence_digest,
        "alignment": {
            "method": "global_sequence_exact_then_ca_kabsch",
            "mapped_residues": len(alignment.exact_pairs),
            "sequence_identity": alignment.sequence_identity,
            "symmetric_mapping_fraction": alignment.symmetric_mapping_fraction,
            "aligned_ca_rmsd_angstrom": aligned_rmsd,
            "row_vector_rotation": rotation.tolist(),
            "translation_angstrom": translation.tolist(),
        },
        "provenance": {
            "apo_mmcif": str(apo_path),
            "apo_mmcif_sha256": file_sha256(apo_path),
            "holo_mmcif": str(holo_path),
            "holo_mmcif_sha256": file_sha256(holo_path),
            "preflight_results": str(preflight.get("record_key")),
        },
    }
    (sample_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "record_key": source_record["record_key"],
        "sample_id": sample_id,
        "sample_dir": str(sample_dir),
        "sequence_sha256": sequence_digest,
        "ligand": ligand,
        "alignment": meta["alignment"],
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection = load_jsonl(args.selection_jsonl)
    preflight_results = load_jsonl(args.preflight_results_jsonl)
    selected_by_key = {str(row["record_key"]): row for row in selection}
    if len(selected_by_key) != len(selection):
        raise ValueError("Selection contains duplicate record keys")
    preflight_by_key = {str(row["record_key"]): row for row in preflight_results}
    if set(preflight_by_key) != set(selected_by_key):
        raise ValueError("Selection and preflight record-key sets differ")

    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in selection:
        key = str(record["record_key"])
        preflight = preflight_by_key[key]
        if not preflight.get("eligible"):
            continue
        try:
            exported.append(
                export_one(
                    source_record=record,
                    preflight=preflight,
                    structures_dir=args.structures_dir,
                    ccd_dir=args.ccd_dir,
                    samples_dir=samples_dir,
                )
            )
        except Exception as exc:
            ligand_identity = (preflight.get("ligand_resolution") or {}).get("ligand")
            sample_id = (
                sample_id_for(record, ligand_identity)
                if isinstance(ligand_identity, Mapping)
                else None
            )
            if sample_id is not None:
                shutil.rmtree(samples_dir / sample_id, ignore_errors=True)
            failures.append(
                {
                    "record_key": key,
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                }
            )

    screening_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for row in exported:
        sample_id = str(row["sample_id"])
        config = {**DEFAULT_SCREEN_CONFIG, "data_dir": str(args.output_dir)}
        screening = screen_sample(sample_id, config)
        screening_rows.append(screening)
        source_record = selected_by_key[str(row["record_key"])]
        preflight = preflight_by_key[str(row["record_key"])]
        manifest_rows.append(
            make_transition_record(
                source_record=source_record,
                preflight=preflight,
                sample_dir=Path(str(row["sample_dir"])),
                sample_id=sample_id,
                sequence_digest=str(row["sequence_sha256"]),
                ligand=row["ligand"],
                screening=screening,
            )
        )

    write_jsonl(args.output_dir / "exported_records.jsonl", exported)
    write_jsonl(args.output_dir / "screening_results.jsonl", screening_rows)
    write_jsonl(args.output_dir / "transition_manifest.jsonl", manifest_rows)
    eligible_manifest = [
        manifest
        for manifest, screening in zip(manifest_rows, screening_rows)
        if screening.get("eligible") is True
    ]
    write_jsonl(
        args.output_dir / "screening_eligible_transition_manifest.jsonl",
        eligible_manifest,
    )
    (args.output_dir / "exported_sample_ids.txt").write_text(
        "".join(f"{row['sample_id']}\n" for row in exported), encoding="utf-8"
    )
    (args.output_dir / "screening_eligible_sample_ids.txt").write_text(
        "".join(
            f"{row['sample_id']}\n"
            for row, screening in zip(exported, screening_rows)
            if screening.get("eligible") is True
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if failures else "complete",
        "inputs": {
            "selection": str(args.selection_jsonl),
            "selection_sha256": file_sha256(args.selection_jsonl),
            "preflight_results": str(args.preflight_results_jsonl),
            "preflight_results_sha256": file_sha256(args.preflight_results_jsonl),
        },
        "counts": {
            "selected": len(selection),
            "preflight_eligible": sum(
                result.get("eligible") is True for result in preflight_results
            ),
            "exported": len(exported),
            "export_failed": len(failures),
            "screening_eligible": len(eligible_manifest),
            "screening_rejected": len(screening_rows) - len(eligible_manifest),
        },
        "screening_reason_counts": {
            reason: sum(str(row.get("reason")) == reason for row in screening_rows)
            for reason in sorted({str(row.get("reason")) for row in screening_rows})
        },
        "failures": failures,
        "claim_boundary": (
            "Exported triplets are aligned endpoint inputs for setup screening. "
            "They have no prepared topology, MD path, accepted replicas, or Path-3 labels."
        ),
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Failed to export {len(failures)} preflight-eligible systems")
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
