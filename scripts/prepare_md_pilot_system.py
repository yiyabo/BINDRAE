#!/usr/bin/env python3
"""Prepare and minimize one ligand-bound OpenMM pilot system."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--transition-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--padding-nm", type=float, default=1.0)
    parser.add_argument("--ionic-strength-molar", type=float, default=0.15)
    parser.add_argument("--small-molecule-forcefield", default="openff-2.2.1")
    parser.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU"], default="CUDA")
    parser.add_argument("--max-minimization-iterations", type=int, default=500)
    parser.add_argument("--solvent-minimization-iterations", type=int, default=1000)
    parser.add_argument("--solute-restraint-k-kj-mol-nm2", type=float, default=1000.0)
    parser.add_argument("--max-residue-net-force-kj-mol-nm", type=float, default=500.0)
    parser.add_argument("--add-missing-residues", action="store_true")
    return parser.parse_args()


def load_candidate(
    path: str | Path, *, candidate_index: int, transition_id: str | None
) -> Dict[str, Any]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Candidate manifest is empty: {path}")
    if transition_id is not None:
        matches = [record for record in records if record.get("transition_id") == transition_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one transition_id={transition_id!r}, found {len(matches)}")
        return dict(matches[0])
    if candidate_index < 0 or candidate_index >= len(records):
        raise IndexError(f"candidate_index={candidate_index} outside [0, {len(records) - 1}]")
    return dict(records[candidate_index])


def _positions_angstrom(positions: Sequence[Any]) -> np.ndarray:
    from openmm import unit

    return np.asarray(positions.value_in_unit(unit.angstrom), dtype=np.float64)


def prepare_protein(
    pdb_path: str | Path, *, ph: float, add_missing_residues: bool
) -> Tuple[Any, Dict[str, Any]]:
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.removeHeterogens(keepWater=False)
    fixer.findNonstandardResidues()
    nonstandard = [(str(residue), str(replacement)) for residue, replacement in fixer.nonstandardResidues]
    if fixer.nonstandardResidues:
        fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    missing_residue_count = sum(len(names) for names in fixer.missingResidues.values())
    if not add_missing_residues:
        fixer.missingResidues = {}
    fixer.findMissingAtoms()
    missing_atom_count = sum(len(atoms) for atoms in fixer.missingAtoms.values())
    missing_terminal_atom_count = sum(len(atoms) for atoms in fixer.missingTerminals.values())
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    diagnostics = {
        "input_pdb": str(pdb_path),
        "add_missing_residues": add_missing_residues,
        "missing_residues_detected": missing_residue_count,
        "missing_atoms_detected": missing_atom_count,
        "missing_terminal_atoms_detected": missing_terminal_atom_count,
        "nonstandard_residues_replaced": nonstandard,
        "prepared_protein_atoms": sum(1 for _ in fixer.topology.atoms()),
        "prepared_protein_residues": sum(1 for _ in fixer.topology.residues()),
    }
    return fixer, diagnostics


def choose_ligand_fragment(
    sdf_path: str | Path, protein_positions_angstrom: np.ndarray
) -> Tuple[Any, Dict[str, Any]]:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None:
        raise ValueError(f"Could not sanitize ligand SDF: {sdf_path}")
    fragments = list(Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True))
    organic: List[Tuple[float, str, Any]] = []
    for fragment in fragments:
        if not any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms()):
            continue
        xyz = np.asarray(fragment.GetConformer().GetPositions(), dtype=np.float64)
        heavy_mask = np.asarray([atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms()])
        heavy_xyz = xyz[heavy_mask]
        squared = np.sum(
            (heavy_xyz[:, None, :] - protein_positions_angstrom[None, :, :]) ** 2,
            axis=-1,
        )
        nearest = float(np.sqrt(np.min(squared)))
        smiles = str(Chem.MolToSmiles(Chem.RemoveHs(fragment), isomericSmiles=True))
        organic.append((nearest, smiles, fragment))
    if not organic:
        raise ValueError(f"No organic ligand fragment found in {sdf_path}")
    unique_smiles = {row[1] for row in organic}
    if len(unique_smiles) != 1:
        raise ValueError(f"Multiple distinct organic fragments found: {sorted(unique_smiles)}")

    nearest, smiles, fragment = min(organic, key=lambda row: row[0])
    fragment = Chem.AddHs(fragment, addCoords=True)
    diagnostics = {
        "input_sdf": str(sdf_path),
        "source_fragment_count": len(fragments),
        "organic_copy_count": len(organic),
        "selected_nearest_protein_distance_angstrom": nearest,
        "canonical_smiles": smiles,
        "formal_charge": int(Chem.GetFormalCharge(fragment)),
        "atoms_with_hydrogen": int(fragment.GetNumAtoms()),
        "heavy_atoms": int(fragment.GetNumHeavyAtoms()),
    }
    return fragment, diagnostics


def _maximum_force_kj_mol_nm(forces: Any) -> float:
    from openmm import unit

    values = np.asarray(forces.value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    return float(np.linalg.norm(values, axis=1).max())


def _force_summary(
    forces: Any, *, protein_atoms: int, solute_atoms: int
) -> Dict[str, Dict[str, float]]:
    from openmm import unit

    values = np.asarray(forces.value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    norms = np.linalg.norm(values, axis=1)
    slices = {
        "protein": norms[:protein_atoms],
        "ligand": norms[protein_atoms:solute_atoms],
        "solvent": norms[solute_atoms:],
        "all": norms,
    }


def _residue_net_force_summary(
    forces: Any, topology: Any, *, protein_atoms: int, solute_atoms: int
) -> Dict[str, Dict[str, float]]:
    from openmm import unit

    values = np.asarray(forces.value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    grouped: Dict[str, List[float]] = {
        "protein": [],
        "ligand": [],
        "water": [],
        "ions": [],
    }
    for residue in topology.residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        if not atom_indices:
            continue
        if atom_indices[0] < protein_atoms:
            category = "protein"
        elif atom_indices[0] < solute_atoms:
            category = "ligand"
        elif residue.name in {"HOH", "WAT"}:
            category = "water"
        else:
            category = "ions"
        grouped[category].append(float(np.linalg.norm(values[atom_indices].sum(axis=0))))
    return {
        name: {
            "count": len(component),
            "max_kj_mol_nm": float(np.max(component)),
            "p99_kj_mol_nm": float(np.quantile(component, 0.99)),
            "mean_kj_mol_nm": float(np.mean(component)),
        }
        for name, component in grouped.items()
        if component
    }
    return {
        name: {
            "max_kj_mol_nm": float(component.max()),
            "p99_kj_mol_nm": float(np.quantile(component, 0.99)),
            "mean_kj_mol_nm": float(component.mean()),
        }
        for name, component in slices.items()
        if len(component) > 0
    }


def prepare_and_minimize(record: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    from openff.toolkit import Molecule
    from openmm import (
        CustomExternalForce,
        LangevinMiddleIntegrator,
        Platform,
        XmlSerializer,
        unit,
    )
    from openmm import app
    from openmmforcefields.generators import SystemGenerator

    endpoints = dict(record.get("endpoints") or {})
    holo_path = Path(str(endpoints.get("holo_structure_path") or ""))
    if not holo_path.is_file():
        raise FileNotFoundError(f"Holo endpoint does not exist: {holo_path}")
    ligand_path = holo_path.parent / "ligand.sdf"
    if not ligand_path.is_file():
        raise FileNotFoundError(f"Ligand SDF does not exist: {ligand_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/7] Repairing protein endpoint: {holo_path}", flush=True)
    fixer, protein_diagnostics = prepare_protein(
        holo_path, ph=args.ph, add_missing_residues=args.add_missing_residues
    )
    protein_xyz = _positions_angstrom(fixer.positions)
    print("[2/7] Selecting the nearest chemically identical ligand copy", flush=True)
    rdkit_ligand, ligand_diagnostics = choose_ligand_fragment(ligand_path, protein_xyz)
    print(
        f"Selected ligand with {ligand_diagnostics['heavy_atoms']} heavy atoms at "
        f"{ligand_diagnostics['selected_nearest_protein_distance_angstrom']:.2f} A",
        flush=True,
    )
    off_ligand = Molecule.from_rdkit(
        rdkit_ligand,
        allow_undefined_stereo=True,
        hydrogens_are_explicit=True,
    )
    off_ligand.name = str((record.get("ligand") or {}).get("comp_id") or "LIG")

    print(f"[3/7] Registering {args.small_molecule_forcefield} parameters", flush=True)
    system_generator = SystemGenerator(
        forcefields=["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"],
        small_molecule_forcefield=args.small_molecule_forcefield,
        molecules=[off_ligand],
        forcefield_kwargs={
            "constraints": app.HBonds,
            "rigidWater": True,
            "removeCMMotion": True,
        },
        periodic_forcefield_kwargs={
            "nonbondedMethod": app.PME,
            "nonbondedCutoff": 1.0 * unit.nanometer,
            "ewaldErrorTolerance": 5e-4,
        },
        cache=str(args.output_dir / "small_molecule_system_cache.json"),
    )

    modeller = app.Modeller(fixer.topology, fixer.positions)
    ligand_topology = off_ligand.to_topology().to_openmm()
    ligand_xyz = np.asarray(rdkit_ligand.GetConformer().GetPositions(), dtype=np.float64)
    modeller.add(ligand_topology, unit.Quantity(ligand_xyz, unit.angstrom))
    with (args.output_dir / "repaired_complex_unsolvated.pdb").open("w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    protein_atoms = int(protein_diagnostics["prepared_protein_atoms"])
    pre_solvent_atoms = sum(1 for _ in modeller.topology.atoms())
    solute_atoms = list(modeller.topology.atoms())
    solute_positions_nm = np.asarray(
        modeller.positions.value_in_unit(unit.nanometer), dtype=np.float64
    )
    restrained_indices = [
        atom.index
        for atom in solute_atoms
        if atom.element is not None and atom.element.symbol != "H"
    ]
    print(f"[4/7] Adding explicit solvent around {pre_solvent_atoms} solute atoms", flush=True)
    modeller.addSolvent(
        system_generator.forcefield,
        model="tip3p",
        padding=args.padding_nm * unit.nanometer,
        ionicStrength=args.ionic_strength_molar * unit.molar,
        neutralize=True,
    )
    print("[5/7] Creating the periodic OpenMM system", flush=True)
    system = system_generator.create_system(modeller.topology)
    platform = Platform.getPlatformByName(args.platform)
    properties: Dict[str, str] = {}
    if args.platform in {"CUDA", "OpenCL"}:
        properties["Precision"] = "mixed"

    def make_simulation() -> Any:
        integrator = LangevinMiddleIntegrator(
            300.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        return app.Simulation(modeller.topology, system, integrator, platform, properties)

    restraint = CustomExternalForce(
        "0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    restraint.addGlobalParameter(
        "k",
        args.solute_restraint_k_kj_mol_nm2
        * unit.kilojoule_per_mole
        / unit.nanometer**2,
    )
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)
    for atom_index in restrained_indices:
        restraint.addParticle(atom_index, solute_positions_nm[atom_index])
    restraint_index = system.addForce(restraint)

    simulation = make_simulation()
    simulation.context.setPositions(modeller.positions)
    print("[6/7] Evaluating the initial energy", flush=True)
    initial = simulation.context.getState(getEnergy=True, getForces=True)
    initial_energy = float(initial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    initial_force = _maximum_force_kj_mol_nm(initial.getForces(asNumpy=True))
    if not math.isfinite(initial_energy) or not math.isfinite(initial_force):
        raise RuntimeError("Initial system energy or force is non-finite")

    print(
        f"[7/7] Restrained solvent minimization from {initial_energy:.3f} kJ/mol and "
        f"max force {initial_force:.3f} kJ/mol/nm",
        flush=True,
    )
    simulation.minimizeEnergy(maxIterations=args.solvent_minimization_iterations)
    restrained_state = simulation.context.getState(
        getEnergy=True, getForces=True, getPositions=True
    )
    restrained_energy = float(
        restrained_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    restrained_force = _maximum_force_kj_mol_nm(
        restrained_state.getForces(asNumpy=True)
    )
    restrained_positions = restrained_state.getPositions()
    del simulation

    system.removeForce(restraint_index)
    simulation = make_simulation()
    simulation.context.setPositions(restrained_positions)
    print(
        f"Full-system minimization from {restrained_energy:.3f} kJ/mol and "
        f"max force {restrained_force:.3f} kJ/mol/nm",
        flush=True,
    )
    simulation.minimizeEnergy(maxIterations=args.max_minimization_iterations)
    final = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
    final_energy = float(final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    final_force = _maximum_force_kj_mol_nm(final.getForces(asNumpy=True))
    if not math.isfinite(final_energy) or not math.isfinite(final_force):
        raise RuntimeError("Minimized system energy or force is non-finite")

    with (args.output_dir / "minimized_solvated.pdb").open("w") as handle:
        app.PDBFile.writeFile(modeller.topology, final.getPositions(), handle, keepIds=True)
    (args.output_dir / "system.xml").write_text(XmlSerializer.serialize(system))
    (args.output_dir / "minimized_state.xml").write_text(XmlSerializer.serialize(final))

    residues = list(modeller.topology.residues())
    waters = sum(residue.name in {"HOH", "WAT"} for residue in residues)
    final_forces = final.getForces(asNumpy=True)
    force_summary = _force_summary(
        final_forces,
        protein_atoms=protein_atoms,
        solute_atoms=pre_solvent_atoms,
    )
    residue_net_force_summary = _residue_net_force_summary(
        final_forces,
        modeller.topology,
        protein_atoms=protein_atoms,
        solute_atoms=pre_solvent_atoms,
    )
    maximum_residue_net_force = max(
        summary["max_kj_mol_nm"] for summary in residue_net_force_summary.values()
    )
    minimization_ready = (
        final_energy < initial_energy
        and maximum_residue_net_force <= args.max_residue_net_force_kj_mol_nm
    )
    result = {
        "transition_id": record.get("transition_id"),
        "status": (
            "minimized_ready_for_dynamics" if minimization_ready else "minimized_incomplete"
        ),
        "platform": args.platform,
        "seed": args.seed,
        "forcefields": {
            "protein": "amber14/protein.ff14SB.xml",
            "water": "amber14/tip3p.xml",
            "ligand": args.small_molecule_forcefield,
        },
        "protein": protein_diagnostics,
        "ligand": ligand_diagnostics,
        "system": {
            "pre_solvent_atoms": pre_solvent_atoms,
            "solvated_atoms": sum(1 for _ in modeller.topology.atoms()),
            "solvated_residues": len(residues),
            "waters": waters,
            "padding_nm": args.padding_nm,
            "ionic_strength_molar": args.ionic_strength_molar,
        },
        "minimization": {
            "max_iterations": args.max_minimization_iterations,
            "solvent_minimization_iterations": args.solvent_minimization_iterations,
            "solute_restraint_k_kj_mol_nm2": args.solute_restraint_k_kj_mol_nm2,
            "initial_potential_kj_mol": initial_energy,
            "restrained_potential_kj_mol": restrained_energy,
            "final_potential_kj_mol": final_energy,
            "energy_change_kj_mol": final_energy - initial_energy,
            "initial_max_force_kj_mol_nm": initial_force,
            "restrained_max_force_kj_mol_nm": restrained_force,
            "final_max_force_kj_mol_nm": final_force,
            "final_atomic_force_summary": force_summary,
            "final_residue_net_force_summary": residue_net_force_summary,
            "maximum_residue_net_force_kj_mol_nm": maximum_residue_net_force,
            "max_residue_net_force_threshold_kj_mol_nm": (
                args.max_residue_net_force_kj_mol_nm
            ),
            "ready_for_dynamics_smoke": minimization_ready,
        },
        "outputs": {
            "unsolvated_pdb": str(args.output_dir / "repaired_complex_unsolvated.pdb"),
            "minimized_pdb": str(args.output_dir / "minimized_solvated.pdb"),
            "system_xml": str(args.output_dir / "system.xml"),
            "state_xml": str(args.output_dir / "minimized_state.xml"),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    record = load_candidate(
        args.candidate_manifest,
        candidate_index=args.candidate_index,
        transition_id=args.transition_id,
    )
    print(f"Preparing {record.get('transition_id')} on {args.platform}", flush=True)
    result = prepare_and_minimize(record, args)
    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
