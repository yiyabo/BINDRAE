#!/usr/bin/env python3
"""Relax and score one exported BINDRAE path with an OpenMM Gate-0 contract.

The primary development scorer is an all-solute ff14SB/OpenFF 2.2.1 system
with GBn2 implicit solvent.  Candidate frames are aligned to the prepared holo
topology, injected residue-coherently, restrained on a declared atom subset,
and minimized independently while traversing holo to apo.  Raw and relaxed
energies are both retained; this script does not call either profile a free
energy or a kinetic transition path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_md_pilot_system import choose_ligand_fragment  # noqa: E402
from src.data.openmm_gate0 import (  # noqa: E402
    OPENMM_GATE0_SCORE_SCHEMA_VERSION,
    assess_relaxed_frame_validity,
    align_candidate_to_topology,
    build_candidate_topology_mapping,
    inject_candidate_frame,
    load_frame_reference_cache,
    load_path_candidate,
    mapped_target_indices,
    rms_distance,
    relaxed_frame_validity_contract,
    summarize_energy_profile,
    topology_atom_records,
    validate_openmm_gate0_score_report,
    validate_reference_topology_state,
)


IMPLICIT_SYSTEM_SCHEMA_VERSION = "bindrae_path4_gate0_implicit_system_v1"
RESULT_SCHEMA_VERSION = OPENMM_GATE0_SCORE_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implicit-cache-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--platform", choices=["CPU", "CUDA", "OpenCL"], default="CPU")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--restraint-mode", choices=["ca", "backbone", "heavy"], default="ca")
    parser.add_argument("--restraint-k-kj-mol-nm2", type=float, default=5000.0)
    parser.add_argument("--minimization-tolerance-kj-mol-nm", type=float, default=25.0)
    parser.add_argument("--max-minimization-iterations", type=int, default=250)
    parser.add_argument(
        "--maximum-relaxed-residue-net-force-kj-mol-nm",
        type=float,
        default=500.0,
        help=(
            "Mark a relaxed frame invalid when its maximum protein residue-net "
            "force exceeds this post-minimization threshold."
        ),
    )
    parser.add_argument("--minimum-residue-mapping", type=float, default=0.98)
    parser.add_argument("--minimum-atom-mapping", type=float, default=0.95)
    parser.add_argument(
        "--maximum-reference-atomic-force-kj-mol-nm",
        type=float,
        default=1.0e6,
    )
    parser.add_argument("--severe-clash-distance-angstrom", type=float, default=1.5)
    parser.add_argument(
        "--frame-initialization",
        choices=["reference_cache", "prepared_reference", "previous_relaxed"],
        default="reference_cache",
        help=(
            "Use one frozen all-atom reference per frame by default. The other "
            "modes are retained only for explicit engineering diagnostics."
        ),
    )
    parser.add_argument("--frame-reference-cache", type=Path, default=None)
    parser.add_argument("--diagnostic-force-components", action="store_true")
    parser.add_argument("--force-rebuild-system", action="store_true")
    return parser.parse_args()


def _resolve_path(path: object, project_root: Path) -> Path:
    value = Path(str(path))
    return value if value.is_absolute() else project_root / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def _system_contract(
    unsolvated_pdb: Path,
    ligand_sdf: Path,
    preparation_report: Path,
) -> Dict[str, Any]:
    return {
        "schema_version": IMPLICIT_SYSTEM_SCHEMA_VERSION,
        "inputs": {
            "unsolvated_pdb": str(unsolvated_pdb),
            "unsolvated_pdb_sha256": _sha256(unsolvated_pdb),
            "ligand_sdf": str(ligand_sdf),
            "ligand_sdf_sha256": _sha256(ligand_sdf),
            "preparation_report": str(preparation_report),
            "preparation_report_sha256": _sha256(preparation_report),
        },
        "forcefields": {
            "protein": "amber14/protein.ff14SB.xml",
            "implicit_solvent": "implicit/gbn2.xml",
            "ligand": "openff-2.2.1",
        },
        "system_options": {
            "constraints": "HBonds",
            "nonbonded_method": "NoCutoff",
            "remove_cmmotion": True,
        },
    }


def _system_generator_options(app) -> Dict[str, Dict[str, Any]]:
    """Keep topology-independent and nonperiodic options API-separated."""
    return {
        "forcefield_kwargs": {
            "constraints": app.HBonds,
            "removeCMMotion": True,
        },
        "nonperiodic_forcefield_kwargs": {
            "nonbondedMethod": app.NoCutoff,
        },
    }


def _write_implicit_system_cache(
    *,
    cache_dir: Path,
    contract: Mapping[str, Any],
    unsolvated_pdb: Path,
    ligand_sdf: Path,
    protein_atoms: int,
) -> Tuple[Path, Path]:
    from openff.toolkit import Molecule
    from openmm import XmlSerializer
    from openmm import app, unit
    from openmmforcefields.generators import SystemGenerator

    cache_dir.mkdir(parents=True, exist_ok=True)
    system_path = cache_dir / "implicit_system.xml"
    topology_path = cache_dir / "implicit_topology.pdb"
    metadata_path = cache_dir / "contract.json"
    pdb = app.PDBFile(str(unsolvated_pdb))
    positions_angstrom = np.asarray(
        pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64
    )
    if not 0 < protein_atoms < positions_angstrom.shape[0]:
        raise ValueError(
            f"Invalid prepared protein atom count {protein_atoms} for "
            f"{positions_angstrom.shape[0]} solute atoms"
        )
    rdkit_ligand, _ = choose_ligand_fragment(
        ligand_sdf, positions_angstrom[:protein_atoms]
    )
    off_ligand = Molecule.from_rdkit(
        rdkit_ligand,
        allow_undefined_stereo=True,
        hydrogens_are_explicit=True,
    )
    off_ligand.name = "LIG"
    generator = SystemGenerator(
        forcefields=["amber14/protein.ff14SB.xml", "implicit/gbn2.xml"],
        small_molecule_forcefield="openff-2.2.1",
        molecules=[off_ligand],
        **_system_generator_options(app),
        cache=str(cache_dir / "small_molecule_system_cache.json"),
    )
    system = generator.create_system(pdb.topology)
    system_path.write_text(XmlSerializer.serialize(system))
    with topology_path.open("w") as handle:
        app.PDBFile.writeFile(pdb.topology, pdb.positions, handle, keepIds=True)
    metadata_path.write_text(json.dumps(dict(contract), indent=2, sort_keys=True) + "\n")
    return system_path, topology_path


def ensure_implicit_system(
    *,
    cache_dir: Path,
    preparation: Mapping[str, Any],
    preparation_report: Path,
    project_root: Path,
    force_rebuild: bool,
) -> Tuple[Path, Path, Dict[str, Any]]:
    outputs = dict(preparation.get("outputs") or {})
    ligand = dict(preparation.get("ligand") or {})
    protein = dict(preparation.get("protein") or {})
    unsolvated_pdb = _resolve_path(outputs.get("unsolvated_pdb"), project_root)
    ligand_sdf = _resolve_path(ligand.get("input_sdf"), project_root)
    missing = [str(path) for path in (unsolvated_pdb, ligand_sdf) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing implicit-system inputs: {missing}")
    contract = _system_contract(unsolvated_pdb, ligand_sdf, preparation_report)
    system_path = cache_dir / "implicit_system.xml"
    topology_path = cache_dir / "implicit_topology.pdb"
    metadata_path = cache_dir / "contract.json"
    existing = [path.exists() for path in (system_path, topology_path, metadata_path)]
    if any(existing) and not all(existing) and not force_rebuild:
        raise FileExistsError(
            f"Partial implicit-system cache at {cache_dir}; use --force-rebuild-system"
        )
    if all(existing) and not force_rebuild:
        observed = _load_json(metadata_path)
        if observed != contract:
            raise ValueError(
                f"Stale implicit-system cache contract at {cache_dir}; "
                "use --force-rebuild-system with a new or intentionally replaced cache"
            )
        return system_path, topology_path, contract
    if force_rebuild:
        for path in (system_path, topology_path, metadata_path):
            path.unlink(missing_ok=True)
    return (
        *_write_implicit_system_cache(
            cache_dir=cache_dir,
            contract=contract,
            unsolvated_pdb=unsolvated_pdb,
            ligand_sdf=ligand_sdf,
            protein_atoms=int(protein["prepared_protein_atoms"]),
        ),
        contract,
    )


def _platform(args: argparse.Namespace):
    from openmm import Platform

    platform = Platform.getPlatformByName(args.platform)
    properties: Dict[str, str] = {}
    if args.platform in {"CUDA", "OpenCL"}:
        properties["Precision"] = "mixed"
        properties["DeviceIndex"] = str(args.device_index)
    elif args.platform == "CPU":
        properties["Threads"] = str(args.cpu_threads)
    return platform, properties


def _frame_initial_positions(
    prepared_reference_angstrom: np.ndarray,
    previous_relaxed_angstrom: np.ndarray,
    mode: str,
    cached_reference_angstrom: np.ndarray | None = None,
) -> np.ndarray:
    prepared = np.asarray(prepared_reference_angstrom, dtype=np.float64)
    previous = np.asarray(previous_relaxed_angstrom, dtype=np.float64)
    if prepared.shape != previous.shape or prepared.ndim != 2 or prepared.shape[1] != 3:
        raise ValueError("Frame initialization positions must have equal [A, 3] shapes")
    if mode == "reference_cache":
        if cached_reference_angstrom is None:
            raise ValueError("reference_cache mode requires cached frame positions")
        cached = np.asarray(cached_reference_angstrom, dtype=np.float64)
        if cached.shape != prepared.shape:
            raise ValueError("Cached frame positions do not match the topology")
        selected = cached
    elif mode == "prepared_reference":
        selected = prepared
    elif mode == "previous_relaxed":
        selected = previous
    else:
        raise ValueError(f"Unsupported frame initialization mode: {mode}")
    if not np.isfinite(selected).all():
        raise ValueError("Frame initialization positions are non-finite")
    return selected.copy()


def _add_target_restraint(
    system,
    topology_indices: np.ndarray,
    initial_targets_nm: np.ndarray,
    restraint_k: float,
):
    from openmm import CustomExternalForce, unit

    force = CustomExternalForce(
        "0.5*gate0_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    force.addGlobalParameter(
        "gate0_k",
        float(restraint_k) * unit.kilojoule_per_mole / unit.nanometer**2,
    )
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    for topology_index, target in zip(topology_indices, initial_targets_nm):
        force.addParticle(int(topology_index), [float(value) for value in target])
    system.addForce(force)
    return force


def _update_target_restraint(force, context, targets_nm: np.ndarray) -> None:
    if force.getNumParticles() != targets_nm.shape[0]:
        raise ValueError("OpenMM restraint target count changed across path frames")
    for restraint_index, target in enumerate(targets_nm):
        topology_index, _ = force.getParticleParameters(restraint_index)
        force.setParticleParameters(
            restraint_index,
            topology_index,
            [float(value) for value in target],
        )
    force.updateParametersInContext(context)


def _state_values(context, *, get_positions: bool = True):
    from openmm import unit

    state = context.getState(
        getEnergy=True,
        getForces=True,
        getPositions=get_positions,
    )
    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=np.float64,
    )
    positions = None
    if get_positions:
        positions = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
            dtype=np.float64,
        )
    if not math.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError("OpenMM returned a non-finite energy or force")
    if positions is not None and not np.isfinite(positions).all():
        raise RuntimeError("OpenMM returned non-finite positions")
    return energy, forces, positions


def _make_minimization_reporter(openmm_module):
    """Return an optional iteration recorder without requiring a specific OpenMM."""
    reporter_base = getattr(openmm_module, "MinimizationReporter", None)
    if reporter_base is None:
        return None

    class _IterationReporter(reporter_base):
        def __init__(self):
            super().__init__()
            self.report_calls = 0
            self.last_iteration = None

        def report(self, iteration, x, grad, args):
            del x, grad, args
            self.report_calls += 1
            self.last_iteration = int(iteration)
            return False

    return _IterationReporter()


def _minimization_audit(
    reporter,
    *,
    maximum_iterations: int,
    tolerance_kj_mol_nm: float,
    unrestrained_potential_change_kj_mol: float,
) -> Dict[str, Any]:
    if reporter is None:
        reporter_callback_count = None
        last_reported_iteration_index = None
        termination = "not_reported_by_openmm_api"
        reporter_available = False
    else:
        reporter_available = True
        reporter_callback_count = int(getattr(reporter, "report_calls", 0))
        last_iteration = getattr(reporter, "last_iteration", None)
        last_reported_iteration_index = (
            None if last_iteration is None else int(last_iteration)
        )
        termination = "returned_from_local_energy_minimizer"
    return {
        "reporter_available": reporter_available,
        "reporter_callback_count": reporter_callback_count,
        "last_reported_iteration_index": last_reported_iteration_index,
        "termination": termination,
        "termination_reason_available": False,
        "maximum_iterations": int(maximum_iterations),
        "tolerance_kj_mol_nm": float(tolerance_kj_mol_nm),
        "unrestrained_potential_change_kj_mol": float(
            unrestrained_potential_change_kj_mol
        ),
    }


def _assign_diagnostic_force_groups(system) -> List[Tuple[int, str]]:
    """Assign one OpenMM force group per force for component diagnostics."""
    num_forces = int(system.getNumForces())
    if num_forces > 32:
        raise ValueError(
            f"OpenMM force-component diagnostics support at most 32 forces, got {num_forces}"
        )
    labels: List[Tuple[int, str]] = []
    counts: Dict[str, int] = {}
    for group in range(num_forces):
        force = system.getForce(group)
        force_name = force.__class__.__name__
        occurrence = counts.get(force_name, 0)
        counts[force_name] = occurrence + 1
        label = force_name if occurrence == 0 else f"{force_name}_{occurrence}"
        force.setForceGroup(group)
        labels.append((group, label))
    return labels


def _force_component_energies(context, labels: Sequence[Tuple[int, str]]) -> Dict[str, float]:
    from openmm import unit

    values: Dict[str, float] = {}
    for group, label in labels:
        state = context.getState(getEnergy=True, groups=1 << int(group))
        values[label] = float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
    return values


def _residue_net_force_summary(
    forces: np.ndarray,
    atom_records,
    protein_atoms: int,
) -> Dict[str, float]:
    grouped: Dict[int, np.ndarray] = {}
    for atom in atom_records:
        if atom.index >= protein_atoms:
            continue
        grouped.setdefault(atom.residue_index, np.zeros(3, dtype=np.float64))
        grouped[atom.residue_index] += forces[atom.index]
    norms = np.asarray([np.linalg.norm(value) for value in grouped.values()])
    if norms.size == 0:
        raise ValueError("No protein residue forces were available")
    return {
        "protein_residue_net_force_mean_kj_mol_nm": float(norms.mean()),
        "protein_residue_net_force_p95_kj_mol_nm": float(np.quantile(norms, 0.95)),
        "protein_residue_net_force_max_kj_mol_nm": float(norms.max()),
    }


def _minimum_distances(
    positions_angstrom: np.ndarray,
    atom_records,
    protein_atoms: int,
    severe_threshold: float,
) -> Dict[str, float]:
    protein = [
        atom for atom in atom_records if atom.index < protein_atoms and atom.element != "H"
    ]
    ligand = [
        atom for atom in atom_records if atom.index >= protein_atoms and atom.element != "H"
    ]
    protein_indices = np.asarray([atom.index for atom in protein], dtype=np.int64)
    ligand_indices = np.asarray([atom.index for atom in ligand], dtype=np.int64)
    if protein_indices.size == 0 or ligand_indices.size == 0:
        raise ValueError("Implicit topology is missing protein or ligand heavy atoms")

    protein_ligand = np.linalg.norm(
        positions_angstrom[protein_indices, None, :]
        - positions_angstrom[ligand_indices, :][None, :, :],
        axis=-1,
    )
    protein_residue = np.asarray([atom.residue_index for atom in protein], dtype=np.int64)
    internal_min = math.inf
    internal_severe = 0
    internal_pairs = 0
    protein_xyz = positions_angstrom[protein_indices]
    chunk_size = 512
    for start in range(0, protein_indices.size, chunk_size):
        stop = min(start + chunk_size, protein_indices.size)
        distances = np.linalg.norm(
            protein_xyz[start:stop, None, :] - protein_xyz[None, :, :],
            axis=-1,
        )
        left_indices = np.arange(start, stop, dtype=np.int64)[:, None]
        right_indices = np.arange(protein_indices.size, dtype=np.int64)[None, :]
        valid = right_indices > left_indices
        valid &= np.abs(
            protein_residue[start:stop, None] - protein_residue[None, :]
        ) > 1
        values = distances[valid]
        if values.size:
            internal_min = min(internal_min, float(values.min()))
            internal_severe += int(np.count_nonzero(values < severe_threshold))
            internal_pairs += int(values.size)
    if internal_pairs == 0:
        raise ValueError("No nonlocal protein heavy-atom pairs were available")
    return {
        "protein_ligand_min_distance_angstrom": float(protein_ligand.min()),
        "protein_ligand_severe_clash_pairs": int(
            np.count_nonzero(protein_ligand < severe_threshold)
        ),
        "protein_internal_min_distance_angstrom": float(internal_min),
        "protein_internal_severe_clash_pairs": int(internal_severe),
    }


def score_candidate(
    args: argparse.Namespace,
    *,
    system_path: Path,
    topology_path: Path,
    preparation: Mapping[str, Any],
    system_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    import openmm as openmm_module
    from openmm import Context, LocalEnergyMinimizer, VerletIntegrator, XmlSerializer, unit
    from openmm import app

    candidate = load_path_candidate(args.candidate)
    pdb = app.PDBFile(str(topology_path))
    atom_records = topology_atom_records(pdb.topology)
    topology_positions = np.asarray(
        pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64
    )
    mapping = build_candidate_topology_mapping(
        candidate,
        atom_records,
        minimum_residue_fraction=float(args.minimum_residue_mapping),
        minimum_atom_fraction=float(args.minimum_atom_mapping),
    )
    aligned_path, alignment = align_candidate_to_topology(
        candidate, mapping, topology_positions
    )
    frame_reference_cache = None
    frame_reference_positions = None
    if args.frame_initialization == "reference_cache":
        frame_reference_cache = load_frame_reference_cache(
            args.frame_reference_cache,
            candidate,
            system_contract=system_contract,
            topology_atom_count=topology_positions.shape[0],
        )
        frame_reference_positions = frame_reference_cache.all_atom_pos_angstrom
    restraint_topology, restraint_residue, restraint_atom14 = mapped_target_indices(
        mapping, args.restraint_mode
    )
    initial_targets_nm = (
        aligned_path[-1, restraint_residue, restraint_atom14] / 10.0
    )

    system = XmlSerializer.deserialize(system_path.read_text())
    restraint = _add_target_restraint(
        system,
        restraint_topology,
        initial_targets_nm,
        float(args.restraint_k_kj_mol_nm2),
    )
    force_component_labels = (
        _assign_diagnostic_force_groups(system)
        if args.diagnostic_force_components
        else []
    )
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    platform, properties = _platform(args)
    context = Context(system, integrator, platform, properties)
    previous_relaxed_positions = topology_positions.copy()
    context.setPositions(unit.Quantity(topology_positions / 10.0, unit.nanometer))
    protein_atoms = int((preparation.get("protein") or {})["prepared_protein_atoms"])

    frame_records: List[Dict[str, Any] | None] = [None] * candidate.n_frames
    total_started = time.perf_counter()
    context.setParameter("gate0_k", 0.0)
    reference_energy, reference_forces, _ = _state_values(
        context, get_positions=False
    )
    reference_diagnostics = validate_reference_topology_state(
        reference_energy,
        reference_forces,
        protein_atoms,
        maximum_atomic_force_kj_mol_nm=float(
            args.maximum_reference_atomic_force_kj_mol_nm
        ),
    )
    reference_components = _force_component_energies(
        context, force_component_labels
    )
    energy_calls = 1 + len(force_component_labels)
    for frame_index in reversed(range(candidate.n_frames)):
        frame_started = time.perf_counter()
        target = aligned_path[frame_index]
        initial_positions = _frame_initial_positions(
            topology_positions,
            previous_relaxed_positions,
            args.frame_initialization,
            (
                frame_reference_positions[frame_index]
                if frame_reference_positions is not None
                else None
            ),
        )
        injected = inject_candidate_frame(initial_positions, target, mapping)
        context.setPositions(unit.Quantity(injected / 10.0, unit.nanometer))
        targets_nm = target[restraint_residue, restraint_atom14] / 10.0
        _update_target_restraint(restraint, context, targets_nm)

        context.setParameter("gate0_k", 0.0)
        raw_energy, raw_forces, _ = _state_values(context, get_positions=False)
        energy_calls += 1
        raw_components = _force_component_energies(
            context, force_component_labels
        )
        energy_calls += len(force_component_labels)
        context.setParameter(
            "gate0_k", float(args.restraint_k_kj_mol_nm2)
        )
        minimization_reporter = _make_minimization_reporter(openmm_module)
        minimization_args = (
            context,
            float(args.minimization_tolerance_kj_mol_nm)
            * unit.kilojoule_per_mole
            / unit.nanometer,
            int(args.max_minimization_iterations),
        )
        if minimization_reporter is None:
            LocalEnergyMinimizer.minimize(*minimization_args)
        else:
            LocalEnergyMinimizer.minimize(
                *minimization_args, minimization_reporter
            )
        context.setParameter("gate0_k", 0.0)
        relaxed_energy, relaxed_forces, relaxed_positions = _state_values(context)
        energy_calls += 1
        relaxed_components = _force_component_energies(
            context, force_component_labels
        )
        energy_calls += len(force_component_labels)
        context.setParameter(
            "gate0_k", float(args.restraint_k_kj_mol_nm2)
        )
        previous_relaxed_positions = relaxed_positions
        force_norms = np.linalg.norm(relaxed_forces[:protein_atoms], axis=-1)
        all_force_norms = np.linalg.norm(relaxed_forces, axis=-1)
        target_rms = rms_distance(
            relaxed_positions[restraint_topology],
            target[restraint_residue, restraint_atom14],
        )
        residue_force_summary = _residue_net_force_summary(
            relaxed_forces, atom_records, protein_atoms
        )
        relaxed_validity = assess_relaxed_frame_validity(
            residue_force_summary[
                "protein_residue_net_force_max_kj_mol_nm"
            ],
            maximum_residue_net_force_kj_mol_nm=float(
                args.maximum_relaxed_residue_net_force_kj_mol_nm
            ),
        )
        frame_record: Dict[str, Any] = {
            "frame_index": int(frame_index),
            "time": float(candidate.times[frame_index]),
            "raw_potential_kj_mol": float(raw_energy),
            "relaxed_potential_kj_mol": float(relaxed_energy),
            "raw_force_components_kj_mol": raw_components,
            "relaxed_force_components_kj_mol": relaxed_components,
            "raw_protein_atomic_force_max_kj_mol_nm": float(
                np.linalg.norm(raw_forces[:protein_atoms], axis=-1).max()
            ),
            "relaxed_protein_atomic_force_mean_kj_mol_nm": float(force_norms.mean()),
            "relaxed_protein_atomic_force_p95_kj_mol_nm": float(
                np.quantile(force_norms, 0.95)
            ),
            "relaxed_protein_atomic_force_max_kj_mol_nm": float(force_norms.max()),
            "relaxed_all_atom_force_mean_kj_mol_nm": float(all_force_norms.mean()),
            "relaxed_all_atom_force_rms_kj_mol_nm": float(
                np.sqrt(np.mean(all_force_norms**2))
            ),
            "relaxed_all_atom_force_p95_kj_mol_nm": float(
                np.quantile(all_force_norms, 0.95)
            ),
            "relaxed_all_atom_force_max_kj_mol_nm": float(all_force_norms.max()),
            "restraint_target_rms_angstrom": float(target_rms),
            "relaxed_valid": bool(relaxed_validity["valid"]),
            "relaxed_invalid_reasons": list(
                relaxed_validity["invalid_reasons"]
            ),
            "relaxed_validity": relaxed_validity,
            "minimization": _minimization_audit(
                minimization_reporter,
                maximum_iterations=int(args.max_minimization_iterations),
                tolerance_kj_mol_nm=float(
                    args.minimization_tolerance_kj_mol_nm
                ),
                unrestrained_potential_change_kj_mol=float(
                    relaxed_energy - raw_energy
                ),
            ),
            "wall_seconds": float(time.perf_counter() - frame_started),
            **residue_force_summary,
        }
        frame_record.update(
            _minimum_distances(
                relaxed_positions,
                atom_records,
                protein_atoms,
                float(args.severe_clash_distance_angstrom),
            )
        )
        frame_records[frame_index] = frame_record

    del context
    del integrator
    records = [record for record in frame_records if record is not None]
    raw_profile = summarize_energy_profile(
        candidate.times,
        [float(record["raw_potential_kj_mol"]) for record in records],
    )
    relaxed_profile = summarize_energy_profile(
        candidate.times,
        [float(record["relaxed_potential_kj_mol"]) for record in records],
        interior_valid_mask=[bool(record["relaxed_valid"]) for record in records],
    )
    interior_records = records[1:-1]
    valid_interior_records = [
        record for record in interior_records if bool(record["relaxed_valid"])
    ]
    severe_counts = np.asarray(
        [
            int(record["protein_ligand_severe_clash_pairs"])
            + int(record["protein_internal_severe_clash_pairs"])
            for record in records[1:-1]
        ],
        dtype=np.int64,
    )
    invalid_mask = np.asarray(
        [not bool(record["relaxed_valid"]) for record in interior_records],
        dtype=np.bool_,
    )
    invalid_reason_counts: Dict[str, int] = {}
    for record in interior_records:
        for reason in record["relaxed_invalid_reasons"]:
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1

    def _valid_quantile(field: str, quantile: float) -> float | None:
        if not valid_interior_records:
            return None
        return float(
            np.quantile(
                [float(record[field]) for record in valid_interior_records],
                quantile,
            )
        )
    cache_metadata = None
    if frame_reference_cache is not None:
        cache_metadata = {
            "path": str(frame_reference_cache.source_path),
            "sha256": _sha256(frame_reference_cache.source_path),
            "source_candidate": frame_reference_cache.source_candidate,
            "source_candidate_sha256": frame_reference_cache.source_candidate_sha256,
            "generation_contract": frame_reference_cache.generation_contract,
        }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "sample_id": candidate.sample_id,
        "candidate": str(candidate.source_path),
        "candidate_label": candidate.candidate_label,
        "path_parameterization": candidate.path_parameterization,
        "platform": args.platform,
        "device_index": str(args.device_index),
        "contract": {
            "forcefield": "ff14SB + OpenFF 2.2.1 + GBn2 implicit solvent",
            "restraint_mode": args.restraint_mode,
            "restraint_k_kj_mol_nm2": float(args.restraint_k_kj_mol_nm2),
            "minimization_tolerance_kj_mol_nm": float(
                args.minimization_tolerance_kj_mol_nm
            ),
            "max_minimization_iterations": int(args.max_minimization_iterations),
            "relaxed_frame_validity": relaxed_frame_validity_contract(
                float(args.maximum_relaxed_residue_net_force_kj_mol_nm)
            ),
            "frame_traversal": "holo_to_apo",
            "frame_initialization": args.frame_initialization,
            "frame_reference_cache_sha256": (
                cache_metadata["sha256"] if cache_metadata is not None else None
            ),
            "energy_interpretation": "potential_energy_not_free_energy",
        },
        "frame_reference_cache": cache_metadata,
        "mapping": {
            "mapped_residue_fraction": mapping.mapped_residue_fraction,
            "mapped_atom_fraction": mapping.mapped_atom_fraction,
            "mapped_atoms": int(mapping.topology_atom_indices.size),
            "restraint_atoms": int(restraint_topology.size),
            "ignored_chain_labels": mapping.ignored_chain_labels,
            **alignment,
        },
        "reference_topology": {
            **reference_diagnostics,
            "force_components_kj_mol": reference_components,
        },
        "raw_energy_profile": raw_profile,
        "relaxed_energy_profile": relaxed_profile,
        "relaxed_path": {
            "interior_frames": int(severe_counts.size),
            "valid_interior_frames": int(len(valid_interior_records)),
            "invalid_interior_frames": int(invalid_mask.sum()),
            "invalid_frame_fraction": float(
                invalid_mask.mean() if invalid_mask.size else 0.0
            ),
            "invalid_reason_counts": invalid_reason_counts,
            "severe_clash_frame_fraction": float(
                np.mean(severe_counts > 0) if severe_counts.size else 0.0
            ),
            "invalid_or_severe_clash_frame_fraction": float(
                np.mean(invalid_mask | (severe_counts > 0))
                if severe_counts.size
                else 0.0
            ),
            "severe_clash_pairs_max": int(
                severe_counts.max() if severe_counts.size else 0
            ),
            "restraint_target_rms_p95_angstrom": float(
                _valid_quantile("restraint_target_rms_angstrom", 0.95)
            ) if valid_interior_records else None,
            "protein_residue_net_force_p95_over_frames_kj_mol_nm": (
                _valid_quantile(
                    "protein_residue_net_force_p95_kj_mol_nm", 0.95
                )
            ),
        },
        "energy_force_calls": int(energy_calls),
        "wall_seconds": float(time.perf_counter() - total_started),
        "frames": records,
    }


def main() -> None:
    args = parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if args.restraint_k_kj_mol_nm2 <= 0.0:
        raise ValueError("--restraint-k-kj-mol-nm2 must be positive")
    if args.max_minimization_iterations <= 0:
        raise ValueError("--max-minimization-iterations must be positive")
    if args.maximum_relaxed_residue_net_force_kj_mol_nm <= 0.0:
        raise ValueError(
            "--maximum-relaxed-residue-net-force-kj-mol-nm must be positive"
        )
    if args.maximum_reference_atomic_force_kj_mol_nm <= 0.0:
        raise ValueError(
            "--maximum-reference-atomic-force-kj-mol-nm must be positive"
        )
    if args.frame_initialization == "reference_cache":
        if args.frame_reference_cache is None:
            raise ValueError(
                "--frame-reference-cache is required for reference_cache mode"
            )
    elif args.frame_reference_cache is not None:
        raise ValueError(
            "--frame-reference-cache may only be used with reference_cache mode"
        )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite Gate-0 result: {args.output}")
    preparation = _load_json(args.preparation_report)
    system_path, topology_path, system_contract = ensure_implicit_system(
        cache_dir=args.implicit_cache_dir,
        preparation=preparation,
        preparation_report=args.preparation_report,
        project_root=args.project_root,
        force_rebuild=bool(args.force_rebuild_system),
    )
    result = score_candidate(
        args,
        system_path=system_path,
        topology_path=topology_path,
        preparation=preparation,
        system_contract=system_contract,
    )
    result["implicit_system_contract"] = system_contract
    validate_openmm_gate0_score_report(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    printable = dict(result)
    printable.pop("frames", None)
    printable["frame_records"] = len(result["frames"])
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
