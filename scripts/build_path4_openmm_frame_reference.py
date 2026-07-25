#!/usr/bin/env python3
"""Build a frozen all-atom frame-reference cache from one Path-3 guide."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_path4_openmm_gate0 import (  # noqa: E402
    _add_target_restraint,
    _assign_diagnostic_force_groups,
    _load_json,
    _platform,
    _sha256,
    _state_values,
    ensure_implicit_system,
)
from scripts.optimize_path4_openmm_gate0 import (  # noqa: E402
    _precondition_reference_positions,
)
from src.data.openmm_gate0 import (  # noqa: E402
    FRAME_REFERENCE_CACHE_SCHEMA_VERSION,
    ReferenceTopologyStateError,
    align_candidate_to_topology,
    build_candidate_topology_mapping,
    build_frame_reference_cache_payload,
    load_path_candidate,
    rms_distance,
    topology_atom_records,
    validate_reference_topology_state,
)


REPORT_SCHEMA_VERSION = "bindrae_path4_frame_reference_report_v2"


class FrameReferenceBuildError(RuntimeError):
    """Stage-aware frame-reference failure with serializable evidence."""

    def __init__(
        self,
        *,
        status: str,
        failure_stage: str,
        rejection_type: str,
        rejection_reason: str,
        failure_context: Mapping[str, Any] | None = None,
        partial_report: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(rejection_reason)
        self.status = status
        self.failure_stage = failure_stage
        self.rejection_type = rejection_type
        self.rejection_reason = rejection_reason
        self.failure_context = dict(failure_context or {})
        self.partial_report = dict(partial_report or {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--implicit-cache-dir", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--platform", choices=["CPU", "CUDA", "OpenCL"], default="CPU")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--reference-relaxation-iterations", type=int, default=25)
    parser.add_argument(
        "--reference-restraint-k-kj-mol-nm2", type=float, default=100000.0
    )
    parser.add_argument(
        "--reference-minimization-tolerance-kj-mol-nm", type=float, default=500.0
    )
    parser.add_argument("--minimum-residue-mapping", type=float, default=0.98)
    parser.add_argument("--minimum-atom-mapping", type=float, default=0.95)
    parser.add_argument(
        "--maximum-reference-atomic-force-kj-mol-nm",
        type=float,
        default=1.0e6,
    )
    parser.add_argument(
        "--maximum-frame-reference-atomic-force-kj-mol-nm",
        type=float,
        default=1.0e6,
    )
    parser.add_argument(
        "--diagnostic-atom-force-threshold-kj-mol-nm",
        type=float,
        help="Record atom identities and force groups only above this threshold.",
    )
    parser.add_argument("--diagnostic-top-force-atoms", type=int, default=5)
    parser.add_argument("--force-rebuild-system", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "cpu_threads",
        "reference_relaxation_iterations",
        "reference_restraint_k_kj_mol_nm2",
        "reference_minimization_tolerance_kj_mol_nm",
        "maximum_reference_atomic_force_kj_mol_nm",
        "maximum_frame_reference_atomic_force_kj_mol_nm",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("minimum_residue_mapping", "minimum_atom_mapping"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    if (
        args.diagnostic_atom_force_threshold_kj_mol_nm is not None
        and args.diagnostic_atom_force_threshold_kj_mol_nm <= 0.0
    ):
        raise ValueError(
            "--diagnostic-atom-force-threshold-kj-mol-nm must be positive"
        )
    if args.diagnostic_top_force_atoms <= 0:
        raise ValueError("--diagnostic-top-force-atoms must be positive")


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_cache(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _atom_scope(
    topology_index: int, protein_atom_count: int, mapped_atom_indices: set[int]
) -> str:
    if topology_index >= int(protein_atom_count):
        return "environment"
    if topology_index in mapped_atom_indices:
        return "mapped_protein"
    return "hidden_protein"


def _atom_identity_payload(atom, *, scope: str) -> Dict[str, Any]:
    return {
        "topology_atom_index": int(atom.index),
        "scope": scope,
        "chain_id": atom.chain_id,
        "residue_number": int(atom.residue_number),
        "insertion_code": atom.insertion_code,
        "residue_name": atom.residue_name,
        "atom_name": atom.atom_name,
        "element": atom.element,
    }


def _bonded_neighbor_sets(topology, atom_count: int) -> List[set[int]]:
    neighbors = [set() for _ in range(int(atom_count))]
    for left, right in topology.bonds():
        left_index = int(left.index)
        right_index = int(right.index)
        neighbors[left_index].add(right_index)
        neighbors[right_index].add(left_index)
    return neighbors


def _top_force_atom_diagnostics(
    positions_angstrom: np.ndarray,
    forces_kj_mol_nm: np.ndarray,
    atom_records: Sequence[Any],
    bonded_neighbors: Sequence[set[int]],
    *,
    protein_atom_count: int,
    mapped_atom_indices: set[int],
    top_k: int,
) -> List[Dict[str, Any]]:
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    forces = np.asarray(forces_kj_mol_nm, dtype=np.float64)
    if positions.shape != forces.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Atom-force diagnostics require equal [A, 3] arrays")
    if len(atom_records) != positions.shape[0] or len(bonded_neighbors) != positions.shape[0]:
        raise ValueError("Atom-force diagnostics do not match the topology atom count")
    records_by_index = {int(atom.index): atom for atom in atom_records}
    if set(records_by_index) != set(range(positions.shape[0])):
        raise ValueError("Topology atom records do not cover a contiguous atom axis")

    norms = np.linalg.norm(forces, axis=-1)
    order = np.argsort(-norms, kind="stable")[: min(int(top_k), norms.size)]
    diagnostics: List[Dict[str, Any]] = []
    for topology_index_value in order:
        topology_index = int(topology_index_value)
        atom = records_by_index[topology_index]
        record = _atom_identity_payload(
            atom,
            scope=_atom_scope(
                topology_index, protein_atom_count, mapped_atom_indices
            ),
        )
        record.update(
            {
                "force_norm_kj_mol_nm": float(norms[topology_index]),
                "force_vector_kj_mol_nm": [
                    float(value) for value in forces[topology_index]
                ],
            }
        )
        excluded = set(bonded_neighbors[topology_index])
        excluded.add(topology_index)
        candidate_mask = np.ones(positions.shape[0], dtype=np.bool_)
        candidate_mask[np.asarray(sorted(excluded), dtype=np.int64)] = False
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size:
            distances = np.linalg.norm(
                positions[candidate_indices] - positions[topology_index], axis=-1
            )
            nearest_offset = int(np.argmin(distances))
            nearest_index = int(candidate_indices[nearest_offset])
            nearest_atom = records_by_index[nearest_index]
            record["nearest_not_directly_bonded_atom"] = {
                **_atom_identity_payload(
                    nearest_atom,
                    scope=_atom_scope(
                        nearest_index, protein_atom_count, mapped_atom_indices
                    ),
                ),
                "distance_angstrom": float(distances[nearest_offset]),
            }
        diagnostics.append(record)
    return diagnostics


def _force_component_atom_diagnostics(
    context,
    labels: Sequence[tuple[int, str]],
    atom_records: Sequence[Any],
    *,
    protein_atom_count: int,
    mapped_atom_indices: set[int],
    total_max_atom_index: int,
) -> Dict[str, Dict[str, Any]]:
    from openmm import unit

    records_by_index = {int(atom.index): atom for atom in atom_records}
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for group, label in labels:
        state = context.getState(
            getEnergy=True, getForces=True, groups=1 << int(group)
        )
        component_forces = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            ),
            dtype=np.float64,
        )
        norms = np.linalg.norm(component_forces, axis=-1)
        maximum_index = int(np.argmax(norms))
        maximum_atom = records_by_index[maximum_index]
        diagnostics[label] = {
            "potential_kj_mol": float(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ),
            "atomic_force_max_kj_mol_nm": float(norms[maximum_index]),
            "maximum_force_atom": _atom_identity_payload(
                maximum_atom,
                scope=_atom_scope(
                    maximum_index, protein_atom_count, mapped_atom_indices
                ),
            ),
            "force_on_total_max_atom_kj_mol_nm": float(
                norms[int(total_max_atom_index)]
            ),
        }
    return diagnostics


def _classify_failure(stage: str, error: Exception) -> tuple[str, str]:
    message = str(error).lower()
    if stage == "candidate_topology_mapping":
        if "residue identity mismatch" in message:
            return "rejected", "residue_identity_mismatch"
        return "rejected", "candidate_topology_mapping_failure"
    if stage in {"candidate_load", "candidate_alignment"}:
        return "rejected", "candidate_coordinate_contract_failure"
    if stage == "prepared_topology_preflight":
        return "rejected", "prepared_topology_physical_failure"
    if stage == "frame_reference_preflight":
        return "rejected", "frame_reference_physical_failure"
    return "failed", f"{stage}_failure"


def _validate_state_for_stage(
    potential_kj_mol: float,
    atomic_forces_kj_mol_nm: np.ndarray,
    protein_atom_count: int,
    *,
    maximum_atomic_force_kj_mol_nm: float,
    state_label: str,
    failure_stage: str,
    rejection_type: str,
    context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    details = dict(context or {})
    try:
        diagnostics = validate_reference_topology_state(
            potential_kj_mol,
            atomic_forces_kj_mol_nm,
            protein_atom_count,
            maximum_atomic_force_kj_mol_nm=maximum_atomic_force_kj_mol_nm,
            state_label=state_label,
        )
    except Exception as error:
        if isinstance(error, ReferenceTopologyStateError):
            details.update(error.diagnostics)
        raise FrameReferenceBuildError(
            status="rejected",
            failure_stage=failure_stage,
            rejection_type=rejection_type,
            rejection_reason=f"{type(error).__name__}: {error}",
            failure_context=details,
        ) from error
    diagnostics.update(details)
    return diagnostics


def _rejection_report(
    error: FrameReferenceBuildError, *, wall_seconds: float
) -> Dict[str, Any]:
    report = dict(error.partial_report)
    report.update(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": error.status,
            "failure_stage": error.failure_stage,
            "rejection_type": error.rejection_type,
            "rejection_reason": error.rejection_reason,
            "failure_context": error.failure_context,
            "wall_seconds": float(wall_seconds),
        }
    )
    if error.failure_stage == "frame_reference_preflight":
        if "frame_index" in error.failure_context:
            report["failed_frame_index"] = int(error.failure_context["frame_index"])
        if "time" in error.failure_context:
            report["failed_frame_time"] = float(error.failure_context["time"])
    observed_forces = [
        float(record["atomic_force_max_kj_mol_nm"])
        for record in report.get("frame_preflight", [])
        if "atomic_force_max_kj_mol_nm" in record
    ]
    if observed_forces:
        report["frame_reference_atomic_force_max_observed_kj_mol_nm"] = max(
            observed_forces
        )
    return report


def build_reference_cache(args: argparse.Namespace) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "running",
        "candidate": str(args.candidate),
        "preparation_report": str(args.preparation_report),
        "implicit_cache_dir": str(args.implicit_cache_dir),
        "output_cache": str(args.output_cache),
        "platform": args.platform,
        "device_index": str(args.device_index),
        "contract": {
            "minimum_residue_mapping": float(args.minimum_residue_mapping),
            "minimum_atom_mapping": float(args.minimum_atom_mapping),
            "maximum_reference_atomic_force_kj_mol_nm": float(
                args.maximum_reference_atomic_force_kj_mol_nm
            ),
            "maximum_frame_reference_atomic_force_kj_mol_nm": float(
                args.maximum_frame_reference_atomic_force_kj_mol_nm
            ),
        },
    }
    stage = "inputs"
    context = None
    integrator = None
    try:
        preparation = _load_json(args.preparation_report)
        stage = "implicit_system"
        system_path, topology_path, system_contract = ensure_implicit_system(
            cache_dir=args.implicit_cache_dir,
            preparation=preparation,
            preparation_report=args.preparation_report,
            project_root=args.project_root,
            force_rebuild=bool(args.force_rebuild_system),
        )
        report["system_contract"] = system_contract

        from openmm import Context, VerletIntegrator, XmlSerializer, unit
        from openmm import app

        stage = "candidate_load"
        candidate = load_path_candidate(args.candidate)
        candidate_sha256 = _sha256(args.candidate)
        report.update(
            {
                "sample_id": candidate.sample_id,
                "candidate_schema_version": candidate.schema_version,
                "source_candidate": str(args.candidate),
                "source_candidate_sha256": candidate_sha256,
                "cache_schema_version": FRAME_REFERENCE_CACHE_SCHEMA_VERSION,
            }
        )
        pdb = app.PDBFile(str(topology_path))
        topology_positions = np.asarray(
            pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64
        )
        atom_records = topology_atom_records(pdb.topology)
        bonded_neighbors = _bonded_neighbor_sets(
            pdb.topology, topology_positions.shape[0]
        )

        stage = "candidate_topology_mapping"
        mapping = build_candidate_topology_mapping(
            candidate,
            atom_records,
            minimum_residue_fraction=float(args.minimum_residue_mapping),
            minimum_atom_fraction=float(args.minimum_atom_mapping),
        )
        report["mapping"] = {
            "mapped_residue_fraction": mapping.mapped_residue_fraction,
            "mapped_atom_fraction": mapping.mapped_atom_fraction,
            "mapped_atoms": int(mapping.topology_atom_indices.size),
            "ignored_chain_labels": bool(mapping.ignored_chain_labels),
        }

        stage = "candidate_alignment"
        aligned_path, alignment = align_candidate_to_topology(
            candidate, mapping, topology_positions
        )
        report["alignment"] = alignment

        generation_contract = {
            "source": "frozen_path3_all_atom_preconditioning",
            "platform": args.platform,
            "device_index": str(args.device_index),
            "frame_traversal": "holo_to_apo",
            "reference_relaxation_iterations": int(
                args.reference_relaxation_iterations
            ),
            "reference_restraint_k_kj_mol_nm2": float(
                args.reference_restraint_k_kj_mol_nm2
            ),
            "reference_minimization_tolerance_kj_mol_nm": float(
                args.reference_minimization_tolerance_kj_mol_nm
            ),
            "maximum_frame_reference_atomic_force_kj_mol_nm": float(
                args.maximum_frame_reference_atomic_force_kj_mol_nm
            ),
            "implicit_system_sha256": _sha256(system_path),
            "implicit_topology_sha256": _sha256(topology_path),
        }
        report["generation_contract"] = generation_contract
        diagnostic_threshold = args.diagnostic_atom_force_threshold_kj_mol_nm
        if diagnostic_threshold is not None:
            diagnostic_contract = {
                "atomic_force_threshold_kj_mol_nm": float(diagnostic_threshold),
                "top_force_atoms": int(args.diagnostic_top_force_atoms),
                "nearest_atom_excludes_direct_bonds_only": True,
                "force_components": True,
            }
            generation_contract["atom_force_diagnostics"] = diagnostic_contract
            report["atom_force_diagnostic_contract"] = diagnostic_contract

        stage = "reference_context"
        system = XmlSerializer.deserialize(system_path.read_text())
        restraint = _add_target_restraint(
            system,
            mapping.topology_atom_indices,
            aligned_path[
                -1,
                mapping.candidate_residue_indices,
                mapping.candidate_atom14_indices,
            ]
            / 10.0,
            float(args.reference_restraint_k_kj_mol_nm2),
        )
        force_component_labels = (
            _assign_diagnostic_force_groups(system)
            if diagnostic_threshold is not None
            else []
        )
        integrator = VerletIntegrator(0.001 * unit.picoseconds)
        platform, properties = _platform(args)
        context = Context(system, integrator, platform, properties)
        context.setParameter("gate0_k", 0.0)
        context.setPositions(
            unit.Quantity(topology_positions / 10.0, unit.nanometer)
        )
        protein_atoms = int(
            (preparation.get("protein") or {})["prepared_protein_atoms"]
        )
        initial_energy, initial_forces, _ = _state_values(
            context, get_positions=False
        )

        stage = "prepared_topology_preflight"
        try:
            prepared_preflight = _validate_state_for_stage(
                initial_energy,
                initial_forces,
                protein_atoms,
                maximum_atomic_force_kj_mol_nm=float(
                    args.maximum_reference_atomic_force_kj_mol_nm
                ),
                state_label="Prepared OpenMM reference topology",
                failure_stage=stage,
                rejection_type="prepared_topology_physical_failure",
            )
        except FrameReferenceBuildError as error:
            report["prepared_topology_preflight"] = {
                **error.failure_context,
                "accepted": False,
            }
            raise
        report["prepared_topology_preflight"] = prepared_preflight

        stage = "reference_preconditioning"
        references, relaxation = _precondition_reference_positions(
            context,
            restraint,
            topology_positions,
            aligned_path,
            mapping,
            args,
            protein_atom_count=protein_atoms,
        )
        report["relaxation"] = relaxation

        frame_preflight: List[Dict[str, Any]] = []
        report["frame_preflight"] = frame_preflight
        mapped_atom_indices = {
            int(value) for value in mapping.topology_atom_indices.tolist()
        }
        context.setParameter("gate0_k", 0.0)
        for frame_index, reference_positions in enumerate(references):
            stage = "frame_reference_preflight"
            context.setPositions(
                unit.Quantity(reference_positions / 10.0, unit.nanometer)
            )
            energy, forces, _ = _state_values(context, get_positions=False)
            target = aligned_path[
                frame_index,
                mapping.candidate_residue_indices,
                mapping.candidate_atom14_indices,
            ]
            frame_context = {
                "frame_index": int(frame_index),
                "time": float(candidate.times[frame_index]),
                "mapped_heavy_rms_angstrom": rms_distance(
                    reference_positions[mapping.topology_atom_indices], target
                ),
            }
            force_norms = np.linalg.norm(forces, axis=-1)
            if (
                diagnostic_threshold is not None
                and float(force_norms.max()) > float(diagnostic_threshold)
            ):
                top_force_atoms = _top_force_atom_diagnostics(
                    reference_positions,
                    forces,
                    atom_records,
                    bonded_neighbors,
                    protein_atom_count=protein_atoms,
                    mapped_atom_indices=mapped_atom_indices,
                    top_k=int(args.diagnostic_top_force_atoms),
                )
                frame_context["atom_force_diagnostics"] = {
                    "top_force_atoms": top_force_atoms,
                    "force_components": _force_component_atom_diagnostics(
                        context,
                        force_component_labels,
                        atom_records,
                        protein_atom_count=protein_atoms,
                        mapped_atom_indices=mapped_atom_indices,
                        total_max_atom_index=int(np.argmax(force_norms)),
                    ),
                }
            try:
                diagnostics = _validate_state_for_stage(
                    energy,
                    forces,
                    protein_atoms,
                    maximum_atomic_force_kj_mol_nm=float(
                        args.maximum_frame_reference_atomic_force_kj_mol_nm
                    ),
                    state_label=f"All-atom frame reference {frame_index}",
                    failure_stage=stage,
                    rejection_type="frame_reference_physical_failure",
                    context=frame_context,
                )
            except FrameReferenceBuildError as error:
                frame_preflight.append(
                    {**error.failure_context, "accepted": False}
                )
                report["frame_preflight_completed"] = len(frame_preflight) - 1
                raise
            frame_preflight.append(diagnostics)
        report["frame_preflight_completed"] = len(frame_preflight)

        stage = "cache_payload"
        payload = build_frame_reference_cache_payload(
            candidate,
            references,
            source_candidate_sha256=candidate_sha256,
            system_contract=system_contract,
            generation_contract=generation_contract,
            frame_preflight=frame_preflight,
        )
        stage = "cache_write"
        _write_cache(args.output_cache, payload)
        report.update(
            {
                "status": "completed",
                "output_cache_sha256": _sha256(args.output_cache),
                "frame_reference_atomic_force_max_kj_mol_nm": float(
                    max(
                        record["atomic_force_max_kj_mol_nm"]
                        for record in frame_preflight
                    )
                ),
            }
        )
        return report
    except FrameReferenceBuildError as error:
        error.partial_report = dict(report)
        raise
    except Exception as error:
        status, rejection_type = _classify_failure(stage, error)
        raise FrameReferenceBuildError(
            status=status,
            failure_stage=stage,
            rejection_type=rejection_type,
            rejection_reason=f"{type(error).__name__}: {error}",
            partial_report=report,
        ) from error
    finally:
        context = None
        integrator = None


def main() -> None:
    args = parse_args()
    _validate_args(args)
    for path in (args.output_cache, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frame reference output: {path}")
    started = time.perf_counter()
    try:
        report = build_reference_cache(args)
    except FrameReferenceBuildError as error:
        rejection = _rejection_report(
            error, wall_seconds=time.perf_counter() - started
        )
        _write_json(args.report, rejection)
        raise
    report["wall_seconds"] = float(time.perf_counter() - started)
    _write_json(args.report, report)
    printable = dict(report)
    printable.pop("frame_preflight", None)
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
