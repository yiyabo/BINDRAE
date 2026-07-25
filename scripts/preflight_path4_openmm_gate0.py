#!/usr/bin/env python3
"""Preflight one prepared system for Path-4 without building a frame cache."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_path4_openmm_gate0 import (  # noqa: E402
    _load_json,
    _platform,
    _state_values,
    ensure_implicit_system,
)
from src.data.openmm_gate0 import (  # noqa: E402
    align_candidate_to_topology,
    build_candidate_topology_mapping,
    load_path_candidate,
    topology_atom_records,
    validate_reference_topology_state,
)


SCHEMA_VERSION = "bindrae_path4_openmm_preflight_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--implicit-cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--platform", choices=["CPU", "CUDA", "OpenCL"], default="CPU")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--minimum-residue-mapping", type=float, default=0.98)
    parser.add_argument("--minimum-atom-mapping", type=float, default=0.95)
    parser.add_argument(
        "--maximum-reference-atomic-force-kj-mol-nm",
        type=float,
        default=1.0e6,
    )
    parser.add_argument("--force-rebuild-system", action="store_true")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def classify_failure(stage: str, error: Exception) -> tuple[str, str]:
    message = str(error).lower()
    if stage == "candidate_topology_mapping":
        if "residue identity mismatch" in message:
            return "rejected", "residue_identity_mismatch"
        return "rejected", "candidate_topology_mapping_failure"
    if stage == "reference_topology_state":
        return "rejected", "reference_topology_physical_failure"
    if stage in {"candidate_load", "candidate_alignment"}:
        return "rejected", "candidate_coordinate_contract_failure"
    return "failed", f"{stage}_failure"


def run_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if not 0.0 < args.minimum_residue_mapping <= 1.0:
        raise ValueError("--minimum-residue-mapping must be in (0, 1]")
    if not 0.0 < args.minimum_atom_mapping <= 1.0:
        raise ValueError("--minimum-atom-mapping must be in (0, 1]")
    if args.maximum_reference_atomic_force_kj_mol_nm <= 0.0:
        raise ValueError("--maximum-reference-atomic-force-kj-mol-nm must be positive")

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": _now(),
        "candidate": str(args.candidate),
        "preparation_report": str(args.preparation_report),
        "implicit_cache_dir": str(args.implicit_cache_dir),
        "platform": args.platform,
        "device_index": str(args.device_index),
        "contract": {
            "minimum_residue_mapping": float(args.minimum_residue_mapping),
            "minimum_atom_mapping": float(args.minimum_atom_mapping),
            "maximum_reference_atomic_force_kj_mol_nm": float(
                args.maximum_reference_atomic_force_kj_mol_nm
            ),
            "frame_reference_preconditioning": False,
        },
    }
    stage = "inputs"
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
        report.update(
            {
                "sample_id": candidate.sample_id,
                "candidate_schema_version": candidate.schema_version,
                "residue_identity_hash": candidate.residue_identity_hash,
            }
        )
        pdb = app.PDBFile(str(topology_path))
        topology_positions = np.asarray(
            pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64
        )
        atom_records = topology_atom_records(pdb.topology)

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
            "ignored_chain_labels": mapping.ignored_chain_labels,
        }

        stage = "candidate_alignment"
        _, alignment = align_candidate_to_topology(
            candidate, mapping, topology_positions
        )
        report["alignment"] = alignment

        stage = "reference_context"
        system = XmlSerializer.deserialize(system_path.read_text())
        integrator = VerletIntegrator(0.001 * unit.picoseconds)
        platform, properties = _platform(args)
        context = Context(system, integrator, platform, properties)
        context.setPositions(unit.Quantity(topology_positions / 10.0, unit.nanometer))
        reference_energy, reference_forces, _ = _state_values(
            context, get_positions=False
        )
        protein_atoms = int((preparation.get("protein") or {})["prepared_protein_atoms"])
        stage = "reference_topology_state"
        report["reference_topology"] = validate_reference_topology_state(
            reference_energy,
            reference_forces,
            protein_atoms,
            maximum_atomic_force_kj_mol_nm=float(
                args.maximum_reference_atomic_force_kj_mol_nm
            ),
        )
        report["status"] = "completed"
        report["completed_at"] = _now()
        return report
    except Exception as error:
        status, rejection_type = classify_failure(stage, error)
        report.update(
            {
                "status": status,
                "failed_at": _now(),
                "failure_stage": stage,
                "rejection_type": rejection_type,
                "rejection_reason": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        _write_json_atomic(args.report, report)


def main() -> None:
    args = parse_args()
    try:
        report = run_preflight(args)
    except Exception:
        if args.report.is_file():
            print(args.report.read_text(), flush=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
