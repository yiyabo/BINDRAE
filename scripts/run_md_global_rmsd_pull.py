#!/usr/bin/env python3
"""Generate a biased holo-to-apo atomistic path using a global CA-RMSD CV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.register_md_context_replica import load_one_candidate  # noqa: E402
from scripts.run_md_pilot_dynamics_smoke import (  # noqa: E402
    aligned_endpoint_rmsds,
    degrees_of_freedom,
    kabsch_transform,
)
from src.data.md_pilot_selection import parse_ca_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--transition-id", required=True)
    parser.add_argument("--npt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU"], default="CUDA")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--resample-initial-velocities",
        action="store_true",
        help=(
            "Replace velocities restored from the NPT state with a seeded Maxwell-Boltzmann "
            "sample. Enable this for independent trajectory replicas."
        ),
    )
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--pre-equilibration-steps", type=int, default=500)
    parser.add_argument("--pulling-steps", type=int, default=5000)
    parser.add_argument("--endpoint-hold-steps", type=int, default=1000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--rmsd-k-kj-mol-nm2", type=float, default=5000.0)
    parser.add_argument("--final-target-rmsd-nm", type=float, default=0.05)
    parser.add_argument("--max-final-apo-ca-rmsd-a", type=float, default=1.0)
    parser.add_argument("--max-final-ligand-rmsd-a", type=float, default=5.0)
    parser.add_argument("--min-target-progress-fraction", type=float, default=0.5)
    return parser.parse_args()


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    rotation, translation = kabsch_transform(mobile, target)
    aligned = mobile @ rotation + translation
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))


def run_pull(args: argparse.Namespace) -> Dict[str, Any]:
    from openmm import (
        CustomCVForce,
        LangevinMiddleIntegrator,
        Platform,
        RMSDForce,
        XmlSerializer,
        unit,
    )
    from openmm import app

    required = {
        "system": args.npt_dir / "npt_system.xml",
        "state": args.npt_dir / "final_npt_state.xml",
        "topology": args.npt_dir / "final_npt.pdb",
        "npt_report": args.npt_dir / "npt_report.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing NPT inputs: {missing}")
    npt_report = json.loads(required["npt_report"].read_text())
    if npt_report.get("status") != "npt_smoke_passed":
        raise ValueError(f"NPT input did not pass: {npt_report.get('status')!r}")

    candidate = load_one_candidate(args.candidate_manifest, args.transition_id)
    endpoints = dict(candidate.get("endpoints") or {})
    apo_path = Path(str(endpoints.get("apo_structure_path") or ""))
    holo_path = Path(str(endpoints.get("holo_structure_path") or ""))
    if not apo_path.is_file() or not holo_path.is_file():
        raise FileNotFoundError(f"Missing endpoint PDBs: apo={apo_path}, holo={holo_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    system = XmlSerializer.deserialize(required["system"].read_text())
    initial_state = XmlSerializer.deserialize(required["state"].read_text())
    pdb = app.PDBFile(str(required["topology"]))
    preparation = json.loads((Path(str(npt_report["system_dir"])) / "preparation_report.json").read_text())
    protein_atoms = int(preparation["protein"]["prepared_protein_atoms"])
    solute_atoms = int(preparation["system"]["pre_solvent_atoms"])
    topology_atoms = list(pdb.topology.atoms())
    ca_indices = [
        atom.index for atom in topology_atoms[:protein_atoms] if atom.name == "CA"
    ]
    apo = parse_ca_records(apo_path)
    holo = parse_ca_records(holo_path)
    apo_xyz = np.asarray(apo["xyz"], dtype=np.float64)
    holo_xyz = np.asarray(holo["xyz"], dtype=np.float64)
    if len(ca_indices) != len(apo_xyz) or apo_xyz.shape != holo_xyz.shape:
        raise ValueError(
            f"CA mapping mismatch: topology={len(ca_indices)}, apo={len(apo_xyz)}, holo={len(holo_xyz)}"
        )

    initial_positions = initial_state.getPositions(asNumpy=True)
    initial_angstrom = np.asarray(initial_positions.value_in_unit(unit.angstrom))
    current_ca = initial_angstrom[ca_indices]
    endpoint_rotation, endpoint_translation = kabsch_transform(holo_xyz, current_ca)
    apo_target_angstrom = apo_xyz @ endpoint_rotation + endpoint_translation
    reference_nm = np.asarray(initial_positions.value_in_unit(unit.nanometer)).copy()
    reference_nm[ca_indices] = apo_target_angstrom / 10.0

    rmsd_force = RMSDForce(unit.Quantity(reference_nm, unit.nanometer), ca_indices)
    pull_force = CustomCVForce(
        "0.5*rmsd_k*(apo_rmsd-target_rmsd_nm)^2"
    )
    pull_force.addGlobalParameter(
        "rmsd_k", args.rmsd_k_kj_mol_nm2 * unit.kilojoule_per_mole / unit.nanometer**2
    )
    pull_force.addGlobalParameter("target_rmsd_nm", 0.0)
    pull_force.addCollectiveVariable("apo_rmsd", rmsd_force)
    system.addForce(pull_force)

    platform = Platform.getPlatformByName(args.platform)
    properties: Dict[str, str] = {}
    if args.platform in {"CUDA", "OpenCL"}:
        properties["Precision"] = "mixed"
    integrator = LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        args.timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setState(initial_state)
    if args.resample_initial_velocities:
        simulation.context.setVelocitiesToTemperature(
            args.temperature_k * unit.kelvin, args.seed
        )
    dof = degrees_of_freedom(system)
    start_rmsd_nm = float(pull_force.getCollectiveVariableValues(simulation.context)[0])
    final_target_nm = min(args.final_target_rmsd_nm, start_rmsd_nm)
    simulation.context.setParameter("target_rmsd_nm", start_rmsd_nm)

    records: List[Dict[str, Any]] = []
    total_step = 0
    metric_fields = [
        "step",
        "time_ps",
        "stage",
        "progress",
        "target_rmsd_nm",
        "cv_apo_rmsd_nm",
        "apo_ca_rmsd_angstrom",
        "holo_ca_rmsd_angstrom",
        "ligand_heavy_rmsd_angstrom",
        "temperature_k",
        "potential_kj_mol",
        "kinetic_kj_mol",
    ]
    metrics_handle = (args.output_dir / "rmsd_pull_metrics.csv").open("w", newline="")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=metric_fields)
    metrics_writer.writeheader()
    dcd_handle = (args.output_dir / "rmsd_pull.dcd").open("wb")
    dcd = app.DCDFile(
        dcd_handle,
        pdb.topology,
        args.timestep_fs * unit.femtoseconds,
        firstStep=0,
        interval=args.report_interval,
    )

    def capture(stage: str, progress: float, target_rmsd_nm: float) -> Any:
        state = simulation.context.getState(
            getEnergy=True,
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=True,
        )
        positions_angstrom = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        )
        current = positions_angstrom[ca_indices]
        stability = aligned_endpoint_rmsds(
            initial_angstrom,
            positions_angstrom,
            pdb.topology,
            protein_atoms=protein_atoms,
            solute_atoms=solute_atoms,
            periodic_box_vectors_angstrom=np.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
            ),
        )
        potential = float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        kinetic = float(state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole))
        temperature = float(
            (2.0 * state.getKineticEnergy() / (dof * unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(
                unit.kelvin
            )
        )
        cv_value = float(pull_force.getCollectiveVariableValues(simulation.context)[0])
        values = (potential, kinetic, temperature, cv_value)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite RMSD-pull state at step {total_step}: {values}")
        row = {
            "step": total_step,
            "time_ps": total_step * args.timestep_fs / 1000.0,
            "stage": stage,
            "progress": progress,
            "target_rmsd_nm": target_rmsd_nm,
            "cv_apo_rmsd_nm": cv_value,
            "apo_ca_rmsd_angstrom": kabsch_rmsd(current, apo_target_angstrom),
            "holo_ca_rmsd_angstrom": stability["protein_ca_rmsd_angstrom"],
            "ligand_heavy_rmsd_angstrom": stability["ligand_heavy_rmsd_angstrom"],
            "temperature_k": temperature,
            "potential_kj_mol": potential,
            "kinetic_kj_mol": kinetic,
        }
        records.append(row)
        metrics_writer.writerow(row)
        metrics_handle.flush()
        dcd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())
        return state

    def step_chunks(stage: str, steps: int, start_progress: float, end_progress: float) -> Any:
        nonlocal total_step
        remaining = steps
        state = None
        completed = 0
        while remaining > 0:
            chunk = min(args.report_interval, remaining)
            progress = start_progress + (end_progress - start_progress) * (
                (completed + chunk) / max(steps, 1)
            )
            target = start_rmsd_nm + progress * (final_target_nm - start_rmsd_nm)
            simulation.context.setParameter("target_rmsd_nm", target)
            simulation.step(chunk)
            total_step += chunk
            completed += chunk
            remaining -= chunk
            state = capture(stage, progress, target)
        return state

    print(
        f"Initial apo CA RMSD CV: {start_rmsd_nm * 10.0:.3f} A; "
        f"target: {final_target_nm * 10.0:.3f} A",
        flush=True,
    )
    capture("initial", 0.0, start_rmsd_nm)
    print("[1/3] CV-equilibration at the holo endpoint", flush=True)
    step_chunks("pre_equilibration", args.pre_equilibration_steps, 0.0, 0.0)
    print("[2/3] Global CA-RMSD pulling toward apo", flush=True)
    step_chunks("rmsd_pull", args.pulling_steps, 0.0, 1.0)
    print("[3/3] Biased apo-endpoint hold", flush=True)
    final_state = step_chunks("endpoint_hold", args.endpoint_hold_steps, 1.0, 1.0)
    dcd_handle.close()
    metrics_handle.close()

    final_row = records[-1]
    initial_apo_rmsd = float(records[0]["apo_ca_rmsd_angstrom"])
    final_apo_rmsd = float(final_row["apo_ca_rmsd_angstrom"])
    final_target_angstrom = final_target_nm * 10.0
    target_progress_fraction = (initial_apo_rmsd - final_apo_rmsd) / max(
        initial_apo_rmsd - final_target_angstrom, 1e-8
    )
    passed = (
        final_apo_rmsd <= args.max_final_apo_ca_rmsd_a
        and final_apo_rmsd < float(final_row["holo_ca_rmsd_angstrom"])
        and float(final_row["ligand_heavy_rmsd_angstrom"])
        <= args.max_final_ligand_rmsd_a
        and 200.0 <= float(final_row["temperature_k"]) <= 400.0
        and target_progress_fraction >= args.min_target_progress_fraction
    )
    with (args.output_dir / "final_pulled.pdb").open("w") as handle:
        app.PDBFile.writeFile(pdb.topology, final_state.getPositions(), handle, keepIds=True)
    (args.output_dir / "final_pulled_state.xml").write_text(
        XmlSerializer.serialize(final_state)
    )
    (args.output_dir / "pull_system.xml").write_text(XmlSerializer.serialize(system))

    return {
        "status": "rmsd_pull_smoke_passed" if passed else "rmsd_pull_smoke_failed",
        "evidence_tier": "silver_enhanced_sampling",
        "biased_sampling": True,
        "physical_time_interpretable": False,
        "transition_id": candidate["transition_id"],
        "direction_generated": "holo_to_apo",
        "reverse_for_model_direction": True,
        "platform": args.platform,
        "seed": args.seed,
        "initial_velocities": (
            "seeded_maxwell_boltzmann"
            if args.resample_initial_velocities
            else "restored_from_npt_state"
        ),
        "protocol": {
            "temperature_k": args.temperature_k,
            "timestep_fs": args.timestep_fs,
            "report_interval": args.report_interval,
            "rmsd_k_kj_mol_nm2": args.rmsd_k_kj_mol_nm2,
            "final_target_rmsd_nm": args.final_target_rmsd_nm,
        },
        "steps": {
            "pre_equilibration": args.pre_equilibration_steps,
            "pulling": args.pulling_steps,
            "endpoint_hold": args.endpoint_hold_steps,
            "total": total_step,
        },
        "initial_apo_ca_rmsd_angstrom": initial_apo_rmsd,
        "final": {
            "apo_ca_rmsd_angstrom": final_apo_rmsd,
            "holo_ca_rmsd_angstrom": float(final_row["holo_ca_rmsd_angstrom"]),
            "ligand_heavy_rmsd_angstrom": float(final_row["ligand_heavy_rmsd_angstrom"]),
            "temperature_k": float(final_row["temperature_k"]),
            "target_progress_fraction": target_progress_fraction,
        },
        "gates": {
            "max_final_apo_ca_rmsd_angstrom": args.max_final_apo_ca_rmsd_a,
            "max_final_ligand_rmsd_angstrom": args.max_final_ligand_rmsd_a,
            "min_target_progress_fraction": args.min_target_progress_fraction,
            "requires_apo_closer_than_holo": True,
            "passed": passed,
        },
        "usage": {
            "phase_supervision": False,
            "heldout_benchmark": False,
            "kinetics_claims": False,
            "note": "Requires path/event audit before any low-weight geometry supervision.",
        },
        "outputs": {
            "trajectory_dcd": str(args.output_dir / "rmsd_pull.dcd"),
            "metrics_csv": str(args.output_dir / "rmsd_pull_metrics.csv"),
            "final_pdb": str(args.output_dir / "final_pulled.pdb"),
            "final_state_xml": str(args.output_dir / "final_pulled_state.xml"),
            "system_xml": str(args.output_dir / "pull_system.xml"),
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    result = run_pull(args)
    (args.output_dir / "rmsd_pull_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "rmsd_pull_smoke_passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
