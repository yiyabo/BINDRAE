#!/usr/bin/env python3
"""Run restrained heating and a short unrestrained NVT stability smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU"], default="CUDA")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--heating-temperatures-k", default="50,100,150,200,250,300")
    parser.add_argument("--heating-steps-per-stage", type=int, default=250)
    parser.add_argument("--restrained-equilibration-steps", type=int, default=1000)
    parser.add_argument("--unrestrained-nvt-steps", type=int, default=1000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--solute-restraint-k-kj-mol-nm2", type=float, default=1000.0)
    parser.add_argument("--min-final-temperature-k", type=float, default=200.0)
    parser.add_argument("--max-final-temperature-k", type=float, default=400.0)
    parser.add_argument("--max-protein-ca-rmsd-a", type=float, default=2.5)
    parser.add_argument("--max-ligand-heavy-rmsd-a", type=float, default=3.0)
    return parser.parse_args()


def parse_temperatures(value: str) -> List[float]:
    temperatures = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not temperatures or any(item <= 0.0 for item in temperatures):
        raise ValueError("Heating temperatures must be positive")
    return temperatures


def degrees_of_freedom(system: Any) -> int:
    from openmm import CMMotionRemover

    dof = 3 * system.getNumParticles() - system.getNumConstraints()
    if any(isinstance(system.getForce(index), CMMotionRemover) for index in range(system.getNumForces())):
        dof -= 3
    if dof <= 0:
        raise ValueError(f"Invalid system degrees of freedom: {dof}")
    return dof


def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Expected matching [N, 3] coordinates, got {mobile.shape} and {target.shape}")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_mobile = mobile - mobile_center
    centered_target = target - target_center
    left, _, right = np.linalg.svd(centered_mobile.T @ centered_target)
    correction = np.sign(np.linalg.det(left @ right))
    rotation = left @ np.diag([1.0, 1.0, correction]) @ right
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def closest_periodic_image(
    points: np.ndarray, reference: np.ndarray, box_vectors: np.ndarray
) -> np.ndarray:
    if box_vectors.shape != (3, 3):
        raise ValueError(f"Expected [3, 3] box vectors, got {box_vectors.shape}")
    centroid_delta = points.mean(axis=0) - reference.mean(axis=0)
    fractional_delta = centroid_delta @ np.linalg.inv(box_vectors)
    lattice_shift = np.round(fractional_delta) @ box_vectors
    return points - lattice_shift


def aligned_endpoint_rmsds(
    reference_angstrom: np.ndarray,
    final_angstrom: np.ndarray,
    topology: Any,
    *,
    protein_atoms: int,
    solute_atoms: int,
    periodic_box_vectors_angstrom: np.ndarray | None = None,
) -> Dict[str, float]:
    atoms = list(topology.atoms())
    ca_indices = [
        atom.index for atom in atoms[:protein_atoms] if atom.name == "CA"
    ]
    if len(ca_indices) < 3:
        raise ValueError("At least three protein CA atoms are required for stability RMSD")
    rotation, translation = kabsch_transform(
        final_angstrom[ca_indices], reference_angstrom[ca_indices]
    )
    aligned_final = final_angstrom @ rotation + translation
    protein_delta = aligned_final[ca_indices] - reference_angstrom[ca_indices]
    ligand_indices = [
        atom.index
        for atom in atoms[protein_atoms:solute_atoms]
        if atom.element is not None and atom.element.symbol != "H"
    ]
    if not ligand_indices:
        raise ValueError("No ligand heavy atoms found in the prepared topology")
    aligned_ligand = aligned_final[ligand_indices]
    if periodic_box_vectors_angstrom is not None:
        rotated_box = np.asarray(periodic_box_vectors_angstrom, dtype=np.float64) @ rotation
        aligned_ligand = closest_periodic_image(
            aligned_ligand,
            reference_angstrom[ligand_indices],
            rotated_box,
        )
    ligand_delta = aligned_ligand - reference_angstrom[ligand_indices]
    return {
        "protein_ca_rmsd_angstrom": float(
            np.sqrt(np.mean(np.sum(protein_delta**2, axis=1)))
        ),
        "ligand_heavy_rmsd_angstrom": float(
            np.sqrt(np.mean(np.sum(ligand_delta**2, axis=1)))
        ),
    }


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    from openmm import (
        CustomExternalForce,
        LangevinMiddleIntegrator,
        Platform,
        XmlSerializer,
        unit,
    )
    from openmm import app

    required = {
        "system": args.input_dir / "system.xml",
        "state": args.input_dir / "minimized_state.xml",
        "topology": args.input_dir / "minimized_solvated.pdb",
        "report": args.input_dir / "preparation_report.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prepared-system files: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    system = XmlSerializer.deserialize(required["system"].read_text())
    minimized_state = XmlSerializer.deserialize(required["state"].read_text())
    pdb = app.PDBFile(str(required["topology"]))
    preparation = json.loads(required["report"].read_text())
    protein_atoms = int(preparation["protein"]["prepared_protein_atoms"])
    solute_atoms = int(preparation["system"]["pre_solvent_atoms"])
    temperatures = parse_temperatures(args.heating_temperatures_k)
    reference_positions = minimized_state.getPositions(asNumpy=True)
    reference_nm = np.asarray(reference_positions.value_in_unit(unit.nanometer))
    topology_atoms = list(pdb.topology.atoms())

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
    restrained_indices = [
        atom.index
        for atom in topology_atoms[:solute_atoms]
        if atom.element is not None and atom.element.symbol != "H"
    ]
    for atom_index in restrained_indices:
        restraint.addParticle(atom_index, reference_nm[atom_index])
    restraint_index = system.addForce(restraint)

    platform = Platform.getPlatformByName(args.platform)
    platform_properties: Dict[str, str] = {}
    if args.platform in {"CUDA", "OpenCL"}:
        platform_properties["Precision"] = "mixed"
    dof = degrees_of_freedom(system)

    def make_simulation(seed_offset: int) -> Tuple[Any, Any]:
        integrator = LangevinMiddleIntegrator(
            temperatures[-1] * unit.kelvin,
            1.0 / unit.picosecond,
            args.timestep_fs * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(args.seed + seed_offset)
        simulation = app.Simulation(
            pdb.topology, system, integrator, platform, platform_properties
        )
        return simulation, integrator

    records: List[Dict[str, Any]] = []
    total_step = 0
    dcd_handle = (args.output_dir / "dynamics_smoke.dcd").open("wb")
    dcd = app.DCDFile(
        dcd_handle,
        pdb.topology,
        args.timestep_fs * unit.femtoseconds,
        firstStep=0,
        interval=args.report_interval,
    )

    def capture(simulation: Any, stage: str, target_temperature: float) -> Any:
        state = simulation.context.getState(
            getEnergy=True,
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=True,
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
        box = np.asarray(
            state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        )
        volume = float(abs(np.linalg.det(box)))
        values = (potential, kinetic, temperature, volume)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite dynamics state at step {total_step}: {values}")
        records.append(
            {
                "step": total_step,
                "time_ps": total_step * args.timestep_fs / 1000.0,
                "stage": stage,
                "target_temperature_k": target_temperature,
                "temperature_k": temperature,
                "potential_kj_mol": potential,
                "kinetic_kj_mol": kinetic,
                "total_energy_kj_mol": potential + kinetic,
                "volume_nm3": volume,
            }
        )
        dcd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())
        return state

    def advance(
        simulation: Any,
        *,
        stage: str,
        target_temperature: float,
        steps: int,
    ) -> Any:
        nonlocal total_step
        remaining = steps
        state = None
        while remaining > 0:
            chunk = min(args.report_interval, remaining)
            simulation.step(chunk)
            total_step += chunk
            remaining -= chunk
            state = capture(simulation, stage, target_temperature)
        return state

    print("[1/3] Restrained heating", flush=True)
    restrained_simulation, restrained_integrator = make_simulation(0)
    restrained_simulation.context.setState(minimized_state)
    restrained_simulation.context.setVelocitiesToTemperature(
        temperatures[0] * unit.kelvin, args.seed
    )
    capture(restrained_simulation, "initial", temperatures[0])
    for temperature in temperatures:
        print(f"Heating target: {temperature:.1f} K", flush=True)
        restrained_integrator.setTemperature(temperature * unit.kelvin)
        advance(
            restrained_simulation,
            stage="heating_restrained",
            target_temperature=temperature,
            steps=args.heating_steps_per_stage,
        )
    print("[2/3] Restrained 300 K equilibration", flush=True)
    restrained_integrator.setTemperature(temperatures[-1] * unit.kelvin)
    restrained_final = advance(
        restrained_simulation,
        stage="nvt_restrained",
        target_temperature=temperatures[-1],
        steps=args.restrained_equilibration_steps,
    )
    restrained_positions = restrained_final.getPositions()
    restrained_velocities = restrained_final.getVelocities()
    del restrained_simulation, restrained_integrator

    system.removeForce(restraint_index)
    dof = degrees_of_freedom(system)
    print("[3/3] Unrestrained 300 K NVT smoke", flush=True)
    simulation, integrator = make_simulation(1)
    simulation.context.setPositions(restrained_positions)
    simulation.context.setVelocities(restrained_velocities)
    integrator.setTemperature(temperatures[-1] * unit.kelvin)
    final_state = advance(
        simulation,
        stage="nvt_unrestrained",
        target_temperature=temperatures[-1],
        steps=args.unrestrained_nvt_steps,
    )
    dcd_handle.close()

    final_angstrom = np.asarray(
        final_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    )
    reference_angstrom = np.asarray(reference_positions.value_in_unit(unit.angstrom))
    rmsds = aligned_endpoint_rmsds(
        reference_angstrom,
        final_angstrom,
        pdb.topology,
        protein_atoms=protein_atoms,
        solute_atoms=solute_atoms,
        periodic_box_vectors_angstrom=np.asarray(
            final_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        ),
    )
    final_temperature = float(records[-1]["temperature_k"])
    passed = (
        args.min_final_temperature_k <= final_temperature <= args.max_final_temperature_k
        and rmsds["protein_ca_rmsd_angstrom"] <= args.max_protein_ca_rmsd_a
        and rmsds["ligand_heavy_rmsd_angstrom"] <= args.max_ligand_heavy_rmsd_a
    )

    with (args.output_dir / "final_nvt.pdb").open("w") as handle:
        app.PDBFile.writeFile(pdb.topology, final_state.getPositions(), handle, keepIds=True)
    (args.output_dir / "final_nvt_state.xml").write_text(XmlSerializer.serialize(final_state))
    with (args.output_dir / "dynamics_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    return {
        "status": "nvt_smoke_passed" if passed else "nvt_smoke_failed",
        "input_dir": str(args.input_dir),
        "platform": args.platform,
        "seed": args.seed,
        "steps": {
            "heating": len(temperatures) * args.heating_steps_per_stage,
            "restrained_equilibration": args.restrained_equilibration_steps,
            "unrestrained_nvt": args.unrestrained_nvt_steps,
            "total": total_step,
        },
        "final": {
            "temperature_k": final_temperature,
            **rmsds,
        },
        "gates": {
            "temperature_k": [args.min_final_temperature_k, args.max_final_temperature_k],
            "max_protein_ca_rmsd_angstrom": args.max_protein_ca_rmsd_a,
            "max_ligand_heavy_rmsd_angstrom": args.max_ligand_heavy_rmsd_a,
            "passed": passed,
        },
        "outputs": {
            "trajectory_dcd": str(args.output_dir / "dynamics_smoke.dcd"),
            "metrics_csv": str(args.output_dir / "dynamics_metrics.csv"),
            "final_pdb": str(args.output_dir / "final_nvt.pdb"),
            "final_state_xml": str(args.output_dir / "final_nvt_state.xml"),
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    result = run_smoke(args)
    (args.output_dir / "dynamics_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "nvt_smoke_passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
