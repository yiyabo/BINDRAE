#!/usr/bin/env python3
"""Run a short restrained-to-unrestrained NPT endpoint stability smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_md_pilot_dynamics_smoke import (  # noqa: E402
    aligned_endpoint_rmsds,
    degrees_of_freedom,
)

DALTON_PER_NM3_TO_G_ML = 1.66053906660e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-dir", type=Path, required=True)
    parser.add_argument("--nvt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU"], default="CUDA")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--barostat-interval", type=int, default=25)
    parser.add_argument("--restrained-equilibration-steps", type=int, default=2500)
    parser.add_argument("--unrestrained-production-steps", type=int, default=5000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--solute-restraint-k-kj-mol-nm2", type=float, default=200.0)
    parser.add_argument("--min-density-g-ml", type=float, default=0.80)
    parser.add_argument("--max-density-g-ml", type=float, default=1.20)
    parser.add_argument("--max-production-volume-cv", type=float, default=0.10)
    parser.add_argument("--max-protein-ca-rmsd-a", type=float, default=3.0)
    parser.add_argument("--max-ligand-heavy-rmsd-a", type=float, default=4.0)
    return parser.parse_args()


def coefficient_of_variation(values: List[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or float(array.mean()) == 0.0:
        return 0.0
    return float(array.std(ddof=1) / abs(array.mean()))


def density_g_ml(total_mass_dalton: float, volume_nm3: float) -> float:
    if total_mass_dalton <= 0.0 or volume_nm3 <= 0.0:
        raise ValueError("Mass and volume must be positive")
    return float(total_mass_dalton * DALTON_PER_NM3_TO_G_ML / volume_nm3)


def run_npt_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    from openmm import (
        CustomExternalForce,
        LangevinMiddleIntegrator,
        MonteCarloBarostat,
        Platform,
        XmlSerializer,
        unit,
    )
    from openmm import app

    required = {
        "system": args.system_dir / "system.xml",
        "preparation": args.system_dir / "preparation_report.json",
        "state": args.nvt_dir / "final_nvt_state.xml",
        "topology": args.nvt_dir / "final_nvt.pdb",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing NPT input files: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    system = XmlSerializer.deserialize(required["system"].read_text())
    nvt_state = XmlSerializer.deserialize(required["state"].read_text())
    pdb = app.PDBFile(str(required["topology"]))
    preparation = json.loads(required["preparation"].read_text())
    protein_atoms = int(preparation["protein"]["prepared_protein_atoms"])
    solute_atoms = int(preparation["system"]["pre_solvent_atoms"])
    reference_positions = nvt_state.getPositions(asNumpy=True)
    reference_nm = np.asarray(reference_positions.value_in_unit(unit.nanometer))
    topology_atoms = list(pdb.topology.atoms())

    barostat = MonteCarloBarostat(
        args.pressure_bar * unit.bar,
        args.temperature_k * unit.kelvin,
        args.barostat_interval,
    )
    barostat.setRandomNumberSeed(args.seed)
    system.addForce(barostat)
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
    for atom in topology_atoms[:solute_atoms]:
        if atom.element is not None and atom.element.symbol != "H":
            restraint.addParticle(atom.index, reference_nm[atom.index])
    restraint_index = system.addForce(restraint)

    platform = Platform.getPlatformByName(args.platform)
    properties: Dict[str, str] = {}
    if args.platform in {"CUDA", "OpenCL"}:
        properties["Precision"] = "mixed"
    dof = degrees_of_freedom(system)
    total_mass_dalton = sum(
        float(system.getParticleMass(index).value_in_unit(unit.dalton))
        for index in range(system.getNumParticles())
    )

    def make_simulation(seed_offset: int) -> Tuple[Any, Any]:
        integrator = LangevinMiddleIntegrator(
            args.temperature_k * unit.kelvin,
            1.0 / unit.picosecond,
            args.timestep_fs * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(args.seed + seed_offset)
        return (
            app.Simulation(pdb.topology, system, integrator, platform, properties),
            integrator,
        )

    records: List[Dict[str, Any]] = []
    total_step = 0
    metric_fields = [
        "step",
        "time_ps",
        "stage",
        "temperature_k",
        "potential_kj_mol",
        "kinetic_kj_mol",
        "total_energy_kj_mol",
        "volume_nm3",
        "density_g_ml",
    ]
    metrics_handle = (args.output_dir / "npt_metrics.csv").open("w", newline="")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=metric_fields)
    metrics_writer.writeheader()
    metrics_handle.flush()
    dcd_handle = (args.output_dir / "npt_smoke.dcd").open("wb")
    dcd = app.DCDFile(
        dcd_handle,
        pdb.topology,
        args.timestep_fs * unit.femtoseconds,
        firstStep=0,
        interval=args.report_interval,
    )

    def capture(simulation: Any, stage: str) -> Any:
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
        volume_nm3 = float(abs(np.linalg.det(box)))
        density = density_g_ml(total_mass_dalton, volume_nm3)
        values = (potential, kinetic, temperature, volume_nm3, density)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite NPT state at step {total_step}: {values}")
        row = {
            "step": total_step,
            "time_ps": total_step * args.timestep_fs / 1000.0,
            "stage": stage,
            "temperature_k": temperature,
            "potential_kj_mol": potential,
            "kinetic_kj_mol": kinetic,
            "total_energy_kj_mol": potential + kinetic,
            "volume_nm3": volume_nm3,
            "density_g_ml": density,
        }
        records.append(row)
        metrics_writer.writerow(row)
        metrics_handle.flush()
        dcd.writeModel(state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors())
        return state

    def advance(simulation: Any, *, stage: str, steps: int) -> Any:
        nonlocal total_step
        remaining = steps
        state = None
        while remaining > 0:
            chunk = min(args.report_interval, remaining)
            simulation.step(chunk)
            total_step += chunk
            remaining -= chunk
            state = capture(simulation, stage)
        return state

    print("[1/2] Restrained NPT density equilibration", flush=True)
    restrained_simulation, restrained_integrator = make_simulation(0)
    restrained_simulation.context.setState(nvt_state)
    capture(restrained_simulation, "initial")
    restrained_final = advance(
        restrained_simulation,
        stage="npt_restrained",
        steps=args.restrained_equilibration_steps,
    )
    positions = restrained_final.getPositions()
    velocities = restrained_final.getVelocities()
    box_vectors = restrained_final.getPeriodicBoxVectors()
    del restrained_simulation, restrained_integrator

    system.removeForce(restraint_index)
    dof = degrees_of_freedom(system)
    print("[2/2] Unrestrained NPT stability smoke", flush=True)
    simulation, integrator = make_simulation(1)
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)
    final_state = advance(
        simulation,
        stage="npt_unrestrained",
        steps=args.unrestrained_production_steps,
    )
    dcd_handle.close()
    metrics_handle.close()

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
    production = [row for row in records if row["stage"] == "npt_unrestrained"]
    mean_density = float(np.mean([row["density_g_ml"] for row in production]))
    volume_cv = coefficient_of_variation([row["volume_nm3"] for row in production])
    final_temperature = float(records[-1]["temperature_k"])
    passed = (
        args.min_density_g_ml <= mean_density <= args.max_density_g_ml
        and volume_cv <= args.max_production_volume_cv
        and 200.0 <= final_temperature <= 400.0
        and rmsds["protein_ca_rmsd_angstrom"] <= args.max_protein_ca_rmsd_a
        and rmsds["ligand_heavy_rmsd_angstrom"] <= args.max_ligand_heavy_rmsd_a
    )

    with (args.output_dir / "final_npt.pdb").open("w") as handle:
        app.PDBFile.writeFile(pdb.topology, final_state.getPositions(), handle, keepIds=True)
    (args.output_dir / "final_npt_state.xml").write_text(XmlSerializer.serialize(final_state))
    (args.output_dir / "npt_system.xml").write_text(XmlSerializer.serialize(system))
    return {
        "status": "npt_smoke_passed" if passed else "npt_smoke_failed",
        "system_dir": str(args.system_dir),
        "nvt_dir": str(args.nvt_dir),
        "platform": args.platform,
        "seed": args.seed,
        "steps": {
            "restrained_equilibration": args.restrained_equilibration_steps,
            "unrestrained_production": args.unrestrained_production_steps,
            "total": total_step,
        },
        "final": {
            "temperature_k": final_temperature,
            "mean_production_density_g_ml": mean_density,
            "production_volume_cv": volume_cv,
            **rmsds,
        },
        "gates": {
            "density_g_ml": [args.min_density_g_ml, args.max_density_g_ml],
            "max_production_volume_cv": args.max_production_volume_cv,
            "max_protein_ca_rmsd_angstrom": args.max_protein_ca_rmsd_a,
            "max_ligand_heavy_rmsd_angstrom": args.max_ligand_heavy_rmsd_a,
            "passed": passed,
        },
        "outputs": {
            "trajectory_dcd": str(args.output_dir / "npt_smoke.dcd"),
            "metrics_csv": str(args.output_dir / "npt_metrics.csv"),
            "final_pdb": str(args.output_dir / "final_npt.pdb"),
            "final_state_xml": str(args.output_dir / "final_npt_state.xml"),
            "system_xml": str(args.output_dir / "npt_system.xml"),
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    result = run_npt_smoke(args)
    (args.output_dir / "npt_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "npt_smoke_passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
