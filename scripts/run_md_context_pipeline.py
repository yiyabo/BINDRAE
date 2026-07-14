#!/usr/bin/env python3
"""Run one resumable setup, NVT, NPT, and context-registration matrix row."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--platform", choices=["CPU", "CUDA", "OpenCL"], default="CPU")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_record(path: Path, index: int) -> Dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not 0 <= index < len(records):
        raise IndexError(f"Matrix index {index} is outside [0, {len(records)})")
    record = records[index]
    if int(record.get("matrix_index", -1)) != index:
        raise ValueError(f"Matrix row/index mismatch at {index}")
    return record


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return record


def passed(path: Path, expected_status: str) -> bool:
    record = load_json(path)
    return record is not None and record.get("status") == expected_status


def write_status(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def run_stage(
    *,
    name: str,
    command: Sequence[str],
    output: Path,
    expected_status: str,
    force: bool,
    pipeline_status: Path,
    state: Dict[str, Any],
) -> None:
    if not force and passed(output, expected_status):
        state["stages"][name] = {"status": "skipped_passed", "output": str(output)}
        write_status(pipeline_status, state)
        return
    if output.exists() and not force:
        state["status"] = "failed_existing_output"
        state["failed_stage"] = name
        state["stages"][name] = {"status": "existing_output_not_passed", "output": str(output)}
        write_status(pipeline_status, state)
        raise RuntimeError(f"Existing {name} output did not pass: {output}; use --force to rerun")

    state["status"] = "running"
    state["current_stage"] = name
    state["stages"][name] = {"status": "running", "started_at": utc_now()}
    write_status(pipeline_status, state)
    print(f"[{name}] {' '.join(command)}", flush=True)
    result = subprocess.run(list(command), cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0 or not passed(output, expected_status):
        state["status"] = "failed"
        state["failed_stage"] = name
        state["stages"][name] = {
            "status": "failed",
            "returncode": result.returncode,
            "finished_at": utc_now(),
            "output": str(output),
        }
        write_status(pipeline_status, state)
        raise SystemExit(result.returncode or 2)
    state["stages"][name] = {
        "status": "passed",
        "returncode": result.returncode,
        "finished_at": utc_now(),
        "output": str(output),
    }
    write_status(pipeline_status, state)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    record = load_record(args.matrix, args.index)
    setup_dir = Path(record["setup_dir"])
    nvt_dir = Path(record["nvt_dir"])
    npt_dir = Path(record["npt_dir"])
    context_record = Path(record["context_record"])
    for directory in (setup_dir, nvt_dir, npt_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pipeline_status = setup_dir.parent / "pipeline_status.json"
    state: Dict[str, Any] = {
        "schema_version": "bindrae_md_context_pipeline_v1",
        "status": "starting",
        "matrix": str(args.matrix),
        "matrix_index": args.index,
        "system_sample_id": record["system_sample_id"],
        "transition_id": record["transition_id"],
        "seed": record["seed"],
        "protocol": record["protocol"],
        "started_at": utc_now(),
        "stages": {},
    }
    protocol = dict(record["protocol"])
    setup = dict(protocol["setup"])
    nvt = dict(protocol["nvt"])
    npt = dict(protocol["npt"])
    python = sys.executable
    seed = str(record["seed"])
    stages: List[tuple[str, List[str], Path, str]] = [
        (
            "setup",
            [
                python,
                "scripts/prepare_md_pilot_system.py",
                "--candidate-manifest", str(record["candidate_manifest"]),
                "--transition-id", str(record["transition_id"]),
                "--output-dir", str(setup_dir),
                "--platform", args.platform,
                "--seed", seed,
                "--solvent-minimization-iterations", str(setup["solvent_minimization_iterations"]),
                "--max-minimization-iterations", str(setup["max_minimization_iterations"]),
            ],
            setup_dir / "preparation_report.json",
            "minimized_ready_for_dynamics",
        ),
        (
            "nvt",
            [
                python,
                "scripts/run_md_pilot_dynamics_smoke.py",
                "--input-dir", str(setup_dir),
                "--output-dir", str(nvt_dir),
                "--platform", args.platform,
                "--seed", seed,
                "--heating-steps-per-stage", str(nvt["heating_steps_per_stage"]),
                "--restrained-equilibration-steps", str(nvt["restrained_equilibration_steps"]),
                "--unrestrained-nvt-steps", str(nvt["unrestrained_nvt_steps"]),
            ],
            nvt_dir / "dynamics_report.json",
            "nvt_smoke_passed",
        ),
        (
            "npt",
            [
                python,
                "scripts/run_md_pilot_npt_smoke.py",
                "--system-dir", str(setup_dir),
                "--nvt-dir", str(nvt_dir),
                "--output-dir", str(npt_dir),
                "--platform", args.platform,
                "--seed", seed,
                "--restrained-equilibration-steps", str(npt["restrained_equilibration_steps"]),
                "--unrestrained-production-steps", str(npt["unrestrained_production_steps"]),
            ],
            npt_dir / "npt_report.json",
            "npt_smoke_passed",
        ),
        (
            "register_context",
            [
                python,
                "scripts/register_md_context_replica.py",
                "--candidate-manifest", str(record["candidate_manifest"]),
                "--transition-id", str(record["transition_id"]),
                "--npt-dir", str(npt_dir),
                "--output", str(context_record),
                "--replica-index", "0",
            ],
            context_record,
            "prepared",
        ),
    ]
    for name, command, output, expected_status in stages:
        run_stage(
            name=name,
            command=command,
            output=output,
            expected_status=expected_status,
            force=args.force,
            pipeline_status=pipeline_status,
            state=state,
        )
    state["status"] = "completed"
    state["current_stage"] = None
    state["finished_at"] = utc_now()
    write_status(pipeline_status, state)
    return state


def main() -> None:
    args = parse_args()
    result = run_pipeline(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
