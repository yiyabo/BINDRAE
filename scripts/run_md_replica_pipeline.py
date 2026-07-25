#!/usr/bin/env python3
"""Run one resumable pull, audit, and target-export task from a replica matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--platform", choices=["CPU", "CUDA", "OpenCL"], default="CPU")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="OpenMM CPU Platform threads per replica; 0 keeps the default",
    )
    parser.add_argument(
        "--residual-envelope", choices=["sin2", "poly"], default="sin2"
    )
    parser.add_argument(
        "--normal-projection-mode",
        choices=["product", "block"],
        default="product",
    )
    parser.add_argument(
        "--canonical-data-dir",
        type=Path,
        default=Path("processed_data/triplets"),
        help=(
            "Triplet root containing samples/<base_sample_id>/torsion_apo.npz; "
            "validated before any replica stage is run"
        ),
    )
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
    return json.loads(path.read_text())


def passed(path: Path, *, status: str) -> bool:
    record = load_json(path)
    if record is None:
        return False
    return record.get("status") == status and bool(record.get("passed", True))


def write_status(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_canonical_cache(
    data_dir: Path, record: Mapping[str, Any]
) -> Dict[str, Any]:
    from scripts.export_md_phase_normal_targets import _load_canonical_residue_axis

    system_sample_id = str(record["system_sample_id"])
    replica_sample_id = str(record["sample_id"])
    derived_system_id = replica_sample_id.split("__silver_r", 1)[0]
    if derived_system_id != system_sample_id:
        raise ValueError(
            f"Replica sample {replica_sample_id!r} does not belong to "
            f"system {system_sample_id!r}"
        )
    resolved_data_dir = data_dir.resolve()
    cache_path = resolved_data_dir / "samples" / system_sample_id / "torsion_apo.npz"
    keys, names = _load_canonical_residue_axis(
        resolved_data_dir, replica_sample_id
    )
    return {
        "status": "passed",
        "data_dir": str(resolved_data_dir),
        "cache": str(cache_path),
        "cache_sha256": file_sha256(cache_path),
        "n_residues": len(keys),
        "first_residue_key": list(keys[0]),
        "last_residue_key": list(keys[-1]),
        "first_residue_name": names[0],
        "last_residue_name": names[-1],
    }


def run_command(
    *,
    name: str,
    command: Sequence[str],
    output: Path,
    expected_status: str,
    force: bool,
    pipeline_status: Path,
    state: Dict[str, Any],
) -> None:
    if not force and passed(output, status=expected_status):
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
    if result.returncode != 0 or not passed(output, status=expected_status):
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
    pull_dir = Path(record["pull_dir"])
    target_dir = Path(record["target_dir"])
    pull_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    pipeline_status = pull_dir / "pipeline_status.json"
    previous_state = load_json(pipeline_status)
    state: Dict[str, Any] = {
        "schema_version": "bindrae_md_replica_pipeline_v2",
        "status": "starting",
        "matrix": str(args.matrix),
        "matrix_index": args.index,
        "sample_id": record["sample_id"],
        "system_sample_id": record["system_sample_id"],
        "transition_id": record["transition_id"],
        "replica_index": record["replica_index"],
        "seed": record["seed"],
        "protocol": record["protocol"],
        "residual_envelope": args.residual_envelope,
        "cpu_threads": args.cpu_threads,
        "canonical_data_dir": str(args.canonical_data_dir.resolve()),
        "started_at": utc_now(),
        "stages": {},
    }
    if previous_state is not None:
        state["resumed_from"] = {
            "schema_version": previous_state.get("schema_version"),
            "status": previous_state.get("status"),
            "failed_stage": previous_state.get("failed_stage"),
            "started_at": previous_state.get("started_at"),
            "finished_at": previous_state.get("finished_at"),
        }
    try:
        canonical_cache = preflight_canonical_cache(
            args.canonical_data_dir, record
        )
    except Exception as error:
        state["status"] = "failed"
        state["failed_stage"] = "canonical_cache_preflight"
        state["stages"]["canonical_cache_preflight"] = {
            "status": "failed",
            "finished_at": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_status(pipeline_status, state)
        raise
    state["canonical_cache"] = canonical_cache
    state["stages"]["canonical_cache_preflight"] = {
        "status": "passed",
        "finished_at": utc_now(),
        "output": canonical_cache["cache"],
    }
    write_status(pipeline_status, state)
    protocol = dict(record["protocol"])
    python = sys.executable

    pull_command: List[str] = [
        python,
        "scripts/run_md_global_rmsd_pull.py",
        "--candidate-manifest", str(record["candidate_manifest"]),
        "--transition-id", str(record["transition_id"]),
        "--npt-dir", str(record["npt_dir"]),
        "--output-dir", str(pull_dir),
        "--platform", args.platform,
        "--seed", str(record["seed"]),
        "--pre-equilibration-steps", str(protocol["pre_equilibration_steps"]),
        "--pulling-steps", str(protocol["pulling_steps"]),
        "--endpoint-hold-steps", str(protocol["endpoint_hold_steps"]),
        "--report-interval", str(protocol["report_interval"]),
        "--rmsd-k-kj-mol-nm2", str(protocol["rmsd_k_kj_mol_nm2"]),
        "--final-target-rmsd-nm", str(protocol["final_target_rmsd_nm"]),
        "--min-mapping-fraction", str(protocol.get("min_mapping_fraction", 0.95)),
    ]
    if protocol.get("resample_initial_velocities"):
        pull_command.append("--resample-initial-velocities")
    if args.cpu_threads > 0:
        pull_command.extend(["--cpu-threads", str(args.cpu_threads)])

    stages = [
        (
            "pull",
            pull_command,
            pull_dir / "rmsd_pull_report.json",
            "rmsd_pull_smoke_passed",
        ),
        (
            "path_audit",
            [
                python,
                "scripts/audit_md_rmsd_pull.py",
                "--pull-dir", str(pull_dir),
                "--output", str(pull_dir / "path_metrics_audit.json"),
            ],
            pull_dir / "path_metrics_audit.json",
            "path_metrics_passed",
        ),
        (
            "atomistic_audit",
            [
                python,
                "scripts/audit_md_atomistic_path.py",
                "--pull-dir", str(pull_dir),
                "--preparation-report", str(record["preparation_report"]),
                "--output", str(pull_dir / "atomistic_path_audit.json"),
            ],
            pull_dir / "atomistic_path_audit.json",
            "atomistic_path_passed",
        ),
        (
            "target_export",
            [
                python,
                "scripts/export_md_phase_normal_targets.py",
                "--pull-dir", str(pull_dir),
                "--candidate-manifest", str(record["candidate_manifest"]),
                "--transition-id", str(record["transition_id"]),
                "--preparation-report", str(record["preparation_report"]),
                "--output-dir", str(target_dir),
                "--sample-id", str(record["sample_id"]),
                "--data-dir", canonical_cache["data_dir"],
                "--residual-envelope", args.residual_envelope,
                "--normal-projection-mode", args.normal_projection_mode,
            ],
            target_dir / "target_audit.json",
            "md_phase_normal_targets_passed",
        ),
    ]
    for name, command, output, expected_status in stages:
        run_command(
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
    if args.cpu_threads < 0:
        raise ValueError("cpu_threads must be >= 0")
    record = load_record(args.matrix, args.index)
    lock_path = Path(record["pull_dir"]) / ".pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result = {
                "matrix_index": args.index,
                "replica_index": record["replica_index"],
                "sample_id": record["sample_id"],
                "status": "skipped_locked",
                "system_sample_id": record["system_sample_id"],
            }
        else:
            result = run_pipeline(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
