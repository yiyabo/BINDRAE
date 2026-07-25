#!/usr/bin/env python3
"""Run a failure-isolated Path-4 OpenMM development panel.

This is an engineering panel over training-scope systems.  It is deliberately
separate from the frozen validation/test lanes and from the future formal
40--60-system Gate-0 cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANEL_SCHEMA_VERSION = "bindrae_path4_gate0_dev_panel_v1"

# Optimizer settings are frozen from the completed one-system run 147166.  A
# zero tail scale preserves that run's per-system automatic scale selection.
OPTIMIZER_CONTRACT: Mapping[str, object] = {
    "time_basis_rank": 2,
    "num_starts": 3,
    "iterations": 4,
    "seed": 20260720,
    "route_seed_scale_angstrom": 0.05,
    "max_residue_translation_angstrom": 0.75,
    "step_size_angstrom": 0.05,
    "line_search_steps": 5,
    "line_search_shrink": 0.5,
    "chain_smoothing_steps": 2,
    "softmax_beta": 8.0,
    "tail_scale_kj_mol": 0.0,
    "magnitude_penalty_kj_mol_a2": 500.0,
    "temporal_penalty_kj_mol_a2": 100.0,
    "acceptance_tolerance_kj_mol": 1.0e-3,
}

REFERENCE_CONTRACT: Mapping[str, object] = {
    # Re-frozen after the formal reserve6 reference-only run on 2026-07-24:
    # all 126 frames passed the unchanged 1e6 force threshold at this cap.
    "reference_relaxation_iterations": 250,
    "reference_restraint_k_kj_mol_nm2": 100000.0,
    "reference_minimization_tolerance_kj_mol_nm": 500.0,
    "minimum_residue_mapping": 0.98,
    "minimum_atom_mapping": 0.95,
    "maximum_reference_atomic_force_kj_mol_nm": 1.0e6,
    "maximum_frame_reference_atomic_force_kj_mol_nm": 1.0e6,
}

SCORER_CONTRACT: Mapping[str, object] = {
    "restraint_mode": "ca",
    "restraint_k_kj_mol_nm2": 5000.0,
    "minimization_tolerance_kj_mol_nm": 25.0,
    "max_minimization_iterations": 250,
    "maximum_relaxed_residue_net_force_kj_mol_nm": 500.0,
    "minimum_residue_mapping": 0.98,
    "minimum_atom_mapping": 0.95,
    "severe_clash_distance_angstrom": 1.5,
    "maximum_endpoint_energy_difference_kj_mol": 1.0,
}


def _scientific_contract() -> Dict[str, Any]:
    return {
        "split": "train",
        "candidate": "frozen_path3_cached_phase_cummax",
        "sentinels_excluded_from_primary_summary": True,
        "reference": dict(REFERENCE_CONTRACT),
        "optimizer": dict(OPTIMIZER_CONTRACT),
        "scorer": dict(SCORER_CONTRACT),
    }


def _validate_resume_scientific_contract(manifest: Mapping[str, Any]) -> None:
    observed = manifest.get("scientific_contract")
    expected = _scientific_contract()
    if not isinstance(observed, Mapping):
        raise ValueError("Resume manifest misses its scientific contract")
    for section, expected_value in expected.items():
        if observed.get(section) != expected_value:
            raise ValueError(
                f"Resume scientific-contract mismatch in {section}: "
                f"requested={expected_value!r}, manifest={observed.get(section)!r}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-samples-file", type=Path, required=True)
    parser.add_argument(
        "--exclude-samples-file", type=Path, action="append", default=[]
    )
    parser.add_argument("--phase-cache-dir", type=Path, required=True)
    parser.add_argument("--md-reference-cache-dir", type=Path)
    parser.add_argument("--prepared-systems-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=12)
    parser.add_argument("--sentinel", action="append", default=[])
    parser.add_argument("--minimum-residues", type=int, default=60)
    parser.add_argument("--maximum-residues", type=int, default=400)
    parser.add_argument("--platform", choices=["CUDA", "CPU", "OpenCL"], default="CUDA")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--conda-executable",
        type=Path,
        default=Path(
            "/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda"
        ),
    )
    parser.add_argument("--bindrae-env", default="BINDRAE")
    parser.add_argument("--openmm-env", default="BINDRAE-MD")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument(
        "--execution-mode",
        choices=["full", "preflight"],
        default="full",
        help="Run the complete optimizer/scorer chain or prepared-system preflight only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write the frozen selection manifest without running OpenMM.",
    )
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_ids(value: object) -> list[str]:
    if isinstance(value, list):
        ids = []
        for row in value:
            if isinstance(row, str):
                ids.append(row)
            elif isinstance(row, Mapping):
                if row.get("sample_id"):
                    ids.append(str(row["sample_id"]))
                elif row.get("id"):
                    ids.append(str(row["id"]))
        return ids
    if isinstance(value, Mapping):
        if value.get("sample_id"):
            return [str(value["sample_id"])]
        if value.get("id"):
            return [str(value["id"])]
        for key in ("systems", "samples", "ids", "train", "val", "test"):
            if key in value:
                ids = _extract_ids(value[key])
                if ids:
                    return ids
    return []


def read_sample_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Sample list does not exist: {path}")
    if path.suffix.lower() == ".json":
        values = _extract_ids(json.loads(path.read_text()))
    else:
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"Sample list is empty or unsupported: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Sample list contains duplicates: {path}")
    return values


def _sample_tokens(sample_id: str) -> tuple[str, str]:
    fields = sample_id.split("-")
    pdb_id = fields[0].lower() if fields else sample_id.lower()
    ligand = fields[2].upper() if len(fields) >= 3 else sample_id.upper()
    return pdb_id, ligand


def _sample_metadata(
    sample_id: str, phase_cache_dir: Path, prepared_systems_dir: Path
) -> Dict[str, Any]:
    safe_id = _safe_sample_id(sample_id)
    phase_path = phase_cache_dir / f"{safe_id}.npz"
    preparation_report = (
        prepared_systems_dir / sample_id / "setup" / "preparation_report.json"
    )
    if not phase_path.is_file() or not preparation_report.is_file():
        raise FileNotFoundError(
            f"Missing phase/preparation asset for {sample_id}: "
            f"phase={phase_path.is_file()} preparation={preparation_report.is_file()}"
        )
    with np.load(phase_path, allow_pickle=False) as loaded:
        cached_id = str(np.asarray(loaded["sample_id"]).item())
        n_residues = int(np.asarray(loaded["n_residues"]).item())
        n_phase_interior = int(np.asarray(loaded["t_values"]).size)
    if cached_id != sample_id:
        raise ValueError(
            f"Phase-cache identity mismatch for {sample_id}: {cached_id}"
        )
    preparation = json.loads(preparation_report.read_text())
    protein = preparation.get("protein") or {}
    pdb_id, ligand_code = _sample_tokens(sample_id)
    return {
        "sample_id": sample_id,
        "safe_sample_id": safe_id,
        "pdb_id": pdb_id,
        "ligand_code": ligand_code,
        "n_residues": n_residues,
        "n_phase_interior": n_phase_interior,
        "prepared_protein_atoms": protein.get("prepared_protein_atoms"),
        "phase_cache": str(phase_path),
        "preparation_report": str(preparation_report),
    }


def select_diverse_rows(rows: Sequence[Mapping[str, Any]], count: int) -> list[Dict[str, Any]]:
    """Choose deterministic length-spread rows while preferring new PDB/ligand IDs."""
    if count < 0:
        raise ValueError("Selection count must be non-negative")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (int(row["n_residues"]), str(row["sample_id"])),
    )
    if len(ordered) < count:
        raise ValueError(f"Need {count} eligible systems, found {len(ordered)}")
    if count == 0:
        return []

    targets = np.linspace(0, len(ordered) - 1, num=count)
    selected: list[Dict[str, Any]] = []
    used_ids: set[str] = set()
    used_pdbs: set[str] = set()
    used_ligands: set[str] = set()
    for target in targets:
        candidates = [row for row in ordered if str(row["sample_id"]) not in used_ids]
        row = min(
            candidates,
            key=lambda item: (
                int(str(item["pdb_id"]) in used_pdbs)
                + int(str(item["ligand_code"]) in used_ligands),
                int(str(item["pdb_id"]) in used_pdbs),
                int(str(item["ligand_code"]) in used_ligands),
                abs(ordered.index(item) - float(target)),
                str(item["sample_id"]),
            ),
        )
        selected.append(row)
        used_ids.add(str(row["sample_id"]))
        used_pdbs.add(str(row["pdb_id"]))
        used_ligands.add(str(row["ligand_code"]))
    return selected


def build_selection_manifest(
    *,
    train_samples_file: Path,
    exclude_samples_files: Sequence[Path],
    phase_cache_dir: Path,
    prepared_systems_dir: Path,
    panel_size: int,
    sentinels: Sequence[str],
    minimum_residues: int,
    maximum_residues: int,
    execution_mode: str = "full",
) -> Dict[str, Any]:
    if panel_size <= 0:
        raise ValueError("panel_size must be positive")
    if minimum_residues <= 0 or maximum_residues < minimum_residues:
        raise ValueError("Invalid residue-count bounds")
    if len(sentinels) != len(set(sentinels)):
        raise ValueError("Duplicate sentinel IDs")
    if len(sentinels) >= panel_size:
        raise ValueError("panel_size must exceed the number of sentinels")
    if execution_mode not in {"full", "preflight"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    train_ids = read_sample_ids(train_samples_file)
    train_set = set(train_ids)
    excluded_by_file: Dict[str, list[str]] = {}
    excluded: set[str] = set()
    for path in exclude_samples_files:
        values = read_sample_ids(path)
        excluded_by_file[str(path)] = values
        excluded.update(values)

    phase_ids = {path.stem for path in phase_cache_dir.glob("*.npz")}
    prepared_ids = {
        path.parent.parent.name
        for path in prepared_systems_dir.glob("*/setup/preparation_report.json")
    }
    asset_intersection = train_set & phase_ids & prepared_ids

    sentinel_rows = []
    for sample_id in sentinels:
        if sample_id not in asset_intersection:
            raise ValueError(
                f"Sentinel is not in the training/phase/prepared intersection: {sample_id}"
            )
        row = _sample_metadata(sample_id, phase_cache_dir, prepared_systems_dir)
        row["role"] = "sentinel"
        row["excluded_from_primary_summary"] = True
        row["explicit_exclusion_overlap"] = sorted(
            path for path, ids in excluded_by_file.items() if sample_id in set(ids)
        )
        sentinel_rows.append(row)

    primary_ids = sorted(asset_intersection - excluded - set(sentinels))
    eligible_rows = []
    invalid_metadata = []
    for sample_id in primary_ids:
        try:
            row = _sample_metadata(sample_id, phase_cache_dir, prepared_systems_dir)
        except Exception as exc:
            invalid_metadata.append(
                {
                    "sample_id": sample_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if minimum_residues <= int(row["n_residues"]) <= maximum_residues:
            eligible_rows.append(row)

    primary_count = panel_size - len(sentinel_rows)
    primary_rows = select_diverse_rows(eligible_rows, primary_count)
    for row in primary_rows:
        row["role"] = "primary"
        row["excluded_from_primary_summary"] = False
        row["explicit_exclusion_overlap"] = []

    selected_ids = [str(row["sample_id"]) for row in [*sentinel_rows, *primary_rows]]
    primary_selected = {str(row["sample_id"]) for row in primary_rows}
    if primary_selected & excluded:
        raise RuntimeError("Explicitly excluded IDs survived primary selection")
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Panel selection produced duplicate IDs")

    input_files = [train_samples_file, *exclude_samples_files]
    return {
        "schema_version": PANEL_SCHEMA_VERSION,
        "created_at": _now(),
        "purpose": "training_scope_development_panel_not_formal_gate0",
        "execution_mode": execution_mode,
        "selection": {
            "algorithm": "deterministic_residue_length_spread_prefer_unique_pdb_ligand_v1",
            "panel_size": panel_size,
            "primary_systems": primary_count,
            "sentinel_systems": len(sentinel_rows),
            "minimum_residues": minimum_residues,
            "maximum_residues": maximum_residues,
        },
        "inputs": [
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "count": len(read_sample_ids(path)),
                "role": "train_scope" if path == train_samples_file else "exclude",
            }
            for path in input_files
        ],
        "asset_counts": {
            "train_scope": len(train_set),
            "phase_cache": len(phase_ids),
            "prepared_systems": len(prepared_ids),
            "train_phase_prepared_intersection": len(asset_intersection),
            "explicitly_excluded_union": len(excluded),
            "eligible_after_bounds": len(eligible_rows),
            "invalid_metadata": len(invalid_metadata),
        },
        "exclusion_audit": {
            "primary_overlap": sorted(primary_selected & excluded),
            "sentinel_overlap_is_non_headline": sorted(set(sentinels) & excluded),
            "passed": not bool(primary_selected & excluded),
        },
        "scientific_contract": _scientific_contract(),
        "systems": [*sentinel_rows, *primary_rows],
        "invalid_metadata": invalid_metadata,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _json_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(value, Mapping):
        return str(value.get("status") or "")
    return None


def _log_tail(path: Path, maximum_chars: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    return text[-maximum_chars:]


def _conda_python(conda: Path, environment: str) -> list[str]:
    return [str(conda), "run", "--no-capture-output", "-n", environment, "python"]


def _run_logged(command: Sequence[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(f"\n[{_now()}] $ {shlex.join(str(part) for part in command)}\n")
        handle.flush()
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
        handle.write(f"[{_now()}] exit_code={completed.returncode}\n")
    return int(completed.returncode)


def _option_args(values: Mapping[str, object]) -> list[str]:
    result = []
    for name, value in values.items():
        result.extend([f"--{name.replace('_', '-')}", str(value)])
    return result


def _initial_state(
    manifest: Mapping[str, Any], output_root: Path, execution_mode: str
) -> Dict[str, Any]:
    return {
        "schema_version": PANEL_SCHEMA_VERSION,
        "status": "running",
        "execution_mode": execution_mode,
        "started_at": _now(),
        "updated_at": _now(),
        "output_root": str(output_root),
        "systems": {
            str(row["sample_id"]): {
                "role": row["role"],
                "status": "pending",
                "completed_stages": [],
            }
            for row in manifest["systems"]
        },
    }


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = _now()
    _write_json_atomic(path, state)


def _record_failure(
    state: Dict[str, Any],
    state_path: Path,
    sample_id: str,
    *,
    status: str,
    stage: str,
    log_path: Path,
    report_path: Path | None = None,
) -> None:
    record = state["systems"][sample_id]
    record.update(
        {
            "status": status,
            "failure_stage": stage,
            "log": str(log_path),
            "log_tail": _log_tail(log_path),
        }
    )
    if report_path is not None and report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            report = None
        if isinstance(report, Mapping):
            report_failure_stage = report.get("failure_stage")
            if report_failure_stage:
                record["runner_failure_stage"] = stage
                record["failure_stage"] = report_failure_stage
            record["rejection_type"] = report.get("rejection_type")
            record["rejection_reason"] = report.get("rejection_reason")
            for key in (
                "failed_frame_index",
                "failed_frame_time",
                "failure_context",
                "frame_preflight_completed",
            ):
                if key in report:
                    record[key] = report[key]
            record["report"] = str(report_path)
    _save_state(state_path, state)


def _mark_stage(
    state: Dict[str, Any], state_path: Path, sample_id: str, stage: str
) -> None:
    completed = state["systems"][sample_id]["completed_stages"]
    if stage not in completed:
        completed.append(stage)
    state["systems"][sample_id]["status"] = "running"
    _save_state(state_path, state)


def _run_system(
    args: argparse.Namespace,
    row: Mapping[str, Any],
    state: Dict[str, Any],
    state_path: Path,
    bindrae_python: Sequence[str],
    openmm_python: Sequence[str],
) -> bool:
    root = args.project_root
    sample_id = str(row["sample_id"])
    safe_id = str(row["safe_sample_id"])
    system_root = args.output_root / "systems" / safe_id
    candidate_dir = system_root / "candidates"
    path3_candidate = candidate_dir / "path3" / f"{safe_id}.npz"
    optimized_candidate = candidate_dir / "openmm_multistart" / f"{safe_id}.npz"
    export_manifest = system_root / "path3_manifest.json"
    subset_file = system_root / "sample.txt"
    prepared_preflight_report = system_root / "reference" / "preflight.json"
    reference_cache = system_root / "reference" / "path3_all_atom_reference.npz"
    reference_report = system_root / "reference" / "report.json"
    optimizer_report = system_root / "optimizer" / "report.json"
    path3_score = system_root / "openmm" / "path3.json"
    optimized_score = system_root / "openmm" / "optimized.json"
    pair_summary = system_root / "paired_summary.json"
    path3_md_reference = system_root / "md_reference" / "path3.json"
    optimized_md_reference = system_root / "md_reference" / "optimized.json"
    md_reference_comparison = system_root / "md_reference" / "comparison.json"
    implicit_cache = args.output_root / "implicit_cache" / f"{safe_id}_gbn2_v1"
    preparation_report = Path(str(row["preparation_report"]))
    subset_file.parent.mkdir(parents=True, exist_ok=True)
    subset_file.write_text(sample_id + "\n")

    record = state["systems"][sample_id]
    record["status"] = "running"
    record["system_root"] = str(system_root)
    _save_state(state_path, state)
    print(f"[{_now()}] {sample_id}: start ({row['role']})", flush=True)

    export_log = system_root / "logs" / "01_export_path3.log"
    if not (args.resume and path3_candidate.is_file() and export_manifest.is_file()):
        command = [
            *bindrae_python,
            "scripts/export_cached_phase_path_candidate.py",
            "--checkpoint",
            str(args.checkpoint),
            "--data-dir",
            str(args.data_dir),
            "--split",
            "train",
            "--valid-samples-file",
            str(subset_file),
            "--phase-cache-dir",
            str(args.phase_cache_dir),
            "--device",
            "cpu",
            "--output-dir",
            str(path3_candidate.parent),
            "--manifest",
            str(export_manifest),
            "--candidate-label",
            "frozen_path3_cummax_gate0_dev_panel",
            "--candidate-mode",
            "path3",
            "--phase-tau-postprocess",
            "cummax",
            "--max-samples",
            "1",
        ]
        if _run_logged(command, cwd=root, log_path=export_log) != 0:
            _record_failure(
                state,
                state_path,
                sample_id,
                status="failed",
                stage="path3_export",
                log_path=export_log,
            )
            return False
    _mark_stage(state, state_path, sample_id, "path3_export")

    if args.execution_mode == "preflight":
        preflight_log = system_root / "logs" / "02_prepared_preflight.log"
        if not (
            args.resume and _json_status(prepared_preflight_report) == "completed"
        ):
            command = [
                *openmm_python,
                "scripts/preflight_path4_openmm_gate0.py",
                "--candidate",
                str(path3_candidate),
                "--preparation-report",
                str(preparation_report),
                "--implicit-cache-dir",
                str(implicit_cache),
                "--report",
                str(prepared_preflight_report),
                "--platform",
                args.platform,
                "--device-index",
                str(args.device_index),
                "--cpu-threads",
                str(args.cpu_threads),
                "--minimum-residue-mapping",
                str(REFERENCE_CONTRACT["minimum_residue_mapping"]),
                "--minimum-atom-mapping",
                str(REFERENCE_CONTRACT["minimum_atom_mapping"]),
                "--maximum-reference-atomic-force-kj-mol-nm",
                str(REFERENCE_CONTRACT["maximum_reference_atomic_force_kj_mol_nm"]),
            ]
            if _run_logged(command, cwd=root, log_path=preflight_log) != 0:
                report_status = _json_status(prepared_preflight_report)
                _record_failure(
                    state,
                    state_path,
                    sample_id,
                    status=(
                        "rejected_preflight"
                        if report_status == "rejected"
                        else "failed"
                    ),
                    stage="prepared_reference_preflight",
                    log_path=preflight_log,
                    report_path=prepared_preflight_report,
                )
                return False
        _mark_stage(state, state_path, sample_id, "prepared_reference_preflight")
        record.update(
            {
                "status": "preflight_completed",
                "path3_candidate": str(path3_candidate),
                "preflight_report": str(prepared_preflight_report),
            }
        )
        _save_state(state_path, state)
        print(f"[{_now()}] {sample_id}: preflight completed", flush=True)
        return True

    reference_log = system_root / "logs" / "02_reference_preflight.log"
    if not (
        args.resume
        and reference_cache.is_file()
        and _json_status(reference_report) == "completed"
    ):
        command = [
            *openmm_python,
            "scripts/build_path4_openmm_frame_reference.py",
            "--candidate",
            str(path3_candidate),
            "--preparation-report",
            str(preparation_report),
            "--implicit-cache-dir",
            str(implicit_cache),
            "--output-cache",
            str(reference_cache),
            "--report",
            str(reference_report),
            "--platform",
            args.platform,
            "--device-index",
            str(args.device_index),
            "--cpu-threads",
            str(args.cpu_threads),
            *_option_args(REFERENCE_CONTRACT),
        ]
        if _run_logged(command, cwd=root, log_path=reference_log) != 0:
            report_status = _json_status(reference_report)
            _record_failure(
                state,
                state_path,
                sample_id,
                status=(
                    "rejected_reference_preflight"
                    if report_status == "rejected"
                    else "failed"
                ),
                stage="reference_preflight",
                log_path=reference_log,
                report_path=reference_report,
            )
            return False
    _mark_stage(state, state_path, sample_id, "reference_preflight")

    optimizer_log = system_root / "logs" / "03_optimize.log"
    if not (
        args.resume
        and optimized_candidate.is_file()
        and _json_status(optimizer_report) == "completed"
    ):
        optimizer_reference = {
            key: value
            for key, value in REFERENCE_CONTRACT.items()
            if key != "maximum_frame_reference_atomic_force_kj_mol_nm"
        }
        command = [
            *openmm_python,
            "scripts/optimize_path4_openmm_gate0.py",
            "--candidate",
            str(path3_candidate),
            "--preparation-report",
            str(preparation_report),
            "--implicit-cache-dir",
            str(implicit_cache),
            "--output-candidate",
            str(optimized_candidate),
            "--report",
            str(optimizer_report),
            "--candidate-label",
            "openmm_multistart_gate0_dev_panel",
            "--platform",
            args.platform,
            "--device-index",
            str(args.device_index),
            "--cpu-threads",
            str(args.cpu_threads),
            *_option_args(OPTIMIZER_CONTRACT),
            *_option_args(optimizer_reference),
        ]
        if _run_logged(command, cwd=root, log_path=optimizer_log) != 0:
            _record_failure(
                state,
                state_path,
                sample_id,
                status="failed",
                stage="optimizer",
                log_path=optimizer_log,
                report_path=optimizer_report,
            )
            return False
    _mark_stage(state, state_path, sample_id, "optimizer")

    common_score = [
        "--preparation-report",
        str(preparation_report),
        "--implicit-cache-dir",
        str(implicit_cache),
        "--platform",
        args.platform,
        "--device-index",
        str(args.device_index),
        "--cpu-threads",
        str(args.cpu_threads),
        "--restraint-mode",
        str(SCORER_CONTRACT["restraint_mode"]),
        "--frame-initialization",
        "reference_cache",
        "--frame-reference-cache",
        str(reference_cache),
        "--restraint-k-kj-mol-nm2",
        str(SCORER_CONTRACT["restraint_k_kj_mol_nm2"]),
        "--minimization-tolerance-kj-mol-nm",
        str(SCORER_CONTRACT["minimization_tolerance_kj_mol_nm"]),
        "--max-minimization-iterations",
        str(SCORER_CONTRACT["max_minimization_iterations"]),
        "--maximum-relaxed-residue-net-force-kj-mol-nm",
        str(SCORER_CONTRACT["maximum_relaxed_residue_net_force_kj_mol_nm"]),
        "--maximum-reference-atomic-force-kj-mol-nm",
        str(REFERENCE_CONTRACT["maximum_reference_atomic_force_kj_mol_nm"]),
        "--minimum-residue-mapping",
        str(SCORER_CONTRACT["minimum_residue_mapping"]),
        "--minimum-atom-mapping",
        str(SCORER_CONTRACT["minimum_atom_mapping"]),
        "--severe-clash-distance-angstrom",
        str(SCORER_CONTRACT["severe_clash_distance_angstrom"]),
        "--diagnostic-force-components",
    ]
    for stage, candidate, output, log_name in (
        ("path3_score", path3_candidate, path3_score, "04_score_path3.log"),
        (
            "optimized_score",
            optimized_candidate,
            optimized_score,
            "05_score_optimized.log",
        ),
    ):
        score_log = system_root / "logs" / log_name
        if not (args.resume and _json_status(output) == "completed"):
            command = [
                *openmm_python,
                "scripts/evaluate_path4_openmm_gate0.py",
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                *common_score,
            ]
            if _run_logged(command, cwd=root, log_path=score_log) != 0:
                _record_failure(
                    state,
                    state_path,
                    sample_id,
                    status="failed",
                    stage=stage,
                    log_path=score_log,
                    report_path=output,
                )
                return False
        _mark_stage(state, state_path, sample_id, stage)

    pair_log = system_root / "logs" / "06_pair_summary.log"
    if not (args.resume and pair_summary.is_file()):
        command = [
            *bindrae_python,
            "scripts/summarize_path4_gate0_pairs.py",
            "--path3-glob",
            str(path3_score),
            "--candidate-glob",
            str(optimized_score),
            "--optimizer-glob",
            str(optimizer_report),
            "--output",
            str(pair_summary),
            "--bootstrap-resamples",
            str(args.bootstrap_resamples),
            "--maximum-endpoint-energy-difference-kj-mol",
            str(SCORER_CONTRACT["maximum_endpoint_energy_difference_kj_mol"]),
        ]
        if _run_logged(command, cwd=root, log_path=pair_log) != 0:
            _record_failure(
                state,
                state_path,
                sample_id,
                status="failed",
                stage="paired_summary",
                log_path=pair_log,
                report_path=pair_summary,
            )
            return False
    _mark_stage(state, state_path, sample_id, "paired_summary")

    for stage, candidate_path, output, log_name in (
        (
            "path3_md_reference",
            path3_candidate,
            path3_md_reference,
            "07_md_reference_path3.log",
        ),
        (
            "optimized_md_reference",
            optimized_candidate,
            optimized_md_reference,
            "08_md_reference_optimized.log",
        ),
    ):
        md_log = system_root / "logs" / log_name
        if not (args.resume and _json_status(output) == "completed"):
            command = [
                *bindrae_python,
                "scripts/evaluate_path4_candidate_md_reference.py",
                "--candidate",
                str(candidate_path),
                "--data-dir",
                str(args.data_dir),
                "--split",
                "train",
                "--valid-samples-file",
                str(subset_file),
                "--md-reference-cache-dir",
                str(args.md_reference_cache_dir),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
            if _run_logged(command, cwd=root, log_path=md_log) != 0:
                _record_failure(
                    state,
                    state_path,
                    sample_id,
                    status="failed",
                    stage=stage,
                    log_path=md_log,
                    report_path=output,
                )
                return False
        _mark_stage(state, state_path, sample_id, stage)

    md_compare_log = system_root / "logs" / "09_md_reference_compare.log"
    if not (args.resume and md_reference_comparison.is_file()):
        command = [
            *bindrae_python,
            "scripts/compare_stage2_md_reference_results.py",
            "--baseline",
            str(path3_md_reference),
            "--candidate",
            str(optimized_md_reference),
            "--output",
            str(md_reference_comparison),
            "--baseline-name",
            "frozen_path3",
            "--candidate-name",
            "path4_openmm_optimized",
            "--metrics",
            ",".join(
                (
                    "md_path_product_rmse",
                    "md_path_translation_mae_a",
                    "md_path_rotation_mae_rad",
                    "md_path_chi_mae_rad",
                )
            ),
            "--bootstrap-samples",
            str(args.bootstrap_resamples),
            "--noninferiority-margin-percent",
            "1.0",
        ]
        if _run_logged(command, cwd=root, log_path=md_compare_log) != 0:
            _record_failure(
                state,
                state_path,
                sample_id,
                status="failed",
                stage="md_reference_comparison",
                log_path=md_compare_log,
                report_path=md_reference_comparison,
            )
            return False
    _mark_stage(state, state_path, sample_id, "md_reference_comparison")
    record.update(
        {
            "status": "completed",
            "path3_score": str(path3_score),
            "optimized_score": str(optimized_score),
            "optimizer_report": str(optimizer_report),
            "paired_summary": str(pair_summary),
            "path3_md_reference": str(path3_md_reference),
            "optimized_md_reference": str(optimized_md_reference),
            "md_reference_comparison": str(md_reference_comparison),
        }
    )
    _save_state(state_path, state)
    print(f"[{_now()}] {sample_id}: completed", flush=True)
    return True


def _aggregate(
    *,
    args: argparse.Namespace,
    state: Mapping[str, Any],
    roles: Iterable[str],
    output: Path,
    log_path: Path,
    bindrae_python: Sequence[str],
) -> int:
    allowed_roles = set(roles)
    completed = [
        record
        for record in state["systems"].values()
        if record.get("status") == "completed" and record.get("role") in allowed_roles
    ]
    if not completed:
        return 3
    command = [*bindrae_python, "scripts/summarize_path4_gate0_pairs.py"]
    for record in completed:
        command.extend(["--path3-glob", str(record["path3_score"])])
        command.extend(["--candidate-glob", str(record["optimized_score"])])
        command.extend(["--optimizer-glob", str(record["optimizer_report"])])
    command.extend(
        [
            "--output",
            str(output),
            "--bootstrap-resamples",
            str(args.bootstrap_resamples),
            "--maximum-endpoint-energy-difference-kj-mol",
            str(SCORER_CONTRACT["maximum_endpoint_energy_difference_kj_mol"]),
        ]
    )
    return _run_logged(command, cwd=args.project_root, log_path=log_path)


def _aggregate_md_reference(
    *,
    args: argparse.Namespace,
    state: Mapping[str, Any],
    roles: Iterable[str],
    output_dir: Path,
    bindrae_python: Sequence[str],
) -> tuple[int, Path | None]:
    allowed_roles = set(roles)
    completed = [
        record
        for record in state["systems"].values()
        if record.get("status") == "completed" and record.get("role") in allowed_roles
    ]
    if not completed:
        return 3, None

    arm_outputs = {}
    for arm, record_key in (
        ("path3", "path3_md_reference"),
        ("optimized", "optimized_md_reference"),
    ):
        records = []
        sources = []
        for record in completed:
            source = Path(str(record[record_key]))
            payload = json.loads(source.read_text())
            if payload.get("schema_version") != "bindrae_path4_candidate_md_reference_eval_v1":
                raise ValueError(f"Unexpected candidate MD-reference schema: {source}")
            arm_records = payload.get("records")
            if not isinstance(arm_records, list) or not arm_records:
                raise ValueError(f"Candidate MD-reference output has no records: {source}")
            records.extend(arm_records)
            sources.append(str(source))
        arm_output = output_dir / f"md_reference_{arm}_primary.json"
        _write_json_atomic(
            arm_output,
            {
                "schema_version": "bindrae_path4_candidate_md_reference_collection_v1",
                "arm": arm,
                "systems": len(completed),
                "sources": sources,
                "records": records,
            },
        )
        arm_outputs[arm] = arm_output

    comparison = output_dir / "md_reference_comparison_primary.json"
    command = [
        *bindrae_python,
        "scripts/compare_stage2_md_reference_results.py",
        "--baseline",
        str(arm_outputs["path3"]),
        "--candidate",
        str(arm_outputs["optimized"]),
        "--output",
        str(comparison),
        "--baseline-name",
        "frozen_path3",
        "--candidate-name",
        "path4_openmm_optimized",
        "--metrics",
        ",".join(
            (
                "md_path_product_rmse",
                "md_path_translation_mae_a",
                "md_path_rotation_mae_rad",
                "md_path_chi_mae_rad",
            )
        ),
        "--bootstrap-samples",
        str(args.bootstrap_resamples),
        "--noninferiority-margin-percent",
        "1.0",
    ]
    code = _run_logged(
        command,
        cwd=args.project_root,
        log_path=output_dir / "md_reference_comparison_primary.log",
    )
    return code, comparison if code == 0 else None


def _write_valid_primary_samples(
    output_root: Path, state: Mapping[str, Any]
) -> Path:
    accepted_statuses = {"preflight_completed", "completed"}
    sample_ids = sorted(
        sample_id
        for sample_id, record in state["systems"].items()
        if record.get("role") == "primary"
        and record.get("status") in accepted_statuses
    )
    path = output_root / "valid_primary_samples.txt"
    path.write_text("".join(f"{sample_id}\n" for sample_id in sample_ids))
    return path


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    for name in (
        "checkpoint",
        "data_dir",
        "train_samples_file",
        "phase_cache_dir",
        "prepared_systems_dir",
        "output_root",
        "conda_executable",
    ):
        setattr(args, name, _resolve(args.project_root, getattr(args, name)))
    if args.md_reference_cache_dir is not None:
        args.md_reference_cache_dir = _resolve(
            args.project_root, args.md_reference_cache_dir
        )
    args.exclude_samples_file = [
        _resolve(args.project_root, path) for path in args.exclude_samples_file
    ]
    if args.cpu_threads <= 0 or args.bootstrap_resamples <= 0:
        raise ValueError("cpu_threads and bootstrap_resamples must be positive")
    if (
        args.execution_mode == "full"
        and not args.selection_only
        and args.md_reference_cache_dir is None
    ):
        raise ValueError("full execution requires --md-reference-cache-dir")
    for path in (
        args.checkpoint,
        args.data_dir,
        args.train_samples_file,
        args.phase_cache_dir,
        args.prepared_systems_dir,
        args.conda_executable,
        *args.exclude_samples_file,
        *(
            [args.md_reference_cache_dir]
            if args.md_reference_cache_dir is not None
            else []
        ),
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest_path = args.output_root / "selection_manifest.json"
    state_path = args.output_root / "panel_state.json"
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to reuse development-panel output without --resume: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not manifest_path.is_file() or not state_path.is_file():
            raise FileNotFoundError("Resume requires selection_manifest.json and panel_state.json")
        manifest = json.loads(manifest_path.read_text())
        state = json.loads(state_path.read_text())
        if manifest.get("schema_version") != PANEL_SCHEMA_VERSION:
            raise ValueError("Resume manifest schema mismatch")
        manifest_mode = str(manifest.get("execution_mode") or "full")
        state_mode = str(state.get("execution_mode") or manifest_mode)
        if manifest_mode != args.execution_mode or state_mode != args.execution_mode:
            raise ValueError(
                "Resume execution-mode mismatch: "
                f"requested={args.execution_mode}, manifest={manifest_mode}, "
                f"state={state_mode}"
            )
        _validate_resume_scientific_contract(manifest)
    else:
        manifest = build_selection_manifest(
            train_samples_file=args.train_samples_file,
            exclude_samples_files=args.exclude_samples_file,
            phase_cache_dir=args.phase_cache_dir,
            prepared_systems_dir=args.prepared_systems_dir,
            panel_size=args.panel_size,
            sentinels=args.sentinel,
            minimum_residues=args.minimum_residues,
            maximum_residues=args.maximum_residues,
            execution_mode=args.execution_mode,
        )
        _write_json_atomic(manifest_path, manifest)
        state = _initial_state(manifest, args.output_root, args.execution_mode)
        _save_state(state_path, state)

    print(
        json.dumps(
            {
                "selection_manifest": str(manifest_path),
                "output_root": str(args.output_root),
                "systems": [
                    {"sample_id": row["sample_id"], "role": row["role"]}
                    for row in manifest["systems"]
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.selection_only:
        state["status"] = "selection_only_completed"
        _save_state(state_path, state)
        return

    bindrae_python = _conda_python(args.conda_executable, args.bindrae_env)
    openmm_python = _conda_python(args.conda_executable, args.openmm_env)
    for row in manifest["systems"]:
        sample_id = str(row["sample_id"])
        completed_status = (
            "preflight_completed" if args.execution_mode == "preflight" else "completed"
        )
        if args.resume and state["systems"][sample_id].get("status") == completed_status:
            print(f"[{_now()}] {sample_id}: resume skip completed", flush=True)
            continue
        _run_system(
            args,
            row,
            state,
            state_path,
            bindrae_python,
            openmm_python,
        )

    counts: Dict[str, int] = {}
    for record in state["systems"].values():
        status = str(record.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    valid_primary_samples = _write_valid_primary_samples(args.output_root, state)
    state["counts"] = dict(sorted(counts.items()))
    state["valid_primary_samples"] = str(valid_primary_samples)
    if args.execution_mode == "preflight":
        state["status"] = (
            "completed"
            if counts.get("preflight_completed", 0) == len(state["systems"])
            else "completed_with_rejections_or_failures"
        )
        state["finished_at"] = _now()
        _save_state(state_path, state)
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "counts": counts,
                    "valid_primary_samples": str(valid_primary_samples),
                },
                sort_keys=True,
            )
        )
        return

    aggregate_dir = args.output_root / "aggregate"
    primary_output = aggregate_dir / "paired_summary_primary.json"
    all_output = aggregate_dir / "paired_summary_with_sentinel.json"
    primary_code = _aggregate(
        args=args,
        state=state,
        roles=["primary"],
        output=primary_output,
        log_path=aggregate_dir / "primary_summary.log",
        bindrae_python=bindrae_python,
    )
    all_code = _aggregate(
        args=args,
        state=state,
        roles=["primary", "sentinel"],
        output=all_output,
        log_path=aggregate_dir / "with_sentinel_summary.log",
        bindrae_python=bindrae_python,
    )
    md_code, md_output = _aggregate_md_reference(
        args=args,
        state=state,
        roles=["primary"],
        output_dir=aggregate_dir,
        bindrae_python=bindrae_python,
    )
    state["aggregate"] = {
        "primary": str(primary_output) if primary_code == 0 else None,
        "primary_exit_code": primary_code,
        "with_sentinel": str(all_output) if all_code == 0 else None,
        "with_sentinel_exit_code": all_code,
        "md_reference_primary": str(md_output) if md_output is not None else None,
        "md_reference_primary_exit_code": md_code,
        "sentinel_is_excluded_from_primary": True,
    }
    state["status"] = (
        "completed"
        if counts.get("completed", 0) == len(state["systems"])
        else "completed_with_rejections_or_failures"
    )
    state["finished_at"] = _now()
    _save_state(state_path, state)
    print(json.dumps({"status": state["status"], "counts": counts}, sort_keys=True))
    if primary_code != 0 or md_code != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
