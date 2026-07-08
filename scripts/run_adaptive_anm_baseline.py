#!/usr/bin/env python3
"""Run an adaptive CA-ANM projection baseline on BINDRAE apo/holo samples."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_anm_baseline import (  # noqa: E402
    append_jsonl,
    build_anm_hessian,
    clean_run_dir,
    parse_ca_records,
    select_candidates,
    write_ca_pdb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive CA-ANM projection baseline")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--sample_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_output", default=None)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--max_samples", type=int, default=10_000)
    parser.add_argument("--candidate_scan_limit", type=int, default=10_000)
    parser.add_argument("--min_residues", type=int, default=50)
    parser.add_argument("--max_residues", type=int, default=512)
    parser.add_argument("--min_ca_rmsd", type=float, default=0.0)
    parser.add_argument("--max_ca_rmsd", type=float, default=1.0e6)
    parser.add_argument("--same_chain_only", action="store_true")
    parser.add_argument("--cutoff", type=float, default=15.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--n_modes", type=int, default=20)
    parser.add_argument("--mode_extra", type=int, default=24)
    parser.add_argument("--eig_min", type=float, default=1.0e-6)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument(
        "--step_mode",
        choices=["fraction", "rescale_to_linear_rmsd"],
        default="fraction",
        help="fraction advances by projected displacement / remaining steps; rescale keeps low-mode direction but matches linear RMSD step size.",
    )
    parser.add_argument("--max_rescale", type=float, default=5.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def low_mode_project(
    coords: np.ndarray,
    target_disp: np.ndarray,
    n_modes: int,
    cutoff: float,
    gamma: float,
    mode_extra: int,
    eig_min: float,
) -> Tuple[np.ndarray, int, float, float]:
    hessian = build_anm_hessian(coords, cutoff=cutoff, gamma=gamma)
    n_dof = hessian.shape[0]
    upper = min(n_dof - 1, 6 + int(n_modes) + int(mode_extra))
    eigvals, eigvecs = scipy.linalg.eigh(hessian, subset_by_index=[0, upper], check_finite=False)
    keep = np.where(eigvals > float(eig_min))[0]
    if keep.size == 0:
        raise ValueError("no nonzero ANM modes")
    keep = keep[: int(n_modes)]
    modes = eigvecs[:, keep]
    target = target_disp.reshape(-1)
    coeff = modes.T @ target
    projected = (modes @ coeff).reshape(coords.shape)
    captured = float(np.sum(coeff * coeff) / max(float(np.sum(target * target)), 1.0e-12))
    return projected, int(keep.size), float(eigvals[keep[0]]), captured


def rmsd_vector(vec: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(vec * vec, axis=-1))))


def adaptive_anm_path(
    apo: np.ndarray,
    holo: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    current = apo.astype(np.float64).copy()
    frames: List[np.ndarray] = []
    captures: List[float] = []
    used_modes: List[int] = []
    first_eigs: List[float] = []

    for frame_idx in range(int(args.n_frames)):
        remaining_frames = int(args.n_frames) - frame_idx
        remaining = holo - current
        projected, n_used, first_eig, captured = low_mode_project(
            current,
            remaining,
            n_modes=int(args.n_modes),
            cutoff=float(args.cutoff),
            gamma=float(args.gamma),
            mode_extra=int(args.mode_extra),
            eig_min=float(args.eig_min),
        )
        captures.append(captured)
        used_modes.append(n_used)
        first_eigs.append(first_eig)

        if args.step_mode == "fraction":
            step = projected / max(float(remaining_frames), 1.0)
        else:
            target_step_rmsd = rmsd_vector(remaining) / max(float(remaining_frames), 1.0)
            projected_rmsd = rmsd_vector(projected)
            if projected_rmsd <= 1.0e-8:
                step = np.zeros_like(projected)
            else:
                scale = min(float(args.max_rescale), target_step_rmsd / projected_rmsd)
                step = projected * scale

        current = current + step
        frames.append(current.copy())

    return frames, {
        "mean_projected_motion_fraction": float(np.mean(captures)) if captures else math.nan,
        "min_projected_motion_fraction": float(np.min(captures)) if captures else math.nan,
        "mean_used_modes": float(np.mean(used_modes)) if used_modes else math.nan,
        "first_nonzero_eig_mean": float(np.mean(first_eigs)) if first_eigs else math.nan,
        "final_ca_rmsd_to_holo": rmsd_vector(holo - current),
    }


def run_one(args: argparse.Namespace, info: Dict[str, object]) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    run_dir = output_dir / str(info["sample_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_run_dir(run_dir)
    shutil.copy2(str(info["apo_pdb"]), run_dir / "apo.pdb")
    shutil.copy2(str(info["holo_pdb"]), run_dir / "holo.pdb")

    existing = sorted(run_dir.glob("DIMS_MD*.pdb"))
    if args.skip_existing and len(existing) >= int(args.n_frames):
        return {**info, "status": "skipped_existing", "run_dir": str(run_dir), "frames": len(existing)}

    start = time.time()
    try:
        apo_records = parse_ca_records(Path(str(info["apo_pdb"])), str(info["apo_chain"]))
        holo_records = parse_ca_records(Path(str(info["holo_pdb"])), str(info["holo_chain"]))
        apo = np.stack([rec["xyz"] for rec in apo_records], axis=0)
        holo = np.stack([rec["xyz"] for rec in holo_records], axis=0)
        frames_xyz, diag = adaptive_anm_path(apo, holo, args)
        for frame_idx, coords in enumerate(frames_xyz, start=1):
            write_ca_pdb(run_dir / f"DIMS_MD{frame_idx:04d}.pdb", apo_records, coords)
        frames = sorted(run_dir.glob("DIMS_MD*.pdb"))
        status = "success" if len(frames) >= int(args.n_frames) else "success_too_few_frames"
        error = None
    except Exception as exc:  # noqa: BLE001 - manifest should record per-sample failures.
        frames = sorted(run_dir.glob("DIMS_MD*.pdb"))
        diag = {
            "mean_projected_motion_fraction": math.nan,
            "min_projected_motion_fraction": math.nan,
            "mean_used_modes": math.nan,
            "first_nonzero_eig_mean": math.nan,
            "final_ca_rmsd_to_holo": math.nan,
        }
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    wall_sec = time.time() - start

    return {
        **info,
        "status": status,
        "run_dir": str(run_dir),
        "frames": len(frames),
        "first_frame": frames[0].name if frames else None,
        "last_frame": frames[-1].name if frames else None,
        "cutoff": float(args.cutoff),
        "gamma": float(args.gamma),
        "n_modes": int(args.n_modes),
        "step_mode": str(args.step_mode),
        "wall_sec": wall_sec,
        "error": error,
        **diag,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_output) if args.manifest_output else output_dir / "run_manifest.jsonl"
    summary_path = Path(args.summary_output) if args.summary_output else output_dir / "summary.json"

    selected, rejected = select_candidates(args)
    if manifest_path.exists():
        manifest_path.unlink()

    rows: List[Dict[str, object]] = []
    for info in selected:
        row = run_one(args, info)
        rows.append(row)
        append_jsonl(manifest_path, row)
        print(json.dumps(row, sort_keys=True), flush=True)

    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "data_dir": args.data_dir,
        "sample_file": args.sample_file,
        "output_dir": str(output_dir),
        "manifest_output": str(manifest_path),
        "requested_max_samples": int(args.max_samples),
        "candidate_scan_limit": int(args.candidate_scan_limit),
        "selected": len(selected),
        "rejected": len(rejected),
        "status_counts": status_counts,
        "cutoff": float(args.cutoff),
        "gamma": float(args.gamma),
        "n_modes": int(args.n_modes),
        "n_frames": int(args.n_frames),
        "step_mode": str(args.step_mode),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
