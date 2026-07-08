#!/usr/bin/env python3
"""Evaluate CA-only path baselines against ligand-pocket CA distance metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


AA_MASS = {
    "ALA": 71.0,
    "ARG": 156.0,
    "ASN": 114.0,
    "ASP": 115.0,
    "CYS": 103.0,
    "GLU": 129.0,
    "GLH": 129.0,
    "GLN": 128.0,
    "GLY": 57.0,
    "HIS": 137.0,
    "HIE": 137.0,
    "HID": 137.0,
    "HIP": 137.0,
    "ILE": 113.0,
    "LEU": 113.0,
    "LYS": 128.0,
    "LYN": 128.0,
    "MET": 131.0,
    "PHE": 147.0,
    "PRO": 97.0,
    "SER": 87.0,
    "THR": 101.0,
    "TRP": 186.0,
    "TYR": 163.0,
    "VAL": 99.0,
}

FRAME_RE = re.compile(r"DIMS_MD(\d+)\.pdb$")


class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, values: np.ndarray, mask: np.ndarray) -> None:
        mask = mask.astype(bool)
        if values.ndim > mask.ndim:
            mask = np.expand_dims(mask, axis=-1)
        selected = values[mask]
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            return
        self.total += float(selected.sum())
        self.count += int(selected.size)

    def add_scalar(self, value: float, count: int = 1) -> None:
        if count <= 0 or not math.isfinite(value):
            return
        self.total += float(value) * int(count)
        self.count += int(count)

    def mean(self) -> Optional[float]:
        return self.total / self.count if self.count else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CA-only path baselines")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--run_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per_sample_output", default=None)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--n_path_frames", type=int, default=16)
    parser.add_argument("--active_delta", type=float, default=0.75)
    parser.add_argument("--contact_dist", type=float, default=4.5)
    parser.add_argument("--pocket_threshold", type=float, default=0.3)
    parser.add_argument("--path_dist_cap", type=float, default=20.0)
    parser.add_argument("--improvement_min_delta", type=float, default=0.25)
    parser.add_argument("--status_allow", default="success,partial_success,timeout_partial,success_too_few_frames")
    parser.add_argument("--method_prefix", default="ebdims2_ca")
    parser.add_argument("--frame_offset", choices=["apo_mass_com", "none"], default="apo_mass_com")
    parser.add_argument("--append_holo_endpoint", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_ca_records(path: Path, chain_filter: Optional[str] = None) -> Tuple[np.ndarray, List[str], np.ndarray]:
    coords: List[List[float]] = []
    resnames: List[str] = []
    chains = set(chain_filter) if chain_filter else None
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            chain = line[21].strip() or "_"
            if chains is not None and chain not in chains:
                continue
            try:
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
            except ValueError:
                continue
            resnames.append(line[17:20].strip())
    if not coords:
        return np.zeros((0, 3), dtype=np.float64), [], np.zeros((0,), dtype=np.float64)
    masses = np.array([AA_MASS.get(name, 100.0) for name in resnames], dtype=np.float64)
    return np.array(coords, dtype=np.float64), resnames, masses


def frame_sort_key(path: Path) -> Tuple[int, str]:
    match = FRAME_RE.search(path.name)
    return (int(match.group(1)) if match else -1, path.name)


def list_frames(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("DIMS_MD*.pdb"), key=frame_sort_key)


def resample_indices(n: int, target: int) -> np.ndarray:
    if n <= target:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, n - 1, target)).astype(np.int64))


def min_ligand_dist(ca: np.ndarray, ligand: np.ndarray) -> np.ndarray:
    if ligand.size == 0:
        return np.full((ca.shape[0],), np.inf, dtype=np.float64)
    diff = ca[:, None, :] - ligand[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=1)


def class_masks(
    w_res: np.ndarray,
    d_apo: np.ndarray,
    d_holo: np.ndarray,
    pocket_threshold: float,
    active_delta: float,
    contact_dist: float,
) -> Dict[str, np.ndarray]:
    pocket = w_res > float(pocket_threshold)
    finite = pocket & np.isfinite(d_apo) & np.isfinite(d_holo) & (d_apo < 49.0) & (d_holo < 49.0)
    delta = d_holo - d_apo
    active = finite & (np.abs(delta) >= float(active_delta))
    return {
        "all_pocket": finite,
        "active": active,
        "approach": active & (delta < 0),
        "release": active & (delta > 0),
        "formed_contact": active & (d_apo > float(contact_dist)) & (d_holo <= float(contact_dist)),
        "released_contact": active & (d_apo <= float(contact_dist)) & (d_holo > float(contact_dist)),
        "stable_contact": finite & (d_apo <= float(contact_dist)) & (d_holo <= float(contact_dist)),
        "stable_noncontact": finite & (d_apo > float(contact_dist)) & (d_holo > float(contact_dist)),
    }


def add_direction(stats: Dict[str, RunningMean], key: str, actual: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
    valid = mask & np.isfinite(actual) & np.isfinite(target) & (np.abs(target) > 1.0e-6)
    if not valid.any():
        return
    ok = (actual[valid] * target[valid]) > 0.0
    stats[key].add_scalar(float(ok.mean()), int(ok.size))


def summarize(stats: Dict[str, RunningMean]) -> Dict[str, Optional[float]]:
    return {key: value.mean() for key, value in sorted(stats.items())}


def align_lengths(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = min(arr.shape[0] for arr in arrays)
    return tuple(arr[:n] for arr in arrays)


def load_ligand(sample_dir: Path) -> np.ndarray:
    path = sample_dir / "ligand_coords.npy"
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float64)
    ligand = np.load(path).astype(np.float64)
    ligand = ligand.reshape(-1, 3)
    return ligand[np.isfinite(ligand).all(axis=1)]


def load_w_res(sample_dir: Path, n: int) -> np.ndarray:
    path = sample_dir / "w_res.npy"
    if not path.exists():
        return np.ones((n,), dtype=np.float64)
    w_res = np.load(path).astype(np.float64).reshape(-1)
    if w_res.shape[0] >= n:
        return w_res[:n]
    out = np.zeros((n,), dtype=np.float64)
    out[: w_res.shape[0]] = w_res
    return out


def evaluate_one(args: argparse.Namespace, row: Dict[str, object], stats: Dict[str, RunningMean]) -> Optional[Dict[str, object]]:
    sample_id = str(row["sample_id"])
    sample_dir = Path(args.data_dir) / "samples" / sample_id
    run_dir = Path(str(row["run_dir"]))
    frames = list_frames(run_dir)
    if len(frames) < int(args.min_frames):
        return None

    apo_chain = str(row.get("apo_chain") or "")
    holo_chain = str(row.get("holo_chain") or "")
    apo_ca, _, masses = parse_ca_records(sample_dir / "apo.pdb", apo_chain)
    holo_ca, _, _ = parse_ca_records(sample_dir / "holo.pdb", holo_chain)
    if apo_ca.size == 0 or holo_ca.size == 0:
        return None
    n = min(apo_ca.shape[0], holo_ca.shape[0], masses.shape[0])
    apo_ca = apo_ca[:n]
    holo_ca = holo_ca[:n]
    masses = masses[:n]
    mass_com = (apo_ca * masses[:, None]).sum(axis=0) / masses.sum()
    frame_offset = mass_com if args.frame_offset == "apo_mass_com" else np.zeros((3,), dtype=np.float64)

    indices = resample_indices(len(frames), int(args.n_path_frames))
    path: List[np.ndarray] = [apo_ca]
    for idx in indices:
        frame_ca, _, _ = parse_ca_records(frames[int(idx)], apo_chain)
        if frame_ca.shape[0] < n:
            continue
        path.append(frame_ca[:n] + frame_offset)
    if args.append_holo_endpoint:
        path.append(holo_ca)
    if len(path) < 2:
        return None

    ligand = load_ligand(sample_dir)
    w_res = load_w_res(sample_dir, n)
    d_apo = min_ligand_dist(apo_ca, ligand)
    d_holo = min_ligand_dist(holo_ca, ligand)
    path_dists = [min_ligand_dist(frame, ligand) for frame in path]
    masks = class_masks(
        w_res,
        d_apo,
        d_holo,
        float(args.pocket_threshold),
        float(args.active_delta),
        float(args.contact_dist),
    )

    cap = float(args.path_dist_cap)
    d_final = path_dists[-1]
    delta_target = d_holo - d_apo
    delta_actual = d_final - d_apo
    endpoint_abs = np.abs(np.minimum(d_final, cap) - np.minimum(d_holo, cap))
    initial_abs = np.abs(np.minimum(d_apo, cap) - np.minimum(d_holo, cap))
    improvement = np.full_like(initial_abs, np.nan)
    improve_mask = initial_abs >= float(args.improvement_min_delta)
    improvement[improve_mask] = (
        initial_abs[improve_mask] - endpoint_abs[improve_mask]
    ) / np.maximum(initial_abs[improve_mask], 1.0e-6)
    path_mae = np.zeros_like(d_apo)
    for t_val, d_t in zip(np.linspace(0.0, 1.0, len(path)), path_dists):
        target_t = d_apo + float(t_val) * delta_target
        path_mae += np.abs(np.minimum(d_t, cap) - np.minimum(target_t, cap))
    path_mae /= max(len(path), 1)
    ca_endpoint_err = np.linalg.norm(path[-1] - holo_ca, axis=1)
    step_sizes = [
        np.linalg.norm(path[i + 1] - path[i], axis=1)
        for i in range(len(path) - 1)
    ]
    ca_step = np.stack(step_sizes, axis=0).mean(axis=0) if step_sizes else np.zeros((n,), dtype=np.float64)
    pred_contact = d_final <= float(args.contact_dist)

    prefix = str(args.method_prefix).rstrip("/") + "/"
    for name, mask in masks.items():
        stats[prefix + f"{name}/endpoint_abs_dist_A"].add(endpoint_abs, mask)
        stats[prefix + f"{name}/path_mae_dist_A"].add(path_mae, mask)
        stats[prefix + f"{name}/improvement_to_holo"].add(improvement, mask)
        stats[prefix + f"{name}/ca_endpoint_err_A"].add(ca_endpoint_err, mask)
        stats[prefix + f"{name}/mean_ca_step_A"].add(ca_step, mask)
        add_direction(stats, prefix + f"{name}/direction_acc", delta_actual, delta_target, mask)
        if name == "formed_contact":
            stats[prefix + "formed_contact/recall"].add(pred_contact.astype(np.float64), mask)
        if name == "released_contact":
            stats[prefix + "released_contact/release_success"].add((~pred_contact).astype(np.float64), mask)
        if name == "stable_contact":
            stats[prefix + "stable_contact/retention"].add(pred_contact.astype(np.float64), mask)

    return {
        "sample_id": sample_id,
        "run_dir": str(run_dir),
        "frames": len(frames),
        "used_path_frames": len(path),
        "last_convergence": row.get("last_convergence"),
        "class_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "all_pocket_endpoint_abs_dist_A": float(endpoint_abs[masks["all_pocket"]].mean()) if masks["all_pocket"].any() else None,
        "active_direction_acc": (
            float(((delta_actual[masks["active"]] * delta_target[masks["active"]]) > 0).mean())
            if masks["active"].any()
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    allowed = {item.strip() for item in str(args.status_allow).split(",") if item.strip()}
    rows = [
        row
        for row in read_jsonl(Path(args.run_manifest))
        if str(row.get("status")) in allowed
    ]
    stats: Dict[str, RunningMean] = defaultdict(RunningMean)
    per_sample: List[Dict[str, object]] = []
    skipped = 0
    for row in rows:
        result = evaluate_one(args, row, stats)
        if result is None:
            skipped += 1
        else:
            per_sample.append(result)

    summary = {
        "run_manifest": args.run_manifest,
        "data_dir": args.data_dir,
        "samples": len(per_sample),
        "skipped": skipped,
        "min_frames": int(args.min_frames),
        "n_path_frames": int(args.n_path_frames),
        "improvement_min_delta": float(args.improvement_min_delta),
        "method_prefix": str(args.method_prefix).rstrip("/"),
        "frame_offset": str(args.frame_offset),
        "append_holo_endpoint": bool(args.append_holo_endpoint),
        **summarize(stats),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.per_sample_output:
        per_path = Path(args.per_sample_output)
        per_path.parent.mkdir(parents=True, exist_ok=True)
        with per_path.open("w", encoding="utf-8") as handle:
            for row in per_sample:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
