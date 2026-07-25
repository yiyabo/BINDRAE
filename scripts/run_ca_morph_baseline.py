#!/usr/bin/env python3
"""Run simple CA morphing baselines on BINDRAE apo/holo samples."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scripts.ca_baseline_common import canonical_ca_pair
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from ca_baseline_common import canonical_ca_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple CA morphing baselines")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--sample_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_output", default=None)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--method", choices=["static", "linear", "smoothstep"], required=True)
    parser.add_argument("--max_samples", type=int, default=10_000)
    parser.add_argument("--candidate_scan_limit", type=int, default=10_000)
    parser.add_argument("--min_residues", type=int, default=50)
    parser.add_argument("--max_residues", type=int, default=512)
    parser.add_argument("--min_ca_rmsd", type=float, default=0.0)
    parser.add_argument("--max_ca_rmsd", type=float, default=1.0e6)
    parser.add_argument("--same_chain_only", action="store_true")
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def read_sample_ids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_ca_records(path: Path, chain_filter: Optional[str] = None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not path.exists():
        return records
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
                xyz = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=np.float64,
                )
            except ValueError:
                continue
            records.append(
                {
                    "chain": chain,
                    "resseq": line[22:26].strip(),
                    "icode": line[26].strip(),
                    "resname": line[17:20].strip(),
                    "xyz": xyz,
                }
            )
    return records


def chain_counts(records: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in records:
        chain = str(rec["chain"])
        counts[chain] = counts.get(chain, 0) + 1
    return counts


def one_chain(counts: Dict[str, int]) -> Optional[str]:
    nonempty = [chain for chain, n in counts.items() if n > 0]
    return nonempty[0] if len(nonempty) == 1 else None


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - a.mean(axis=0, keepdims=True)
    bb = b - b.mean(axis=0, keepdims=True)
    cov = aa.T @ bb
    v, _, wt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1.0, 1.0, sign]) @ wt
    aligned = aa @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - bb) ** 2, axis=1))))


def sample_info(data_dir: Path, sample_id: str, same_chain_only: bool) -> Dict[str, object]:
    sample_dir = data_dir / "samples" / sample_id
    apo_pdb = sample_dir / "apo.pdb"
    holo_pdb = sample_dir / "holo.pdb"
    apo = parse_ca_records(apo_pdb)
    holo = parse_ca_records(holo_pdb)
    if not apo or not holo:
        return {"sample_id": sample_id, "eligible": False, "reason": "missing_ca"}
    apo_chain = one_chain(chain_counts(apo))
    holo_chain = one_chain(chain_counts(holo))
    if apo_chain is None or holo_chain is None:
        return {"sample_id": sample_id, "eligible": False, "reason": "not_single_chain"}
    if same_chain_only and apo_chain != holo_chain:
        return {
            "sample_id": sample_id,
            "eligible": False,
            "reason": "chain_mismatch",
            "apo_chain": apo_chain,
            "holo_chain": holo_chain,
        }
    if len(apo) != len(holo):
        return {
            "sample_id": sample_id,
            "eligible": False,
            "reason": "ca_count_mismatch",
            "apo_n_ca": len(apo),
            "holo_n_ca": len(holo),
            "apo_chain": apo_chain,
            "holo_chain": holo_chain,
        }
    apo_xyz, holo_xyz, coordinate_source = canonical_ca_pair(sample_dir, apo, holo)
    raw_rmsd = float(np.sqrt(np.mean(np.sum((apo_xyz - holo_xyz) ** 2, axis=1))))
    return {
        "sample_id": sample_id,
        "eligible": True,
        "sample_dir": str(sample_dir),
        "apo_pdb": str(apo_pdb),
        "holo_pdb": str(holo_pdb),
        "apo_chain": apo_chain,
        "holo_chain": holo_chain,
        "n_ca": len(apo),
        "ca_rmsd": kabsch_rmsd(apo_xyz, holo_xyz),
        "ca_raw_rmsd": raw_rmsd,
        "coordinate_source": coordinate_source,
    }


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def select_candidates(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    data_dir = Path(args.data_dir)
    ids = read_sample_ids(Path(args.sample_file))[: int(args.candidate_scan_limit)]
    eligible: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    for sample_id in ids:
        info = sample_info(data_dir, sample_id, bool(args.same_chain_only))
        if not info.get("eligible"):
            rejected.append(info)
            continue
        n_ca = int(info["n_ca"])
        ca_rmsd = float(info["ca_rmsd"])
        if n_ca < int(args.min_residues) or n_ca > int(args.max_residues):
            info.update({"eligible": False, "reason": "residue_count_filter"})
            rejected.append(info)
            continue
        if ca_rmsd < float(args.min_ca_rmsd) or ca_rmsd > float(args.max_ca_rmsd):
            info.update({"eligible": False, "reason": "ca_rmsd_filter"})
            rejected.append(info)
            continue
        eligible.append(info)
    return eligible[: int(args.max_samples)], rejected


def write_ca_pdb(path: Path, template: Sequence[Dict[str, object]], coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for idx, (rec, xyz) in enumerate(zip(template, coords), start=1):
        chain = " " if str(rec["chain"]) == "_" else str(rec["chain"])[:1]
        resseq = str(rec["resseq"])[:4].rjust(4)
        icode = str(rec["icode"])[:1] or " "
        resname = str(rec["resname"])[:3].rjust(3)
        x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        lines.append(
            f"ATOM  {idx:5d}  CA  {resname} {chain}{resseq}{icode}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def interpolation_fraction(method: str, t_val: float) -> float:
    if method == "static":
        return 0.0
    if method == "linear":
        return float(t_val)
    if method == "smoothstep":
        return float(3.0 * t_val * t_val - 2.0 * t_val * t_val * t_val)
    raise ValueError(f"unsupported method: {method}")


def clean_run_dir(run_dir: Path) -> None:
    for pattern in ["DIMS_MD*.pdb", "apo.pdb", "holo.pdb"]:
        for path in run_dir.glob(pattern):
            path.unlink()


def run_one(args: argparse.Namespace, info: Dict[str, object]) -> Dict[str, object]:
    run_dir = Path(args.output_dir) / str(info["sample_id"])
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
        apo, holo, coordinate_source = canonical_ca_pair(
            Path(str(info["sample_dir"])), apo_records, holo_records
        )
        disp = holo - apo
        for frame_idx, t_val in enumerate(np.linspace(1.0 / int(args.n_frames), 1.0, int(args.n_frames)), start=1):
            frac = interpolation_fraction(str(args.method), float(t_val))
            write_ca_pdb(run_dir / f"DIMS_MD{frame_idx:04d}.pdb", apo_records, apo + frac * disp)
        frames = sorted(run_dir.glob("DIMS_MD*.pdb"))
        status = "success" if len(frames) >= int(args.n_frames) else "success_too_few_frames"
        error = None
    except Exception as exc:  # noqa: BLE001 - keep failures local to each sample.
        frames = sorted(run_dir.glob("DIMS_MD*.pdb"))
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    wall_sec = time.time() - start

    return {
        **info,
        "status": status,
        "method": str(args.method),
        "run_dir": str(run_dir),
        "frames": len(frames),
        "first_frame": frames[0].name if frames else None,
        "last_frame": frames[-1].name if frames else None,
        "wall_sec": wall_sec,
        "error": error,
        "coordinate_source": coordinate_source if status != "failed" else info.get("coordinate_source"),
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
        "method": str(args.method),
        "requested_max_samples": int(args.max_samples),
        "candidate_scan_limit": int(args.candidate_scan_limit),
        "selected": len(selected),
        "rejected": len(rejected),
        "status_counts": status_counts,
        "n_frames": int(args.n_frames),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
