#!/usr/bin/env python3
"""Run eBDIMS2 on BINDRAE apo/holo samples.

This is a lightweight external-baseline runner. eBDIMS2 is a CPU/OpenMP
endpoint-conditioned CA-path method, not a neural model, so the reproducible
unit is the binary plus input PDBs and run parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scripts.ca_baseline_common import canonical_ca_pair, write_ca_only_pdb
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from ca_baseline_common import canonical_ca_pair, write_ca_only_pdb


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

CONV_RE = re.compile(r"DIMS conv:\s*([0-9.+\-Ee]+)%")
FRAME_RE = re.compile(r"DIMS_MD(\d+)\.pdb$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eBDIMS2 on BINDRAE samples")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--sample_file", required=True)
    parser.add_argument("--ebdims2_bin", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_output", default=None)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--candidate_scan_limit", type=int, default=512)
    parser.add_argument(
        "--selection",
        choices=["manifest_order", "ca_rmsd_desc"],
        default="ca_rmsd_desc",
    )
    parser.add_argument("--min_residues", type=int, default=50)
    parser.add_argument("--max_residues", type=int, default=512)
    parser.add_argument("--min_ca_rmsd", type=float, default=0.0)
    parser.add_argument("--max_ca_rmsd", type=float, default=1.0e6)
    parser.add_argument("--same_chain_only", action="store_true")
    parser.add_argument("--save_freq", type=int, default=25)
    parser.add_argument("--convergence", type=float, default=99.9)
    parser.add_argument("--timeout_sec", type=int, default=1800)
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
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


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - a.mean(axis=0, keepdims=True)
    bb = b - b.mean(axis=0, keepdims=True)
    cov = aa.T @ bb
    v, _, wt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1.0, 1.0, sign]) @ wt
    aligned = aa @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - bb) ** 2, axis=1))))


def one_chain(counts: Dict[str, int]) -> Optional[str]:
    nonempty = [chain for chain, n in counts.items() if n > 0]
    if len(nonempty) != 1:
        return None
    return nonempty[0]


def sample_info(data_dir: Path, sample_id: str, same_chain_only: bool) -> Dict[str, object]:
    sample_dir = data_dir / "samples" / sample_id
    apo_pdb = sample_dir / "apo.pdb"
    holo_pdb = sample_dir / "holo.pdb"
    apo = parse_ca_records(apo_pdb)
    holo = parse_ca_records(holo_pdb)
    if not apo or not holo:
        return {"sample_id": sample_id, "eligible": False, "reason": "missing_ca"}
    apo_counts = chain_counts(apo)
    holo_counts = chain_counts(holo)
    apo_chain = one_chain(apo_counts)
    holo_chain = one_chain(holo_counts)
    if apo_chain is None or holo_chain is None:
        return {
            "sample_id": sample_id,
            "eligible": False,
            "reason": "not_single_chain",
            "apo_chain_counts": apo_counts,
            "holo_chain_counts": holo_counts,
        }
    if same_chain_only and apo_chain != holo_chain:
        return {
            "sample_id": sample_id,
            "eligible": False,
            "reason": "chain_mismatch",
            "apo_chain": apo_chain,
            "holo_chain": holo_chain,
        }
    n = min(len(apo), len(holo))
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
        "n_ca": n,
        "ca_rmsd": kabsch_rmsd(apo_xyz, holo_xyz),
        "ca_raw_rmsd": raw_rmsd,
        "coordinate_source": coordinate_source,
    }


def frame_sort_key(path: Path) -> Tuple[int, str]:
    match = FRAME_RE.search(path.name)
    return (int(match.group(1)) if match else -1, path.name)


def list_frames(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("DIMS_MD*.pdb"), key=frame_sort_key)


def clean_run_dir(run_dir: Path) -> None:
    patterns = [
        "DIMS_MD*.pdb",
        "log.txt",
        "log_time.txt",
        "eBDIMS_confrms_list.txt",
        "apo_ATOM.pdb",
        "apo_CA.pdb",
        "holo_ATOM.pdb",
        "holo_CA.pdb",
    ]
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            path.unlink()


def parse_last_convergence(run_dir: Path) -> Optional[float]:
    candidates = [run_dir / "log.txt"]
    values: List[float] = []
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            match = CONV_RE.search(line)
            if match:
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    pass
    return values[-1] if values else None


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def select_candidates(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    data_dir = Path(args.data_dir)
    ids = read_sample_ids(Path(args.sample_file))[: int(args.candidate_scan_limit)]
    ids = ids[int(args.shard_index) :: int(args.num_shards)]
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
    if args.selection == "ca_rmsd_desc":
        eligible.sort(key=lambda row: float(row["ca_rmsd"]), reverse=True)
    return eligible[: int(args.max_samples)], rejected


def run_one(args: argparse.Namespace, info: Dict[str, object]) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    run_dir = output_dir / str(info["sample_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_run_dir(run_dir)
    apo_records = parse_ca_records(Path(str(info["apo_pdb"])), str(info["apo_chain"]))
    holo_records = parse_ca_records(Path(str(info["holo_pdb"])), str(info["holo_chain"]))
    apo_ca, holo_ca, coordinate_source = canonical_ca_pair(
        Path(str(info["sample_dir"])), apo_records, holo_records
    )
    write_ca_only_pdb(run_dir / "apo.pdb", apo_records, apo_ca)
    write_ca_only_pdb(run_dir / "holo.pdb", holo_records, holo_ca)

    existing_frames = list_frames(run_dir)
    if args.skip_existing and len(existing_frames) >= int(args.min_frames):
        return {
            **info,
            "status": "skipped_existing",
            "run_dir": str(run_dir),
            "frames": len(existing_frames),
            "last_convergence": parse_last_convergence(run_dir),
        }

    if args.dry_run:
        return {**info, "status": "dry_run", "run_dir": str(run_dir), "frames": len(existing_frames)}

    cmd = [
        str(Path(args.ebdims2_bin).resolve()),
        "apo",
        str(info["apo_chain"]),
        "holo",
        str(info["holo_chain"]),
        str(int(args.save_freq)),
        str(float(args.convergence)),
    ]
    start = time.time()
    timed_out = False
    returncode: Optional[int] = None
    stdout = ""
    stderr = ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(args.timeout_sec),
            check=False,
        )
        returncode = int(proc.returncode)
        stdout = proc.stdout[-8000:]
        stderr = proc.stderr[-8000:]
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else ""
    wall_sec = time.time() - start

    frames = list_frames(run_dir)
    last_conv = parse_last_convergence(run_dir)
    if timed_out:
        status = "timeout_partial" if len(frames) >= int(args.min_frames) else "timeout_no_frames"
    elif returncode == 0 and len(frames) >= int(args.min_frames):
        status = "success"
    elif len(frames) >= int(args.min_frames):
        status = "partial_success"
    elif returncode == 0:
        status = "success_too_few_frames"
    else:
        status = "failed"

    (run_dir / "runner_stdout_tail.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "runner_stderr_tail.txt").write_text(stderr, encoding="utf-8")
    return {
        **info,
        "source_apo_pdb": str(info["apo_pdb"]),
        "source_holo_pdb": str(info["holo_pdb"]),
        "apo_pdb": str(run_dir / "apo.pdb"),
        "holo_pdb": str(run_dir / "holo.pdb"),
        "coordinate_source": coordinate_source,
        "status": status,
        "run_dir": str(run_dir),
        "frames": len(frames),
        "first_frame": frames[0].name if frames else None,
        "last_frame": frames[-1].name if frames else None,
        "last_convergence": last_conv,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_sec": wall_sec,
        "cmd": cmd,
    }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"shard_index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
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
        "ebdims2_bin": args.ebdims2_bin,
        "output_dir": str(output_dir),
        "manifest_output": str(manifest_path),
        "requested_max_samples": int(args.max_samples),
        "candidate_scan_limit": int(args.candidate_scan_limit),
        "selected": len(selected),
        "rejected": len(rejected),
        "status_counts": status_counts,
        "save_freq": int(args.save_freq),
        "convergence": float(args.convergence),
        "timeout_sec": int(args.timeout_sec),
        "min_frames": int(args.min_frames),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
