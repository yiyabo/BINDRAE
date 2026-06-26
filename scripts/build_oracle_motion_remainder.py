#!/usr/bin/env python3
"""Build OracleMotion partial manifests and clean remainder sample lists.

This helper is intentionally separate from the exporter. The exporter should
hard-fail on bad ligand chemistry; this script prepares a clean continuation
list after such a failure without deleting any already exported cache files.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.export_oracle_motion_features import FEATURE_NAMES, SCHEMA_VERSION  # noqa: E402

try:
    from rdkit import Chem, RDLogger  # noqa: E402

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - cluster env should have RDKit
    Chem = None
    RDKIT_AVAILABLE = False


class LigandCheckTimeout(TimeoutError):
    """Raised when a single ligand check exceeds the configured timeout."""


def _raise_ligand_check_timeout(signum, frame):
    raise LigandCheckTimeout("ligand check timed out")


class ligand_check_timeout:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.previous_handler = None

    def __enter__(self):
        if (
            self.seconds <= 0
            or not hasattr(signal, "SIGALRM")
            or threading.current_thread() is not threading.main_thread()
        ):
            return self
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_ligand_check_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write manifest for a partial OracleMotion cache and build a clean remainder list."
    )
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--candidate_list", required=True)
    parser.add_argument("--partial_cache_dir", required=True)
    parser.add_argument("--partial_manifest_out", default=None)
    parser.add_argument("--remainder_out", required=True)
    parser.add_argument("--rejects_out", required=True)
    parser.add_argument("--merged_manifest_out", default=None)
    parser.add_argument(
        "--extra_cache_dir",
        action="append",
        default=[],
        help="Additional completed cache dirs to include in merged_manifest_out.",
    )
    parser.add_argument(
        "--skip_ligand_check",
        action="store_true",
        help="Only skip already completed samples; do not validate ligand SDFs.",
    )
    parser.add_argument(
        "--ligand_timeout_seconds",
        type=float,
        default=5.0,
        help="Per-sample wall-clock timeout for RDKit ligand validation.",
    )
    parser.add_argument(
        "--fast_manifest",
        action="store_true",
        help="Read only sample_id/n_residues from npz files; leave per-mask counts empty.",
    )
    return parser.parse_args()


def read_ids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_record(npz_path: Path, fast: bool = False) -> Optional[Dict]:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            sample_id = str(data["sample_id"].item())
            n_res = int(data["n_residues"].item())
            if fast:
                counts = {}
            else:
                counts = {
                    "contact_apo": int(np.asarray(data["contact_apo"]).sum()),
                    "contact_holo": int(np.asarray(data["contact_holo"]).sum()),
                    "dist_valid_mask": int(np.asarray(data["dist_valid_mask"]).sum()),
                    "formed_contact": int(np.asarray(data["formed_contact"]).sum()),
                    "motion_active": int(np.asarray(data["motion_active"]).sum()),
                    "pocket_mask": int(np.asarray(data["pocket_mask"]).sum()),
                    "released_contact": int(np.asarray(data["released_contact"]).sum()),
                    "valid_mask": int(np.asarray(data["node_mask"]).sum()),
                }
    except Exception as exc:
        print(f"[WARN] failed to read cache file {npz_path}: {exc}", file=sys.stderr)
        return None
    return {
        "sample_id": sample_id,
        "path": str(npz_path.resolve()),
        "relative_path": npz_path.name,
        "n_residues": n_res,
        "counts": counts,
    }


def merge_counts(records: Iterable[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        for key, value in record.get("counts", {}).items():
            counts[key] = counts.get(key, 0) + int(value)
    return counts


def load_cache_records(cache_dir: Path, fast: bool = False) -> List[Dict]:
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        records = []
        for record in manifest.get("records", []):
            sample_id = record.get("sample_id")
            path_raw = record.get("path") or record.get("relative_path")
            if not sample_id or not path_raw:
                continue
            path = Path(path_raw)
            if not path.is_absolute():
                path = cache_dir / path
            new_record = dict(record)
            new_record["path"] = str(path.resolve())
            new_record["relative_path"] = path.name
            records.append(new_record)
        return records

    records = []
    for npz_path in sorted(cache_dir.glob("*.npz")):
        record = load_record(npz_path, fast=fast)
        if record is not None:
            records.append(record)
    return records


def write_manifest(path: Path, records: List[Dict], args: argparse.Namespace, source_dirs: List[Path]) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": "holo_truth",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "feature_names": list(FEATURE_NAMES),
        "feature_dim": len(FEATURE_NAMES),
        "num_samples": len(records),
        "counts": merge_counts(records),
        "source_cache_dirs": [str(p.resolve()) for p in source_dirs],
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def ligand_is_loadable(data_dir: Path, sample_id: str, timeout_seconds: float = 5.0) -> Tuple[bool, str]:
    sample_dir = data_dir / "samples" / sample_id
    coords_path = sample_dir / "ligand_coords.npy"
    sdf_path = sample_dir / "ligand.sdf"
    if not coords_path.exists():
        return False, "missing ligand_coords.npy"
    if not sdf_path.exists():
        return False, "missing ligand.sdf"
    if not RDKIT_AVAILABLE:
        return False, "rdkit unavailable"
    try:
        coords = np.load(coords_path)
        with ligand_check_timeout(timeout_seconds):
            supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
            mol = supplier[0] if len(supplier) > 0 else None
            if mol is None:
                return False, "rdkit returned None"
            if mol.GetNumAtoms() != len(coords):
                return False, f"atom count mismatch sdf={mol.GetNumAtoms()} coords={len(coords)}"
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    partial_cache_dir = Path(args.partial_cache_dir)
    candidate_list = Path(args.candidate_list)
    remainder_out = Path(args.remainder_out)
    rejects_out = Path(args.rejects_out)

    candidate_ids = read_ids(candidate_list)
    partial_records = load_cache_records(partial_cache_dir, fast=args.fast_manifest)
    completed: Set[str] = {str(record["sample_id"]) for record in partial_records}

    if args.partial_manifest_out:
        write_manifest(Path(args.partial_manifest_out), partial_records, args, [partial_cache_dir])

    remainder: List[str] = []
    rejects: List[Tuple[str, str]] = []
    for sample_id in candidate_ids:
        if sample_id in completed:
            continue
        if args.skip_ligand_check:
            remainder.append(sample_id)
            continue
        ok, reason = ligand_is_loadable(data_dir, sample_id, args.ligand_timeout_seconds)
        if ok:
            remainder.append(sample_id)
        else:
            rejects.append((sample_id, reason))

    remainder_out.parent.mkdir(parents=True, exist_ok=True)
    remainder_out.write_text("\n".join(remainder) + ("\n" if remainder else ""))
    rejects_out.parent.mkdir(parents=True, exist_ok=True)
    rejects_out.write_text("\n".join(f"{sample_id}\t{reason}" for sample_id, reason in rejects) + ("\n" if rejects else ""))

    if args.merged_manifest_out:
        all_records = list(partial_records)
        source_dirs = [partial_cache_dir]
        seen = set(completed)
        for raw_dir in args.extra_cache_dir:
            cache_dir = Path(raw_dir)
            source_dirs.append(cache_dir)
            for record in load_cache_records(cache_dir, fast=args.fast_manifest):
                sample_id = str(record.get("sample_id"))
                if not sample_id or sample_id in seen:
                    continue
                seen.add(sample_id)
                all_records.append(record)
        write_manifest(Path(args.merged_manifest_out), all_records, args, source_dirs)

    print(
        json.dumps(
            {
                "candidate_samples": len(candidate_ids),
                "partial_records": len(partial_records),
                "completed_unique": len(completed),
                "remainder": len(remainder),
                "rejected": len(rejects),
                "partial_manifest": args.partial_manifest_out or "",
                "merged_manifest": args.merged_manifest_out or "",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
