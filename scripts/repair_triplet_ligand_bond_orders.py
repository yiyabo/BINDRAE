#!/usr/bin/env python3
"""Repair connectivity-only ligand SDFs in existing AHoJ triplet directories.

``prepare_ahojdb_triplets.py`` built ``ligand.sdf`` with
``Chem.MolFromPDBBlock``, and the PDB format carries no bond orders.  Every bond
was therefore written as order 1 with no ``M  CHG`` record.  Polyphosphate
ligands become illegal-valence phosphorus that RDKit completes with a hydride,
which OpenFF then parameterises as a neutral ``P-H`` species; the solvated
system reaches a non-finite initial energy and MD setup aborts.  Non-phosphate
ligands do not abort but are silently simulated as fully saturated molecules.

This script rebuilds bond orders and formal charges from the PDB Chemical
Component Dictionary, preserving observed atom order and coordinates so
``ligand_coords.npy`` and every index-aligned downstream tensor stay valid.

Two modes:

  prefetch  Download the CCD reference SDFs for every resname in scope.  Run
            this where there is outbound network (login node or laptop); the
            repair itself is fully offline.
  repair    Rewrite ``ligand.sdf`` in place, keeping the original alongside as
            ``ligand.legacy_connectivity_only.sdf``.

``repair`` is a dry run unless ``--apply`` is passed, and it never touches
``ligand_coords.npy`` -- it verifies against it instead.

Examples
--------
    # 1. On a host with network access
    python scripts/repair_triplet_ligand_bond_orders.py prefetch \
        --triplet-root processed_data/triplets \
        --ccd-dir processed_data/ccd_cache

    # 2. Dry run, then apply
    python scripts/repair_triplet_ligand_bond_orders.py repair \
        --triplet-root processed_data/triplets \
        --ccd-dir processed_data/ccd_cache \
        --report-dir logs/ligand_bond_order_repair/dryrun_v1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ligand_bond_orders import (  # noqa: E402
    SCHEMA_VERSION,
    LigandBondOrderError,
    ccd_cache_path,
    download_ccd_sdf,
    file_sha256,
    reconstruct_ligand_sdf,
)

REPORT_SCHEMA_VERSION = "bindrae_ligand_bond_order_repair_v1"
LEGACY_SDF_NAME = "ligand.legacy_connectivity_only.sdf"


# --------------------------------------------------------------------------
# Sample discovery
# --------------------------------------------------------------------------


def iter_sample_dirs(
    triplet_root: Path, sample_list: Path | None, limit: int | None,
    *, require_ligand: bool = True,
) -> list[Path]:
    """Enumerate sample directories.

    ``os.scandir`` is used rather than ``Path.iterdir`` plus ``is_dir``: on a
    shared parallel filesystem holding ~90k samples the latter costs one extra
    metadata round trip per entry, and the per-entry ``ligand.sdf`` existence
    check costs another. ``scandir`` answers the directory question from the
    entry it already fetched, and ``require_ligand`` lets callers that do not
    need the file (resname collection) skip the second check entirely.
    """

    if sample_list is not None:
        names = [
            line.strip()
            for line in Path(sample_list).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        directories = [
            Path(name) if Path(name).is_absolute() else triplet_root / name
            for name in names
        ]
    else:
        with os.scandir(triplet_root) as entries:
            directories = sorted(
                Path(entry.path) for entry in entries if entry.is_dir()
            )
    if require_ligand:
        directories = [path for path in directories if (path / "ligand.sdf").is_file()]
    if limit is not None:
        directories = directories[: int(limit)]
    return directories


def resname_candidates(sample_dir: Path) -> list[str]:
    """Resnames to try, most authoritative first.

    ``meta.json`` records the resname the exporter *intended*.  It is not always
    the residue that was actually written: the exporter falls back from the holo
    structure to the query structure, and some AHoJ entries name a ligand that
    resolves to a different component (for example a sample keyed ``NAI`` whose
    ``holo.pdb`` contains ``NAD``).  The HETATM records in ``holo.pdb`` are the
    fallback, and the reconstruction itself is the arbiter: a wrong resname
    fails the subgraph match rather than producing a wrong molecule.
    """

    candidates: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip().upper()
        if text and text not in candidates:
            candidates.append(text)

    meta_path = sample_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        add(meta.get("ligand_resname"))
        ligand = meta.get("ligand")
        if isinstance(ligand, dict):
            add(ligand.get("resname"))

    parts = sample_dir.name.split("-")
    if len(parts) >= 3:
        add(parts[-2])

    holo = sample_dir / "holo.pdb"
    if holo.is_file():
        seen: dict[str, int] = {}
        for line in holo.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("HETATM") or len(line) < 20:
                continue
            residue = line[17:20].strip().upper()
            if residue in {"HOH", "WAT", "DOD", "H2O"}:
                continue
            seen[residue] = seen.get(residue, 0) + 1
        for residue, _ in sorted(seen.items(), key=lambda row: -row[1]):
            add(residue)

    return candidates


def resname_candidate_tiers(sample_dir: Path) -> list[list[str]]:
    """Candidate resnames grouped by how strong the evidence for them is.

    Tier 0 is what names *this sample*: ``meta.json`` and the sample id.
    Tier 1 is HETATM residue names scraped from ``holo.pdb``, which exist only
    to rescue a stale or missing tier 0.

    The tiers must not be flattened.  AHoJ pairs an apo/holo entry with a
    *query* entry, and the two deposited structures routinely contain
    chemically related but distinct components -- 1t26 has NAI while its holo
    1t2d has NAD, 4rc7 has PL3 while 4rc8 has STE.  ``holo.pdb`` therefore
    contributes the holo component's name even when the ligand was extracted
    from the query structure, and a flat candidate list reports that as a
    chemistry ambiguity when it is only a provenance artifact.
    """

    named: list[str] = []
    meta_path = sample_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        for key in ("effective_ligand_resname", "ligand_resname"):
            value = str(meta.get(key) or "").strip().upper()
            if value and value not in named:
                named.append(value)
    parts = sample_dir.name.split("-")
    if len(parts) >= 3:
        value = parts[-2].strip().upper()
        if value and value not in named:
            named.append(value)

    fallback = [r for r in resname_candidates(sample_dir) if r not in named]
    return [tier for tier in (named, fallback) if tier]


# --------------------------------------------------------------------------
# Per-sample repair
# --------------------------------------------------------------------------


def residue_groups_from_holo(sample_dir: Path, sdf_path: Path) -> list[tuple[int, ...]] | None:
    """Recover per-residue atom groups for an already-written ligand.sdf.

    ``SDWriter`` drops the PDB residue information, so an oligosaccharide -- N
    component copies joined by glycosidic bonds -- reads back as one connected
    fragment N times the component size, and connectivity alone cannot split it.
    ``holo.pdb`` still carries the HETATM residue numbers, so the grouping is
    recovered by matching coordinates.

    Returns ``None`` unless *every* SDF atom matches exactly one HETATM record,
    so a frame mismatch degrades to fragment-based grouping instead of producing
    a silently wrong partition.
    """

    import numpy as np
    from rdkit import Chem

    holo = sample_dir / "holo.pdb"
    if not holo.is_file():
        return None
    molecule = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=False)
    if molecule is None or molecule.GetNumConformers() == 0:
        return None
    positions = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)

    records: dict[tuple[int, int, int], list[tuple[Any, ...]]] = {}
    for line in holo.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("HETATM") or len(line) < 54:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        key = (int(round(x * 10)), int(round(y * 10)), int(round(z * 10)))
        residue = (
            line[21:22].strip(),
            line[22:26].strip(),
            line[26:27].strip(),
            line[17:20].strip().upper(),
        )
        records.setdefault(key, []).append(residue)

    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, (x, y, z) in enumerate(positions):
        key = (int(round(x * 10)), int(round(y * 10)), int(round(z * 10)))
        hits = records.get(key)
        if not hits or len({tuple(hit) for hit in hits}) != 1:
            return None
        groups.setdefault(tuple(hits[0]), []).append(index)
    if sum(len(v) for v in groups.values()) != molecule.GetNumAtoms():
        return None
    return [tuple(sorted(indices)) for _, indices in sorted(groups.items())]


def verify_against_coords(sample_dir: Path, sdf_path: Path) -> dict[str, Any]:
    """Confirm the repaired SDF still matches the frozen ``ligand_coords.npy``."""

    import numpy as np
    from rdkit import Chem

    coords_path = sample_dir / "ligand_coords.npy"
    if not coords_path.is_file():
        return {"checked": False, "reason": "ligand_coords_missing"}
    reference = np.load(coords_path).astype(np.float64)
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    molecule = next((value for value in supplier if value is not None), None)
    if molecule is None:
        raise LigandBondOrderError(
            "repaired_sdf_unreadable", f"Could not re-read repaired SDF {sdf_path}"
        )
    molecule = Chem.RemoveAllHs(molecule)
    positions = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
    if positions.shape != reference.shape:
        raise LigandBondOrderError(
            "ligand_coords_shape_mismatch",
            f"Repaired SDF has {positions.shape} atoms but ligand_coords.npy has "
            f"{reference.shape}",
        )
    deviation = float(np.max(np.abs(positions - reference)))
    if deviation > 1e-3:
        raise LigandBondOrderError(
            "ligand_coords_drift",
            f"Repaired SDF drifts {deviation:.6f} A from ligand_coords.npy",
        )
    return {
        "checked": True,
        "atoms": int(reference.shape[0]),
        "max_abs_deviation_angstrom": deviation,
    }


def repair_sample(
    sample_dir: Path,
    *,
    ccd_dir: Path,
    apply_changes: bool,
    allow_download: bool,
    missing_bond_policy: str,
    min_observed_heavy_atom_fraction: float,
    keep_backup: bool,
) -> dict[str, Any]:
    ligand_sdf = sample_dir / "ligand.sdf"
    row: dict[str, Any] = {
        "sample_id": sample_dir.name,
        "sample_dir": str(sample_dir),
        "input_sdf_sha256": file_sha256(ligand_sdf),
    }

    tiers = resname_candidate_tiers(sample_dir)
    candidates = [name for tier in tiers for name in tier]
    if not candidates:
        row.update({"status": "rejected", "reason": "resname_unresolved"})
        return row

    # Every candidate is tried, not just the first that works.  Some components
    # are graph-identical and differ only in bond orders and charge -- NAI
    # (NADH, neutral, saturated nicotinamide) and NAD (NAD+, aromatic
    # pyridinium, charge +1) share the same 44-atom heavy graph -- so a
    # subgraph match cannot discriminate them and silently picking the first
    # would commit to a charge state the data does not support.
    # Oligosaccharides need residue grouping; everything else falls back to
    # connected fragments inside reconstruct_bond_orders().
    atom_groups = residue_groups_from_holo(sample_dir, ligand_sdf)
    row["residue_grouping"] = "holo_pdb" if atom_groups else "connected_fragments"

    staged = sample_dir / "ligand.bond_order_repair.tmp.sdf"

    def candidate_sdf(name: str) -> Path:
        return sample_dir / f"ligand.bond_order_repair.tmp.{name}.sdf"

    attempts: list[dict[str, str]] = []
    accepted: list[tuple[str, dict[str, Any]]] = []
    for tier_index, tier in enumerate(tiers):
        if accepted:
            break
        row["resolved_tier"] = tier_index
        for resname in tier:
            candidate_path = candidate_sdf(resname)
            try:
                record = reconstruct_ligand_sdf(
                    input_sdf=ligand_sdf,
                    resname=resname,
                    output_sdf=candidate_path,
                    ccd_dir=ccd_dir,
                    allow_download=allow_download,
                    missing_bond_policy=missing_bond_policy,
                    min_observed_heavy_atom_fraction=min_observed_heavy_atom_fraction,
                    atom_groups=atom_groups,
                )
                accepted.append((resname, record))
            except LigandBondOrderError as exc:
                candidate_path.unlink(missing_ok=True)
                attempts.append(
                    {"resname": resname, "reason": exc.reason, "detail": str(exc)}
                )

    if not accepted:
        row.update(
            {
                "status": "rejected",
                "reason": attempts[0]["reason"] if attempts else "unknown",
                "resname_candidates": candidates,
                "attempts": attempts,
            }
        )
        return row

    resolved_resname, record = accepted[0]
    row["resolved_resname"] = resolved_resname
    alternatives = [
        {
            "resname": name,
            "formal_charge": int(alt["after"]["formal_charge"]),
            "canonical_smiles": alt["after"]["canonical_smiles"],
        }
        for name, alt in accepted
    ]
    row["accepted_resnames"] = alternatives
    distinct_chemistry = {
        (row_["formal_charge"], row_["canonical_smiles"]) for row_ in alternatives
    }
    row["resname_ambiguous"] = len(distinct_chemistry) > 1
    if row["resname_ambiguous"]:
        # Do not guess a charge state: leave the sample untouched and surface it.
        for name, _ in accepted:
            candidate_sdf(name).unlink(missing_ok=True)
        row.update(
            {
                "status": "rejected",
                "reason": "resname_chemistry_ambiguous",
                "resname_candidates": candidates,
                "attempts": attempts,
            }
        )
        return row

    for name, _ in accepted:
        if name != resolved_resname:
            candidate_sdf(name).unlink(missing_ok=True)
    os.replace(candidate_sdf(resolved_resname), staged)

    try:
        row["coords_check"] = verify_against_coords(sample_dir, staged)
    except LigandBondOrderError as exc:
        staged.unlink(missing_ok=True)
        row.update({"status": "rejected", "reason": exc.reason, "detail": str(exc)})
        return row

    row.update(
        {
            "resname_candidates": candidates,
            "failed_resname_attempts": attempts,
            "ccd": record["ccd"],
            "reconstruction": record["reconstruction"],
            "before": record["before"],
            "after": record["after"],
            "repaired_sdf_sha256": record["output_sdf_sha256"],
            "defect_present_before": bool(record["before"]["sentinel_hydrides"]),
            "bond_orders_changed": int(record["reconstruction"]["bond_orders_changed"]),
        }
    )

    if not apply_changes:
        staged.unlink(missing_ok=True)
        row["status"] = "would_repair"
        return row

    if keep_backup:
        legacy = sample_dir / LEGACY_SDF_NAME
        if not legacy.exists():
            shutil.copy2(ligand_sdf, legacy)
        row["legacy_sdf"] = str(legacy)
    os.replace(staged, ligand_sdf)
    row["status"] = "repaired"
    return row


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def run_prefetch(args: argparse.Namespace) -> dict[str, Any]:
    sample_dirs = iter_sample_dirs(
        args.triplet_root, args.sample_list, args.limit, require_ligand=False
    )
    # Prefetch only needs the de-duplicated resname set, and a sample id encodes
    # it as <pdb>-<chain>-<RESNAME>-<num>. Parsing the directory name costs zero
    # I/O; meta.json is opened only for the minority whose name does not parse.
    # Scanning holo.pdb for the tier-1 fallback is skipped outright: it would
    # mean reading every structure file in the corpus for a handful of extra
    # components, which repair can fetch on demand instead.
    wanted: dict[str, list[str]] = {}
    unparsed = 0
    for sample_dir in sample_dirs:
        parts = sample_dir.name.split("-")
        resname = parts[-2].strip().upper() if len(parts) >= 3 else ""
        if not resname:
            unparsed += 1
            tiers = resname_candidate_tiers(sample_dir)
            for name in (tiers[0] if tiers else []):
                wanted.setdefault(name, []).append(sample_dir.name)
            continue
        wanted.setdefault(resname, []).append(sample_dir.name)
    print(
        f"  scanned {len(sample_dirs)} samples, {len(wanted)} distinct components, "
        f"{unparsed} needed meta.json",
        flush=True,
    )

    ordered = sorted(wanted)
    cached = [
        r for r in ordered
        if ccd_cache_path(r, args.ccd_dir).is_file()
        and ccd_cache_path(r, args.ccd_dir).stat().st_size > 0
    ]
    todo = [r for r in ordered if r not in set(cached)]
    print(f"  {len(cached)} already cached, {len(todo)} to fetch", flush=True)

    # Fetching is network bound, not CPU bound, so threads are the right tool
    # and the GIL is irrelevant here. The cache is published atomically, so
    # concurrent writers of the same component are safe.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    downloaded: list[str] = []
    failed: list[dict[str, str]] = []

    def fetch(resname: str) -> tuple[str, str | None, str]:
        try:
            download_ccd_sdf(resname, args.ccd_dir, timeout_seconds=args.timeout_seconds)
            return resname, None, ""
        except LigandBondOrderError as exc:
            return resname, exc.reason, str(exc)

    workers = max(1, int(args.workers))
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, r) for r in todo]
            for done, future in enumerate(as_completed(futures), start=1):
                resname, reason, detail = future.result()
                if reason is None:
                    downloaded.append(resname)
                else:
                    failed.append({"resname": resname, "reason": reason, "detail": detail})
                if done % 500 == 0 or done == len(todo):
                    print(
                        f"  {done}/{len(todo)} fetched "
                        f"({len(downloaded)} ok, {len(failed)} failed)",
                        flush=True,
                    )
    return {
        "mode": "prefetch",
        "ccd_dir": str(args.ccd_dir),
        "samples_scanned": len(sample_dirs),
        "resnames_requested": len(wanted),
        "downloaded": downloaded,
        "already_cached": cached,
        "failed": failed,
    }


def _repair_one(job: tuple[Path, dict[str, Any]]) -> dict[str, Any]:
    """Worker entry point. One bad sample must never stop the sweep."""

    sample_dir, options = job
    try:
        return repair_sample(sample_dir, **options)
    except Exception as exc:  # noqa: BLE001
        return {
            "sample_id": sample_dir.name,
            "sample_dir": str(sample_dir),
            "status": "error",
            "reason": type(exc).__name__,
            "detail": str(exc),
        }


def run_repair(args: argparse.Namespace) -> dict[str, Any]:
    sample_dirs = iter_sample_dirs(args.triplet_root, args.sample_list, args.limit)
    options = {
        "ccd_dir": args.ccd_dir,
        "apply_changes": args.apply,
        "allow_download": args.allow_download,
        "missing_bond_policy": args.missing_bond_policy,
        "min_observed_heavy_atom_fraction": args.min_observed_heavy_atom_fraction,
        "keep_backup": not args.no_backup,
    }
    jobs = [(d, options) for d in sample_dirs]
    rows: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        results = (_repair_one(job) for job in jobs)
    else:
        import multiprocessing

        pool = multiprocessing.Pool(processes=workers)
        results = pool.imap_unordered(_repair_one, jobs, chunksize=8)
    for index, row in enumerate(results, start=1):
        rows.append(row)
        if index % 500 == 0 or index == len(jobs):
            print(f"  {index}/{len(jobs)} processed", flush=True)
    if workers > 1:
        pool.close()
        pool.join()

    status_counts: dict[str, int] = {}
    reject_ledger: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        if row["status"] in {"rejected", "error"}:
            reason = str(row.get("reason", "unknown"))
            reject_ledger[reason] = reject_ledger.get(reason, 0) + 1

    succeeded = [row for row in rows if row["status"] in {"repaired", "would_repair"}]
    return {
        "mode": "repair",
        "applied": bool(args.apply),
        "samples_scanned": len(sample_dirs),
        "status_counts": dict(sorted(status_counts.items())),
        "reject_ledger": dict(sorted(reject_ledger.items(), key=lambda row: -row[1])),
        "defect_present_before": sum(
            1 for row in succeeded if row.get("defect_present_before")
        ),
        "bond_orders_changed_total": sum(
            int(row.get("bond_orders_changed", 0)) for row in succeeded
        ),
        "samples_with_bond_order_changes": sum(
            1 for row in succeeded if int(row.get("bond_orders_changed", 0)) > 0
        ),
        "partial_observation": sum(
            1
            for row in succeeded
            if row.get("reconstruction", {}).get("partial_observation")
        ),
        "resname_resolved_by_fallback": sum(
            1 for row in succeeded if row.get("failed_resname_attempts")
        ),
        "resname_chemistry_ambiguous": sum(
            1 for row in rows if row.get("resname_ambiguous")
        ),
        "rows": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["prefetch", "repair"])
    parser.add_argument("--triplet-root", type=Path, required=True)
    parser.add_argument("--ccd-dir", type=Path, required=True)
    parser.add_argument("--sample-list", type=Path, default=None,
                        help="Optional newline-delimited sample ids or directories")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--apply", action="store_true",
                        help="repair mode: actually rewrite ligand.sdf (default is a dry run)")
    parser.add_argument("--allow-download", action="store_true",
                        help="repair mode: permit CCD downloads instead of requiring a warm cache")
    parser.add_argument("--no-backup", action="store_true",
                        help=f"repair mode: do not keep the original as {LEGACY_SDF_NAME}")
    parser.add_argument("--missing-bond-policy", default="restore",
                        choices=["restore", "reject", "ignore"],
                        help="What to do with a CCD bond absent from the perceived graph")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for repair mode. Reconstruction is "
                             "CPU bound and per-sample independent, so this scales nearly "
                             "linearly; the CCD cache is shared read-only and published "
                             "atomically.")
    parser.add_argument("--min-observed-heavy-atom-fraction", type=float, default=0.80,
                        help="Per-copy coverage of the CCD component required to accept a "
                             "resname. Without it a small ligand matches as a subgraph of a "
                             "much larger component (G3H inside NAD). 0.80 matches the "
                             "existing APObind preflight threshold.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_prefetch(args) if args.mode == "prefetch" else run_repair(args)
    summary["schema_version"] = REPORT_SCHEMA_VERSION
    summary["reconstruction_schema_version"] = SCHEMA_VERSION

    rows = summary.pop("rows", None)
    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if rows is not None:
            (args.report_dir / "records.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
        print(f"\nReport written to {args.report_dir}")

    print("\n" + json.dumps(summary, indent=2, sort_keys=True))
    if args.mode == "repair":
        failures = summary["status_counts"].get("error", 0)
        return 1 if failures else 0
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
