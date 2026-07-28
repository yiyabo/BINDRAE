#!/usr/bin/env python3
"""Repair triplet samples whose ligand.sdf holds more than one component copy.

``extract_ligand_from_pdb`` selected HETATM records by chain and residue *name*
only, so every copy of the component anywhere in the chain was concatenated into
one "ligand".  41,599 of 91,189 samples are affected, with fragment centroids a
median of 72 A apart.  ``src/data/ligand_residue_selection`` fixes the generator;
this script fixes what the generator already wrote.

Why repair in place rather than regenerate
------------------------------------------
The AHoJ source is on disk, so regeneration is available -- it is just not worth
it.  Regenerating rewrites ``apo.pdb``, ``holo.pdb`` and ``esm.pt``, which means
an ESM pass over 91,189 samples to correct two files each.  This script touches
only ``ligand.sdf`` and ``ligand_coords.npy`` and leaves every other file
byte-identical.

How the correct atoms are located
---------------------------------
The written files do not record which copy the sample is named after: ``SDWriter``
drops PDB residue numbers.  So the deposited residue is re-extracted from
``ahojdb_v2c/pdb_files`` with the fixed selection, and matched back into the SDF.

The match does **not** need the alignment matrices.  ``apply_rt`` is a single
rigid transform and rigid transforms preserve intramolecular distances, so the
sorted vector of internal distances identifies the copy without knowing the
transform.  Validated on three samples: the true copy matched to 0.8-1.1e-4 A
while the closest wrong copy was 0.028-0.236 A away, a margin of 250x or better.

Skipping the matrices removes a failure mode rather than adding one: an entry
with missing or ambiguous alignment files is still repairable.

What is preserved
-----------------
The repair *subsets* the existing molecule rather than rebuilding it, so the CCD
bond-order reconstruction already applied to part of the corpus survives, and the
surviving atoms keep their original order.  ``ligand_coords.npy`` is sliced with
the same indices, so the two stay index-aligned.

Every write is gated on a mechanical check, not on judgement: unique geometric
match, coordinate agreement between the sliced array and the subset conformer,
and a strict re-parse of the written SDF.  A sample failing any gate is left
untouched and recorded with a reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - hosts without RDKit can still import
    # Import must stay soft: the path-resolution and ledger helpers below carry
    # their own tests, and a hard SystemExit here would break test collection on
    # any host without RDKit. main() refuses to run instead.
    Chem = None  # type: ignore[assignment]
    RDLogger = None  # type: ignore[assignment]
    RDKIT_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ligand_copy_identification import (  # noqa: E402
    CopyIdentificationError,
    identify_ligand_atom_indices,
)
from src.data.ligand_residue_selection import (  # noqa: E402
    LigandResidueNotFound,
    parse_residue_number,
)

SCHEMA_VERSION = "bindrae_ligand_extraction_repair_v1"

LOG = logging.getLogger("repair_triplet_ligand_extraction")

#: Kept beside the repaired file, mirroring the bond-order repair's convention.
BACKUP_SDF_NAME = "ligand.multi_copy_extraction.sdf"
BACKUP_COORDS_NAME = "ligand_coords.multi_copy_extraction.npy"

#: Ceiling on the disagreement between the sliced ``ligand_coords.npy`` and the
#: subset conformer.  Both derive from the same float32 source, so a genuine
#: match is at the 1e-5 level; 1e-3 leaves room without admitting a real error.
COORDINATE_AGREEMENT_TOLERANCE_ANGSTROM = 1.0e-3


def iter_sample_dirs(samples_root: Path) -> Iterator[Path]:
    """Yield sample directories that actually hold a ligand.

    ``os.scandir`` rather than ``Path.iterdir`` because the corpus has ~91k
    entries on a network filesystem, where the per-entry ``stat`` calls of the
    pathlib route dominate the runtime.
    """
    with os.scandir(samples_root) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            if not os.path.exists(os.path.join(entry.path, "ligand.sdf")):
                continue
            yield Path(entry.path)


def load_meta(sample_dir: Path) -> Dict:
    try:
        return json.loads((sample_dir / "meta.json").read_text())
    except Exception:
        return {}


def resolve_query_pdb(pdb_files_dir: Path, query_pdb: str) -> Optional[Path]:
    for name in (f"{query_pdb.lower()}.pdb", f"{query_pdb.upper()}.pdb",
                 f"{query_pdb.lower()}.ent", f"pdb{query_pdb.lower()}.ent"):
        candidate = pdb_files_dir / name
        if candidate.exists():
            return candidate
    return None


def subset_molecule(mol: Chem.Mol, keep_indices: Tuple[int, ...]) -> Chem.Mol:
    """Drop every atom outside ``keep_indices``, preserving order and coordinates.

    Removal is used rather than rebuilding so that bond orders and formal charges
    -- which for part of the corpus are the output of the CCD reconstruction --
    are carried through untouched.
    """
    keep = set(keep_indices)
    editable = Chem.RWMol(mol)
    editable.BeginBatchEdit()
    for index in range(mol.GetNumAtoms() - 1, -1, -1):
        if index not in keep:
            editable.RemoveAtom(index)
    editable.CommitBatchEdit()
    return editable.GetMol()


def conformer_positions(mol: Chem.Mol) -> np.ndarray:
    conformer = mol.GetConformer()
    return np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )


def _reject(sample_dir: Path, reason: str, detail: str, **extra) -> Dict:
    row = {
        "sample_id": sample_dir.name,
        "sample_dir": str(sample_dir),
        "status": "rejected",
        "reject_reason": reason,
        "detail": detail,
    }
    row.update(extra)
    return row


def ligand_ids_from_entry(entry_dir: Optional[Path], resname: str) -> List[Tuple[str, int]]:
    """``(chain, resnum)`` pairs for ``resname`` in the entry's ligands.json.

    The generator preferred the holo structure whenever ligands.json named a
    matching component, falling back to the query only otherwise.  A sample built
    on that preferred path holds the *holo* residue, which no query-side
    reference can reproduce, so those ids have to be tried too.
    """
    if entry_dir is None:
        return []
    path = entry_dir / "ligands.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []

    found: List[Tuple[str, int]] = []
    seen = set()

    def walk(node) -> None:
        if isinstance(node, str):
            parts = node.split("_")
            if len(parts) >= 3 and parts[1].upper() == resname.upper():
                try:
                    key = (parts[0], int(parts[2]))
                except ValueError:
                    return
                if key not in seen:
                    seen.add(key)
                    found.append(key)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


def _reference_candidates(
    *,
    pdb_files_dir: Path,
    entry_dir: Optional[Path],
    resname: str,
    query_pdb: str,
    query_chain: str,
    query_resnum: int,
    holo_pdb: str,
) -> Iterator[Tuple[str, Path, str, int, bool]]:
    """Yield ``(label, pdb_path, chain, resnum, use_covalent_closure)`` in order.

    Trying several references is safe because acceptance is mechanical -- an
    exact internal-geometry match at 1e-3 A with a 10x margin -- rather than a
    judgement call.  The order matters only for speed and for which label ends up
    in the ledger.

    Both closure settings are offered for every source.  Covalent closure is
    needed for an oligosaccharide written one unit per residue, but it
    over-reaches on some structures: on ``3hxy-A-MDN-443`` it grew a 9-atom
    diphosphonate into 68 atoms, where the deposited ligand is the single
    residue.  Which is right is decided by the match, not in advance.
    """
    query_path = resolve_query_pdb(pdb_files_dir, query_pdb)
    if query_path is not None:
        yield ("query_closure", query_path, query_chain, query_resnum, True)
        yield ("query_seed_only", query_path, query_chain, query_resnum, False)

    holo_path = resolve_query_pdb(pdb_files_dir, holo_pdb) if holo_pdb else None
    if holo_path is not None:
        for chain, resnum in ligand_ids_from_entry(entry_dir, resname):
            yield ("holo_ligand_id_closure", holo_path, chain, resnum, True)
            yield ("holo_ligand_id_seed_only", holo_path, chain, resnum, False)


def repair_one(
    sample_dir: Path,
    *,
    pdb_files_dir: Path,
    apply_changes: bool,
    entry_dir: Optional[Path] = None,
) -> Dict:
    """Repair a single sample. Returns a ledger row; never raises for data reasons."""
    from scripts.prepare_ahojdb_triplets import extract_ligand_from_pdb

    sample_id = sample_dir.name
    resnum = parse_residue_number(sample_id)
    if resnum is None:
        return _reject(sample_dir, "residue_number_unparseable",
                       f"entry key {sample_id!r} has no integer tail")

    meta = load_meta(sample_dir)
    query_pdb = str(meta.get("query_pdb") or "").strip()
    query_chain = str(meta.get("query_chain") or "").strip()
    resname = str(meta.get("ligand_resname") or "").strip()
    holo_pdb = str(meta.get("holo_pdb") or "").strip()
    if not (query_pdb and query_chain and resname):
        return _reject(sample_dir, "meta_incomplete",
                       "meta.json lacks query_pdb, query_chain or ligand_resname")

    candidates = list(_reference_candidates(
        pdb_files_dir=pdb_files_dir, entry_dir=entry_dir, resname=resname,
        query_pdb=query_pdb, query_chain=query_chain, query_resnum=resnum,
        holo_pdb=holo_pdb,
    ))
    if not candidates:
        return _reject(sample_dir, "query_pdb_missing",
                       f"no structure for {query_pdb} or {holo_pdb} under {pdb_files_dir}")

    sdf_path = sample_dir / "ligand.sdf"
    coords_path = sample_dir / "ligand_coords.npy"
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=False)
    if mol is None:
        return _reject(sample_dir, "observed_sdf_unreadable", f"{sdf_path} did not parse")
    if mol.GetNumConformers() == 0:
        return _reject(sample_dir, "observed_sdf_has_no_conformer", str(sdf_path))

    observed = conformer_positions(mol)
    fragments = Chem.GetMolFrags(mol)

    # A single connected fragment cannot be the scattered-copies defect: copies
    # a median of 72 A apart are never bonded to each other. Such a sample is out
    # of scope here even when its atom count exceeds one component, because that
    # is a *linked* oligosaccharide -- 6t0i-B-XYP-2 holds three bonded xylose
    # units, 9n+1 = 28 atoms -- and where a glycan's ligand boundary lies is a
    # separate open question, not something this repair should settle silently.
    if len(fragments) <= 1:
        return {
            "sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "status": "single_fragment_not_in_scope",
            "observed_atoms": int(mol.GetNumAtoms()),
            "observed_fragments": len(fragments),
        }

    try:
        stored = np.load(coords_path)
    except Exception as exc:  # noqa: BLE001
        return _reject(sample_dir, "coords_unreadable", f"{type(exc).__name__}: {exc}")
    if stored.shape[0] != mol.GetNumAtoms():
        return _reject(sample_dir, "ligand_coords_shape_mismatch",
                       f"npy has {stored.shape[0]} atoms, sdf has {mol.GetNumAtoms()}",
                       observed_atoms=int(mol.GetNumAtoms()),
                       stored_atoms=int(stored.shape[0]))

    identified = None
    selection: Dict = {}
    reference_coords = np.zeros((0, 3), dtype=np.float64)
    resolved_source = None
    attempts: List[Dict] = []
    for label, path, chain, candidate_resnum, use_closure in candidates:
        try:
            coords, _, diagnostics = extract_ligand_from_pdb(
                path, resname, chain,
                residue_number=candidate_resnum,
                select_single_residue=True,
                covalent_closure=use_closure,
            )
        except LigandResidueNotFound as exc:
            attempts.append({"source": label, "outcome": "residue_absent", "detail": str(exc)[:160]})
            continue
        except Exception as exc:  # noqa: BLE001 - any parse failure moves on
            attempts.append({"source": label, "outcome": "extraction_failed",
                             "detail": f"{type(exc).__name__}: {exc}"[:160]})
            continue

        try:
            candidate_match = identify_ligand_atom_indices(
                [coords.astype(np.float64)], observed, fragments
            )
        except CopyIdentificationError as exc:
            attempts.append({"source": label, "outcome": exc.reason,
                             "reference_atoms": int(coords.shape[0])})
            continue

        identified = candidate_match
        selection = diagnostics
        reference_coords = coords
        resolved_source = label
        break

    if identified is None:
        return _reject(sample_dir, "no_reference_reproduced_the_ligand",
                       f"none of {len(attempts)} reference candidates matched",
                       observed_atoms=int(mol.GetNumAtoms()),
                       observed_fragments=len(fragments),
                       attempts=attempts)

    row: Dict = {
        "sample_id": sample_id,
        "sample_dir": str(sample_dir),
        "query_pdb": query_pdb,
        "query_chain": query_chain,
        "resname": resname,
        "residue_number": resnum,
        "resolved_source": resolved_source,
        "reference_attempts": len(attempts) + 1,
        "observed_atoms": int(mol.GetNumAtoms()),
        "observed_fragments": len(fragments),
        "reference_atoms": int(reference_coords.shape[0]),
        "selected_atoms": len(identified.atom_indices),
        "identification": identified.as_diagnostics(),
        # The generator-side view of how many copies the chain holds. It can
        # disagree with observed_fragments -- 2hdr-A-4A3-506 has 15 fragments but
        # only 8 same-name residues in the query chain -- and the cause is not
        # established, so both numbers are recorded rather than reconciled.
        "reextraction_discarded_copies": selection.get("discarded_same_resname"),
        "reextraction_discarded_max_centroid_angstrom": selection.get(
            "discarded_max_centroid_distance_angstrom"
        ),
        "fragment_count_matches_reextraction": (
            len(fragments) == (selection.get("discarded_same_resname") or 0) + 1
        ),
    }

    if len(identified.atom_indices) == mol.GetNumAtoms():
        row["status"] = "no_change_needed"
        return row

    subset = subset_molecule(mol, identified.atom_indices)
    subset_positions = conformer_positions(subset)
    sliced = stored[list(identified.atom_indices)]

    drift = float(np.max(np.abs(subset_positions - sliced.astype(np.float64))))
    row["coordinate_agreement_angstrom"] = drift
    if drift > COORDINATE_AGREEMENT_TOLERANCE_ANGSTROM:
        return _reject(sample_dir, "subset_coordinates_disagree",
                       f"sliced npy and subset conformer differ by {drift:.6f} A",
                       **{k: row[k] for k in ("observed_atoms", "selected_atoms")})

    row["status"] = "would_repair" if not apply_changes else "repaired"
    if not apply_changes:
        return row

    backup_sdf = sample_dir / BACKUP_SDF_NAME
    backup_coords = sample_dir / BACKUP_COORDS_NAME
    try:
        if not backup_sdf.exists():
            backup_sdf.write_bytes(sdf_path.read_bytes())
        if not backup_coords.exists():
            backup_coords.write_bytes(coords_path.read_bytes())

        # Atomic publish: a crash between the two writes would otherwise leave
        # ligand.sdf and ligand_coords.npy describing different atom sets.
        temp_sdf = sdf_path.with_suffix(f".{os.getpid()}.tmp")
        writer = Chem.SDWriter(str(temp_sdf))
        writer.write(subset)
        writer.close()
        reparsed = Chem.MolFromMolFile(str(temp_sdf), removeHs=False, sanitize=False)
        if reparsed is None or reparsed.GetNumAtoms() != len(identified.atom_indices):
            temp_sdf.unlink(missing_ok=True)
            return _reject(sample_dir, "written_sdf_failed_reparse",
                           "the subset SDF did not read back with the expected atoms")

        temp_coords = coords_path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(temp_coords, sliced.astype(np.float32))
        os.replace(temp_sdf, sdf_path)
        os.replace(temp_coords, coords_path)
    except Exception as exc:  # noqa: BLE001
        return _reject(sample_dir, "write_failed", f"{type(exc).__name__}: {exc}",
                       **{k: row[k] for k in ("observed_atoms", "selected_atoms")})

    return row


def build_entry_index(ahoj_data_dir: Path) -> Dict[str, str]:
    """Map entry key to entry directory in one sweep of the shards.

    561 shards hold ~91k entries, so a per-sample glob would repeat the same
    directory walk tens of thousands of times on a network filesystem.
    """
    index: Dict[str, str] = {}
    if not ahoj_data_dir.is_dir():
        return index
    with os.scandir(ahoj_data_dir) as shards:
        for shard in shards:
            if not shard.is_dir():
                continue
            with os.scandir(shard.path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        index[entry.name] = entry.path
    return index


def _worker(payload: Tuple[str, str, bool, Optional[str]]) -> Dict:
    sample_dir, pdb_files_dir, apply_changes, entry_dir = payload
    RDLogger.DisableLog("rdApp.*")
    try:
        return repair_one(
            Path(sample_dir),
            pdb_files_dir=Path(pdb_files_dir),
            apply_changes=apply_changes,
            entry_dir=Path(entry_dir) if entry_dir else None,
        )
    except Exception as exc:  # noqa: BLE001 - a worker must never take down the pool
        return _reject(Path(sample_dir), "worker_exception", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples-root", type=Path,
                        default=PROJECT_ROOT / "processed_data" / "triplets" / "samples")
    parser.add_argument("--pdb-files-dir", type=Path,
                        default=PROJECT_ROOT / "ahojdb_v2c" / "pdb_files",
                        help="AHoJ pdb_files directory holding the deposited structures")
    parser.add_argument("--ahoj-data-dir", type=Path,
                        default=PROJECT_ROOT / "ahojdb_v2c" / "data",
                        help="AHoJ data shards, read for ligands.json when the "
                             "sample's ligand came from the holo structure")
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="Directory for records.jsonl and summary.json")
    parser.add_argument("--sample-ids-file", type=Path, default=None,
                        help="Restrict to the sample ids listed one per line")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--apply", action="store_true",
                        help="Write the repair. Without this the run is a dry run "
                             "and no file under --samples-root is modified.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    if not RDKIT_AVAILABLE:
        LOG.error("RDKit is required to run the repair")
        return 2
    RDLogger.DisableLog("rdApp.*")

    if not args.pdb_files_dir.is_dir():
        LOG.error("pdb_files directory not found: %s", args.pdb_files_dir)
        return 2

    if args.sample_ids_file:
        wanted = [
            line.strip()
            for line in args.sample_ids_file.read_text().splitlines()
            if line.strip()
        ]
        sample_dirs = [args.samples_root / sid for sid in wanted]
        sample_dirs = [d for d in sample_dirs if (d / "ligand.sdf").exists()]
        missing = len(wanted) - len(sample_dirs)
        if missing:
            LOG.warning("%d of %d listed samples have no ligand.sdf", missing, len(wanted))
    else:
        sample_dirs = list(iter_sample_dirs(args.samples_root))
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    LOG.info("%s over %d samples (workers=%d)",
             "APPLY" if args.apply else "dry run", len(sample_dirs), args.workers)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.report_dir / "records.jsonl"
    started = time.time()

    entry_index = build_entry_index(args.ahoj_data_dir)
    LOG.info("entry index holds %d AHoJ entries from %s",
             len(entry_index), args.ahoj_data_dir)

    payloads = [
        (str(d), str(args.pdb_files_dir), args.apply, entry_index.get(d.name))
        for d in sample_dirs
    ]
    rows: List[Dict] = []
    with records_path.open("w") as handle:
        if args.workers > 1:
            with multiprocessing.Pool(args.workers) as pool:
                iterator = pool.imap_unordered(_worker, payloads, chunksize=8)
                for index, row in enumerate(iterator, 1):
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    if index % 2000 == 0:
                        LOG.info("  %d/%d", index, len(payloads))
        else:
            for index, payload in enumerate(payloads, 1):
                row = _worker(payload)
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                if index % 2000 == 0:
                    LOG.info("  %d/%d", index, len(payloads))

    statuses: Dict[str, int] = {}
    rejects: Dict[str, int] = {}
    atoms_before = atoms_after = 0
    fragment_mismatches = 0
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        if row["status"] == "rejected":
            rejects[row["reject_reason"]] = rejects.get(row["reject_reason"], 0) + 1
        elif "selected_atoms" in row:
            # Only rows that actually resolved a ligand contribute to the atom
            # tally; counting an out-of-scope row as "0 atoms after" would report
            # its whole molecule as discarded.
            atoms_before += int(row["observed_atoms"])
            atoms_after += int(row["selected_atoms"])
            if row.get("fragment_count_matches_reextraction") is False:
                fragment_mismatches += 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "apply" if args.apply else "dry_run",
        "samples_root": str(args.samples_root),
        "pdb_files_dir": str(args.pdb_files_dir),
        "samples_seen": len(rows),
        "status_counts": statuses,
        "reject_ledger": rejects,
        "ligand_atoms_before": atoms_before,
        "ligand_atoms_after": atoms_after,
        "ligand_atoms_discarded": atoms_before - atoms_after,
        "fragment_count_disagreements": fragment_mismatches,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOG.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
