#!/usr/bin/env python3
"""
Prepare AHoJ-DB apo/holo/ligand triplets for Stage-1/Stage-2.

This script:
1) Parses AHoJ db_entries.json (streaming) to enumerate entries.
2) Selects one apo and one holo candidate per entry (from *_filtered_sorted_results.csv).
3) Applies alignment matrices to place apo/holo/ligand into a shared frame (apo frame).
4) Writes triplet samples to a target directory with minimal metadata.

Expected AHoJ layout:
  <ahoj_root>/
    db_entries.json
    entries.csv
    data/<shard>/<entry_key>/
      apo_filtered_sorted_results.csv
      holo_filtered_sorted_results.csv
      ligands.json (optional but recommended)
      matrices/
        aln_<src><chain>_to_<dst><chain>.txt
      structure_files/ (optional; may fall back to pdb_files/)

Notes:
- This script requires extracted AHoJ data/ (not the tarball).
- It enforces strict data integrity; entries missing required files are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import glob
import re

import numpy as np

try:
    from Bio.PDB import PDBParser, PDBIO, Select
except ImportError as exc:
    raise SystemExit(f"❌ BioPython not available: {exc}")

try:
    from rdkit import Chem
except ImportError as exc:
    raise SystemExit(f"❌ RDKit not available: {exc}")


LOG = logging.getLogger("prepare_ahojdb_triplets")


@dataclass
class Candidate:
    pdb_id: str
    chain_id: str
    aln_matrix: Optional[str] = None


class ChainSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain) -> bool:
        return chain.id == self.chain_id


def iter_json_array(path: Path) -> Iterator[Dict]:
    """Stream a JSON array of objects without loading the full file."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        buf: List[str] = []
        depth = 0
        in_obj = False
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            for ch in chunk:
                if not in_obj:
                    if ch == "{":
                        in_obj = True
                        depth = 1
                        buf = [ch]
                else:
                    buf.append(ch)
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            yield json.loads("".join(buf))
                            in_obj = False
                            buf = []


def resolve_entry_dir(entry: Dict, ahoj_root: Path) -> Optional[Path]:
    entry_dir = entry.get("entry_dir")
    if not entry_dir:
        return None
    entry_dir = str(entry_dir)

    p = Path(entry_dir)
    if p.is_absolute() and p.exists():
        return p

    if "/data/" in entry_dir:
        rel = entry_dir.split("/data/", 1)[1]
        cand = ahoj_root / "data" / rel
        if cand.exists():
            return cand

    cand = ahoj_root / entry_dir
    if cand.exists():
        return cand
    return None


def _pick_column(row: Dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v).strip()
    return None


def _split_chain(field: str) -> Optional[str]:
    if not field:
        return None
    field = field.strip()
    # Some AHoJ fields use chain lists like "A,B"
    for sep in [",", ";", " ", "|", "/"]:
        if sep in field:
            return field.split(sep)[0].strip()
    return field


_STRUCTURE_RE = re.compile(r"^(?:pdb)?([0-9A-Za-z]{4})(.*)$")


def parse_structure_field(structure: str) -> Tuple[str, str]:
    """Parse a candidate 'structure' field into (pdb_id, chain_id)."""
    if not structure:
        return "", ""
    token = structure.strip().split()[0]
    m = _STRUCTURE_RE.match(token)
    if not m:
        m2 = re.search(r"[0-9A-Za-z]{4}", token)
        if not m2:
            return "", ""
        pdb_id = m2.group(0).lower()
        rest = token[m2.end():]
    else:
        pdb_id = m.group(1).lower()
        rest = m.group(2) or ""

    rest = rest.strip()
    chain_id = ""
    if rest:
        if rest[0] in "-_:":
            chain_id = rest[1:2]
        else:
            chain_id = rest[0:1]
    chain_id = chain_id.strip()
    chain_id = _split_chain(chain_id) or ""
    return pdb_id, chain_id


def load_candidate(csv_path: Path) -> Optional[Candidate]:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            structure = _pick_column(row, ["structure", "pdb_id", "pdb", "structure_id"])
            if not structure:
                continue
            pdb_id, chain_id = parse_structure_field(structure)
            if not pdb_id:
                continue

            chain = _pick_column(row, ["chain", "chains", "chain_id", "chainID"])
            if chain:
                chain_id = _split_chain(chain) or chain_id
            if not chain_id:
                chain_hint = _pick_column(row, ["aln_q_chains", "chains2", "chains3", "chains4"])
                chain_id = _split_chain(chain_hint) or ""

            aln_matrix = _pick_column(row, ["aln_matrix", "alignment_matrix", "matrix"])
            return Candidate(pdb_id=pdb_id, chain_id=chain_id, aln_matrix=aln_matrix)
    return None


def load_alignment_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    nums: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for tok in line.strip().split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    continue
    if len(nums) == 16:
        mat = np.array(nums, dtype=np.float32).reshape(4, 4)
        R = mat[:3, :3]
        t = mat[:3, 3]
        return R, t
    if len(nums) == 12:
        mat = np.array(nums, dtype=np.float32).reshape(3, 4)
        R = mat[:, :3]
        t = mat[:, 3]
        return R, t
    raise ValueError(f"Unexpected matrix format in {path} (len={len(nums)})")


def invert_rt(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def apply_rt(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return coords @ R.T + t


def find_alignment_matrix(entry_dir: Path,
                          src: Candidate,
                          dst: Candidate) -> Optional[Path]:
    matrices_dir = entry_dir / "matrices"
    if not matrices_dir.exists():
        return None
    patterns = [
        f"aln_{src.pdb_id}{src.chain_id}_to_{dst.pdb_id}{dst.chain_id}.txt",
        f"aln_{src.pdb_id}{src.chain_id}_to_{dst.pdb_id}.txt",
        f"aln_{src.pdb_id}_to_{dst.pdb_id}{dst.chain_id}.txt",
        f"aln_{src.pdb_id}_to_{dst.pdb_id}.txt",
    ]
    for name in patterns:
        candidate = matrices_dir / name
        if candidate.exists():
            return candidate
    # Fallback: scan for src/dst ids
    for path in matrices_dir.glob("aln_*_to_*.txt"):
        if src.pdb_id in path.name and dst.pdb_id in path.name:
            return path
    return None


def resolve_alignment_matrix(entry_dir: Path,
                             src: Candidate,
                             dst: Candidate) -> Optional[Path]:
    if src.aln_matrix:
        raw = src.aln_matrix.strip()
        if raw:
            p = Path(raw)
            if p.is_absolute() and p.exists():
                return p
            for base in [entry_dir, entry_dir / "matrices"]:
                cand = base / p
                if cand.exists():
                    return cand
    return find_alignment_matrix(entry_dir, src, dst)


def find_pdb_file(entry_dir: Path,
                  pdb_id: str,
                  chain_id: str,
                  pdb_dir: Optional[Path]) -> Optional[Path]:
    structure_dir = entry_dir / "structure_files"
    chain_id = (chain_id or "").strip()
    pdb_id = pdb_id.lower()
    pdb_pat = glob.escape(pdb_id)
    chain_pat = glob.escape(chain_id) if chain_id else ""
    if structure_dir.exists():
        patterns = []
        if chain_id:
            patterns.extend([
                f"{pdb_id}_{chain_id}.pdb",
                f"{pdb_id}{chain_id}.pdb",
                f"{pdb_id}.pdb",
                f"{pdb_id}_{chain_id}.cif",
                f"{pdb_id}{chain_id}.cif",
                f"{pdb_id}.cif",
                f"{pdb_id}_{chain_id}.pdb.gz",
                f"{pdb_id}{chain_id}.pdb.gz",
                f"{pdb_id}.pdb.gz",
                f"{pdb_id}_{chain_id}.cif.gz",
                f"{pdb_id}{chain_id}.cif.gz",
                f"{pdb_id}.cif.gz",
            ])
        else:
            patterns.extend([
                f"{pdb_id}.pdb",
                f"{pdb_id}.cif",
                f"{pdb_id}.pdb.gz",
                f"{pdb_id}.cif.gz",
            ])
        for name in patterns:
            path = structure_dir / name
            if path.exists():
                return path
        if chain_id:
            matches = list(structure_dir.glob(f"*{pdb_pat}*{chain_pat}*.pdb*"))
            if matches:
                return matches[0]
            matches = list(structure_dir.glob(f"*{pdb_pat}*{chain_pat}*.cif*"))
            if matches:
                return matches[0]
        matches = list(structure_dir.glob(f"*{pdb_pat}*.pdb*"))
        if matches:
            return matches[0]
        matches = list(structure_dir.glob(f"*{pdb_pat}*.cif*"))
        if matches:
            return matches[0]
    if pdb_dir is not None:
        candidates: List[Path] = []
        for pid in [pdb_id.lower(), pdb_id.upper()]:
            names = [
                f"{pid}.pdb",
                f"{pid}.cif",
                f"{pid}.pdb.gz",
                f"{pid}.cif.gz",
                f"{pid}.ent",
                f"{pid}.ent.gz",
                f"pdb{pid}.ent",
                f"pdb{pid}.ent.gz",
            ]
            candidates.extend([pdb_dir / n for n in names])
        for p in candidates:
            if p.exists():
                return p
        matches = list(pdb_dir.glob(f"{pdb_id}*"))
        if matches:
            return matches[0]
    return None


def load_chain_structure(pdb_path: Path, chain_id: str):
    suffixes = "".join(pdb_path.suffixes).lower()
    if suffixes.endswith(".cif") or suffixes.endswith(".cif.gz"):
        from Bio.PDB import MMCIFParser  # type: ignore
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    if suffixes.endswith(".gz"):
        import gzip
        with gzip.open(pdb_path, "rt", encoding="utf-8", errors="ignore") as fh:
            structure = parser.get_structure(pdb_path.stem, fh)
    else:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    if chain_id:
        for model in structure:
            if chain_id in model:
                return structure, chain_id
    # fallback: first chain
    for model in structure:
        for chain in model:
            return structure, chain.id
    raise ValueError(f"No chain found in {pdb_path}")


def transform_structure(structure, R: np.ndarray, t: np.ndarray):
    for atom in structure.get_atoms():
        coord = atom.get_coord()
        atom.set_coord(coord @ R.T + t)


def write_chain_pdb(structure, chain_id: str, out_path: Path):
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), select=ChainSelect(chain_id))


def extract_ligand_from_pdb(pdb_path: Path, ligand_resname: str, chain_id: str) -> Tuple[np.ndarray, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    coords = []
    for model in structure:
        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_id()[0] == " ":
                    continue
                if residue.get_resname().strip() != ligand_resname:
                    continue
                for atom in residue:
                    element = (atom.element or "").strip().upper()
                    name = atom.get_name().strip().upper()
                    if element == "H" or name.startswith("H"):
                        continue
                    coords.append(atom.get_coord())
    if not coords:
        raise ValueError(f"Ligand {ligand_resname} not found in {pdb_path}")

    coords = np.array(coords, dtype=np.float32)

    # Build PDB block (RDKit) for this residue
    io = PDBIO()
    io.set_structure(structure)
    from io import StringIO
    fh = StringIO()
    class LigandSelect(Select):
        def accept_residue(self, residue) -> bool:
            return residue.get_id()[0] != " " and residue.get_resname().strip() == ligand_resname
    io.save(fh, select=LigandSelect())
    pdb_block = fh.getvalue()
    return coords, pdb_block


def pdb_block_to_sdf(pdb_block: str, coords: np.ndarray, out_path: Path) -> None:
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError("RDKit failed to build molecule from PDB block")
    # Remove hydrogens to match ligand_coords.npy (heavy atoms only)
    mol = Chem.RemoveAllHs(mol)
    if mol.GetNumAtoms() != len(coords):
        raise ValueError(
            f"Ligand atom count mismatch (mol={mol.GetNumAtoms()}, coords={len(coords)})"
        )
    conf = mol.GetConformer()
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, xyz.tolist())
    w = Chem.SDWriter(str(out_path))
    w.write(mol)
    w.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare AHoJ triplets")
    parser.add_argument("--ahoj-root", type=str, required=True,
                        help="AHoJ root directory (must contain db_entries.json and data/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output dataset directory (apo_holo_triplets)")
    parser.add_argument("--pdb-dir", type=str, default=None,
                        help="Fallback directory with downloaded PDBs (pdb_files)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Stop after writing this many samples")
    parser.add_argument("--limit", type=int, default=None,
                        help="(deprecated) alias for --max-samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="0.9,0.05,0.05",
                        help="Train/val/test split ratio")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    max_samples = args.max_samples if args.max_samples is not None else args.limit

    ahoj_root = Path(args.ahoj_root)
    if not (ahoj_root / "db_entries.json").exists():
        raise FileNotFoundError(f"db_entries.json not found under {ahoj_root}")
    if not (ahoj_root / "data").exists():
        raise FileNotFoundError(f"data/ directory not found under {ahoj_root} (extract data.tar.gz first)")

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else None

    splits = [float(x) for x in args.split.split(",")]
    if len(splits) != 3 or abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError("split must be three comma-separated ratios summing to 1.0")

    entry_iter = iter_json_array(ahoj_root / "db_entries.json")
    written = []
    skipped = 0
    total = 0

    for entry in entry_iter:
        if max_samples is not None and len(written) >= max_samples:
            break
        total += 1

        entry_dir = resolve_entry_dir(entry, ahoj_root)
        if entry_dir is None or not entry_dir.exists():
            skipped += 1
            continue

        query_pdb = entry.get("target_pdb_id", "").lower()[:4]
        query_chain = _split_chain(entry.get("query_chain", "")) or ""
        ligand_resname = entry.get("target_ligand", "").strip()
        if not query_pdb or not query_chain or not ligand_resname:
            skipped += 1
            continue

        try:
            apo_csv = entry_dir / "apo_filtered_sorted_results.csv"
            holo_csv = entry_dir / "holo_filtered_sorted_results.csv"
            apo = load_candidate(apo_csv)
            holo = load_candidate(holo_csv)
            if apo is None or holo is None:
                raise ValueError("missing apo/holo candidates")

            query = Candidate(pdb_id=query_pdb, chain_id=query_chain)

            apo_mat = resolve_alignment_matrix(entry_dir, apo, query)
            holo_mat = resolve_alignment_matrix(entry_dir, holo, query)
            if apo_mat is None or holo_mat is None:
                raise ValueError("missing alignment matrices")

            apo_pdb = find_pdb_file(entry_dir, apo.pdb_id, apo.chain_id, pdb_dir)
            holo_pdb = find_pdb_file(entry_dir, holo.pdb_id, holo.chain_id, pdb_dir)
            query_pdb_path = find_pdb_file(entry_dir, query.pdb_id, query.chain_id, pdb_dir)
            if apo_pdb is None or holo_pdb is None or query_pdb_path is None:
                raise FileNotFoundError("missing PDB files")

            # Load structures
            apo_struct, apo_chain = load_chain_structure(apo_pdb, apo.chain_id)
            holo_struct, holo_chain = load_chain_structure(holo_pdb, holo.chain_id)
            _, query_chain = load_chain_structure(query_pdb_path, query.chain_id)

            # Transforms
            R_aq, t_aq = load_alignment_matrix(apo_mat)
            R_hq, t_hq = load_alignment_matrix(holo_mat)
            R_qa, t_qa = invert_rt(R_aq, t_aq)

            # Map holo -> apo
            R_ha = R_qa @ R_hq
            t_ha = R_qa @ t_hq + t_qa

            # Transform structures
            transform_structure(holo_struct, R_ha, t_ha)

            sample_id = entry.get("entry_key", f"entry_{total}")
            out_dir = samples_dir / sample_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # Write apo (native) and holo (aligned)
            write_chain_pdb(apo_struct, apo_chain, out_dir / "apo.pdb")
            write_chain_pdb(holo_struct, holo_chain, out_dir / "holo.pdb")

            # Ligand extraction from query structure (aligned to apo frame)
            lig_coords, lig_pdb_block = extract_ligand_from_pdb(
                query_pdb_path, ligand_resname, query_chain
            )
            lig_coords = apply_rt(lig_coords, R_qa, t_qa)
            np.save(out_dir / "ligand_coords.npy", lig_coords.astype(np.float32))

            # Write ligand.sdf (strict)
            sdf_path = out_dir / "ligand.sdf"
            pdb_block_to_sdf(lig_pdb_block, lig_coords, sdf_path)

            # Metadata
            meta = {
                "entry_key": sample_id,
                "query_pdb": query_pdb,
                "query_chain": query_chain,
                "apo_pdb": apo.pdb_id,
                "apo_chain": apo_chain,
                "holo_pdb": holo.pdb_id,
                "holo_chain": holo_chain,
                "ligand_resname": ligand_resname,
                "entry_dir": str(entry_dir),
                "apo_matrix": str(apo_mat),
                "holo_matrix": str(holo_mat),
            }
            with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            written.append(sample_id)
        except Exception as exc:
            LOG.warning("Skip %s: %s", entry.get("entry_key", f"entry_{total}"), exc)
            skipped += 1
            continue

    LOG.info("Processed entries: %d", total)
    LOG.info("Written samples: %d", len(written))
    LOG.info("Skipped entries: %d", skipped)

    # Write index + splits
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(written, f, indent=2)

    # Deterministic split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(written)).tolist()
    n_train = int(len(written) * splits[0])
    n_val = int(len(written) * splits[1])

    train_ids = [written[i] for i in perm[:n_train]]
    val_ids = [written[i] for i in perm[n_train:n_train + n_val]]
    test_ids = [written[i] for i in perm[n_train + n_val:]]

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.json").write_text(json.dumps(train_ids, indent=2), encoding="utf-8")
    (splits_dir / "val.json").write_text(json.dumps(val_ids, indent=2), encoding="utf-8")
    (splits_dir / "test.json").write_text(json.dumps(test_ids, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
