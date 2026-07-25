#!/usr/bin/env python3
"""Download and conservatively preflight an APObind structure-smoke cohort."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.md_pilot_selection import METAL_ATOMIC_NUMBERS, kabsch_align
from src.data.residue_alignment import align_residue_names, canonical_resname
from src.data.residue_identity import STANDARD_AA3_TO_1


SCHEMA_VERSION = "bindrae_apobind_structure_preflight_v1"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
RCSB_CCD_URLS = (
    "https://files.rcsb.org/ligands/download/{resname}_ideal.sdf",
    "https://files.rcsb.org/ligands/download/{resname}_model.sdf",
)
WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "H2O"})
COMMON_CRYSTALLIZATION_RESNAMES = frozenset(
    {
        "BME",
        "DMS",
        "EDO",
        "GOL",
        "HEP",
        "MES",
        "MPD",
        "PEG",
        "PGE",
        "PG4",
        "TRS",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--min-sequence-identity", type=float, default=0.95)
    parser.add_argument("--min-mapping-fraction", type=float, default=0.95)
    parser.add_argument("--min-site-signature-match", type=float, default=0.90)
    parser.add_argument("--min-site-mapping-fraction", type=float, default=0.90)
    parser.add_argument("--min-global-rmsd", type=float, default=0.35)
    parser.add_argument("--max-global-rmsd", type=float, default=5.0)
    parser.add_argument("--min-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-heavy-atoms", type=int, default=120)
    parser.add_argument("--max-abs-charge", type=int, default=2)
    parser.add_argument("--min-observed-heavy-atom-fraction", type=float, default=0.80)
    parser.add_argument("--ligand-site-distance", type=float, default=4.5)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def _download_bytes(url: str, *, timeout: float, retries: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "BINDRAE-data-audit/1"})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _valid_gzip_cif(path: Path) -> bool:
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            prefix = handle.read(256)
        return prefix.lstrip().startswith("data_")
    except (OSError, EOFError):
        return False


def download_structure(
    pdb_id: str,
    *,
    structures_dir: Path,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    pdb_id = pdb_id.upper()
    output_path = structures_dir / f"{pdb_id}.cif.gz"
    base = {"pdb_id": pdb_id, "path": str(output_path)}
    if output_path.is_file() and _valid_gzip_cif(output_path):
        return {
            **base,
            "status": "cached",
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
        }

    url = RCSB_CIF_URL.format(pdb_id=pdb_id)
    try:
        content = _download_bytes(url, timeout=timeout, retries=retries)
        with tempfile.NamedTemporaryFile(
            dir=structures_dir, prefix=f".{pdb_id}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
        if not _valid_gzip_cif(temporary_path):
            temporary_path.unlink(missing_ok=True)
            raise ValueError("response is not a valid gzipped mmCIF file")
        os.replace(temporary_path, output_path)
        return {
            **base,
            "status": "downloaded",
            "url": url,
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
        }
    except Exception as exc:
        return {**base, "status": "failed", "url": url, "detail": str(exc)}


def download_selected_structures(
    records: Sequence[Mapping[str, Any]],
    *,
    structures_dir: Path,
    workers: int,
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    structures_dir.mkdir(parents=True, exist_ok=True)
    pdb_ids = sorted(
        {
            str(endpoint["pdb_id"]).upper()
            for record in records
            for endpoint in record["endpoints"]
        }
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_structure,
                pdb_id,
                structures_dir=structures_dir,
                timeout=timeout,
                retries=retries,
            ): pdb_id
            for pdb_id in pdb_ids
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: str(row["pdb_id"]))


def load_structure_context(path: Path) -> tuple[Any, dict[str, str]]:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        structure = parser.get_structure(path.stem, handle)
    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError("structure has no model")
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        cif = MMCIF2Dict(handle)
    component_ids = cif.get("_chem_comp.id", [])
    component_types = cif.get("_chem_comp.type", [])
    if isinstance(component_ids, str):
        component_ids = [component_ids]
    if isinstance(component_types, str):
        component_types = [component_types]
    type_by_id = {
        str(component_id).strip().upper(): str(component_type).strip()
        for component_id, component_type in zip(component_ids, component_types)
    }
    return model, type_by_id


def _atom_element(atom: Any) -> str:
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element
    name = "".join(character for character in atom.get_name() if character.isalpha())
    return name[:1].upper()


def _heavy_coordinates(residue: Any) -> np.ndarray:
    coordinates = [
        np.asarray(atom.get_coord(), dtype=np.float64)
        for atom in residue.get_atoms()
        if _atom_element(atom) not in {"H", "D"}
    ]
    return np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)


def protein_residue_records(model: Any, chain_id: str) -> list[dict[str, Any]]:
    if chain_id not in model:
        available = sorted(str(chain.id) for chain in model)
        raise KeyError(f"chain {chain_id!r} not found; available={available}")
    records: list[dict[str, Any]] = []
    for residue in model[chain_id]:
        name = canonical_resname(residue.get_resname())
        if name not in STANDARD_AA3_TO_1 or "CA" not in residue:
            continue
        _, sequence_id, insertion_code = residue.get_id()
        records.append(
            {
                "bio_residue": residue,
                "name": name,
                "one_letter": STANDARD_AA3_TO_1[name],
                "author_sequence_id": int(sequence_id),
                "insertion_code": str(insertion_code).strip(),
                "ca": np.asarray(residue["CA"].get_coord(), dtype=np.float64),
                "heavy_xyz": _heavy_coordinates(residue),
            }
        )
    if not records:
        raise ValueError(f"chain {chain_id!r} has no standard residues with CA atoms")
    return records


def _integer_tokens(values: Sequence[object]) -> list[int] | None:
    parsed: list[int] = []
    for value in values:
        try:
            parsed.append(int(str(value).strip()))
        except ValueError:
            return None
    return parsed


def resolve_binding_site_signature(
    residues: Sequence[Mapping[str, Any]],
    source_indices: Sequence[object],
    source_residues: Sequence[object],
) -> dict[str, Any]:
    """Resolve APObind site indices without assuming their numbering convention."""

    if not source_indices or len(source_indices) != len(source_residues):
        return {
            "eligible": False,
            "reason": "binding_site_signature_length_mismatch",
            "source_index_count": len(source_indices),
            "source_residue_count": len(source_residues),
        }
    gap_tokens = {"", "-", "."}
    retained_indices: list[object] = []
    retained_residues: list[object] = []
    source_gap_count = 0
    for raw_index, raw_residue in zip(source_indices, source_residues):
        index_is_gap = str(raw_index).strip() in gap_tokens
        residue_is_gap = str(raw_residue).strip() in gap_tokens
        if index_is_gap or residue_is_gap:
            if index_is_gap and residue_is_gap:
                source_gap_count += 1
                continue
            return {
                "eligible": False,
                "reason": "inconsistent_binding_site_gap",
                "source_index": str(raw_index),
                "source_residue": str(raw_residue),
            }
        retained_indices.append(raw_index)
        retained_residues.append(raw_residue)
    if not retained_indices:
        return {"eligible": False, "reason": "empty_binding_site_after_gaps"}

    integer_indices = _integer_tokens(retained_indices)
    if integer_indices is None:
        return {"eligible": False, "reason": "noninteger_binding_site_index"}
    expected_letters = [str(value).strip().upper() for value in retained_residues]

    author_lookup: dict[int, list[int]] = {}
    for index, residue in enumerate(residues):
        author_lookup.setdefault(int(residue["author_sequence_id"]), []).append(index)

    schemes: dict[str, list[int] | None] = {
        "author_sequence_id": [
            author_lookup[value][0]
            if len(author_lookup.get(value, [])) == 1
            else -1
            for value in integer_indices
        ],
        "ordinal_zero_based": [
            value if 0 <= value < len(residues) else -1 for value in integer_indices
        ],
        "ordinal_one_based": [
            value - 1 if 1 <= value <= len(residues) else -1
            for value in integer_indices
        ],
    }
    evaluated: list[dict[str, Any]] = []
    for scheme, mapped in schemes.items():
        assert mapped is not None
        resolved_pairs = [
            (source_index, chain_index)
            for source_index, chain_index in enumerate(mapped)
            if chain_index >= 0
        ]
        resolved_count = len(resolved_pairs)
        match_count = sum(
            residues[chain_index]["one_letter"] == expected_letters[source_index]
            for source_index, chain_index in resolved_pairs
        )
        evaluated.append(
            {
                "scheme": scheme,
                "chain_indices": [chain_index for _, chain_index in resolved_pairs],
                "source_positions": [source_index for source_index, _ in resolved_pairs],
                "coverage": resolved_count / len(retained_indices),
                "signature_match_fraction": (
                    match_count / resolved_count if resolved_count else 0.0
                ),
                "signature_match_count": match_count,
            }
        )

    best_score = max(
        (row["signature_match_fraction"], row["coverage"]) for row in evaluated
    )
    best = [
        row
        for row in evaluated
        if (row["signature_match_fraction"], row["coverage"]) == best_score
    ]
    distinct_mappings = {tuple(row["chain_indices"]) for row in best}
    if len(distinct_mappings) != 1:
        return {
            "eligible": False,
            "reason": "ambiguous_binding_site_index_semantics",
            "best_schemes": [row["scheme"] for row in best],
            "coverage": best_score[1],
            "signature_match_fraction": best_score[0],
        }
    chosen_mapping = next(iter(distinct_mappings))
    chosen_schemes = [
        row["scheme"] for row in best if tuple(row["chain_indices"]) == chosen_mapping
    ]
    representative = next(
        row for row in best if tuple(row["chain_indices"]) == chosen_mapping
    )
    return {
        "eligible": True,
        "reason": "ok",
        "scheme": chosen_schemes[0],
        "equivalent_schemes": chosen_schemes,
        "chain_indices": list(chosen_mapping),
        "source_gap_count": source_gap_count,
        "coverage": representative["coverage"],
        "signature_match_fraction": representative["signature_match_fraction"],
        "signature_match_count": representative["signature_match_count"],
    }


def _minimum_distance(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 or second.size == 0:
        return math.inf
    squared = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=-1)
    return float(np.sqrt(np.min(squared)))


def enumerate_hetero_candidates(
    model: Any,
    site_residues: Sequence[Mapping[str, Any]],
    *,
    component_types: Mapping[str, str],
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    site_distance: float,
) -> list[dict[str, Any]]:
    site_xyz = np.concatenate(
        [np.asarray(residue["heavy_xyz"], dtype=np.float64) for residue in site_residues],
        axis=0,
    )
    candidates: list[dict[str, Any]] = []
    for chain in model:
        for residue in chain:
            hetero_flag, sequence_id, insertion_code = residue.get_id()
            resname = str(residue.get_resname()).strip().upper()
            canonical = canonical_resname(resname)
            component_type = str(component_types.get(resname, "")).strip()
            if resname in WATER_RESNAMES:
                continue
            if (
                hetero_flag == " "
                or canonical in STANDARD_AA3_TO_1
                or "LINKING" in component_type.upper()
            ):
                continue
            heavy_xyz = _heavy_coordinates(residue)
            elements = [
                _atom_element(atom)
                for atom in residue.get_atoms()
                if _atom_element(atom) not in {"H", "D"}
            ]
            heavy_atoms = len(elements)
            carbon_atoms = sum(element == "C" for element in elements)
            minimum_distance = _minimum_distance(heavy_xyz, site_xyz)
            contact_residue_count = sum(
                _minimum_distance(heavy_xyz, np.asarray(site["heavy_xyz"]))
                <= site_distance
                for site in site_residues
            )
            exclusion_reasons: list[str] = []
            if resname in COMMON_CRYSTALLIZATION_RESNAMES:
                exclusion_reasons.append("common_crystallization_component")
            if carbon_atoms == 0:
                exclusion_reasons.append("no_carbon")
            if not min_heavy_atoms <= heavy_atoms <= max_heavy_atoms:
                exclusion_reasons.append("heavy_atom_count")
            if minimum_distance > site_distance:
                exclusion_reasons.append("outside_annotated_site")
            candidates.append(
                {
                    "chain_id": str(chain.id),
                    "resname": resname,
                    "chem_comp_type": component_type or None,
                    "author_sequence_id": int(sequence_id),
                    "insertion_code": str(insertion_code).strip(),
                    "heavy_atoms": heavy_atoms,
                    "carbon_atoms": carbon_atoms,
                    "minimum_site_distance_angstrom": minimum_distance,
                    "contact_site_residues": int(contact_residue_count),
                    "plausible_site_ligand": not exclusion_reasons,
                    "exclusion_reasons": exclusion_reasons,
                }
            )
    return candidates


def choose_unique_site_ligand(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    plausible = [dict(candidate) for candidate in candidates if candidate["plausible_site_ligand"]]
    plausible.sort(
        key=lambda row: (
            float(row["minimum_site_distance_angstrom"]),
            -int(row["contact_site_residues"]),
            str(row["chain_id"]),
            int(row["author_sequence_id"]),
            str(row["resname"]),
        )
    )
    if not plausible:
        return {
            "eligible": False,
            "reason": "no_plausible_site_ligand",
            "plausible_candidates": [],
        }
    if len(plausible) != 1:
        return {
            "eligible": False,
            "reason": "ambiguous_multiple_site_ligands",
            "plausible_candidates": plausible,
        }
    return {
        "eligible": True,
        "reason": "ok",
        "ligand": plausible[0],
        "plausible_candidates": plausible,
    }


def download_ccd(
    resname: str,
    *,
    ccd_dir: Path,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    resname = resname.upper()
    output_path = ccd_dir / f"{resname}.sdf"
    if output_path.is_file() and output_path.stat().st_size > 0:
        return {
            "status": "cached",
            "resname": resname,
            "path": str(output_path),
            "sha256": file_sha256(output_path),
        }
    last_error: Exception | None = None
    for template in RCSB_CCD_URLS:
        url = template.format(resname=resname)
        try:
            content = _download_bytes(url, timeout=timeout, retries=retries)
            if b"V2000" not in content and b"V3000" not in content:
                raise ValueError("response is not an MDL structure file")
            ccd_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)
            return {
                "status": "downloaded",
                "resname": resname,
                "path": str(output_path),
                "url": url,
                "sha256": file_sha256(output_path),
            }
        except Exception as exc:
            last_error = exc
    return {
        "status": "failed",
        "resname": resname,
        "path": str(output_path),
        "detail": str(last_error),
    }


def validate_ccd_sdf(path: Path) -> dict[str, Any]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        return {"eligible": False, "reason": "rdkit_unavailable", "detail": str(exc)}
    molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if molecule is None:
        return {"eligible": False, "reason": "ccd_sdf_parse_failed"}
    heavy_atoms = int(molecule.GetNumHeavyAtoms())
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    contains_metal = any(
        atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS for atom in molecule.GetAtoms()
    )
    return {
        "eligible": heavy_atoms > 0 and carbon_atoms > 0,
        "reason": "ok" if heavy_atoms > 0 and carbon_atoms > 0 else "ccd_not_organic",
        "heavy_atoms": heavy_atoms,
        "carbon_atoms": carbon_atoms,
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "contains_metal": contains_metal,
        "bonds": int(molecule.GetNumBonds()),
        "canonical_smiles": str(
            Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=True, isomericSmiles=True)
        ),
    }


def _public_site_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "chain_indices"}


def _site_mapping_fraction(
    apo_site_indices: Sequence[int],
    holo_site_indices: Sequence[int],
    exact_pairs: Sequence[tuple[int, int]],
) -> float:
    apo_to_holo = dict(exact_pairs)
    holo_site = set(holo_site_indices)
    matched = sum(apo_to_holo.get(apo_index) in holo_site for apo_index in apo_site_indices)
    return matched / max(min(len(apo_site_indices), len(holo_site_indices)), 1)


def preflight_record(
    record: Mapping[str, Any],
    *,
    structures_dir: Path,
    ccd_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    endpoints = record["endpoints"]
    apo_endpoint, holo_endpoint = endpoints
    base = {
        "schema_version": SCHEMA_VERSION,
        "record_key": record["record_key"],
        "selection_rank": record.get("smoke_selection", {}).get("selection_rank"),
        "source_rows": record.get("source_rows", []),
        "apo": {
            "pdb_id": apo_endpoint["pdb_id"],
            "chain_id": apo_endpoint["chains"][0],
        },
        "holo": {
            "pdb_id": holo_endpoint["pdb_id"],
            "chain_id": holo_endpoint["chains"][0],
        },
        "metadata_proxy": record["metadata"],
    }
    rejection_reasons: list[str] = []
    apo_path = structures_dir / f"{str(apo_endpoint['pdb_id']).upper()}.cif.gz"
    holo_path = structures_dir / f"{str(holo_endpoint['pdb_id']).upper()}.cif.gz"
    missing = [str(path) for path in (apo_path, holo_path) if not path.is_file()]
    if missing:
        return {
            **base,
            "eligible": False,
            "rejection_reasons": ["missing_endpoint_structure"],
            "detail": {"missing": missing},
        }

    try:
        apo_model, _ = load_structure_context(apo_path)
        holo_model, holo_component_types = load_structure_context(holo_path)
        apo_residues = protein_residue_records(apo_model, apo_endpoint["chains"][0])
        holo_residues = protein_residue_records(holo_model, holo_endpoint["chains"][0])
    except KeyError as exc:
        return {
            **base,
            "eligible": False,
            "rejection_reasons": ["declared_chain_missing"],
            "detail": str(exc),
        }
    except Exception as exc:
        return {
            **base,
            "eligible": False,
            "rejection_reasons": ["structure_parse_failed"],
            "detail": str(exc),
        }

    alignment = align_residue_names(
        [residue["name"] for residue in apo_residues],
        [residue["name"] for residue in holo_residues],
    )
    mapping = {
        "method": "single_chain_global_sequence_exact",
        "apo_residues": len(apo_residues),
        "holo_residues": len(holo_residues),
        "mapped_exact_residues": len(alignment.exact_pairs),
        "sequence_identity": alignment.sequence_identity,
        "apo_mapping_fraction": alignment.reference_mapping_fraction,
        "holo_mapping_fraction": alignment.query_mapping_fraction,
        "symmetric_mapping_fraction": alignment.symmetric_mapping_fraction,
    }
    if alignment.sequence_identity < float(config["min_sequence_identity"]):
        rejection_reasons.append("sequence_identity_below_threshold")
    if alignment.symmetric_mapping_fraction < float(config["min_mapping_fraction"]):
        rejection_reasons.append("residue_mapping_below_threshold")

    site = record["metadata"]["binding_site"]
    apo_site = resolve_binding_site_signature(
        apo_residues, site["apo_indices"], site["apo_residues"]
    )
    holo_site = resolve_binding_site_signature(
        holo_residues, site["holo_indices"], site["holo_residues"]
    )
    for role, result in (("apo", apo_site), ("holo", holo_site)):
        if not result["eligible"]:
            rejection_reasons.append(f"{role}_{result['reason']}")
        elif result["coverage"] < 1.0:
            rejection_reasons.append(f"{role}_binding_site_incomplete")
        elif result["signature_match_fraction"] < float(
            config["min_site_signature_match"]
        ):
            rejection_reasons.append(f"{role}_binding_site_signature_mismatch")

    motion: dict[str, Any] | None = None
    site_mapping_fraction: float | None = None
    ligand_resolution: dict[str, Any] | None = None
    hetero_candidates: list[dict[str, Any]] = []
    ccd: dict[str, Any] | None = None

    if alignment.exact_pairs:
        apo_indices = np.asarray([pair[0] for pair in alignment.exact_pairs], dtype=int)
        holo_indices = np.asarray([pair[1] for pair in alignment.exact_pairs], dtype=int)
        apo_ca = np.asarray([residue["ca"] for residue in apo_residues])[apo_indices]
        holo_ca = np.asarray([residue["ca"] for residue in holo_residues])[holo_indices]
        aligned_holo, global_rmsd = kabsch_align(holo_ca, apo_ca)
        displacements = np.linalg.norm(aligned_holo - apo_ca, axis=1)
        motion = {
            "global_ca_rmsd_angstrom": global_rmsd,
            "max_ca_displacement_angstrom": float(displacements.max()),
            "median_ca_displacement_angstrom": float(np.median(displacements)),
            "metadata_backbone_rmsd_angstrom": float(
                record["metadata"]["backbone_rmsd"]
            ),
        }
        if not float(config["min_global_rmsd"]) <= global_rmsd <= float(
            config["max_global_rmsd"]
        ):
            rejection_reasons.append("structure_global_rmsd_outside_threshold")

    if apo_site.get("eligible") and holo_site.get("eligible"):
        site_mapping_fraction = _site_mapping_fraction(
            apo_site["chain_indices"],
            holo_site["chain_indices"],
            alignment.exact_pairs,
        )
        if site_mapping_fraction < float(config["min_site_mapping_fraction"]):
            rejection_reasons.append("binding_site_mapping_below_threshold")

        holo_site_residues = [
            holo_residues[index] for index in holo_site["chain_indices"]
        ]
        hetero_candidates = enumerate_hetero_candidates(
            holo_model,
            holo_site_residues,
            component_types=holo_component_types,
            min_heavy_atoms=int(config["min_heavy_atoms"]),
            max_heavy_atoms=int(config["max_heavy_atoms"]),
            site_distance=float(config["ligand_site_distance"]),
        )
        ligand_resolution = choose_unique_site_ligand(hetero_candidates)
        if not ligand_resolution["eligible"]:
            rejection_reasons.append(str(ligand_resolution["reason"]))
        else:
            ligand = ligand_resolution["ligand"]
            ccd_download = download_ccd(
                str(ligand["resname"]),
                ccd_dir=ccd_dir,
                timeout=float(config["timeout_seconds"]),
                retries=int(config["download_retries"]),
            )
            ccd = {"download": ccd_download}
            if ccd_download["status"] == "failed":
                rejection_reasons.append("ccd_download_failed")
            else:
                validation = validate_ccd_sdf(Path(ccd_download["path"]))
                observed_fraction = float(ligand["heavy_atoms"]) / max(
                    int(validation.get("heavy_atoms", 0)), 1
                )
                validation["observed_heavy_atom_fraction"] = observed_fraction
                ccd["validation"] = validation
                if not validation["eligible"]:
                    rejection_reasons.append(str(validation["reason"]))
                if observed_fraction < float(
                    config["min_observed_heavy_atom_fraction"]
                ):
                    rejection_reasons.append("ligand_observed_heavy_atoms_incomplete")
                if abs(int(validation.get("formal_charge", 0))) > int(
                    config["max_abs_charge"]
                ):
                    rejection_reasons.append("ligand_formal_charge")
                if validation.get("contains_metal"):
                    rejection_reasons.append("ligand_contains_metal")

    return {
        **base,
        "eligible": not rejection_reasons,
        "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
        "mapping": mapping,
        "binding_site": {
            "apo": _public_site_result(apo_site),
            "holo": _public_site_result(holo_site),
            "mapped_fraction": site_mapping_fraction,
        },
        "motion": motion,
        "hetero_candidates": hetero_candidates,
        "ligand_resolution": ligand_resolution,
        "ccd": ccd,
        "claim_boundary": (
            "Structure-smoke eligibility covers endpoint files, declared chains, "
            "sequence/mapping, binding-site resolution, one site-local organic "
            "component, and parseable CCD chemistry only. It is not leakage, "
            "prepared-topology, MD-path, or Path-3 supervision acceptance."
        ),
    }


def summarize(
    records: Sequence[Mapping[str, Any]],
    downloads: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    selection_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rejection_counts: Counter[str] = Counter()
    ligand_resnames: Counter[str] = Counter()
    for result in results:
        rejection_counts.update(str(reason) for reason in result["rejection_reasons"])
        ligand = (result.get("ligand_resolution") or {}).get("ligand")
        if ligand:
            ligand_resnames[str(ligand["resname"])] += 1
    eligible = [result for result in results if result["eligible"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "input": {
            "path": str(selection_path),
            "sha256": file_sha256(selection_path),
            "selected_systems": len(records),
        },
        "config": dict(config),
        "counts": {
            "selected_systems": len(records),
            "endpoint_structures": len(downloads),
            "download_success": sum(row["status"] != "failed" for row in downloads),
            "download_failed": sum(row["status"] == "failed" for row in downloads),
            "preflight_eligible": len(eligible),
            "preflight_rejected": len(results) - len(eligible),
        },
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "resolved_ligand_resnames": dict(sorted(ligand_resnames.items())),
        "eligible_record_keys": [str(result["record_key"]) for result in eligible],
        "claim_boundary": (
            "Passing this local structure smoke does not add a system to the "
            "3,000-pair planning pool or the Path-3 training corpus. Frozen "
            "family/scaffold leakage, prepared-system, atomistic path, and two-"
            "replica consensus gates remain outstanding."
        ),
    }


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), default=_json_default)
                + "\n"
            )


def write_summary_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "selection_rank",
        "record_key",
        "eligible",
        "apo_pdb",
        "apo_chain",
        "holo_pdb",
        "holo_chain",
        "sequence_identity",
        "mapping_fraction",
        "site_mapping_fraction",
        "global_ca_rmsd_angstrom",
        "ligand_resname",
        "ligand_chain",
        "ligand_author_sequence_id",
        "rejection_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            ligand = (result.get("ligand_resolution") or {}).get("ligand") or {}
            writer.writerow(
                {
                    "selection_rank": result.get("selection_rank"),
                    "record_key": result["record_key"],
                    "eligible": result["eligible"],
                    "apo_pdb": result["apo"]["pdb_id"],
                    "apo_chain": result["apo"]["chain_id"],
                    "holo_pdb": result["holo"]["pdb_id"],
                    "holo_chain": result["holo"]["chain_id"],
                    "sequence_identity": (result.get("mapping") or {}).get(
                        "sequence_identity"
                    ),
                    "mapping_fraction": (result.get("mapping") or {}).get(
                        "symmetric_mapping_fraction"
                    ),
                    "site_mapping_fraction": (result.get("binding_site") or {}).get(
                        "mapped_fraction"
                    ),
                    "global_ca_rmsd_angstrom": (result.get("motion") or {}).get(
                        "global_ca_rmsd_angstrom"
                    ),
                    "ligand_resname": ligand.get("resname"),
                    "ligand_chain": ligand.get("chain_id"),
                    "ligand_author_sequence_id": ligand.get("author_sequence_id"),
                    "rejection_reasons": ";".join(result["rejection_reasons"]),
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    for name in (
        "min_sequence_identity",
        "min_mapping_fraction",
        "min_site_signature_match",
        "min_site_mapping_fraction",
        "min_observed_heavy_atom_fraction",
    ):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    records = load_jsonl(args.selection_jsonl)
    structures_dir = args.output_dir / "structures"
    ccd_dir = args.output_dir / "ccd"
    downloads = download_selected_structures(
        records,
        structures_dir=structures_dir,
        workers=args.workers,
        timeout=args.timeout_seconds,
        retries=args.download_retries,
    )
    write_json(args.output_dir / "download_manifest.json", downloads)

    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"selection_jsonl", "output_dir", "workers"}
    }
    results = [
        preflight_record(
            record,
            structures_dir=structures_dir,
            ccd_dir=ccd_dir,
            config=config,
        )
        for record in records
    ]
    report = summarize(
        records,
        downloads,
        results,
        selection_path=args.selection_jsonl,
        config=config,
    )
    write_jsonl(args.output_dir / "structure_preflight_results.jsonl", results)
    write_json(args.output_dir / "structure_preflight_report.json", report)
    write_summary_csv(args.output_dir / "structure_preflight_summary.csv", results)
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
