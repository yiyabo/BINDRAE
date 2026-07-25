"""Screen endpoint pairs for small, parameterizable MD pilot systems."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.data.residue_alignment import align_residue_names


METAL_ATOMIC_NUMBERS = frozenset(
    {
        3,
        4,
        11,
        12,
        13,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
    }
)


def read_sample_ids(path: str | Path, *, scan_limit: int, seed: int) -> List[str]:
    sample_ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    sample_ids = list(dict.fromkeys(sample_ids))
    if scan_limit > 0 and len(sample_ids) > scan_limit:
        rng = random.Random(seed)
        sample_ids = rng.sample(sample_ids, scan_limit)
    return sample_ids


def parse_ca_records(path: str | Path) -> Dict[str, Any]:
    coordinates: List[List[float]] = []
    residue_names: List[str] = []
    residue_keys: List[Tuple[str, str, str]] = []
    chains: set[str] = set()
    with Path(path).open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            if line[16].strip() not in {"", "A"}:
                continue
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError:
                continue
            chain = line[21].strip() or "_"
            chains.add(chain)
            coordinates.append(xyz)
            residue_names.append(line[17:20].strip())
            residue_keys.append((chain, line[22:26].strip(), line[26].strip()))
    return {
        "xyz": np.asarray(coordinates, dtype=np.float64).reshape(-1, 3),
        "residue_names": residue_names,
        "residue_keys": residue_keys,
        "chains": sorted(chains),
    }


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Expected matching [N, 3] arrays, got {mobile.shape} and {target.shape}")
    if mobile.shape[0] < 3:
        raise ValueError("At least three coordinates are required for alignment")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_mobile = mobile - mobile_center
    centered_target = target - target_center
    left, _, right = np.linalg.svd(centered_mobile.T @ centered_target)
    correction = np.sign(np.linalg.det(left @ right))
    rotation = left @ np.diag([1.0, 1.0, correction]) @ right
    aligned = centered_mobile @ rotation + target_center
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return aligned, rmsd


def _canonical_fragment_smiles(fragment: Any, chem: Any) -> str:
    return str(chem.MolToSmiles(chem.RemoveHs(fragment), canonical=True, isomericSmiles=True))


def describe_ligand(
    sdf_path: str | Path,
    *,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    max_abs_charge: int,
) -> Dict[str, Any]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        return {"eligible": False, "reason": "rdkit_unavailable", "detail": str(exc)}

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None:
        return {"eligible": False, "reason": "ligand_parse_failed"}

    try:
        fragments = list(Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True))
    except Exception as exc:
        return {"eligible": False, "reason": "ligand_fragment_failed", "detail": str(exc)}

    fragment_rows: List[Tuple[Any, str, int, int]] = []
    for fragment in fragments:
        smiles = _canonical_fragment_smiles(fragment, Chem)
        heavy_atoms = int(fragment.GetNumHeavyAtoms())
        carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())
        fragment_rows.append((fragment, smiles, heavy_atoms, carbon_atoms))
    organic = [row for row in fragment_rows if row[3] > 0]
    if not organic:
        return {
            "eligible": False,
            "reason": "no_organic_fragment",
            "fragment_count": len(fragment_rows),
        }

    unique_organic = sorted({row[1] for row in organic})
    if len(unique_organic) != 1:
        return {
            "eligible": False,
            "reason": "multiple_unique_organic_fragments",
            "fragment_count": len(fragment_rows),
            "unique_organic_fragments": len(unique_organic),
        }

    representative = next(row[0] for row in organic if row[1] == unique_organic[0])
    heavy_atoms = int(representative.GetNumHeavyAtoms())
    formal_charge = int(Chem.GetFormalCharge(representative))
    contains_metal = any(
        atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS for atom in representative.GetAtoms()
    )
    reasons: List[str] = []
    if heavy_atoms < min_heavy_atoms or heavy_atoms > max_heavy_atoms:
        reasons.append("heavy_atom_count")
    if abs(formal_charge) > max_abs_charge:
        reasons.append("formal_charge")
    if contains_metal:
        reasons.append("contains_metal")

    inchikey: Optional[str]
    try:
        inchikey = str(Chem.MolToInchiKey(representative)) or None
    except Exception:
        inchikey = None
    conformer = representative.GetConformer()
    ligand_xyz = np.asarray(conformer.GetPositions(), dtype=np.float64)
    return {
        "eligible": not reasons,
        "reason": ";".join(reasons) if reasons else "ok",
        "heavy_atoms": heavy_atoms,
        "formal_charge": formal_charge,
        "contains_metal": contains_metal,
        "fragment_count": len(fragment_rows),
        "organic_copy_count": sum(row[1] == unique_organic[0] for row in organic),
        "extra_nonorganic_fragments": len(fragment_rows) - len(organic),
        "canonical_smiles": unique_organic[0],
        "inchikey": inchikey,
        "ligand_xyz": ligand_xyz,
    }


def _minimum_distances(points: np.ndarray, ligand_xyz: np.ndarray) -> np.ndarray:
    squared = np.sum((points[:, None, :] - ligand_xyz[None, :, :]) ** 2, axis=-1)
    return np.sqrt(np.min(squared, axis=1))


def _motion_category(
    *,
    global_rmsd: float,
    pocket_rmsd: float,
    moving_fraction: float,
    contact_changes: int,
) -> str:
    if contact_changes >= 3:
        return "contact_switch"
    if pocket_rmsd >= 0.8 and pocket_rmsd >= 1.35 * max(global_rmsd, 1e-6):
        return "local_pocket"
    if global_rmsd >= 1.5 or moving_fraction >= 0.45:
        return "domain_motion"
    return "moderate_motion"


def _motion_score(
    global_rmsd: float,
    pocket_rmsd: float,
    max_displacement: float,
    contact_changes: int,
) -> float:
    clip = lambda value: min(max(float(value), 0.0), 1.0)
    return float(
        0.30 * clip(global_rmsd / 3.0)
        + 0.35 * clip(pocket_rmsd / 3.0)
        + 0.20 * clip(max_displacement / 8.0)
        + 0.15 * clip(contact_changes / 10.0)
    )


def screen_sample(sample_id: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    sample_dir = Path(str(config["data_dir"])) / "samples" / sample_id
    base: Dict[str, Any] = {"sample_id": sample_id, "sample_dir": str(sample_dir)}
    required = [sample_dir / name for name in ("apo.pdb", "holo.pdb", "ligand.sdf", "meta.json")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return {**base, "eligible": False, "reason": f"missing:{','.join(missing)}"}

    try:
        metadata = json.loads((sample_dir / "meta.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "eligible": False, "reason": "invalid_metadata", "detail": str(exc)}
    base.update(metadata)
    ligand_resname = str(metadata.get("ligand_resname") or "").upper()
    excluded_resnames = {str(value).upper() for value in config.get("excluded_resnames", [])}
    if ligand_resname in excluded_resnames:
        return {**base, "eligible": False, "reason": "excluded_ligand_resname"}

    apo = parse_ca_records(sample_dir / "apo.pdb")
    holo = parse_ca_records(sample_dir / "holo.pdb")
    apo_xyz = apo["xyz"]
    holo_xyz = holo["xyz"]
    if len(apo_xyz) < 3 or len(holo_xyz) < 3:
        return {**base, "eligible": False, "reason": "missing_ca"}
    if len(apo["chains"]) != 1 or len(holo["chains"]) != 1:
        return {**base, "eligible": False, "reason": "not_single_chain"}
    apo_n_residues = int(len(apo_xyz))
    holo_n_residues = int(len(holo_xyz))
    if max(apo_n_residues, holo_n_residues) > int(config["max_residues"]):
        return {
            **base,
            "eligible": False,
            "reason": "residue_count",
            "apo_n_residues": apo_n_residues,
            "holo_n_residues": holo_n_residues,
        }

    alignment = align_residue_names(apo["residue_names"], holo["residue_names"])
    sequence_identity = alignment.sequence_identity
    mapping_fraction = alignment.symmetric_mapping_fraction
    mapped_residues = len(alignment.exact_pairs)
    mapping_fields = {
        "apo_n_residues": apo_n_residues,
        "holo_n_residues": holo_n_residues,
        "mapped_residues": mapped_residues,
        "sequence_identity": sequence_identity,
        "residue_mapping_fraction": mapping_fraction,
        "apo_residue_mapping_fraction": alignment.reference_mapping_fraction,
        "holo_residue_mapping_fraction": alignment.query_mapping_fraction,
        "residue_mapping_method": "single_chain_global_sequence_exact",
    }
    if not int(config["min_residues"]) <= mapped_residues <= int(config["max_residues"]):
        return {
            **base,
            **mapping_fields,
            "eligible": False,
            "reason": "residue_count",
            "n_residues": mapped_residues,
        }
    if sequence_identity < float(config["min_sequence_identity"]):
        return {
            **base,
            **mapping_fields,
            "eligible": False,
            "reason": "sequence_identity",
            "n_residues": mapped_residues,
        }
    if mapping_fraction < float(config["min_residue_mapping_fraction"]):
        return {
            **base,
            **mapping_fields,
            "eligible": False,
            "reason": "residue_mapping",
            "n_residues": mapped_residues,
        }

    apo_indices = np.asarray(
        [reference_index for reference_index, _ in alignment.exact_pairs],
        dtype=np.int64,
    )
    holo_indices = np.asarray(
        [query_index for _, query_index in alignment.exact_pairs],
        dtype=np.int64,
    )
    apo_xyz = apo_xyz[apo_indices]
    holo_xyz = holo_xyz[holo_indices]
    n_residues = mapped_residues

    ligand = describe_ligand(
        sample_dir / "ligand.sdf",
        min_heavy_atoms=int(config["min_heavy_atoms"]),
        max_heavy_atoms=int(config["max_heavy_atoms"]),
        max_abs_charge=int(config["max_abs_charge"]),
    )
    ligand_xyz = ligand.pop("ligand_xyz", None)
    ligand_fields = {f"ligand_{key}": value for key, value in ligand.items()}
    if not ligand.get("eligible") or ligand_xyz is None or len(ligand_xyz) == 0:
        return {
            **base,
            **ligand_fields,
            "eligible": False,
            "reason": f"ligand:{ligand.get('reason', 'unknown')}",
            "n_residues": n_residues,
            **mapping_fields,
        }

    aligned_apo, global_rmsd = kabsch_align(apo_xyz, holo_xyz)
    displacement = np.linalg.norm(aligned_apo - holo_xyz, axis=1)
    raw_rmsd = float(np.sqrt(np.mean(np.sum((apo_xyz - holo_xyz) ** 2, axis=1))))
    apo_distance = _minimum_distances(apo_xyz, ligand_xyz)
    holo_distance = _minimum_distances(holo_xyz, ligand_xyz)
    pocket_mask = np.minimum(apo_distance, holo_distance) <= float(config["pocket_radius"])
    pocket_size = int(pocket_mask.sum())
    if pocket_size < int(config["min_pocket_residues"]):
        return {
            **base,
            **ligand_fields,
            "eligible": False,
            "reason": "pocket_too_small",
            "n_residues": n_residues,
            **mapping_fields,
            "pocket_residues": pocket_size,
            "nearest_ligand_ca": float(min(apo_distance.min(), holo_distance.min())),
        }

    pocket_rmsd = float(np.sqrt(np.mean(displacement[pocket_mask] ** 2)))
    moving_fraction = float(np.mean(displacement >= float(config["moving_threshold"])))
    pocket_moving_fraction = float(
        np.mean(displacement[pocket_mask] >= float(config["moving_threshold"]))
    )
    apo_contact = apo_distance <= float(config["contact_radius"])
    holo_contact = holo_distance <= float(config["contact_radius"])
    formed = int(np.sum(~apo_contact & holo_contact))
    released = int(np.sum(apo_contact & ~holo_contact))
    contact_changes = formed + released
    max_displacement = float(displacement.max())
    minimum_motion = (
        global_rmsd >= float(config["min_global_rmsd"])
        or pocket_rmsd >= float(config["min_pocket_rmsd"])
    ) and max_displacement >= float(config["min_max_displacement"])
    extreme_motion = (
        global_rmsd > float(config["max_global_rmsd"])
        or pocket_rmsd > float(config["max_pocket_rmsd"])
        or max_displacement > float(config["max_max_displacement"])
    )
    motion_ok = minimum_motion and not extreme_motion
    category = _motion_category(
        global_rmsd=global_rmsd,
        pocket_rmsd=pocket_rmsd,
        moving_fraction=moving_fraction,
        contact_changes=contact_changes,
    )
    result = {
        **base,
        **ligand_fields,
        "eligible": bool(motion_ok),
        "reason": "ok" if motion_ok else ("extreme_motion" if extreme_motion else "insufficient_motion"),
        "apo_chain_observed": apo["chains"][0],
        "holo_chain_observed": holo["chains"][0],
        "n_residues": n_residues,
        **mapping_fields,
        "ca_raw_rmsd": raw_rmsd,
        "ca_aligned_rmsd": global_rmsd,
        "pocket_ca_rmsd": pocket_rmsd,
        "max_ca_displacement": max_displacement,
        "moving_fraction": moving_fraction,
        "pocket_moving_fraction": pocket_moving_fraction,
        "pocket_residues": pocket_size,
        "contact_formed": formed,
        "contact_released": released,
        "contact_changes": contact_changes,
        "motion_category": category,
        "motion_score": _motion_score(global_rmsd, pocket_rmsd, max_displacement, contact_changes),
    }
    setup_penalty = (
        0.03 * max(int(ligand.get("organic_copy_count", 1)) - 1, 0)
        + 0.02 * int(ligand.get("extra_nonorganic_fragments", 0))
        + 0.08
        * min(max((max(apo_n_residues, holo_n_residues) - 350) / 150.0, 0.0), 1.0)
    )
    result["pilot_score"] = float(result["motion_score"] - setup_penalty)
    return result


def screen_samples(
    sample_ids: Sequence[str], config: Mapping[str, Any], *, workers: int
) -> List[Dict[str, Any]]:
    if workers <= 1:
        return [screen_sample(sample_id, config) for sample_id in sample_ids]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(screen_sample, sample_ids, repeat(dict(config)), chunksize=16)
        )


def select_diverse_candidates(rows: Iterable[Mapping[str, Any]], count: int) -> List[Dict[str, Any]]:
    eligible = [dict(row) for row in rows if row.get("eligible") is True]
    eligible.sort(key=lambda row: (-float(row.get("pilot_score", 0.0)), str(row["sample_id"])))
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_category[str(row.get("motion_category", "unknown"))].append(row)
    categories = sorted(by_category, key=lambda key: (-len(by_category[key]), key))

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    endpoint_pairs: set[Tuple[str, str]] = set()
    scaffolds: set[str] = set()

    def add(row: Dict[str, Any], *, require_new_scaffold: bool) -> bool:
        sample_id = str(row["sample_id"])
        endpoint_pair = (str(row.get("apo_pdb", "")), str(row.get("holo_pdb", "")))
        scaffold = str(row.get("ligand_inchikey") or row.get("ligand_canonical_smiles") or "")
        if sample_id in selected_ids or endpoint_pair in endpoint_pairs:
            return False
        if require_new_scaffold and scaffold and scaffold in scaffolds:
            return False
        selected.append(row)
        selected_ids.add(sample_id)
        endpoint_pairs.add(endpoint_pair)
        if scaffold:
            scaffolds.add(scaffold)
        return True

    for require_new_scaffold in (True, False):
        progress = True
        while len(selected) < count and progress:
            progress = False
            for category in categories:
                for row in by_category[category]:
                    if add(row, require_new_scaffold=require_new_scaffold):
                        progress = True
                        break
                if len(selected) >= count:
                    break
    for row in eligible:
        if len(selected) >= count:
            break
        add(row, require_new_scaffold=False)

    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    return selected


def candidate_to_transition_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    from src.data.md_transition_manifest import SCHEMA_VERSION

    sample_id = str(row["sample_id"])
    apo_pdb = str(row.get("apo_pdb") or "").upper()
    holo_pdb = str(row.get("holo_pdb") or "").upper()
    ligand_id = str(row.get("ligand_resname") or "unknown")
    sample_dir = Path(str(row["sample_dir"]))
    sequence_identity = float(row.get("sequence_identity", 0.0))
    mapping_fraction = float(
        row.get("residue_mapping_fraction", sequence_identity)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "transition_id": f"ahoj:{sample_id}:pilot",
        "ensemble_id": f"ahoj:{apo_pdb}:{holo_pdb}:{ligand_id}",
        "status": "metadata_verified",
        "source": {
            "name": "AHoJ-DB",
            "record_url": f"https://www.ebi.ac.uk/pdbe/entry/pdb/{str(row.get('query_pdb') or holo_pdb).lower()}",
            "license": "PDB source terms; verify AHoJ-DB redistribution terms",
        },
        "evidence": {
            "tier": "context_equilibrium",
            "contains_endpoint_transition": False,
            "biased_sampling": False,
            "physical_time_interpretable": False,
        },
        "protein": {
            "uniprot_id": None,
            "chain_ids": [
                str(row.get("apo_chain") or row.get("apo_chain_observed") or ""),
                str(row.get("holo_chain") or row.get("holo_chain_observed") or ""),
            ],
            "sequence_sha256": None,
        },
        "ligand": {
            "comp_id": ligand_id,
            "inchikey": row.get("ligand_inchikey"),
            "canonical_smiles": row.get("ligand_canonical_smiles"),
        },
        "endpoints": {
            "apo_pdb_id": apo_pdb or None,
            "holo_pdb_id": holo_pdb or None,
            "apo_structure_path": str(sample_dir / "apo.pdb"),
            "holo_structure_path": str(sample_dir / "holo.pdb"),
        },
        "trajectory": {
            "topology_path": None,
            "coordinate_paths": [],
            "n_frames": None,
            "frame_interval_ps": None,
        },
        "usage": {
            "phase_supervision": False,
            "heldout_benchmark": False,
            "kinetics_claims": False,
        },
        "split": {
            "name": "unassigned",
            "family_group": None,
            "ligand_scaffold_group": row.get("ligand_inchikey"),
        },
        "quality": {
            "endpoint_mapping_verified": bool(
                sequence_identity >= 0.95 and mapping_fraction >= 0.95
            ),
            "residue_mapping_fraction": mapping_fraction,
            "transition_verified": False,
            "notes": [
                "AHoJ apo/holo endpoints selected for MD setup pilot; no intermediate trajectory yet."
            ],
        },
        "screening": {
            key: row.get(key)
            for key in (
                "selection_rank",
                "n_residues",
                "apo_n_residues",
                "holo_n_residues",
                "mapped_residues",
                "sequence_identity",
                "residue_mapping_fraction",
                "residue_mapping_method",
                "ca_aligned_rmsd",
                "pocket_ca_rmsd",
                "max_ca_displacement",
                "contact_changes",
                "motion_category",
                "motion_score",
                "pilot_score",
                "ligand_heavy_atoms",
                "ligand_formal_charge",
                "ligand_fragment_count",
                "ligand_organic_copy_count",
            )
        },
    }


def summarize_screen(rows: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    reasons = Counter(str(row.get("reason", "unknown")) for row in rows)
    categories = Counter(str(row.get("motion_category", "unknown")) for row in selected)
    eligible = [row for row in rows if row.get("eligible") is True]

    def endpoint_pairs(values: Sequence[Mapping[str, Any]]) -> set[Tuple[str, str]]:
        return {
            (str(row.get("apo_pdb") or "").upper(), str(row.get("holo_pdb") or "").upper())
            for row in values
            if row.get("apo_pdb") and row.get("holo_pdb")
        }

    def median(key: str) -> Optional[float]:
        values = [float(row[key]) for row in eligible if row.get(key) is not None]
        return float(np.median(values)) if values else None

    return {
        "screened": len(rows),
        "eligible": len(eligible),
        "eligible_unique_endpoint_pairs": len(endpoint_pairs(eligible)),
        "selected": len(selected),
        "selected_unique_endpoint_pairs": len(endpoint_pairs(selected)),
        "reasons": dict(sorted(reasons.items())),
        "selected_categories": dict(sorted(categories.items())),
        "eligible_medians": {
            "n_residues": median("n_residues"),
            "ca_aligned_rmsd": median("ca_aligned_rmsd"),
            "pocket_ca_rmsd": median("pocket_ca_rmsd"),
            "ligand_heavy_atoms": median("ligand_heavy_atoms"),
        },
    }
