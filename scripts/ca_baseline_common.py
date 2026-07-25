"""Shared coordinate handling for external C-alpha path baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np


def _record_coordinates(records: Sequence[Dict[str, object]]) -> np.ndarray:
    if not records:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([np.asarray(record["xyz"], dtype=np.float64) for record in records])


def _record_tokens(records: Sequence[Dict[str, object]]) -> list[tuple[int, str]]:
    return [
        (int(str(record["resseq"])), str(record["icode"]).strip())
        for record in records
    ]


def _cache_tokens(path: Path) -> list[tuple[int, str]] | None:
    with np.load(path, allow_pickle=False) as data:
        if "residue_keys" not in data:
            return None
        raw_keys = np.asarray(data["residue_keys"]).tolist()
    tokens = []
    for raw_key in raw_keys:
        if isinstance(raw_key, bytes):
            raw_key = raw_key.decode("utf-8")
        parts = str(raw_key).split("|")
        if len(parts) != 3:
            raise ValueError(f"Malformed residue key in {path}: {raw_key!r}")
        tokens.append((int(parts[1]), parts[2].strip()))
    return tokens


def canonical_ca_coordinates(
    sample_dir: Path,
    endpoint: str,
    records: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, str]:
    """Load the same aligned backbone coordinates consumed by Stage-2."""
    if endpoint not in {"apo", "holo"}:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    raw = _record_coordinates(records)
    cache_path = sample_dir / f"{endpoint}_backbone.npz"
    if not cache_path.is_file():
        return raw, "pdb"
    with np.load(cache_path, allow_pickle=False) as data:
        if "Ca" not in data:
            raise ValueError(f"{cache_path} is missing Ca coordinates")
        cached = np.asarray(data["Ca"], dtype=np.float64)
    if cached.shape != raw.shape:
        raise ValueError(
            f"{cache_path} Ca shape={cached.shape}, PDB template shape={raw.shape}"
        )
    if not np.isfinite(cached).all():
        raise ValueError(f"{cache_path} contains non-finite Ca coordinates")
    return cached, "backbone_cache"


def canonical_ca_pair(
    sample_dir: Path,
    apo_records: Sequence[Dict[str, object]],
    holo_records: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, str]:
    apo_cache = sample_dir / "apo_backbone.npz"
    holo_cache = sample_dir / "holo_backbone.npz"
    apo_tokens = _cache_tokens(apo_cache) if apo_cache.is_file() else None
    holo_tokens = _cache_tokens(holo_cache) if holo_cache.is_file() else None
    if apo_tokens is not None and apo_tokens != _record_tokens(apo_records):
        raise ValueError(f"{apo_cache} residue axis does not match apo PDB records")
    if holo_tokens is not None and holo_tokens != _record_tokens(holo_records):
        raise ValueError(f"{holo_cache} residue axis does not match holo PDB records")
    if apo_tokens is not None and holo_tokens is not None and apo_tokens != holo_tokens:
        raise ValueError(
            f"Canonical endpoint residue axes differ under {sample_dir}; "
            "audit the sample before running a path baseline"
        )
    apo, apo_source = canonical_ca_coordinates(sample_dir, "apo", apo_records)
    holo, holo_source = canonical_ca_coordinates(sample_dir, "holo", holo_records)
    if apo.shape != holo.shape:
        raise ValueError(f"Canonical endpoint shape mismatch: apo={apo.shape}, holo={holo.shape}")
    source = apo_source if apo_source == holo_source else f"{apo_source}+{holo_source}"
    return apo, holo, source


def write_ca_only_pdb(
    path: Path,
    template: Sequence[Dict[str, object]],
    coordinates: np.ndarray,
) -> None:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (len(template), 3):
        raise ValueError(
            f"Coordinate/template mismatch: coords={coordinates.shape}, template={len(template)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, (record, xyz) in enumerate(zip(template, coordinates), start=1):
        chain = " " if str(record["chain"]) == "_" else str(record["chain"])[:1]
        resseq = str(record["resseq"])[:4].rjust(4)
        icode = str(record["icode"])[:1] or " "
        resname = str(record["resname"])[:3].rjust(3)
        x, y, z = map(float, xyz)
        lines.append(
            f"ATOM  {index:5d}  CA  {resname} {chain}{resseq}{icode}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")
