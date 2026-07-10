"""Canonical residue identity and residue-axis alignment helpers.

The ESM residue axis follows the apo PDB residue order. Every structure-derived
array must be mapped onto that axis by a full residue key instead of by array
prefix or integer residue-number offsets.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np


ResidueKey = Tuple[str, int, str]

RESIDUE_ALIGNMENT_VERSION = "canonical_residue_key_v2_esm_compatible"
_KEY_SEPARATOR = "|"

STANDARD_AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def normalize_residue_key(key: Sequence[object]) -> ResidueKey:
    if len(key) != 3:
        raise ValueError(f"Residue key must have three fields, got {key!r}")
    chain, residue_number, insertion_code = key
    return (
        str(chain).strip(),
        int(residue_number),
        str(insertion_code).strip(),
    )


def residue_key_from_biopython(chain, residue) -> ResidueKey:
    """Build ``(chain, resseq, insertion_code)`` from a Bio.PDB residue."""
    _, residue_number, insertion_code = residue.get_id()
    return normalize_residue_key((chain.id, residue_number, insertion_code))


def is_standard_residue(residue) -> bool:
    residue_name = residue.get_resname().strip().upper()
    # Match the historical ESM cache contract: Bio.PDB ``is_aa(...,
    # standard=True)`` accepts standard amino-acid names even when a repaired
    # PDB writes the residue as HETATM.
    return residue_name in STANDARD_AA3_TO_1


def iter_standard_residues(structure) -> Iterable[object]:
    """Yield standard amino acids from the first PDB model in file order."""
    model = next(structure.get_models(), None)
    if model is None:
        return
    for chain in model:
        for residue in chain:
            if is_standard_residue(residue):
                yield residue


def residue_names_to_sequence(residue_names: Sequence[str]) -> str:
    return "".join(STANDARD_AA3_TO_1.get(str(name).strip().upper(), "X") for name in residue_names)


def serialize_residue_key(key: Sequence[object]) -> str:
    chain, residue_number, insertion_code = normalize_residue_key(key)
    if _KEY_SEPARATOR in chain or _KEY_SEPARATOR in insertion_code:
        raise ValueError(f"Residue key contains reserved separator {_KEY_SEPARATOR!r}: {key!r}")
    return f"{chain}{_KEY_SEPARATOR}{residue_number}{_KEY_SEPARATOR}{insertion_code}"


def deserialize_residue_key(value: object) -> ResidueKey:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parts = str(value).split(_KEY_SEPARATOR)
    if len(parts) != 3:
        raise ValueError(f"Malformed serialized residue key: {value!r}")
    return normalize_residue_key(parts)


def residue_keys_to_array(keys: Sequence[Sequence[object]]) -> np.ndarray:
    return np.asarray([serialize_residue_key(key) for key in keys], dtype=np.str_)


def residue_keys_from_array(values: object) -> List[ResidueKey]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"residue_keys must be one-dimensional, got shape={array.shape}")
    return [deserialize_residue_key(value) for value in array.tolist()]


def load_residue_keys(container: Mapping[str, object], field: str = "residue_keys") -> List[ResidueKey] | None:
    if field not in container:
        return None
    return residue_keys_from_array(container[field])


def residue_identity_hash(keys: Sequence[Sequence[object]]) -> str:
    payload = "\n".join(
        [RESIDUE_ALIGNMENT_VERSION, *(serialize_residue_key(key) for key in keys)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _single_chain(keys: Sequence[ResidueKey]) -> bool:
    return len({key[0] for key in keys}) <= 1


def _alignment_token(key: ResidueKey, ignore_chain: bool) -> ResidueKey:
    if ignore_chain:
        return ("", key[1], key[2])
    return key


def _build_unique_index(
    keys: Sequence[ResidueKey],
    *,
    label: str,
    ignore_chain: bool,
) -> dict[ResidueKey, int]:
    index: dict[ResidueKey, int] = {}
    for position, key in enumerate(keys):
        token = _alignment_token(key, ignore_chain)
        if token in index:
            raise ValueError(
                f"Duplicate residue identity in {label}: {serialize_residue_key(key)!r} "
                f"at positions {index[token]} and {position}"
            )
        index[token] = position
    return index


def scatter_by_residue_keys(
    values: np.ndarray,
    source_keys: Sequence[Sequence[object]],
    target_keys: Sequence[Sequence[object]],
    *,
    label: str,
    fill_value: object = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter the first axis of ``values`` onto the canonical target axis.

    Chain labels are ignored only when both inputs are single-chain. This
    handles harmless apo/holo chain renaming while retaining full identity for
    multi-chain structures.
    """
    values = np.asarray(values)
    normalized_source = [normalize_residue_key(key) for key in source_keys]
    normalized_target = [normalize_residue_key(key) for key in target_keys]
    if values.ndim < 1:
        raise ValueError(f"{label} must have at least one dimension")
    if values.shape[0] != len(normalized_source):
        raise ValueError(
            f"{label} first dimension {values.shape[0]} does not match "
            f"source residue count {len(normalized_source)}"
        )

    ignore_chain = _single_chain(normalized_source) and _single_chain(normalized_target)
    source_index = _build_unique_index(
        normalized_source,
        label=f"{label} source",
        ignore_chain=ignore_chain,
    )
    _build_unique_index(
        normalized_target,
        label=f"{label} target",
        ignore_chain=ignore_chain,
    )

    output = np.full(
        (len(normalized_target), *values.shape[1:]),
        fill_value,
        dtype=values.dtype,
    )
    present = np.zeros((len(normalized_target),), dtype=np.bool_)
    for target_index, key in enumerate(normalized_target):
        source_index_value = source_index.get(_alignment_token(key, ignore_chain))
        if source_index_value is None:
            continue
        output[target_index] = values[source_index_value]
        present[target_index] = True
    return output, present
