"""Backbone coordinate, alignment, torsion, and pocket-weight helpers."""

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from Bio.PDB import PDBParser

_current_file = Path(__file__).resolve()
project_root = _current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.stage1.data.residue_constants import restype_order
from src.data.residue_identity import (
    RESIDUE_ALIGNMENT_VERSION,
    ResidueKey,
    iter_standard_residues,
    load_residue_keys,
    residue_key_from_biopython,
    scatter_by_residue_keys,
)


def extract_backbone_coords(pdb_file: Path):
    """Extract complete backbone atoms and canonical residue keys."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_file))

    N_coords = []
    Ca_coords = []
    C_coords = []
    sequence = []
    residue_keys: List[ResidueKey] = []

    for residue in iter_standard_residues(structure):
        try:
            N = residue['N'].get_coord()
            Ca = residue['CA'].get_coord()
            C = residue['C'].get_coord()
        except KeyError:
            continue

        N_coords.append(N)
        Ca_coords.append(Ca)
        C_coords.append(C)
        sequence.append(residue.get_resname().strip().upper())
        residue_keys.append(residue_key_from_biopython(residue.get_parent(), residue))

    N_coords = np.array(N_coords, dtype=np.float32)
    Ca_coords = np.array(Ca_coords, dtype=np.float32)
    C_coords = np.array(C_coords, dtype=np.float32)

    return N_coords, Ca_coords, C_coords, sequence, residue_keys


def _load_backbone_npz(path: Path):
    """Load cached backbone coords and canonical residue keys when available."""
    with np.load(path, allow_pickle=False) as data:
        residue_keys = load_residue_keys(data)
        if residue_keys is not None and 'residue_alignment_version' in data:
            version = str(np.asarray(data['residue_alignment_version']).item())
            if version != RESIDUE_ALIGNMENT_VERSION:
                raise ValueError(
                    f"{path} residue_alignment_version={version!r}, "
                    f"expected {RESIDUE_ALIGNMENT_VERSION!r}"
                )
        return data['N'], data['Ca'], data['C'], residue_keys


def _coords_valid_mask(N_coords: np.ndarray,
                       Ca_coords: np.ndarray,
                       C_coords: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    n_norm = np.linalg.norm(N_coords, axis=-1)
    ca_norm = np.linalg.norm(Ca_coords, axis=-1)
    c_norm = np.linalg.norm(C_coords, axis=-1)
    return (n_norm > eps) & (ca_norm > eps) & (c_norm > eps)


def align_by_residue_ids(
    apo_coords,
    apo_residue_keys: Sequence[ResidueKey],
    holo_coords,
    holo_residue_keys: Sequence[ResidueKey],
    target_residue_keys: Sequence[ResidueKey],
):
    """Align apo/holo backbones onto the apo/ESM canonical residue axis."""
    apo_stacked = np.stack(apo_coords, axis=1).astype(np.float32, copy=False)
    holo_stacked = np.stack(holo_coords, axis=1).astype(np.float32, copy=False)
    apo_out, apo_present = scatter_by_residue_keys(
        apo_stacked,
        apo_residue_keys,
        target_residue_keys,
        label="apo backbone",
    )
    holo_out, holo_present = scatter_by_residue_keys(
        holo_stacked,
        holo_residue_keys,
        target_residue_keys,
        label="holo backbone",
    )
    valid_mask = apo_present & holo_present
    return (
        (apo_out[:, 0], apo_out[:, 1], apo_out[:, 2]),
        (holo_out[:, 0], holo_out[:, 1], holo_out[:, 2]),
        valid_mask
    )


def compute_pocket_weights(ca_coords: np.ndarray,
                           lig_coords: np.ndarray,
                           d0: float = 6.0,
                           tau: float = 1.0) -> np.ndarray:
    """Compute pocket weights from apo CA coords and ligand coords."""
    diff = ca_coords[:, None, :] - lig_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    d_min = dists.min(axis=1)
    w_res = 1.0 / (1.0 + np.exp((d_min - d0) / max(tau, 1e-6)))
    return w_res.astype(np.float32)


def _align_len(N_coords: np.ndarray,
               Ca_coords: np.ndarray,
               C_coords: np.ndarray,
               n_res: int):
    if len(N_coords) == n_res:
        return N_coords, Ca_coords, C_coords
    N_out = np.zeros((n_res, 3), dtype=np.float32)
    Ca_out = np.zeros((n_res, 3), dtype=np.float32)
    C_out = np.zeros((n_res, 3), dtype=np.float32)
    n_valid = min(len(N_coords), n_res)
    N_out[:n_valid] = N_coords[:n_valid]
    Ca_out[:n_valid] = Ca_coords[:n_valid]
    C_out[:n_valid] = C_coords[:n_valid]
    return N_out, Ca_out, C_out


def _align_array(arr: np.ndarray, n_res: int) -> np.ndarray:
    if len(arr) == n_res:
        return arr
    out = np.zeros((n_res,), dtype=arr.dtype)
    n_valid = min(len(arr), n_res)
    out[:n_valid] = arr[:n_valid]
    return out


def _align_matrix(arr: np.ndarray, n_res: int, n_cols: int) -> np.ndarray:
    if arr.shape[0] == n_res and arr.shape[1] == n_cols:
        return arr
    out = np.zeros((n_res, n_cols), dtype=arr.dtype)
    n_valid = min(arr.shape[0], n_res)
    out[:n_valid] = arr[:n_valid, :n_cols]
    return out


def _align_nma(arr: np.ndarray, n_res: int) -> np.ndarray:
    if arr.ndim == 1:
        return _align_array(arr, n_res)
    if arr.shape[0] == n_res:
        return arr
    out = np.zeros((n_res, arr.shape[1]), dtype=arr.dtype)
    n_valid = min(arr.shape[0], n_res)
    out[:n_valid] = arr[:n_valid]
    return out


def _load_torsion_residue_keys(path: Path) -> List[ResidueKey]:
    with np.load(path, allow_pickle=False) as data:
        residue_keys = load_residue_keys(data)
        if residue_keys is None:
            raise ValueError(
                f"{path} has no canonical residue_keys. Upgrade or regenerate the "
                "torsion cache before training."
            )
        if 'residue_alignment_version' not in data:
            raise ValueError(f"{path} missing residue_alignment_version")
        version = str(np.asarray(data['residue_alignment_version']).item())
        if version != RESIDUE_ALIGNMENT_VERSION:
            raise ValueError(
                f"{path} residue_alignment_version={version!r}, "
                f"expected {RESIDUE_ALIGNMENT_VERSION!r}"
            )
        return residue_keys


def _load_torsions(
    path: Path,
    target_residue_keys: Sequence[ResidueKey],
) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        source_residue_keys = load_residue_keys(data)
        if source_residue_keys is None:
            raise ValueError(
                f"{path} has no canonical residue_keys. Upgrade or regenerate the "
                "torsion cache before training."
            )
        if 'residue_alignment_version' not in data:
            raise ValueError(f"{path} missing residue_alignment_version")
        version = str(np.asarray(data['residue_alignment_version']).item())
        if version != RESIDUE_ALIGNMENT_VERSION:
            raise ValueError(
                f"{path} residue_alignment_version={version!r}, "
                f"expected {RESIDUE_ALIGNMENT_VERSION!r}"
            )

        n_source = len(source_residue_keys)
        torsion_source = np.zeros((n_source, 7), dtype=np.float32)
        for column, field in enumerate(('phi', 'psi', 'omega')):
            values = np.asarray(data[field], dtype=np.float32)
            if values.shape != (n_source,):
                raise ValueError(f"{path} {field} shape={values.shape}, expected {(n_source,)}")
            torsion_source[:, column] = values
        chi_source = np.asarray(data['chi'], dtype=np.float32)
        if chi_source.ndim != 2 or chi_source.shape[0] != n_source or chi_source.shape[1] < 4:
            raise ValueError(f"{path} chi shape={chi_source.shape}, expected [{n_source}, >=4]")
        torsion_source[:, 3:7] = chi_source[:, :4]

        chi_mask_source = np.asarray(data['chi_mask'])
        if (
            chi_mask_source.ndim != 2
            or chi_mask_source.shape[0] != n_source
            or chi_mask_source.shape[1] < 4
        ):
            raise ValueError(
                f"{path} chi_mask shape={chi_mask_source.shape}, expected [{n_source}, >=4]"
            )
        chi_mask_source = chi_mask_source[:, :4].astype(np.bool_)

        if 'bb_mask' in data:
            bb_mask_source = np.asarray(data['bb_mask'])
            if bb_mask_source.ndim == 1:
                if bb_mask_source.shape[0] != n_source:
                    raise ValueError(
                        f"{path} bb_mask shape={bb_mask_source.shape}, expected [{n_source}]"
                    )
                bb_mask_source = np.repeat(bb_mask_source[:, None], 3, axis=1)
            elif bb_mask_source.shape != (n_source, 3):
                raise ValueError(
                    f"{path} bb_mask shape={bb_mask_source.shape}, expected {(n_source, 3)}"
                )
            bb_mask_source = bb_mask_source.astype(np.bool_)
        else:
            bb_mask_source = np.ones((n_source, 3), dtype=np.bool_)

        aatype_source = np.asarray(data['aatype']) if 'aatype' in data else None
        if aatype_source is not None and aatype_source.shape != (n_source,):
            raise ValueError(
                f"{path} aatype shape={aatype_source.shape}, expected {(n_source,)}"
            )

    torsion_angles, residue_present = scatter_by_residue_keys(
        torsion_source,
        source_residue_keys,
        target_residue_keys,
        label=f"{path.name} angles",
    )
    chi_mask, chi_present = scatter_by_residue_keys(
        chi_mask_source,
        source_residue_keys,
        target_residue_keys,
        label=f"{path.name} chi_mask",
    )
    bb_mask, bb_present = scatter_by_residue_keys(
        bb_mask_source,
        source_residue_keys,
        target_residue_keys,
        label=f"{path.name} bb_mask",
    )
    if not np.array_equal(residue_present, chi_present) or not np.array_equal(
        residue_present, bb_present
    ):
        raise RuntimeError(f"Internal residue scatter mismatch while loading {path}")

    aatype = None
    if aatype_source is not None:
        aatype, aatype_present = scatter_by_residue_keys(
            aatype_source,
            source_residue_keys,
            target_residue_keys,
            label=f"{path.name} aatype",
        )
        if not np.array_equal(residue_present, aatype_present):
            raise RuntimeError(f"Internal aatype scatter mismatch while loading {path}")

    return {
        'angles': torsion_angles,
        'chi_mask': chi_mask.astype(np.bool_),
        'bb_mask': bb_mask.astype(np.bool_),
        'aatype': aatype,
        'residue_present': residue_present,
    }


def _sequence_to_aatype(sequence: str, n_res: int) -> np.ndarray:
    aatype = np.zeros((n_res,), dtype=np.int64)
    if not sequence:
        return aatype
    for i, aa in enumerate(sequence):
        if i >= n_res:
            break
        aatype[i] = restype_order.get(aa, 20)
    return aatype
