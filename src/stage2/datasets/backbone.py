"""Backbone coordinate, alignment, torsion, and pocket-weight helpers."""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
from Bio.PDB import PDBParser

_current_file = Path(__file__).resolve()
project_root = _current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.stage1.data.residue_constants import restype_order


def extract_backbone_coords(pdb_file: Path):
    """Extract N, CA, C coords, sequence and residue IDs from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_file))

    N_coords = []
    Ca_coords = []
    C_coords = []
    sequence = []
    residue_ids = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != ' ':
                    continue
                try:
                    N = residue['N'].get_coord()
                    Ca = residue['CA'].get_coord()
                    C = residue['C'].get_coord()
                except KeyError:
                    continue

                N_coords.append(N)
                Ca_coords.append(Ca)
                C_coords.append(C)
                sequence.append(residue.get_resname())
                residue_ids.append(residue.get_id()[1])

    N_coords = np.array(N_coords, dtype=np.float32)
    Ca_coords = np.array(Ca_coords, dtype=np.float32)
    C_coords = np.array(C_coords, dtype=np.float32)

    return N_coords, Ca_coords, C_coords, sequence, residue_ids


def _load_backbone_npz(path: Path):
    """Load cached backbone coords. Returns residue_ids if stored."""
    data = np.load(path)
    residue_ids = data['residue_ids'].tolist() if 'residue_ids' in data else None
    return data['N'], data['Ca'], data['C'], residue_ids


def _coords_valid_mask(N_coords: np.ndarray,
                       Ca_coords: np.ndarray,
                       C_coords: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    n_norm = np.linalg.norm(N_coords, axis=-1)
    ca_norm = np.linalg.norm(Ca_coords, axis=-1)
    c_norm = np.linalg.norm(C_coords, axis=-1)
    return (n_norm > eps) & (ca_norm > eps) & (c_norm > eps)


def align_by_residue_ids(
    apo_coords, apo_res_ids, holo_coords, holo_res_ids, target_len
):
    """Align apo and holo backbone coords based on residue IDs."""
    N_apo, Ca_apo, C_apo = apo_coords
    N_holo, Ca_holo, C_holo = holo_coords
    
    apo_id_to_idx = {rid: i for i, rid in enumerate(apo_res_ids)}
    holo_id_to_idx = {rid: i for i, rid in enumerate(holo_res_ids)}
    common_ids = set(apo_res_ids) & set(holo_res_ids)
    
    N_apo_out = np.zeros((target_len, 3), dtype=np.float32)
    Ca_apo_out = np.zeros((target_len, 3), dtype=np.float32)
    C_apo_out = np.zeros((target_len, 3), dtype=np.float32)
    N_holo_out = np.zeros((target_len, 3), dtype=np.float32)
    Ca_holo_out = np.zeros((target_len, 3), dtype=np.float32)
    C_holo_out = np.zeros((target_len, 3), dtype=np.float32)
    valid_mask = np.zeros(target_len, dtype=bool)
    
    sorted_common = sorted(common_ids)
    if sorted_common:
        residue_min = sorted_common[0]
        residue_max = sorted_common[-1]
        residue_span = residue_max - residue_min + 1
        residue_offset = residue_min if residue_span <= target_len else residue_max - target_len + 1
    else:
        residue_offset = 0

    placed = 0
    for res_id in sorted_common:
        out_idx = res_id - residue_offset
        if out_idx < 0 or out_idx >= target_len:
            continue
        apo_idx = apo_id_to_idx[res_id]
        holo_idx = holo_id_to_idx[res_id]
        N_apo_out[out_idx] = N_apo[apo_idx]
        Ca_apo_out[out_idx] = Ca_apo[apo_idx]
        C_apo_out[out_idx] = C_apo[apo_idx]
        N_holo_out[out_idx] = N_holo[holo_idx]
        Ca_holo_out[out_idx] = Ca_holo[holo_idx]
        C_holo_out[out_idx] = C_holo[holo_idx]
        valid_mask[out_idx] = True
        placed += 1

    if placed == 0:
        for out_idx, res_id in enumerate(sorted_common):
            if out_idx >= target_len:
                break
            apo_idx = apo_id_to_idx[res_id]
            holo_idx = holo_id_to_idx[res_id]
            N_apo_out[out_idx] = N_apo[apo_idx]
            Ca_apo_out[out_idx] = Ca_apo[apo_idx]
            C_apo_out[out_idx] = C_apo[apo_idx]
            N_holo_out[out_idx] = N_holo[holo_idx]
            Ca_holo_out[out_idx] = Ca_holo[holo_idx]
            C_holo_out[out_idx] = C_holo[holo_idx]
            valid_mask[out_idx] = True
    
    return (
        (N_apo_out, Ca_apo_out, C_apo_out),
        (N_holo_out, Ca_holo_out, C_holo_out),
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


def _load_torsions(path: Path, n_res: int) -> Dict[str, np.ndarray]:
    data = np.load(path)

    torsion_angles = np.zeros((n_res, 7), dtype=np.float32)
    torsion_angles[:, 0] = _align_array(data['phi'], n_res)
    torsion_angles[:, 1] = _align_array(data['psi'], n_res)
    torsion_angles[:, 2] = _align_array(data['omega'], n_res)
    torsion_angles[:, 3:7] = _align_matrix(data['chi'][:, :4], n_res, 4)

    chi_mask = _align_matrix(data['chi_mask'][:, :4], n_res, 4).astype(bool)

    if 'bb_mask' in data:
        bb_mask_raw = data['bb_mask']
        if bb_mask_raw.ndim == 1:
            bb_mask = np.repeat(bb_mask_raw[:, None], 3, axis=1).astype(bool)
        else:
            bb_mask = _align_matrix(bb_mask_raw, n_res, 3).astype(bool)
    else:
        bb_mask = np.ones((n_res, 3), dtype=bool)

    aatype = data['aatype'] if 'aatype' in data else None

    return {
        'angles': torsion_angles,
        'chi_mask': chi_mask,
        'bb_mask': bb_mask,
        'aatype': aatype,
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
