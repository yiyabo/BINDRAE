"""
Stage-2 dataset (apo/holo/ligand triplets) for bridge flow training.

Aligned to current Stage-2 spec:
- Input: apo + holo torsions, apo/holo backbone frames, ESM embeddings, ligand tokens
- Pocket weights computed in apo frame by default
- Optional NMA features for gating/weight closure
"""

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.ligand_utils import build_ligand_tokens_from_file, LIGAND_TYPE_DIM
from src.stage1.data.residue_constants import restype_order


@dataclass
class Stage2Batch:
    """Stage-2 batch (apo/holo/ligand)."""
    # ESM embeddings
    esm: torch.Tensor            # [B, N, 1280]

    # Apo/Holo torsions
    torsion_apo: torch.Tensor    # [B, N, 7]
    torsion_holo: torch.Tensor   # [B, N, 7]

    # Masks
    bb_mask: torch.Tensor        # [B, N, 3]
    chi_mask: torch.Tensor       # [B, N, 4]
    node_mask: torch.Tensor      # [B, N]

    # Backbone coords (for building frames)
    N_apo: torch.Tensor          # [B, N, 3]
    Ca_apo: torch.Tensor         # [B, N, 3]
    C_apo: torch.Tensor          # [B, N, 3]
    N_holo: torch.Tensor         # [B, N, 3]
    Ca_holo: torch.Tensor        # [B, N, 3]
    C_holo: torch.Tensor         # [B, N, 3]

    # Ligand tokens
    lig_points: torch.Tensor     # [B, M, 3]
    lig_types: torch.Tensor      # [B, M, LIGAND_TYPE_DIM]
    lig_mask: torch.Tensor       # [B, M]

    # Pocket weights
    w_res: torch.Tensor          # [B, N]

    # Optional NMA features
    nma_features: Optional[torch.Tensor]  # [B, N, K] or None

    # Sequence / aatype
    aatype: torch.Tensor         # [B, N]
    sequences: List[str]

    # Meta
    pdb_ids: List[str]
    n_residues: List[int]


# -----------------------------
# PDB parsing
# -----------------------------

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
    
    for out_idx, res_id in enumerate(sorted(common_ids)):
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


# -----------------------------
# Pocket weights
# -----------------------------

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


# -----------------------------
# Dataset
# -----------------------------


class ApoHoloBridgeDataset(Dataset):
    """Apo/Holo/Ligand triplet dataset for Stage-2."""

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 index_file: Optional[str] = None,
                 max_lig_tokens: int = 128,
                 require_nma: bool = False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_lig_tokens = max_lig_tokens
        self.require_nma = require_nma

        self.samples = self._load_index(index_file)
        print(f"✓ Stage-2 {split} samples: {len(self.samples)}")

    def _load_index(self, index_file: Optional[str]) -> List[Dict]:
        if index_file is not None:
            path = self.data_dir / index_file
            return self._read_index(path)

        split_path = self.data_dir / 'splits' / f'{self.split}.json'
        if split_path.exists():
            return self._read_index(split_path)

        index_path = self.data_dir / 'index.json'
        if index_path.exists():
            return self._read_index(index_path)

        raise FileNotFoundError(
            f"No index file found under {self.data_dir}. "
            "Provide index_file or create splits/<split>.json or index.json."
        )

    def _read_index(self, path: Path) -> List[Dict]:
        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], str)):
            return [{'id': x} for x in data]

        if isinstance(data, dict) and self.split in data:
            entries = data[self.split]
            if isinstance(entries, list) and (len(entries) == 0 or isinstance(entries[0], str)):
                return [{'id': x} for x in entries]
            return entries

        if isinstance(data, list) and isinstance(data[0], dict):
            if 'split' in data[0]:
                return [x for x in data if x.get('split') == self.split]
            return data

        raise ValueError(f"Unsupported index format in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, sample: Dict, key: str, default_name: str) -> Optional[Path]:
        if key in sample:
            p = Path(sample[key])
            return p if p.is_absolute() else (self.data_dir / p)
        if 'id' in sample:
            return self.data_dir / 'samples' / sample['id'] / default_name
        return None

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        sample_id = sample.get('id', f'sample_{idx}')

        # ESM features
        esm_path = self._resolve_path(sample, 'esm_path', 'esm.pt')
        if esm_path is None or not esm_path.exists():
            raise FileNotFoundError(f"ESM not found for {sample_id}")
        esm_data = torch.load(esm_path, weights_only=False)
        esm_features = esm_data['per_residue'].numpy()
        sequence_str = esm_data.get('sequence_str', '')
        n_res = len(esm_features)

        # Apo/Holo backbone
        apo_pdb = self._resolve_path(sample, 'apo_pdb', 'apo.pdb')
        holo_pdb = self._resolve_path(sample, 'holo_pdb', 'holo.pdb')
        if apo_pdb is None or holo_pdb is None:
            raise FileNotFoundError(f"apo/holo PDB not found for {sample_id}")

        apo_backbone = self._resolve_path(sample, 'apo_backbone', 'apo_backbone.npz')
        holo_backbone = self._resolve_path(sample, 'holo_backbone', 'holo_backbone.npz')

        apo_res_ids = None
        holo_res_ids = None

        if apo_backbone is not None and apo_backbone.exists():
            N_apo, Ca_apo, C_apo, apo_res_ids = _load_backbone_npz(apo_backbone)
        else:
            N_apo, Ca_apo, C_apo, _, apo_res_ids = extract_backbone_coords(apo_pdb)

        if holo_backbone is not None and holo_backbone.exists():
            N_holo, Ca_holo, C_holo, holo_res_ids = _load_backbone_npz(holo_backbone)
        else:
            N_holo, Ca_holo, C_holo, _, holo_res_ids = extract_backbone_coords(holo_pdb)

        # Align using residue IDs if available
        if apo_res_ids is not None and holo_res_ids is not None:
            (N_apo, Ca_apo, C_apo), (N_holo, Ca_holo, C_holo), node_mask = align_by_residue_ids(
                (N_apo, Ca_apo, C_apo), apo_res_ids,
                (N_holo, Ca_holo, C_holo), holo_res_ids,
                n_res
            )
        else:
            N_apo, Ca_apo, C_apo = _align_len(N_apo, Ca_apo, C_apo, n_res)
            N_holo, Ca_holo, C_holo = _align_len(N_holo, Ca_holo, C_holo, n_res)
            node_mask = _coords_valid_mask(N_apo, Ca_apo, C_apo) & _coords_valid_mask(N_holo, Ca_holo, C_holo)

        # Ligand tokens
        lig_coords_path = self._resolve_path(sample, 'ligand_coords', 'ligand_coords.npy')
        lig_sdf_path = self._resolve_path(sample, 'ligand_sdf', 'ligand.sdf')
        if lig_coords_path is None or not lig_coords_path.exists():
            raise FileNotFoundError(f"Ligand coords not found for {sample_id}")

        lig_tokens = build_ligand_tokens_from_file(
            lig_coords_path,
            lig_sdf_path,
            max_tokens=self.max_lig_tokens
        )

        # Torsions
        torsion_apo_path = self._resolve_path(sample, 'torsion_apo', 'torsion_apo.npz')
        torsion_holo_path = self._resolve_path(sample, 'torsion_holo', 'torsion_holo.npz')
        if torsion_apo_path is None or torsion_holo_path is None:
            raise FileNotFoundError(f"torsion files not found for {sample_id}")

        torsion_apo = _load_torsions(torsion_apo_path, n_res)
        torsion_holo = _load_torsions(torsion_holo_path, n_res)

        # AAtype
        aatype = torsion_apo.get('aatype')
        if aatype is None:
            aatype = _sequence_to_aatype(sequence_str, n_res)
        else:
            aatype = _align_array(aatype, n_res)

        # Pocket weights (apo frame)
        w_res_path = self._resolve_path(sample, 'w_res', 'w_res.npy')
        if w_res_path is not None and w_res_path.exists():
            w_res = np.load(w_res_path).astype(np.float32)
            w_res = _align_array(w_res, n_res)
        else:
            w_res = compute_pocket_weights(Ca_apo, lig_tokens['coords'])
        w_res = w_res * node_mask.astype(np.float32)

        bb_mask = torsion_apo['bb_mask'] & node_mask[:, None]
        chi_mask = torsion_apo['chi_mask'] & node_mask[:, None]

        # Optional NMA features
        nma_features = None
        nma_path = self._resolve_path(sample, 'nma_features', 'nma_features.npy')
        if nma_path is not None and nma_path.exists():
            nma_features = np.load(nma_path)
            nma_features = _align_nma(nma_features, n_res)
        elif self.require_nma:
            raise FileNotFoundError(f"NMA features not found for {sample_id}")

        return {
            'id': sample_id,
            'esm': esm_features,
            'sequence': sequence_str,
            'N_apo': N_apo,
            'Ca_apo': Ca_apo,
            'C_apo': C_apo,
            'N_holo': N_holo,
            'Ca_holo': Ca_holo,
            'C_holo': C_holo,
            'torsion_apo': torsion_apo['angles'],
            'torsion_holo': torsion_holo['angles'],
            'bb_mask': bb_mask,
            'chi_mask': chi_mask,
            'aatype': aatype,
            'lig_points': lig_tokens['coords'],
            'lig_types': lig_tokens['types'],
            'w_res': w_res,
            'nma_features': nma_features,
            'n_residues': n_res,
            'node_mask': node_mask,
        }


# -----------------------------
# Collate
# -----------------------------


def collate_stage2_batch(samples: List[Dict]) -> Stage2Batch:
    batch_size = len(samples)
    max_n_res = max(s['n_residues'] for s in samples)
    max_lig = max(len(s['lig_points']) for s in samples)

    esm_batch = np.zeros((batch_size, max_n_res, 1280), dtype=np.float32)
    torsion_apo = np.zeros((batch_size, max_n_res, 7), dtype=np.float32)
    torsion_holo = np.zeros((batch_size, max_n_res, 7), dtype=np.float32)
    bb_mask = np.zeros((batch_size, max_n_res, 3), dtype=bool)
    chi_mask = np.zeros((batch_size, max_n_res, 4), dtype=bool)
    node_mask = np.zeros((batch_size, max_n_res), dtype=bool)

    N_apo = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    Ca_apo = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    C_apo = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    N_holo = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    Ca_holo = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    C_holo = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)

    lig_points = np.zeros((batch_size, max_lig, 3), dtype=np.float32)
    lig_types = np.zeros((batch_size, max_lig, LIGAND_TYPE_DIM), dtype=np.float32)
    lig_mask = np.zeros((batch_size, max_lig), dtype=bool)

    w_res = np.zeros((batch_size, max_n_res), dtype=np.float32)

    nma_dim = None
    for s in samples:
        if s['nma_features'] is not None:
            nma_dim = s['nma_features'].shape[-1] if s['nma_features'].ndim > 1 else 1
            break
    nma_features = None
    if nma_dim is not None:
        nma_features = np.zeros((batch_size, max_n_res, nma_dim), dtype=np.float32)

    aatype = np.zeros((batch_size, max_n_res), dtype=np.int64)

    pdb_ids = []
    n_residues = []
    sequences = []

    for i, sample in enumerate(samples):
        n_res = sample['n_residues']
        n_lig = len(sample['lig_points'])

        esm_batch[i, :n_res] = sample['esm']
        torsion_apo[i, :n_res] = sample['torsion_apo']
        torsion_holo[i, :n_res] = sample['torsion_holo']
        bb_mask[i, :n_res] = sample['bb_mask']
        chi_mask[i, :n_res] = sample['chi_mask']
        sample_mask = sample.get('node_mask')
        if sample_mask is None:
            sample_mask = np.ones((n_res,), dtype=bool)
        node_mask[i, :n_res] = sample_mask

        N_apo[i, :n_res] = sample['N_apo']
        Ca_apo[i, :n_res] = sample['Ca_apo']
        C_apo[i, :n_res] = sample['C_apo']
        N_holo[i, :n_res] = sample['N_holo']
        Ca_holo[i, :n_res] = sample['Ca_holo']
        C_holo[i, :n_res] = sample['C_holo']

        lig_points[i, :n_lig] = sample['lig_points']
        lig_types[i, :n_lig] = sample['lig_types']
        lig_mask[i, :n_lig] = True

        w_res[i, :n_res] = sample['w_res']

        if nma_features is not None and sample['nma_features'] is not None:
            nma = sample['nma_features']
            if nma.ndim == 1:
                nma = nma[:, None]
            nma_features[i, :n_res, :nma.shape[-1]] = nma

        aatype[i, :n_res] = sample['aatype']

        pdb_ids.append(sample['id'])
        n_residues.append(n_res)
        sequences.append(sample.get('sequence', ''))

    return Stage2Batch(
        esm=torch.from_numpy(esm_batch),
        torsion_apo=torch.from_numpy(torsion_apo),
        torsion_holo=torch.from_numpy(torsion_holo),
        bb_mask=torch.from_numpy(bb_mask),
        chi_mask=torch.from_numpy(chi_mask),
        node_mask=torch.from_numpy(node_mask),
        N_apo=torch.from_numpy(N_apo),
        Ca_apo=torch.from_numpy(Ca_apo),
        C_apo=torch.from_numpy(C_apo),
        N_holo=torch.from_numpy(N_holo),
        Ca_holo=torch.from_numpy(Ca_holo),
        C_holo=torch.from_numpy(C_holo),
        lig_points=torch.from_numpy(lig_points),
        lig_types=torch.from_numpy(lig_types),
        lig_mask=torch.from_numpy(lig_mask),
        w_res=torch.from_numpy(w_res),
        nma_features=torch.from_numpy(nma_features) if nma_features is not None else None,
        aatype=torch.from_numpy(aatype),
        sequences=sequences,
        pdb_ids=pdb_ids,
        n_residues=n_residues,
    )


# -----------------------------
# DataLoader factory
# -----------------------------


def create_stage2_dataloader(data_dir: str,
                             split: str = 'train',
                             batch_size: int = 2,
                             shuffle: bool = True,
                             num_workers: int = 0,
                             **kwargs):
    from torch.utils.data import DataLoader

    dataset = ApoHoloBridgeDataset(data_dir, split=split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_stage2_batch,
        pin_memory=True,
    )


# -----------------------------
# Helpers
# -----------------------------


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
