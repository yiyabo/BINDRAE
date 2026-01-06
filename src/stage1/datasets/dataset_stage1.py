"""
Stage-1 triplet dataset (apo/holo/ligand)

This dataset is aligned to the current Stage-1 spec:
- Input: apo backbone + ESM + ligand tokens
- Supervision: holo chi + holo geometry
- Pocket weights computed in apo frame
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# FlashIPA path (项目内 vendor 目录)
flash_ipa_path = str(project_root / 'vendor' / 'flash_ipa' / 'src')
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from utils.ligand_utils import build_ligand_tokens_from_file, LIGAND_TYPE_DIM


@dataclass
class Stage1Batch:
    """Stage-1 batch (apo/holo/ligand)."""
    # ESM
    esm: torch.Tensor            # [B, N, 1280]

    # Apo backbone
    N_apo: torch.Tensor          # [B, N, 3]
    Ca_apo: torch.Tensor         # [B, N, 3]
    C_apo: torch.Tensor          # [B, N, 3]

    # Holo backbone (for true frames / evaluation)
    N_holo: torch.Tensor         # [B, N, 3]
    Ca_holo: torch.Tensor        # [B, N, 3]
    C_holo: torch.Tensor         # [B, N, 3]

    # Masks
    node_mask: torch.Tensor      # [B, N]

    # Ligand tokens
    lig_points: torch.Tensor     # [B, M, 3]
    lig_types: torch.Tensor      # [B, M, LIGAND_TYPE_DIM]
    lig_mask: torch.Tensor       # [B, M]

    # Holo chi supervision
    chi_holo: torch.Tensor       # [B, N, 4]
    chi_mask: torch.Tensor       # [B, N, 4]

    # Apo torsions (phi/psi/omega/chi1-4), used to fill FK
    torsion_apo: torch.Tensor    # [B, N, 7]
    
    # Holo torsions (phi/psi/omega/chi1-4), used for FK-generated atom14
    torsion_holo: torch.Tensor   # [B, N, 7]

    # Pocket weights (computed in apo frame)
    w_res: torch.Tensor          # [B, N]

    # Optional atom14 holo supervision
    atom14_holo: torch.Tensor    # [B, N, 14, 3]
    atom14_holo_mask: torch.Tensor  # [B, N, 14]

    # Meta
    pdb_ids: List[str]
    n_residues: List[int]
    sequences: List[str]


# -----------------------------
# PDB parsing
# -----------------------------

def extract_backbone_coords(pdb_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract N, CA, C coords and sequence from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_file))

    N_coords = []
    Ca_coords = []
    C_coords = []
    sequence = []

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

    N_coords = np.array(N_coords, dtype=np.float32)
    Ca_coords = np.array(Ca_coords, dtype=np.float32)
    C_coords = np.array(C_coords, dtype=np.float32)

    return N_coords, Ca_coords, C_coords, sequence


# -----------------------------
# Pocket weights
# -----------------------------

def compute_pocket_weights(ca_coords: np.ndarray,
                           lig_coords: np.ndarray,
                           d0: float = 6.0,
                           tau: float = 1.0) -> np.ndarray:
    """Compute pocket weights from apo CA coords and ligand coords."""
    # [N, 3] vs [M, 3]
    diff = ca_coords[:, None, :] - lig_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)  # [N, M]
    d_min = dists.min(axis=1)  # [N]
    # sigmoid
    w_res = 1.0 / (1.0 + np.exp((d_min - d0) / max(tau, 1e-6)))
    return w_res.astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------

class ApoHoloTripletDataset(Dataset):
    """Apo/Holo/Ligand triplet dataset for Stage-1."""

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 index_file: Optional[str] = None,
                 max_lig_tokens: int = 128,
                 require_atom14: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_lig_tokens = max_lig_tokens
        self.require_atom14 = require_atom14

        self.samples = self._load_index(index_file)
        print(f"✓ Stage-1 {split} samples: {len(self.samples)}")

    def _load_index(self, index_file: Optional[str]) -> List[Dict]:
        # Prefer explicit index file
        if index_file is not None:
            path = self.data_dir / index_file
            return self._read_index(path)

        # Try split file
        split_path = self.data_dir / 'splits' / f'{self.split}.json'
        if split_path.exists():
            return self._read_index(split_path)

        # Fallback to index.json
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

        # list of ids
        if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], str)):
            return [{'id': x} for x in data]

        # dict with split keys
        if isinstance(data, dict) and self.split in data:
            entries = data[self.split]
            if isinstance(entries, list) and (len(entries) == 0 or isinstance(entries[0], str)):
                return [{'id': x} for x in entries]
            return entries

        # list of dicts
        if isinstance(data, list) and isinstance(data[0], dict):
            # filter by split if present
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
        # default path: data_dir/samples/<id>/<default_name>
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
        esm_features = esm_data['per_residue'].numpy()  # [N, 1280]
        sequence_str = esm_data.get('sequence_str', '')
        n_res = len(esm_features)

        # Apo/Holo backbone
        apo_pdb = self._resolve_path(sample, 'apo_pdb', 'apo.pdb')
        holo_pdb = self._resolve_path(sample, 'holo_pdb', 'holo.pdb')
        if apo_pdb is None or holo_pdb is None:
            raise FileNotFoundError(f"apo/holo PDB not found for {sample_id}")

        N_apo, Ca_apo, C_apo, _ = extract_backbone_coords(apo_pdb)
        N_holo, Ca_holo, C_holo, _ = extract_backbone_coords(holo_pdb)

        # Align to ESM length
        N_apo, Ca_apo, C_apo = _align_len(N_apo, Ca_apo, C_apo, n_res)
        N_holo, Ca_holo, C_holo = _align_len(N_holo, Ca_holo, C_holo, n_res)

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

        chi_holo = torsion_holo['angles'][:, 3:7]
        chi_mask = torsion_holo['chi_mask'][:, :4]

        # Pocket weights (apo frame)
        w_res_path = self._resolve_path(sample, 'w_res', 'w_res.npy')
        if w_res_path is not None and w_res_path.exists():
            w_res = np.load(w_res_path).astype(np.float32)
            w_res = _align_array(w_res, n_res)
        else:
            w_res = compute_pocket_weights(Ca_apo, lig_tokens['coords'])

        # Optional atom14 holo
        atom14_path = self._resolve_path(sample, 'atom14_holo', 'atom14_holo.npy')
        atom14_mask_path = self._resolve_path(sample, 'atom14_holo_mask', 'atom14_holo_mask.npy')
        atom14_holo = None
        atom14_holo_mask = None
        if atom14_path is not None and atom14_path.exists():
            atom14_holo = np.load(atom14_path).astype(np.float32)
            atom14_holo = _align_atom14(atom14_holo, n_res)

            if atom14_mask_path is not None and atom14_mask_path.exists():
                atom14_holo_mask = np.load(atom14_mask_path).astype(bool)
                atom14_holo_mask = _align_atom14_mask(atom14_holo_mask, n_res)
        if self.require_atom14 and atom14_holo is None:
            raise FileNotFoundError(
                f"atom14 holo not found for {sample_id} (require_atom14=True)"
            )

        return {
            'id': sample_id,
            'esm': esm_features,
            'N_apo': N_apo,
            'Ca_apo': Ca_apo,
            'C_apo': C_apo,
            'N_holo': N_holo,
            'Ca_holo': Ca_holo,
            'C_holo': C_holo,
            'sequence': sequence_str,
            'lig_points': lig_tokens['coords'],
            'lig_types': lig_tokens['types'],
            'chi_holo': chi_holo,
            'chi_mask': chi_mask,
            'torsion_apo': torsion_apo['angles'],
            'torsion_holo': torsion_holo['angles'],  # 完整 7 角，用于 FK 生成 atom14
            'w_res': w_res,
            'atom14_holo': atom14_holo,
            'atom14_holo_mask': atom14_holo_mask,
            'n_residues': n_res,
        }


# -----------------------------
# Collate
# -----------------------------

def collate_stage1_batch(samples: List[Dict]) -> Stage1Batch:
    batch_size = len(samples)
    max_n_res = max(s['n_residues'] for s in samples)
    max_lig = max(len(s['lig_points']) for s in samples)

    esm_batch = np.zeros((batch_size, max_n_res, 1280), dtype=np.float32)
    N_apo_batch = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    Ca_apo_batch = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    C_apo_batch = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    N_holo_batch = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    Ca_holo_batch = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    C_holo_batch = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    node_mask = np.zeros((batch_size, max_n_res), dtype=bool)

    lig_points = np.zeros((batch_size, max_lig, 3), dtype=np.float32)
    lig_types = np.zeros((batch_size, max_lig, LIGAND_TYPE_DIM), dtype=np.float32)
    lig_mask = np.zeros((batch_size, max_lig), dtype=bool)

    chi_holo = np.zeros((batch_size, max_n_res, 4), dtype=np.float32)
    chi_mask = np.zeros((batch_size, max_n_res, 4), dtype=bool)
    torsion_apo = np.zeros((batch_size, max_n_res, 7), dtype=np.float32)
    torsion_holo = np.zeros((batch_size, max_n_res, 7), dtype=np.float32)

    w_res = np.zeros((batch_size, max_n_res), dtype=np.float32)

    atom14_holo = np.zeros((batch_size, max_n_res, 14, 3), dtype=np.float32)
    atom14_holo_mask = np.zeros((batch_size, max_n_res, 14), dtype=bool)
    atom14_available = False

    pdb_ids = []
    n_residues = []
    sequences = []

    for i, sample in enumerate(samples):
        n_res = sample['n_residues']
        n_lig = len(sample['lig_points'])

        esm_batch[i, :n_res] = sample['esm']
        N_apo_batch[i, :n_res] = sample['N_apo']
        Ca_apo_batch[i, :n_res] = sample['Ca_apo']
        C_apo_batch[i, :n_res] = sample['C_apo']
        N_holo_batch[i, :n_res] = sample['N_holo']
        Ca_holo_batch[i, :n_res] = sample['Ca_holo']
        C_holo_batch[i, :n_res] = sample['C_holo']
        node_mask[i, :n_res] = True

        lig_points[i, :n_lig] = sample['lig_points']
        lig_types[i, :n_lig] = sample['lig_types']
        lig_mask[i, :n_lig] = True

        chi_holo[i, :n_res] = sample['chi_holo']
        chi_mask[i, :n_res] = sample['chi_mask']
        torsion_apo[i, :n_res] = sample['torsion_apo']
        torsion_holo[i, :n_res] = sample['torsion_holo']

        w_res[i, :n_res] = sample['w_res']

        if sample['atom14_holo'] is not None:
            atom14_available = True
            atom14_holo[i, :n_res] = sample['atom14_holo']
            if sample['atom14_holo_mask'] is not None:
                atom14_holo_mask[i, :n_res] = sample['atom14_holo_mask']
            else:
                atom14_holo_mask[i, :n_res] = True

        pdb_ids.append(sample['id'])
        n_residues.append(n_res)
        sequences.append(sample.get('sequence', ''))

    if not atom14_available:
        atom14_holo = np.zeros((batch_size, max_n_res, 14, 3), dtype=np.float32)
        atom14_holo_mask = np.zeros((batch_size, max_n_res, 14), dtype=bool)

    return Stage1Batch(
        esm=torch.from_numpy(esm_batch),
        N_apo=torch.from_numpy(N_apo_batch),
        Ca_apo=torch.from_numpy(Ca_apo_batch),
        C_apo=torch.from_numpy(C_apo_batch),
        N_holo=torch.from_numpy(N_holo_batch),
        Ca_holo=torch.from_numpy(Ca_holo_batch),
        C_holo=torch.from_numpy(C_holo_batch),
        node_mask=torch.from_numpy(node_mask),
        lig_points=torch.from_numpy(lig_points),
        lig_types=torch.from_numpy(lig_types),
        lig_mask=torch.from_numpy(lig_mask),
        chi_holo=torch.from_numpy(chi_holo),
        chi_mask=torch.from_numpy(chi_mask),
        torsion_apo=torch.from_numpy(torsion_apo),
        torsion_holo=torch.from_numpy(torsion_holo),
        w_res=torch.from_numpy(w_res),
        atom14_holo=torch.from_numpy(atom14_holo),
        atom14_holo_mask=torch.from_numpy(atom14_holo_mask),
        pdb_ids=pdb_ids,
        n_residues=n_residues,
        sequences=sequences,
    )


# -----------------------------
# DataLoader factory
# -----------------------------

def create_stage1_dataloader(data_dir: str,
                             split: str = 'train',
                             batch_size: int = 4,
                             shuffle: bool = True,
                             num_workers: int = 0,
                             **kwargs):
    from torch.utils.data import DataLoader

    dataset = ApoHoloTripletDataset(data_dir, split=split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_stage1_batch,
        pin_memory=True,
    )


# -----------------------------
# Helpers
# -----------------------------

def _align_len(N_coords: np.ndarray,
               Ca_coords: np.ndarray,
               C_coords: np.ndarray,
               n_res: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _align_atom14(atom14: np.ndarray, n_res: int) -> np.ndarray:
    if atom14.shape[0] == n_res:
        return atom14
    out = np.zeros((n_res, 14, 3), dtype=np.float32)
    n_valid = min(atom14.shape[0], n_res)
    out[:n_valid] = atom14[:n_valid]
    return out


def _align_atom14_mask(mask: np.ndarray, n_res: int) -> np.ndarray:
    if mask.shape[0] == n_res:
        return mask
    out = np.zeros((n_res, 14), dtype=bool)
    n_valid = min(mask.shape[0], n_res)
    out[:n_valid] = mask[:n_valid]
    return out


def _load_torsions(path: Path, n_res: int) -> Dict[str, np.ndarray]:
    data = np.load(path)

    torsion_angles = np.zeros((n_res, 7), dtype=np.float32)
    torsion_angles[:, 0] = _align_array(data['phi'], n_res)
    torsion_angles[:, 1] = _align_array(data['psi'], n_res)
    torsion_angles[:, 2] = _align_array(data['omega'], n_res)
    torsion_angles[:, 3:7] = _align_matrix(data['chi'][:, :4], n_res, 4)

    chi_mask = _align_matrix(data['chi_mask'][:, :4], n_res, 4).astype(bool)

    return {
        'angles': torsion_angles,
        'chi_mask': chi_mask,
    }


def _align_matrix(arr: np.ndarray, n_res: int, n_cols: int) -> np.ndarray:
    if arr.shape[0] == n_res and arr.shape[1] == n_cols:
        return arr
    out = np.zeros((n_res, n_cols), dtype=arr.dtype)
    n_valid = min(arr.shape[0], n_res)
    out[:n_valid] = arr[:n_valid, :n_cols]
    return out
