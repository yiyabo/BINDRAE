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
from functools import partial
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

def extract_backbone_coords(pdb_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int]]:
    """Extract N, CA, C coords, sequence and residue IDs from a PDB file.
    
    Returns:
        N_coords: [N, 3] array
        Ca_coords: [N, 3] array
        C_coords: [N, 3] array
        sequence: list of 3-letter residue names
        residue_ids: list of integer residue IDs (for alignment)
        
    Raises:
        ValueError: If PDB only contains CA atoms (missing N/C backbone)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_file))

    N_coords = []
    Ca_coords = []
    C_coords = []
    sequence = []
    residue_ids = []
    
    # Track atom types for diagnostic
    all_atom_types = set()
    n_ca_only_residues = 0

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != ' ':
                    continue
                
                # Track what atoms this residue has
                residue_atoms = set(atom.get_name() for atom in residue)
                all_atom_types.update(residue_atoms)
                
                try:
                    N = residue['N'].get_coord()
                    Ca = residue['CA'].get_coord()
                    C = residue['C'].get_coord()
                except KeyError:
                    # Count CA-only residues for diagnostic
                    if 'CA' in residue_atoms and ('N' not in residue_atoms or 'C' not in residue_atoms):
                        n_ca_only_residues += 1
                    continue

                N_coords.append(N)
                Ca_coords.append(Ca)
                C_coords.append(C)
                sequence.append(residue.get_resname())
                residue_ids.append(residue.get_id()[1])  # residue sequence number

    N_coords = np.array(N_coords, dtype=np.float32)
    Ca_coords = np.array(Ca_coords, dtype=np.float32)
    C_coords = np.array(C_coords, dtype=np.float32)

    # Check for CA-only structures (common issue with some apo predictions)
    if len(N_coords) == 0 and n_ca_only_residues > 0:
        raise ValueError(
            f"PDB file only contains CA atoms ({n_ca_only_residues} residues with CA but missing N/C). "
            f"Atom types found: {sorted(all_atom_types)}"
        )

    return N_coords, Ca_coords, C_coords, sequence, residue_ids


def _load_backbone_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[int]]]:
    """Load cached backbone coords. Returns residue_ids if stored."""
    data = np.load(path)
    residue_ids = data['residue_ids'].tolist() if 'residue_ids' in data else None
    return data['N'], data['Ca'], data['C'], residue_ids


def align_by_residue_ids(
    apo_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    apo_res_ids: List[int],
    holo_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    holo_res_ids: List[int],
    target_len: int
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray],
           np.ndarray]:
    """
    Align apo and holo backbone coords based on residue IDs, not prefix.
    
    Only residues present in BOTH apo and holo (by ID) will be valid.
    This fixes the issue where missing residues in the middle cause misalignment.
    
    Args:
        apo_coords: (N_apo, Ca_apo, C_apo) arrays
        apo_res_ids: list of residue IDs for apo
        holo_coords: (N_holo, Ca_holo, C_holo) arrays
        holo_res_ids: list of residue IDs for holo
        target_len: target output length (typically from ESM)
    
    Returns:
        aligned_apo: (N, Ca, C) aligned to target_len
        aligned_holo: (N, Ca, C) aligned to target_len
        valid_mask: [target_len] boolean mask, True where both apo and holo have data
    """
    N_apo, Ca_apo, C_apo = apo_coords
    N_holo, Ca_holo, C_holo = holo_coords
    
    # Build residue ID -> index mapping
    apo_id_to_idx = {rid: i for i, rid in enumerate(apo_res_ids)}
    holo_id_to_idx = {rid: i for i, rid in enumerate(holo_res_ids)}
    
    # Find common residue IDs
    common_ids = set(apo_res_ids) & set(holo_res_ids)
    
    # Initialize output arrays
    N_apo_out = np.zeros((target_len, 3), dtype=np.float32)
    Ca_apo_out = np.zeros((target_len, 3), dtype=np.float32)
    C_apo_out = np.zeros((target_len, 3), dtype=np.float32)
    N_holo_out = np.zeros((target_len, 3), dtype=np.float32)
    Ca_holo_out = np.zeros((target_len, 3), dtype=np.float32)
    C_holo_out = np.zeros((target_len, 3), dtype=np.float32)
    valid_mask = np.zeros(target_len, dtype=bool)
    
    # Map to output indices (sorted by residue ID for consistency)
    sorted_common = sorted(common_ids)
    
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


def _coords_valid_mask(N_coords: np.ndarray,
                       Ca_coords: np.ndarray,
                       C_coords: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    n_norm = np.linalg.norm(N_coords, axis=-1)
    ca_norm = np.linalg.norm(Ca_coords, axis=-1)
    c_norm = np.linalg.norm(C_coords, axis=-1)
    return (n_norm > eps) & (ca_norm > eps) & (c_norm > eps)


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
                 valid_samples_file: Optional[str] = None,
                 max_lig_tokens: int = 128,
                 require_atom14: bool = True):
        """
        Args:
            data_dir: Root data directory
            split: 'train', 'val', or 'test'
            index_file: Optional explicit index file
            valid_samples_file: Optional file with valid sample IDs (one per line).
                                If provided, only samples in this file will be loaded.
                                Can be generated by scripts/validate_triplets_data.py
            max_lig_tokens: Maximum ligand tokens
            require_atom14: Whether to require atom14 files
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_lig_tokens = max_lig_tokens
        self.require_atom14 = require_atom14

        self.samples = self._load_index(index_file)
        original_count = len(self.samples)
        
        # Filter by valid samples list if provided
        if valid_samples_file is not None:
            valid_path = Path(valid_samples_file)
            if not valid_path.is_absolute():
                valid_path = self.data_dir / valid_samples_file
            
            if valid_path.exists():
                with open(valid_path, 'r') as f:
                    valid_set = set(line.strip() for line in f if line.strip())
                self.samples = [s for s in self.samples if s.get('id', '') in valid_set]
                filtered_count = original_count - len(self.samples)
                print(f"  Filtered {filtered_count} invalid samples using {valid_path.name}")
            else:
                print(f"  [WARN] valid_samples_file not found: {valid_path}")
        
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

    def _getitem_impl(self, idx: int) -> Dict:
        """Internal implementation - may raise exceptions."""
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

        apo_backbone = self._resolve_path(sample, 'apo_backbone', 'apo_backbone.npz')
        holo_backbone = self._resolve_path(sample, 'holo_backbone', 'holo_backbone.npz')

        apo_res_ids = None
        holo_res_ids = None
        
        # Try to load cached backbone with residue IDs
        if apo_backbone is not None and apo_backbone.exists():
            N_apo, Ca_apo, C_apo, apo_res_ids = _load_backbone_npz(apo_backbone)
        else:
            N_apo, Ca_apo, C_apo, _, apo_res_ids = extract_backbone_coords(apo_pdb)

        if holo_backbone is not None and holo_backbone.exists():
            N_holo, Ca_holo, C_holo, holo_res_ids = _load_backbone_npz(holo_backbone)
        else:
            N_holo, Ca_holo, C_holo, _, holo_res_ids = extract_backbone_coords(holo_pdb)

        # Check for empty backbone (PDB parsing failed)
        if len(N_apo) == 0:
            raise ValueError(f"Empty apo backbone for {sample_id} (PDB parsing failed)")
        if len(N_holo) == 0:
            raise ValueError(f"Empty holo backbone for {sample_id} (PDB parsing failed)")

        # Align to ESM length - prefer residue ID alignment if available
        if apo_res_ids is not None and holo_res_ids is not None:
            # Use residue ID based alignment (fixes middle-missing residue issues)
            (N_apo, Ca_apo, C_apo), (N_holo, Ca_holo, C_holo), node_mask = align_by_residue_ids(
                (N_apo, Ca_apo, C_apo), apo_res_ids,
                (N_holo, Ca_holo, C_holo), holo_res_ids,
                n_res
            )
        else:
            # Fallback to prefix alignment (legacy behavior)
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

        chi_holo = torsion_holo['angles'][:, 3:7]
        chi_mask = torsion_holo['chi_mask'][:, :4]
        chi_mask = chi_mask & node_mask[:, None]

        # Pocket weights (apo frame)
        w_res_path = self._resolve_path(sample, 'w_res', 'w_res.npy')
        if w_res_path is not None and w_res_path.exists():
            w_res = np.load(w_res_path).astype(np.float32)
            w_res = _align_array(w_res, n_res)
        else:
            w_res = compute_pocket_weights(Ca_apo, lig_tokens['coords'])
        w_res = w_res * node_mask.astype(np.float32)

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
            'node_mask': node_mask,
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

    def __getitem__(self, idx: int) -> Dict:
        """Safe wrapper that returns a random valid sample on error."""
        import random
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                target_idx = idx if attempt == 0 else random.randint(0, len(self) - 1)
                return self._getitem_impl(target_idx)
            except Exception as e:
                if attempt == 0:
                    import sys
                    sample_id = self.samples[idx].get('id', f'sample_{idx}')
                    print(f"[WARN] Skipping sample {sample_id}: {str(e)[:80]}", file=sys.stderr)
        
        # Last resort: return a minimal dummy sample
        raise RuntimeError(f"Failed to load any valid sample after {max_retries} retries")
# -----------------------------
# Collate
# -----------------------------

def collate_stage1_batch(samples: List[Dict], max_n_res: Optional[int] = None) -> Optional[Stage1Batch]:
    original_count = len(samples)
    if max_n_res is not None:
        samples = [s for s in samples if s['n_residues'] <= max_n_res]
        filtered_count = original_count - len(samples)
        if filtered_count > 0:
            import logging
            logging.getLogger('stage1.data').debug(
                f"max_n_res filter: {filtered_count}/{original_count} samples removed "
                f"(>{max_n_res} residues)"
            )
        if len(samples) == 0:
            return None

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
        sample_mask = sample.get('node_mask')
        if sample_mask is None:
            sample_mask = np.ones((n_res,), dtype=bool)
        node_mask[i, :n_res] = sample_mask

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
                atom14_holo_mask[i, :n_res] = sample_mask[:, None]

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
                             max_n_res: Optional[int] = None,
                             valid_samples_file: Optional[str] = None,
                             **kwargs):
    """
    Create Stage-1 dataloader.
    
    Args:
        data_dir: Root data directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        max_n_res: Maximum residues per sample (filter out larger)
        valid_samples_file: Optional file with valid sample IDs to use.
                           Can be generated by scripts/validate_triplets_data.py
        **kwargs: Additional arguments passed to ApoHoloTripletDataset
    """
    from torch.utils.data import DataLoader

    dataset = ApoHoloTripletDataset(
        data_dir, 
        split=split, 
        valid_samples_file=valid_samples_file,
        **kwargs
    )
    collate_fn = partial(collate_stage1_batch, max_n_res=max_n_res)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


# -----------------------------
# Helpers
# -----------------------------

def _align_len(N_coords: np.ndarray,
               Ca_coords: np.ndarray,
               C_coords: np.ndarray,
               n_res: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align backbone coords to target length n_res."""
    # Handle empty arrays
    if len(N_coords) == 0:
        raise ValueError(f"Empty backbone coords (0 residues parsed from PDB)")
    
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
