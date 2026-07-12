"""Stage-2 batch container and collate function."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_current_file = Path(__file__).resolve()
project_root = _current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.ligand_utils import LIGAND_TYPE_DIM


@dataclass
class Stage2Batch:
    """Stage-2 batch (apo/holo/ligand)."""
    esm: torch.Tensor
    torsion_apo: torch.Tensor
    torsion_holo: torch.Tensor
    bb_mask: torch.Tensor
    chi_mask: torch.Tensor
    node_mask: torch.Tensor
    peptide_bond_mask: torch.Tensor
    N_apo: torch.Tensor
    Ca_apo: torch.Tensor
    C_apo: torch.Tensor
    N_holo: torch.Tensor
    Ca_holo: torch.Tensor
    C_holo: torch.Tensor
    lig_points: torch.Tensor
    lig_types: torch.Tensor
    lig_mask: torch.Tensor
    w_res: torch.Tensor
    stage1v2_posterior_features: Optional[torch.Tensor]
    nma_features: Optional[torch.Tensor]
    aatype: torch.Tensor
    sequences: List[str]
    pdb_ids: List[str]
    n_residues: List[int]
    residue_identity_hashes: List[str]

    def pin_memory(self):
        """Pin every tensor field so non-blocking GPU copies are effective."""
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.pin_memory())
        return self


def collate_stage2_batch(samples: List[Dict]) -> Stage2Batch:
    batch_size = len(samples)
    max_n_res = max(s['n_residues'] for s in samples)
    max_lig = max(len(s['lig_points']) for s in samples)

    esm_tail_shape = tuple(samples[0]['esm'].shape[1:])
    for sample in samples:
        if tuple(sample['esm'].shape[1:]) != esm_tail_shape:
            raise ValueError(
                f"Mixed ESM feature shapes in batch: {esm_tail_shape} and "
                f"{tuple(sample['esm'].shape[1:])}"
            )
    esm_batch = np.zeros((batch_size, max_n_res, *esm_tail_shape), dtype=np.float32)
    torsion_apo = np.zeros((batch_size, max_n_res, 7), dtype=np.float32)
    torsion_holo = np.zeros((batch_size, max_n_res, 7), dtype=np.float32)
    bb_mask = np.zeros((batch_size, max_n_res, 3), dtype=bool)
    chi_mask = np.zeros((batch_size, max_n_res, 4), dtype=bool)
    node_mask = np.zeros((batch_size, max_n_res), dtype=bool)
    peptide_bond_mask = np.zeros((batch_size, max(max_n_res - 1, 0)), dtype=bool)

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

    stage1v2_dim = None
    for s in samples:
        if s.get('stage1v2_posterior_features') is not None:
            stage1v2_dim = s['stage1v2_posterior_features'].shape[-1]
            break
    stage1v2_posterior_features = None
    if stage1v2_dim is not None:
        stage1v2_posterior_features = np.zeros((batch_size, max_n_res, stage1v2_dim), dtype=np.float32)

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
    residue_identity_hashes = []

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
        if n_res > 1:
            sample_bonds = np.asarray(sample['peptide_bond_mask'], dtype=bool)
            if sample_bonds.shape != (n_res - 1,):
                raise ValueError(
                    f"peptide_bond_mask shape={sample_bonds.shape}, expected {(n_res - 1,)}"
                )
            peptide_bond_mask[i, :n_res - 1] = sample_bonds

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

        if stage1v2_posterior_features is not None and sample.get('stage1v2_posterior_features') is not None:
            feat = sample['stage1v2_posterior_features']
            stage1v2_posterior_features[i, :n_res, :feat.shape[-1]] = feat

        if nma_features is not None and sample['nma_features'] is not None:
            nma = sample['nma_features']
            if nma.ndim == 1:
                nma = nma[:, None]
            nma_features[i, :n_res, :nma.shape[-1]] = nma

        aatype[i, :n_res] = sample['aatype']

        pdb_ids.append(sample['id'])
        n_residues.append(n_res)
        sequences.append(sample.get('sequence', ''))
        residue_identity_hashes.append(sample['residue_identity_hash'])

    return Stage2Batch(
        esm=torch.from_numpy(esm_batch),
        torsion_apo=torch.from_numpy(torsion_apo),
        torsion_holo=torch.from_numpy(torsion_holo),
        bb_mask=torch.from_numpy(bb_mask),
        chi_mask=torch.from_numpy(chi_mask),
        node_mask=torch.from_numpy(node_mask),
        peptide_bond_mask=torch.from_numpy(peptide_bond_mask),
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
        stage1v2_posterior_features=(
            torch.from_numpy(stage1v2_posterior_features)
            if stage1v2_posterior_features is not None
            else None
        ),
        nma_features=torch.from_numpy(nma_features) if nma_features is not None else None,
        aatype=torch.from_numpy(aatype),
        sequences=sequences,
        pdb_ids=pdb_ids,
        n_residues=n_residues,
        residue_identity_hashes=residue_identity_hashes,
    )
