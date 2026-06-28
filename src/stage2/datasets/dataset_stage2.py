"""
Stage-2 dataset (apo/holo/ligand triplets) for bridge flow training.

Aligned to current Stage-2 spec:
- Input: apo + holo torsions, apo/holo backbone frames, ESM embeddings, ligand tokens
- Pocket weights computed in apo frame by default
- Optional NMA features for gating/weight closure
"""

import sys
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser

# Add project root
_current_file = Path(__file__).resolve()
project_root = _current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.ligand_utils import build_ligand_tokens_from_file, LIGAND_TYPE_DIM
from src.stage1.data.residue_constants import restype_order
from .batch import Stage2Batch, collate_stage2_batch
from .backbone import (
    extract_backbone_coords,
    _load_backbone_npz,
    _coords_valid_mask,
    align_by_residue_ids,
    compute_pocket_weights,
    _align_len,
    _align_array,
    _align_matrix,
    _align_nma,
    _load_torsions,
    _sequence_to_aatype,
)
from .features import (
    ORACLE_MOTION_FEATURE_MODES,
    STAGE1V2_FILE_FEATURE_MODES,
    SAMPLE_SHUFFLED_FEATURE_MODES,
    SCALAR_STAGE1V2_FEATURE_ALIASES,
    _safe_sample_id,
    _parse_feature_names,
    _load_manifest_paths,
    _resolve_stage1v2_path,
    _scalar_string,
    _scalar_int,
    _validate_vector_len,
    _validate_feature_cache_metadata,
    _read_npz_vector,
    _npz_feature_names,
    load_stage1v2_posterior_features,
    load_oracle_motion_features,
    _esm_features_from_data,
    load_esm_features,
    _stable_int_seed,
    _shuffle_valid_residue_rows,
)


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
                 require_nma: bool = False,
                 valid_samples_file: Optional[str] = None,
                 stage1v2_posterior_cache_dir: Optional[str] = None,
                 stage1v2_posterior_feature_mode: str = "none",
                 stage1v2_posterior_feature_names: Optional[str] = None,
                 esm_num_layers: int = 1):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_lig_tokens = max_lig_tokens
        self.require_nma = require_nma
        self.esm_num_layers = int(esm_num_layers)
        if self.esm_num_layers < 1:
            raise ValueError(f"esm_num_layers must be >= 1, got {esm_num_layers}")
        self.stage1v2_posterior_cache_dir = stage1v2_posterior_cache_dir
        self.stage1v2_posterior_feature_mode = str(stage1v2_posterior_feature_mode or "none")
        self.stage1v2_posterior_feature_names = _parse_feature_names(stage1v2_posterior_feature_names)
        self.stage1v2_posterior_cache_map = _load_manifest_paths(stage1v2_posterior_cache_dir, self.data_dir)

        self.samples = self._load_index(index_file)
        
        # Filter by valid_samples_file if provided
        if valid_samples_file:
            self.samples = self._filter_by_valid_samples(valid_samples_file)
        
        # Filter out samples with missing required files
        self.samples = self._filter_valid_samples()

        if self.stage1v2_posterior_feature_mode in STAGE1V2_FILE_FEATURE_MODES:
            self.samples = self._filter_stage1v2_posterior_samples()
        self.stage1v2_sample_shuffle_sources: Dict[int, int] = {}
        if self.stage1v2_posterior_feature_mode in SAMPLE_SHUFFLED_FEATURE_MODES:
            self.stage1v2_sample_shuffle_sources = self._build_same_length_shuffle_sources()

        print(f"✓ Stage-2 {split} samples: {len(self.samples)}")
        if self.stage1v2_posterior_feature_mode != "none":
            print(
                f"  Stage-1-v2 posterior features: mode={self.stage1v2_posterior_feature_mode} "
                f"dim={len(self.stage1v2_posterior_feature_names)}"
            )
        if self.esm_num_layers > 1:
            print(f"  ESM last-K layers required: K={self.esm_num_layers}")

    def _filter_by_valid_samples(self, valid_samples_file: str) -> List[Dict]:
        """Filter samples by a list of valid sample IDs."""
        valid_path = Path(valid_samples_file)
        if not valid_path.is_absolute() and not valid_path.exists():
            valid_path = self.data_dir / valid_path
        
        if not valid_path.exists():
            raise FileNotFoundError(f"valid_samples_file not found: {valid_path}")
        
        with open(valid_path, 'r') as f:
            valid_ids = {line.strip() for line in f if line.strip()}
        
        before = len(self.samples)
        filtered = [s for s in self.samples if s.get('id', '') in valid_ids]
        print(f"  Filtered {before - len(filtered)} samples using {valid_path.name}")
        return filtered

    def _filter_valid_samples(self) -> List[Dict]:
        """Remove samples with missing required files."""
        required_files = [
            ('esm_path', 'esm.pt'),
            ('torsion_apo', 'torsion_apo.npz'),
            ('torsion_holo', 'torsion_holo.npz'),
            ('ligand_coords', 'ligand_coords.npy'),
            ('apo_pdb', 'apo.pdb'),
            ('holo_pdb', 'holo.pdb'),
        ]
        valid = []
        missing_counts = {name: 0 for name, _ in required_files}
        for s in self.samples:
            skip = False
            for key, default in required_files:
                p = self._resolve_path(s, key, default)
                if p is None or not p.exists():
                    missing_counts[key] += 1
                    skip = True
                    break
            if not skip:
                valid.append(s)
        removed = len(self.samples) - len(valid)
        if removed > 0:
            for key, count in missing_counts.items():
                if count > 0:
                    print(f"  Removed {count} samples with missing {key}")
            print(f"  Total removed: {removed}")
        return valid

    def _filter_stage1v2_posterior_samples(self) -> List[Dict]:
        valid = []
        missing = 0
        for sample in self.samples:
            sample_id = sample.get('id', '')
            path = _resolve_stage1v2_path(
                self.stage1v2_posterior_cache_dir,
                self.stage1v2_posterior_cache_map,
                self.data_dir,
                sample_id,
            )
            if path.exists():
                valid.append(sample)
            else:
                missing += 1
        if missing > 0:
            print(f"  Removed {missing} samples without Stage-1-v2 posterior cache/labels")
        if not valid:
            raise ValueError(
                f"No Stage-1-v2 posterior files found for split={self.split}, "
                f"mode={self.stage1v2_posterior_feature_mode}, dir={self.stage1v2_posterior_cache_dir}"
            )
        return valid

    def _stage1v2_path_for_sample_id(self, sample_id: str) -> Path:
        return _resolve_stage1v2_path(
            self.stage1v2_posterior_cache_dir,
            self.stage1v2_posterior_cache_map,
            self.data_dir,
            sample_id,
        )

    def _feature_cache_n_res(self, sample: Dict) -> int:
        sample_id = sample.get('id', '')
        path = self._stage1v2_path_for_sample_id(sample_id)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            if "sample_id" in data:
                cached_id = _scalar_string(data["sample_id"])
                if cached_id != sample_id:
                    raise ValueError(f"{path} sample_id={cached_id!r}, expected {sample_id!r}")
            if "n_residues" not in data:
                raise ValueError(f"{path} missing n_residues; cannot build strict sample-shuffled control")
            return _scalar_int(data["n_residues"])

    def _build_same_length_shuffle_sources(self) -> Dict[int, int]:
        by_len: Dict[int, List[int]] = {}
        for idx, sample in enumerate(self.samples):
            n_res = self._feature_cache_n_res(sample)
            by_len.setdefault(n_res, []).append(idx)

        singleton = {
            n_res: indices
            for n_res, indices in by_len.items()
            if len(indices) < 2
        }
        if singleton:
            examples = [
                self.samples[indices[0]].get('id', f'sample_{indices[0]}')
                for _, indices in list(singleton.items())[:8]
            ]
            raise ValueError(
                f"{self.stage1v2_posterior_feature_mode} requires same-length donor samples. "
                f"{sum(len(v) for v in singleton.values())} singleton-length samples found; "
                f"examples={examples}. Use a same-length subset or oracle_motion_residue_shuffled."
            )

        source_by_idx: Dict[int, int] = {}
        for indices in by_len.values():
            ordered = sorted(indices)
            for pos, idx in enumerate(ordered):
                source_by_idx[idx] = ordered[(pos + 1) % len(ordered)]
        return source_by_idx

    def _load_stage1v2_features_for_sample(
        self,
        *,
        path: Path,
        sample_id: str,
        n_res: int,
        aatype: Optional[np.ndarray],
        node_mask: Optional[np.ndarray],
        feature_mode: str,
        strict_residue_identity: bool,
    ) -> np.ndarray:
        expected_aatype = aatype if strict_residue_identity else None
        expected_node_mask = node_mask if strict_residue_identity else None
        if feature_mode in ORACLE_MOTION_FEATURE_MODES:
            return load_oracle_motion_features(
                path,
                self.stage1v2_posterior_feature_names,
                n_res,
                expected_sample_id=sample_id,
                expected_aatype=expected_aatype,
                expected_node_mask=expected_node_mask,
            )
        return load_stage1v2_posterior_features(
            path,
            self.stage1v2_posterior_feature_names,
            n_res,
            expected_sample_id=sample_id,
            expected_aatype=expected_aatype,
            expected_node_mask=expected_node_mask,
        )

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
        if not isinstance(esm_data, dict):
            raise ValueError(f"{esm_path} must contain a dict with per_residue features")
        esm_features = _esm_features_from_data(esm_data, esm_path, self.esm_num_layers)
        sequence_str = esm_data.get('sequence_str', '')
        n_res = int(esm_features.shape[0])

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

        stage1v2_posterior_features = None
        if self.stage1v2_posterior_feature_mode == "zero":
            stage1v2_posterior_features = np.zeros(
                (n_res, len(self.stage1v2_posterior_feature_names)),
                dtype=np.float32,
            )
        elif self.stage1v2_posterior_feature_mode in {"student", "oracle_holo_truth", "external_teacher_cached", "oracle_motion"}:
            posterior_path = self._stage1v2_path_for_sample_id(sample_id)
            stage1v2_posterior_features = self._load_stage1v2_features_for_sample(
                path=posterior_path,
                sample_id=sample_id,
                n_res=n_res,
                aatype=aatype,
                node_mask=node_mask,
                feature_mode=self.stage1v2_posterior_feature_mode,
                strict_residue_identity=True,
            )
        elif self.stage1v2_posterior_feature_mode == "oracle_motion_residue_shuffled":
            posterior_path = self._stage1v2_path_for_sample_id(sample_id)
            stage1v2_posterior_features = self._load_stage1v2_features_for_sample(
                path=posterior_path,
                sample_id=sample_id,
                n_res=n_res,
                aatype=aatype,
                node_mask=node_mask,
                feature_mode=self.stage1v2_posterior_feature_mode,
                strict_residue_identity=True,
            )
            stage1v2_posterior_features = _shuffle_valid_residue_rows(
                stage1v2_posterior_features,
                node_mask,
                sample_id,
                self.stage1v2_posterior_feature_mode,
            )
        elif self.stage1v2_posterior_feature_mode in SAMPLE_SHUFFLED_FEATURE_MODES:
            if len(self.samples) < 2:
                raise ValueError(f"{self.stage1v2_posterior_feature_mode} mode requires at least two samples")
            source_idx = self.stage1v2_sample_shuffle_sources.get(idx)
            if source_idx is None:
                raise RuntimeError(f"No same-length shuffled source for sample idx={idx} id={sample_id}")
            source_sample = self.samples[source_idx]
            source_id = source_sample.get('id', f'sample_{source_idx}')
            posterior_path = self._stage1v2_path_for_sample_id(source_id)
            source_mode = (
                "oracle_motion"
                if self.stage1v2_posterior_feature_mode == "oracle_motion_sample_shuffled"
                else "student"
            )
            stage1v2_posterior_features = self._load_stage1v2_features_for_sample(
                path=posterior_path,
                sample_id=source_id,
                n_res=n_res,
                aatype=None,
                node_mask=None,
                feature_mode=source_mode,
                strict_residue_identity=False,
            )
        elif self.stage1v2_posterior_feature_mode != "none":
            raise ValueError(f"Unknown stage1v2_posterior_feature_mode={self.stage1v2_posterior_feature_mode}")
        if stage1v2_posterior_features is not None:
            stage1v2_posterior_features = stage1v2_posterior_features * node_mask[:, None].astype(np.float32)

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
            'stage1v2_posterior_features': stage1v2_posterior_features,
            'nma_features': nma_features,
            'n_residues': n_res,
            'node_mask': node_mask,
        }


# -----------------------------
# Collate
# -----------------------------


# -----------------------------
# DataLoader factory
# -----------------------------


def create_stage2_dataloader(data_dir: str,
                             split: str = 'train',
                             batch_size: int = 2,
                             shuffle: bool = True,
                             num_workers: int = 0,
                             valid_samples_file: Optional[str] = None,
                             **kwargs):
    from torch.utils.data import DataLoader

    dataset = ApoHoloBridgeDataset(
        data_dir, 
        split=split, 
        valid_samples_file=valid_samples_file,
        **kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_stage2_batch,
        pin_memory=True,
    )
