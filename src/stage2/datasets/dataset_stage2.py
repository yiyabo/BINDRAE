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
from dataclasses import dataclass
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


@dataclass
class Stage2Batch:
    """Stage-2 batch (apo/holo/ligand)."""
    # ESM embeddings
    esm: torch.Tensor            # [B, N, 1280] or [B, N, K, 1280]

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

    # Optional Stage-1-v2 posterior scalar features
    stage1v2_posterior_features: Optional[torch.Tensor]  # [B, N, D] or None

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


def _safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def _parse_feature_names(raw: Optional[str]) -> List[str]:
    if raw is None:
        raw = (
            "contact_prob,active_prob,approach_prob,release_prob,confidence,"
            "teacher_min_dist_pred_norm,signed_delta_dist_pred_norm"
        )
    if isinstance(raw, (list, tuple)):
        names = [str(x).strip() for x in raw if str(x).strip()]
    else:
        names = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not names:
        raise ValueError("stage1v2_posterior_feature_names cannot be empty")
    return names


def _load_manifest_paths(cache_dir: Optional[str], data_dir: Path) -> Dict[str, Path]:
    if not cache_dir:
        return {}
    root = Path(cache_dir)
    if not root.is_absolute():
        root = root if root.exists() else data_dir / root
    manifest_path = root / "manifest.json"
    mapping: Dict[str, Path] = {}
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
        for record in manifest.get("records", []):
            sample_id = record.get("sample_id")
            path = record.get("path")
            if not sample_id or not path:
                continue
            p = Path(path)
            if not p.is_absolute():
                p = root / p
            mapping[str(sample_id)] = p
    return mapping


def _resolve_stage1v2_path(cache_dir: Optional[str], cache_map: Dict[str, Path], data_dir: Path, sample_id: str) -> Path:
    if sample_id in cache_map:
        return cache_map[sample_id]
    if not cache_dir:
        return Path("")
    root = Path(cache_dir)
    if not root.is_absolute():
        root = root if root.exists() else data_dir / root
    return root / f"{_safe_sample_id(sample_id)}.npz"


ORACLE_MOTION_FEATURE_MODES = {
    "oracle_motion",
    "oracle_motion_residue_shuffled",
    "oracle_motion_sample_shuffled",
}

STAGE1V2_FILE_FEATURE_MODES = {
    "student",
    "student_shuffled",
    "oracle_holo_truth",
    "external_teacher_cached",
    *ORACLE_MOTION_FEATURE_MODES,
}

SAMPLE_SHUFFLED_FEATURE_MODES = {
    "student_shuffled",
    "oracle_motion_sample_shuffled",
}

SCALAR_STAGE1V2_FEATURE_ALIASES = {
    "active_prob": ("active_prob", "switch_prob"),
    "switch_prob": ("switch_prob", "active_prob"),
    "teacher_min_dist_pred": ("teacher_min_dist_pred", "teacher_min_dist"),
    "teacher_min_dist": ("teacher_min_dist", "teacher_min_dist_pred"),
    "signed_delta_dist_pred": ("signed_delta_dist_pred", "signed_delta_dist"),
    "signed_delta_dist": ("signed_delta_dist", "signed_delta_dist_pred"),
    "teacher_min_dist_pred_norm": ("teacher_min_dist_pred", "teacher_min_dist"),
    "teacher_min_dist_norm": ("teacher_min_dist", "teacher_min_dist_pred"),
    "signed_delta_dist_pred_norm": ("signed_delta_dist_pred", "signed_delta_dist"),
    "signed_delta_dist_norm": ("signed_delta_dist", "signed_delta_dist_pred"),
}


def _scalar_string(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    raise ValueError(f"Expected scalar string field, got shape={arr.shape}")


def _scalar_int(value) -> int:
    arr = np.asarray(value)
    if arr.shape == ():
        return int(arr.item())
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    raise ValueError(f"Expected scalar int field, got shape={arr.shape}")


def _validate_vector_len(name: str, arr: np.ndarray, n_res: int, path: Path) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError(f"{path} field {name} must be 1D, got shape={arr.shape}")
    if arr.shape[0] != n_res:
        raise ValueError(f"{path} field {name} length mismatch: {arr.shape[0]} != expected {n_res}")
    return arr


def _validate_feature_cache_metadata(
    data,
    path: Path,
    *,
    expected_sample_id: str,
    expected_n_res: int,
    expected_aatype: Optional[np.ndarray],
    expected_node_mask: Optional[np.ndarray],
) -> None:
    missing_meta = [
        key for key in ("sample_id", "n_residues")
        if key not in data
    ]
    if missing_meta:
        raise ValueError(f"{path} missing required feature-cache metadata: {missing_meta}")

    sample_id = _scalar_string(data["sample_id"])
    if sample_id != expected_sample_id:
        raise ValueError(f"{path} sample_id={sample_id!r}, expected {expected_sample_id!r}")

    n_res = _scalar_int(data["n_residues"])
    if n_res != int(expected_n_res):
        raise ValueError(f"{path} n_residues={n_res}, expected {expected_n_res}")

    if expected_aatype is not None:
        if "aatype" not in data:
            raise ValueError(f"{path} missing required feature-cache metadata: ['aatype']")
        cache_aatype = _validate_vector_len(
            "aatype",
            np.asarray(data["aatype"]).astype(np.int64),
            int(expected_n_res),
            path,
        )
        expected = np.asarray(expected_aatype).astype(np.int64)
        _validate_vector_len("expected_aatype", expected, int(expected_n_res), path)
        if not np.array_equal(cache_aatype, expected):
            mismatch = np.flatnonzero(cache_aatype != expected)[:8].tolist()
            raise ValueError(f"{path} aatype mismatch at residues {mismatch}")

    if expected_node_mask is not None:
        mask_key = "node_mask" if "node_mask" in data else ("valid_mask" if "valid_mask" in data else None)
        if mask_key is not None:
            cache_mask = _validate_vector_len(
                mask_key,
                np.asarray(data[mask_key]).astype(np.bool_),
                int(expected_n_res),
                path,
            )
            expected_mask = np.asarray(expected_node_mask).astype(np.bool_)
            _validate_vector_len("expected_node_mask", expected_mask, int(expected_n_res), path)
            invalid_claims = cache_mask & (~expected_mask)
            if invalid_claims.any():
                mismatch = np.flatnonzero(invalid_claims)[:8].tolist()
                raise ValueError(
                    f"{path} {mask_key} marks residues valid that Stage-2 marks invalid: {mismatch}"
                )


def _read_npz_vector(data, feature_name: str, n_res: int, path: Path) -> np.ndarray:
    keys = SCALAR_STAGE1V2_FEATURE_ALIASES.get(feature_name, (feature_name,))
    key = next((k for k in keys if k in data), None)
    if key is None:
        raise KeyError(f"{path} missing Stage-1-v2 posterior feature {feature_name!r}")
    arr = np.asarray(data[key]).astype(np.float32)
    arr = _validate_vector_len(feature_name, arr, n_res, path)
    if feature_name in {"teacher_min_dist_pred_norm", "teacher_min_dist_norm"}:
        arr = np.clip(arr, 0.0, 20.0) / 10.0
    elif feature_name in {"signed_delta_dist_pred_norm", "signed_delta_dist_norm"}:
        arr = np.clip(arr, -10.0, 10.0) / 5.0
    if not np.isfinite(arr).all():
        raise ValueError(f"{path} feature {feature_name} contains non-finite values")
    return arr


def load_stage1v2_posterior_features(
    path: Path,
    feature_names: List[str],
    n_res: int,
    *,
    expected_sample_id: str,
    expected_aatype: Optional[np.ndarray],
    expected_node_mask: Optional[np.ndarray],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        _validate_feature_cache_metadata(
            data,
            path,
            expected_sample_id=expected_sample_id,
            expected_n_res=n_res,
            expected_aatype=expected_aatype,
            expected_node_mask=expected_node_mask,
        )
        cols = [_read_npz_vector(data, name, n_res, path) for name in feature_names]
    return np.stack(cols, axis=-1).astype(np.float32)


def _npz_feature_names(data, path: Path) -> List[str]:
    if "feature_names" not in data:
        raise ValueError(f"{path} missing feature_names for oracle_motion_features")
    return [str(x) for x in np.asarray(data["feature_names"]).tolist()]


def load_oracle_motion_features(
    path: Path,
    feature_names: List[str],
    n_res: int,
    *,
    expected_sample_id: str,
    expected_aatype: Optional[np.ndarray],
    expected_node_mask: Optional[np.ndarray],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        _validate_feature_cache_metadata(
            data,
            path,
            expected_sample_id=expected_sample_id,
            expected_n_res=n_res,
            expected_aatype=expected_aatype,
            expected_node_mask=expected_node_mask,
        )
        if "oracle_motion_features" not in data:
            raise KeyError(f"{path} missing oracle_motion_features")
        matrix = np.asarray(data["oracle_motion_features"]).astype(np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"{path} oracle_motion_features must be 2D, got shape={matrix.shape}")
        if matrix.shape[0] != n_res:
            raise ValueError(f"{path} oracle_motion_features n_res mismatch: {matrix.shape[0]} != {n_res}")
        cached_names = _npz_feature_names(data, path)
        name_to_idx = {name: idx for idx, name in enumerate(cached_names)}
        missing = [name for name in feature_names if name not in name_to_idx]
        if missing:
            raise KeyError(f"{path} missing OracleMotion features: {missing}")
        indices = [name_to_idx[name] for name in feature_names]
        out = matrix[:, indices]
    if not np.isfinite(out).all():
        raise ValueError(f"{path} oracle_motion_features contains non-finite values")
    return out.astype(np.float32)


def _esm_features_from_data(data: Dict, path: Path, esm_num_layers: int = 1) -> np.ndarray:
    """Extract single-layer or last-K ESM residue features from loaded esm.pt data."""
    if esm_num_layers < 1:
        raise ValueError(f"esm_num_layers must be >= 1, got {esm_num_layers}")
    if "per_residue" not in data:
        raise KeyError(f"{path} missing per_residue")

    per_residue = data["per_residue"]
    if torch.is_tensor(per_residue):
        per_residue = per_residue.detach().cpu().numpy()
    per_residue = np.asarray(per_residue, dtype=np.float32)
    if per_residue.ndim != 2:
        raise ValueError(f"{path} per_residue must be [N, D], got shape={per_residue.shape}")

    if esm_num_layers == 1:
        out = per_residue
    else:
        if "per_residue_layers" not in data:
            raise KeyError(
                f"{path} missing per_residue_layers required for esm_num_layers={esm_num_layers}"
            )
        layers = data["per_residue_layers"]
        if torch.is_tensor(layers):
            layers = layers.detach().cpu().numpy()
        layers = np.asarray(layers, dtype=np.float32)
        if layers.ndim != 3:
            raise ValueError(
                f"{path} per_residue_layers must be [N, K, D], got shape={layers.shape}"
            )
        if layers.shape[0] != per_residue.shape[0] or layers.shape[-1] != per_residue.shape[-1]:
            raise ValueError(
                f"{path} per_residue_layers shape {layers.shape} is inconsistent with "
                f"per_residue shape {per_residue.shape}"
            )
        if layers.shape[1] < esm_num_layers:
            raise ValueError(
                f"{path} stores K={layers.shape[1]} ESM layers, fewer than requested "
                f"esm_num_layers={esm_num_layers}"
            )
        out = layers[:, -esm_num_layers:, :]

    if not np.isfinite(out).all():
        raise ValueError(f"{path} ESM features contain non-finite values")
    return out.astype(np.float32)


def load_esm_features(path: Path, esm_num_layers: int = 1) -> np.ndarray:
    """Load single-layer or last-K ESM residue features from an esm.pt file."""
    data = torch.load(path, weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a dict with per_residue features")
    return _esm_features_from_data(data, path, esm_num_layers)


def _stable_int_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def _shuffle_valid_residue_rows(features: np.ndarray, node_mask: np.ndarray, sample_id: str, mode: str) -> np.ndarray:
    out = np.array(features, copy=True)
    valid = np.flatnonzero(np.asarray(node_mask).astype(bool))
    if valid.size <= 1:
        return out
    perm = np.array(valid, copy=True)
    rng = np.random.default_rng(_stable_int_seed(sample_id, mode, "residue_shuffle"))
    rng.shuffle(perm)
    out[valid] = features[perm]
    return out


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
            print(f"[WARN] valid_samples_file not found: {valid_path}")
            return self.samples
        
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
    )


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
