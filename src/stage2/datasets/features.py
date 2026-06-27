"""Stage-2 ESM, Stage-1-v2, and OracleMotion feature-cache helpers."""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .esm_cache import _esm_features_from_data, load_esm_features


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
    "teacher_min_dist_pred_norm": ("teacher_min_dist_pred_norm", "teacher_min_dist"),
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
