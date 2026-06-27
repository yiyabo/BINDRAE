"""Teacher posterior label schema for Stage-1-v2."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch


SCHEMA_VERSION = "bindrae_teacher_posterior_v1"

FLOAT_FIELDS = (
    "apo_min_dist",
    "teacher_min_dist",
    "signed_delta_dist",
    "contact_prob",
    "approach_prob",
    "release_prob",
    "switch_prob",
    "confidence",
    "w_res",
)

BOOL_FIELDS = (
    "valid_mask",
    "pocket_mask",
    "contact_apo",
    "contact_teacher",
    "approach",
    "release",
    "formed_contact",
    "released_contact",
    "stable_contact",
    "stable_noncontact",
    "active",
)

META_FIELDS = (
    "schema_version",
    "teacher_source",
    "sample_id",
    "n_residues",
    "aatype",
)

REQUIRED_FIELDS = META_FIELDS + FLOAT_FIELDS + BOOL_FIELDS


@dataclass(frozen=True)
class TeacherPosteriorLabel:
    """Validated per-residue teacher posterior arrays for one sample."""

    path: Path
    sample_id: str
    teacher_source: str
    n_residues: int
    aatype: np.ndarray
    floats: Mapping[str, np.ndarray]
    bools: Mapping[str, np.ndarray]


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


def _require_fields(data: Mapping[str, np.ndarray], path: Path) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValueError(f"{path} missing teacher-posterior fields: {missing}")


def _validate_vector(name: str, arr: np.ndarray, n_residues: int, path: Path) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError(f"{path} field {name} must be 1D, got shape={arr.shape}")
    if arr.shape[0] != n_residues:
        raise ValueError(
            f"{path} field {name} length mismatch: {arr.shape[0]} != n_residues {n_residues}"
        )
    return arr


def validate_teacher_posterior_arrays(
    data: Mapping[str, np.ndarray],
    path: Path,
    expected_sample_id: str | None = None,
) -> TeacherPosteriorLabel:
    """Validate arrays and return a structured label object.

    This intentionally hard-fails on stale or malformed labels. Silent fallback
    would corrupt posterior metrics and Stage-2 guidance experiments.
    """

    _require_fields(data, path)
    schema_version = _scalar_string(data["schema_version"])
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"{path} schema_version={schema_version!r}, expected {SCHEMA_VERSION!r}")

    sample_id = _scalar_string(data["sample_id"])
    if expected_sample_id is not None and sample_id != expected_sample_id:
        raise ValueError(f"{path} sample_id={sample_id!r}, expected {expected_sample_id!r}")

    teacher_source = _scalar_string(data["teacher_source"])
    n_residues = _scalar_int(data["n_residues"])
    if n_residues <= 0:
        raise ValueError(f"{path} n_residues must be positive, got {n_residues}")

    aatype = _validate_vector("aatype", np.asarray(data["aatype"]), n_residues, path).astype(np.int64)
    floats: Dict[str, np.ndarray] = {}
    bools: Dict[str, np.ndarray] = {}

    for field in FLOAT_FIELDS:
        arr = _validate_vector(field, np.asarray(data[field]), n_residues, path).astype(np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{path} field {field} contains non-finite values")
        floats[field] = arr

    for field in BOOL_FIELDS:
        arr = _validate_vector(field, np.asarray(data[field]), n_residues, path).astype(np.bool_)
        bools[field] = arr

    valid = bools["valid_mask"]
    if not valid.any():
        raise ValueError(f"{path} valid_mask is empty")
    if bools["pocket_mask"].sum() == 0:
        raise ValueError(f"{path} pocket_mask is empty; label export likely failed")

    return TeacherPosteriorLabel(
        path=path,
        sample_id=sample_id,
        teacher_source=teacher_source,
        n_residues=n_residues,
        aatype=aatype,
        floats=floats,
        bools=bools,
    )


def load_teacher_posterior_npz(path: str | Path, expected_sample_id: str | None = None) -> TeacherPosteriorLabel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return validate_teacher_posterior_arrays(arrays, path, expected_sample_id=expected_sample_id)


def pad_1d(values: Iterable[np.ndarray], max_len: int, dtype) -> torch.Tensor:
    arrays = list(values)
    out = np.zeros((len(arrays), max_len), dtype=dtype)
    for i, arr in enumerate(arrays):
        out[i, : arr.shape[0]] = arr
    return torch.from_numpy(out)
