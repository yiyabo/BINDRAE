"""ESM feature-cache loading helpers for Stage-2 datasets."""

from pathlib import Path
from typing import Dict

import numpy as np
import torch


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
