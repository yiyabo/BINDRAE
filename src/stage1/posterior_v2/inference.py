"""Inference helpers for Stage-1-v2 posterior checkpoints."""

from pathlib import Path
from typing import Dict, Tuple

import torch

from .model import Stage1PosteriorV2, Stage1PosteriorV2Config


def model_config_from_checkpoint(checkpoint: Dict) -> Stage1PosteriorV2Config:
    """Recreate the student model config saved by the training script."""

    train_config = checkpoint.get("config", {}) or {}
    defaults = Stage1PosteriorV2Config()
    return Stage1PosteriorV2Config(
        esm_dim=int(train_config.get("esm_dim", defaults.esm_dim)),
        c_s=int(train_config.get("c_s", defaults.c_s)),
        d_lig=int(train_config.get("d_lig", defaults.d_lig)),
        num_heads_cross=int(train_config.get("num_heads_cross", defaults.num_heads_cross)),
        num_rbf=int(train_config.get("num_rbf", defaults.num_rbf)),
        rbf_max=float(train_config.get("rbf_max", defaults.rbf_max)),
        hidden_dim=int(train_config.get("hidden_dim", defaults.hidden_dim)),
        num_layers=int(train_config.get("num_layers", defaults.num_layers)),
        dropout=float(train_config.get("dropout", defaults.dropout)),
        output_latent_dim=int(train_config.get("output_latent_dim", defaults.output_latent_dim)),
        use_latent_head=bool(train_config.get("use_latent_head", defaults.use_latent_head)),
        warmup_steps=int(train_config.get("warmup_steps", defaults.warmup_steps)),
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_stage1v2_posterior_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[Stage1PosteriorV2, Dict]:
    """Load a Stage-1-v2 posterior student checkpoint for eval/inference."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = Stage1PosteriorV2(model_config_from_checkpoint(checkpoint))
    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def batch_to_device(batch, device: torch.device):
    """Move tensor fields of a TeacherPosteriorBatch to a device in-place."""

    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            setattr(batch, name, value.to(device))
    batch.teacher_float = {key: value.to(device) for key, value in batch.teacher_float.items()}
    batch.teacher_bool = {key: value.to(device) for key, value in batch.teacher_bool.items()}
    return batch


def ligand_inputs(batch, mode: str = "real"):
    """Return ligand tensors for real and counterfactual posterior evaluation."""

    if mode == "real":
        return batch.lig_points, batch.lig_types, batch.lig_mask, batch.w_res
    if mode == "nolig":
        return (
            torch.zeros_like(batch.lig_points),
            torch.zeros_like(batch.lig_types),
            torch.zeros_like(batch.lig_mask),
            torch.zeros_like(batch.w_res),
        )
    if mode == "shuffled":
        bsz = batch.lig_points.shape[0]
        perm = (
            torch.roll(torch.arange(bsz, device=batch.lig_points.device), shifts=1)
            if bsz > 1
            else torch.arange(bsz, device=batch.lig_points.device)
        )
        return (
            batch.lig_points[perm],
            batch.lig_types[perm],
            batch.lig_mask[perm],
            torch.zeros_like(batch.w_res),
        )
    if mode == "translated":
        offset = batch.lig_points.new_tensor([50.0, -37.0, 23.0]).view(1, 1, 3)
        shifted = batch.lig_points + offset * batch.lig_mask.float().unsqueeze(-1)
        return shifted, batch.lig_types, batch.lig_mask, torch.zeros_like(batch.w_res)
    raise ValueError(f"Unknown ligand counterfactual mode: {mode}")


def predict_posterior(model: Stage1PosteriorV2, batch, mode: str = "real") -> Dict[str, torch.Tensor]:
    lig_points, lig_types, lig_mask, w_res = ligand_inputs(batch, mode=mode)
    return model(
        esm=batch.esm,
        Ca_apo=batch.Ca_apo,
        lig_points=lig_points,
        lig_types=lig_types,
        lig_mask=lig_mask,
        w_res=w_res,
        node_mask=batch.node_mask,
        torsion_apo=batch.torsion_apo,
    )


def posterior_selection_score(metrics: Dict[str, float]) -> float:
    """Same validation score used by the trainer for best checkpoint selection."""

    return (
        0.45 * float(metrics.get("active_balanced_acc", 0.0))
        + 0.25 * float(metrics.get("contact_balanced_acc", 0.0))
        + 0.15 * float(metrics.get("approach_balanced_acc", 0.0))
        + 0.15 * float(metrics.get("release_balanced_acc", 0.0))
        - 0.02 * min(float(metrics.get("pocket_delta_mae", 0.0)), 20.0)
    )
