#!/usr/bin/env python3

import argparse
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.audit_stage1_checkpoint import FK_BUFFER_KEYS
from src.stage1.datasets.dataset_stage1 import Stage1Batch
from src.stage1.training.config import TrainingConfig
from src.stage1.training.trainer import Stage1Trainer
from utils.metrics import wrap_angle_diff


def config_to_dict(config_obj: Any) -> Dict[str, Any]:
    if config_obj is None:
        return {}
    if isinstance(config_obj, TrainingConfig):
        return asdict(config_obj)
    if is_dataclass(config_obj) and not isinstance(config_obj, type):
        return asdict(config_obj)
    if isinstance(config_obj, dict):
        return dict(config_obj)
    return {
        key: getattr(config_obj, key)
        for key in dir(config_obj)
        if not key.startswith('_') and not callable(getattr(config_obj, key))
    }


def make_config(ckpt: Dict[str, Any], args: argparse.Namespace) -> TrainingConfig:
    ckpt_cfg = config_to_dict(ckpt.get('config'))
    fields = TrainingConfig.__dataclass_fields__
    config = TrainingConfig(**{k: v for k, v in ckpt_cfg.items() if k in fields})
    config.distributed = False
    config.resume_from = None
    config.device = args.device
    config.data_dir = args.data_dir or config.data_dir
    config.val_samples_file = args.val_samples_file if args.val_samples_file is not None else config.val_samples_file
    config.valid_samples_file = args.valid_samples_file if args.valid_samples_file is not None else config.valid_samples_file
    config.sample_metadata_file = args.sample_metadata_file if args.sample_metadata_file is not None else config.sample_metadata_file
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.max_n_res = args.max_n_res if args.max_n_res is not None else config.max_n_res
    config.compute_slow_metrics = False
    config.enable_dual_mask_audit = False
    config.log_dir = args.output_dir
    config.save_dir = args.output_dir
    return config


def load_checkpoint_into_trainer(trainer: Stage1Trainer, ckpt: Dict[str, Any], checkpoint_path: str) -> None:
    state_dict = ckpt.get('model_state_dict', ckpt)
    result = trainer.model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in result.unexpected_keys if key not in FK_BUFFER_KEYS]
    missing = [key for key in result.missing_keys if key not in FK_BUFFER_KEYS]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}: unexpected={unexpected}, missing={missing}"
        )


def load_trusted_checkpoint(checkpoint_path: str, device: str, trust_pickle: bool) -> Dict[str, Any]:
    """Load a checkpoint, defaulting to PyTorch's safer tensor-only mode.

    Legacy BINDRAE checkpoints may contain a serialized TrainingConfig object;
    loading those requires pickle and is therefore restricted to explicit opt-in.
    Only use --trust_checkpoint_pickle for checkpoints produced by this project in
    the trusted cluster/workspace.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False if trust_pickle else True)
    except Exception as exc:
        if trust_pickle:
            raise
        raise RuntimeError(
            "Safe checkpoint loading failed. If this is a trusted BINDRAE legacy "
            "checkpoint that stores non-tensor config objects, rerun with "
            "--trust_checkpoint_pickle. Never use that flag for untrusted files."
        ) from exc
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(ckpt).__name__}")
    return ckpt


def batch_to_device(batch: Stage1Batch, device: torch.device) -> Stage1Batch:
    for name in (
        'esm', 'N_apo', 'Ca_apo', 'C_apo', 'N_holo', 'Ca_holo', 'C_holo',
        'node_mask', 'lig_points', 'lig_types', 'lig_mask', 'chi_holo',
        'chi_mask', 'torsion_apo', 'torsion_holo', 'w_res', 'atom14_holo',
        'atom14_holo_mask',
    ):
        value = getattr(batch, name)
        if value is not None:
            setattr(batch, name, value.to(device))
    return batch


def clone_batch(batch: Stage1Batch) -> Stage1Batch:
    values = {}
    for field in Stage1Batch.__dataclass_fields__:
        value = getattr(batch, field)
        if torch.is_tensor(value):
            values[field] = value.clone()
        elif isinstance(value, list):
            values[field] = list(value)
        else:
            values[field] = value
    return Stage1Batch(**values)


def make_variant(batch: Stage1Batch, variant: str, rng: torch.Generator) -> Stage1Batch:
    out = clone_batch(batch)
    if variant == 'correct_ligand':
        return out
    if variant == 'no_ligand':
        out.lig_mask = torch.zeros_like(out.lig_mask, dtype=torch.bool)
        return out
    if variant == 'translated_away':
        offset = torch.tensor([100.0, 100.0, 100.0], device=out.lig_points.device, dtype=out.lig_points.dtype)
        out.lig_points = out.lig_points + offset.view(1, 1, 3)
        return out
    if variant == 'scrambled_types':
        for i in range(out.lig_types.shape[0]):
            valid = torch.nonzero(out.lig_mask[i], as_tuple=False).flatten()
            if valid.numel() > 1:
                perm = valid[torch.randperm(valid.numel(), generator=rng, device=valid.device)]
                out.lig_types[i, valid] = out.lig_types[i, perm]
        return out
    if variant == 'batch_shuffled_ligand':
        if out.lig_points.shape[0] > 1:
            perm = torch.randperm(out.lig_points.shape[0], generator=rng, device=out.lig_points.device)
            out.lig_points = out.lig_points[perm]
            out.lig_types = out.lig_types[perm]
            out.lig_mask = out.lig_mask[perm]
        return out
    raise ValueError(f"Unknown variant: {variant}")


def outputs_from_model(model: torch.nn.Module, batch: Stage1Batch, current_step: int) -> Dict[str, torch.Tensor]:
    return model(batch, current_step=current_step)


def angles_from_outputs(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.atan2(outputs['pred_chi'][..., 0], outputs['pred_chi'][..., 1])


def angles_from_model(model: torch.nn.Module, batch: Stage1Batch, current_step: int) -> torch.Tensor:
    out = model(batch, current_step=current_step)
    return angles_from_outputs(out)


def init_stat() -> Dict[str, float]:
    return {
        'total': 0.0,
        'hits': 0.0,
        'mae_deg_sum': 0.0,
        'apo_correct_total': 0.0,
        'apo_wrong_total': 0.0,
        'rescue_hits': 0.0,
        'harmful_flips': 0.0,
        'pred_changed_total': 0.0,
        'pred_changed_sum_deg': 0.0,
    }


def update_stat(
    stat: Dict[str, float],
    pred: np.ndarray,
    holo: np.ndarray,
    apo: np.ndarray,
    mask: np.ndarray,
    threshold_deg: float,
    reference_pred: Optional[np.ndarray] = None,
) -> None:
    mask = mask.astype(bool)
    if not mask.any():
        return
    pred_valid = pred[mask]
    holo_valid = holo[mask]
    apo_valid = apo[mask]
    pred_err = np.abs(wrap_angle_diff(pred_valid, holo_valid)) * 180.0 / np.pi
    apo_err = np.abs(wrap_angle_diff(apo_valid, holo_valid)) * 180.0 / np.pi
    pred_correct = pred_err < threshold_deg
    apo_correct = apo_err < threshold_deg
    stat['total'] += float(mask.sum())
    stat['hits'] += float(pred_correct.sum())
    stat['mae_deg_sum'] += float(pred_err.sum())
    stat['apo_correct_total'] += float(apo_correct.sum())
    stat['apo_wrong_total'] += float((~apo_correct).sum())
    stat['rescue_hits'] += float(((~apo_correct) & pred_correct).sum())
    stat['harmful_flips'] += float((apo_correct & (~pred_correct)).sum())
    if reference_pred is not None:
        ref_valid = reference_pred[mask]
        changed_deg = np.abs(wrap_angle_diff(pred_valid, ref_valid)) * 180.0 / np.pi
        stat['pred_changed_total'] += float((changed_deg >= threshold_deg).sum())
        stat['pred_changed_sum_deg'] += float(changed_deg.sum())


def finalize_stat(stat: Dict[str, float]) -> Dict[str, float]:
    total = max(stat['total'], 1.0)
    apo_correct_total = max(stat['apo_correct_total'], 1.0)
    apo_wrong_total = max(stat['apo_wrong_total'], 1.0)
    return {
        'n': int(stat['total']),
        'chi1_acc': stat['hits'] / total,
        'chi1_mae_deg': stat['mae_deg_sum'] / total,
        'apo_correct_n': int(stat['apo_correct_total']),
        'apo_wrong_n': int(stat['apo_wrong_total']),
        'rescue_rate': stat['rescue_hits'] / apo_wrong_total,
        'harmful_flip_rate': stat['harmful_flips'] / apo_correct_total,
        'prediction_changed_rate': stat['pred_changed_total'] / total,
        'prediction_changed_mae_deg': stat['pred_changed_sum_deg'] / total,
    }


def init_posterior_stat() -> Dict[str, Any]:
    return {
        'rotamer_conf': [],
        'rotamer_correct': [],
        'rotamer_entropy': [],
        'rotamer_rescue': [],
        'rotamer_harmful': [],
        'contact_score': [],
        'contact_label': [],
    }


def init_rotamer_stat() -> Dict[str, float]:
    return {
        'total': 0.0,
        'hits': 0.0,
        'apo_wrong_total': 0.0,
        'rescue_hits': 0.0,
        'apo_correct_total': 0.0,
        'harmful_flips': 0.0,
        'confidence_sum': 0.0,
        'entropy_sum': 0.0,
        'changed_from_correct_total': 0.0,
    }


def select_rotamer_probs(outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    for key in ('candidate_chi1_probs', 'chi1_rotamer_probs', 'geometry_chi1_probs'):
        value = outputs.get(key)
        if value is not None:
            return value
    return None


def update_rotamer_stat(
    stat: Dict[str, float],
    outputs: Dict[str, torch.Tensor],
    holo_bins: np.ndarray,
    apo_bins: np.ndarray,
    mask: np.ndarray,
    correct_pred: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    probs_t = select_rotamer_probs(outputs)
    if probs_t is None:
        return None
    probs = probs_t.detach().cpu().numpy()
    pred = probs.argmax(axis=-1).astype(np.int64)
    mask = mask.astype(bool)
    if not mask.any():
        return pred

    pred_valid = pred[mask]
    holo_valid = holo_bins[mask]
    apo_valid = apo_bins[mask]
    conf_valid = probs.max(axis=-1)[mask]
    entropy_valid = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)[mask]

    stat['total'] += float(mask.sum())
    stat['hits'] += float((pred_valid == holo_valid).sum())
    apo_correct = apo_valid == holo_valid
    apo_wrong = ~apo_correct
    stat['apo_correct_total'] += float(apo_correct.sum())
    stat['apo_wrong_total'] += float(apo_wrong.sum())
    stat['rescue_hits'] += float(((pred_valid == holo_valid) & apo_wrong).sum())
    stat['harmful_flips'] += float(((pred_valid != holo_valid) & apo_correct).sum())
    stat['confidence_sum'] += float(conf_valid.sum())
    stat['entropy_sum'] += float(entropy_valid.sum())
    if correct_pred is not None:
        stat['changed_from_correct_total'] += float((pred_valid != correct_pred[mask]).sum())
    return pred


def finalize_rotamer_stat(stat: Dict[str, float]) -> Dict[str, float]:
    total = max(stat['total'], 1.0)
    apo_wrong = max(stat['apo_wrong_total'], 1.0)
    apo_correct = max(stat['apo_correct_total'], 1.0)
    return {
        'n': int(stat['total']),
        'rotamer_acc': stat['hits'] / total,
        'rescue_rate': stat['rescue_hits'] / apo_wrong,
        'harmful_flip_rate': stat['harmful_flips'] / apo_correct,
        'net_rescue': (stat['rescue_hits'] / apo_wrong) - (stat['harmful_flips'] / apo_correct),
        'mean_confidence': stat['confidence_sum'] / total,
        'mean_entropy': stat['entropy_sum'] / total,
        'changed_from_correct_rate': stat['changed_from_correct_total'] / total,
    }


def _extend_stat_list(stat: Dict[str, Any], key: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.size == 0:
        return
    stat[key].extend(values.reshape(-1).tolist())


def _rotamer_labels_np(chi1_angles: np.ndarray) -> np.ndarray:
    """Circular nearest-center rotamer bin: 0=g-(-60°), 1=g+(+60°), 2=t(180°)."""
    angle = ((chi1_angles + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-π, π)
    centers = np.array([-np.pi / 3, np.pi / 3, np.pi])  # g-, g+, t
    diff = angle[..., None] - centers  # [..., 3]
    # circular distance
    diff = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    return np.argmin(diff, axis=-1).astype(np.int64)


def _rotamer_bins_from_logits(logits: np.ndarray) -> np.ndarray:
    """Argmax of [B, N, 3] logits → [B, N] bin indices (0=g-, 1=g+, 2=t)."""
    return np.argmax(logits, axis=-1).astype(np.int64)


def _compute_G_i(base_logits: np.ndarray, full_logits: np.ndarray) -> np.ndarray:
    """G_i(k) = log softmax(full)_k - log softmax(base)_k.  Shapes [B, N, 3]."""
    def _log_softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        log_sum = np.log(np.exp(x - x_max).sum(axis=axis, keepdims=True)) + x_max
        return x - log_sum
    return _log_softmax(full_logits) - _log_softmax(base_logits)


def _init_decomp_stat() -> Dict[str, float]:
    return {
        'total': 0.0,
        'rotamer_hits': 0.0,
        'rescue_hits': 0.0,
        'harmful_flips': 0.0,
        'apo_correct_total': 0.0,
        'apo_wrong_total': 0.0,
        'G_holo_sum': 0.0,
        'G_max_sum': 0.0,
        'gate_mean_sum': 0.0,
        'gate_std_sum': 0.0,
        'residual_norm_sum': 0.0,
        'n_with_G': 0.0,
    }


def _update_decomp_stat(
    stat: Dict[str, float],
    pred_bins: np.ndarray,       # [B, N]
    holo_bins: np.ndarray,       # [B, N]
    apo_bins: np.ndarray,        # [B, N]
    mask: np.ndarray,            # [B, N] bool
    G_i: Optional[np.ndarray] = None,  # [B, N, 3] or None
    holo_labels: Optional[np.ndarray] = None,  # [B, N] int (for G on holo bin)
    gate: Optional[np.ndarray] = None,  # [B, N, 3] or [B, N, 1] or None
    residual: Optional[np.ndarray] = None,  # [B, N, 3] or None
) -> None:
    mask = mask.astype(bool)
    if not mask.any():
        return
    p = pred_bins[mask]
    h = holo_bins[mask]
    a = apo_bins[mask]
    stat['total'] += float(mask.sum())
    stat['rotamer_hits'] += float((p == h).sum())
    apo_correct = (a == h)
    apo_wrong = (a != h)
    stat['apo_correct_total'] += float(apo_correct.sum())
    stat['apo_wrong_total'] += float(apo_wrong.sum())
    stat['rescue_hits'] += float((apo_wrong & (p == h)).sum())
    stat['harmful_flips'] += float((apo_correct & (p != h)).sum())
    if G_i is not None and holo_labels is not None:
        G_flat = G_i[mask]  # [n, 3]
        h_flat = holo_labels[mask]  # [n]
        G_on_holo = G_flat[np.arange(len(h_flat)), h_flat]
        stat['G_holo_sum'] += float(G_on_holo.sum())
        stat['G_max_sum'] += float(G_flat.max(axis=-1).sum())
        stat['n_with_G'] += float(len(h_flat))
    if gate is not None:
        g = gate[mask]
        if g.ndim == 3:
            g = g.mean(axis=-1)  # [n]
        elif g.ndim == 2:
            pass
        else:
            g = g.reshape(-1)
        stat['gate_mean_sum'] += float(g.mean() * mask.sum())
        stat['gate_std_sum'] += float(g.std() * mask.sum())
    if residual is not None:
        r = residual[mask]  # [n, 3]
        norms = np.linalg.norm(r, axis=-1)  # [n]
        stat['residual_norm_sum'] += float(norms.sum())


def _finalize_decomp_stat(stat: Dict[str, float]) -> Dict[str, float]:
    total = max(stat['total'], 1.0)
    apo_wrong = max(stat['apo_wrong_total'], 1.0)
    apo_correct = max(stat['apo_correct_total'], 1.0)
    n_G = max(stat['n_with_G'], 1.0)
    return {
        'n': int(stat['total']),
        'rotamer_acc': stat['rotamer_hits'] / total,
        'rescue_rate': stat['rescue_hits'] / apo_wrong,
        'harmful_flip_rate': stat['harmful_flips'] / apo_correct,
        'net_rescue': (stat['rescue_hits'] / apo_wrong) - (stat['harmful_flips'] / apo_correct),
        'mean_G_holo': stat['G_holo_sum'] / n_G if stat['n_with_G'] > 0 else None,
        'mean_G_max': stat['G_max_sum'] / n_G if stat['n_with_G'] > 0 else None,
        'mean_gate': stat['gate_mean_sum'] / total if stat['gate_mean_sum'] != 0 else None,
        'mean_gate_std': stat['gate_std_sum'] / total if stat['gate_std_sum'] != 0 else None,
        'mean_residual_norm': stat['residual_norm_sum'] / total if stat['residual_norm_sum'] != 0 else None,
    }


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    scores = scores.astype(np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            tied = order[start:end]
            ranks[tied] = ranks[tied].mean()
        start = end
    pos_rank_sum = ranks[labels].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    scores = scores.astype(np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float('nan')
    order = np.argsort(-scores, kind='mergesort')
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels, dtype=np.float64)
    precision = tp / (np.arange(sorted_labels.size, dtype=np.float64) + 1.0)
    return float((precision * sorted_labels).sum() / n_pos)


def _ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> Tuple[float, List[Dict[str, float]]]:
    if confidence.size == 0:
        return float('nan'), []
    correct = correct.astype(np.float64)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        weight = n / confidence.size
        ece += weight * abs(avg_conf - avg_acc)
        bins.append({
            'lo': lo,
            'hi': hi,
            'n': n,
            'confidence': avg_conf,
            'accuracy': avg_acc,
        })
    return float(ece), bins


def _coverage_curve(confidence: np.ndarray, correct: np.ndarray, coverages: Iterable[float]) -> List[Dict[str, float]]:
    if confidence.size == 0:
        return []
    correct = correct.astype(np.float64)
    order = np.argsort(-confidence, kind='mergesort')
    curve = []
    for coverage in coverages:
        k = int(math.ceil(confidence.size * coverage))
        k = max(1, min(k, confidence.size))
        selected = order[:k]
        curve.append({
            'coverage': float(k / confidence.size),
            'n': int(k),
            'accuracy': float(correct[selected].mean()),
            'mean_confidence': float(confidence[selected].mean()),
            'min_confidence': float(confidence[selected].min()),
        })
    return curve


def _topk_contact_curve(scores: np.ndarray, labels: np.ndarray, fractions: Iterable[float]) -> List[Dict[str, float]]:
    if scores.size == 0:
        return []
    labels = labels.astype(bool)
    order = np.argsort(-scores, kind='mergesort')
    curve = []
    for fraction in fractions:
        k = int(math.ceil(scores.size * fraction))
        k = max(1, min(k, scores.size))
        selected = order[:k]
        positives = int(labels[selected].sum())
        curve.append({
            'fraction': float(k / scores.size),
            'n': int(k),
            'precision': float(positives / k),
            'recall': float(positives / max(int(labels.sum()), 1)),
            'min_score': float(scores[selected].min()),
            'mean_score': float(scores[selected].mean()),
        })
    return curve


def finalize_posterior_stat(stat: Dict[str, Any], n_bins: int) -> Dict[str, Any]:
    rot_conf = np.asarray(stat['rotamer_conf'], dtype=np.float64)
    rot_correct = np.asarray(stat['rotamer_correct'], dtype=bool)
    rot_entropy = np.asarray(stat['rotamer_entropy'], dtype=np.float64)
    rot_rescue = np.asarray(stat['rotamer_rescue'], dtype=bool)
    rot_harmful = np.asarray(stat['rotamer_harmful'], dtype=bool)
    contact_score = np.asarray(stat['contact_score'], dtype=np.float64)
    contact_label = np.asarray(stat['contact_label'], dtype=bool)

    rot_ece, rot_bins = _ece(rot_conf, rot_correct, n_bins)
    contact_ece, contact_bins = _ece(contact_score, contact_label, n_bins)
    rot_total = max(rot_correct.size, 1)
    contact_total = max(contact_label.size, 1)
    return {
        'rotamer': {
            'n': int(rot_correct.size),
            'accuracy': float(rot_correct.mean()) if rot_correct.size else float('nan'),
            'mean_confidence': float(rot_conf.mean()) if rot_conf.size else float('nan'),
            'mean_entropy': float(rot_entropy.mean()) if rot_entropy.size else float('nan'),
            'ece': rot_ece,
            'coverage_curve': _coverage_curve(rot_conf, rot_correct, (0.05, 0.10, 0.20, 0.40, 0.60, 1.00)),
            'calibration_bins': rot_bins,
            'rescue_rate': float(rot_rescue.sum() / rot_total),
            'harmful_flip_rate': float(rot_harmful.sum() / rot_total),
        },
        'contact': {
            'n': int(contact_label.size),
            'positive_rate': float(contact_label.mean()) if contact_label.size else float('nan'),
            'mean_score': float(contact_score.mean()) if contact_score.size else float('nan'),
            'auroc': _binary_auc(contact_label, contact_score) if contact_label.size else float('nan'),
            'average_precision': _average_precision(contact_label, contact_score) if contact_label.size else float('nan'),
            'ece': contact_ece,
            'topk_curve': _topk_contact_curve(contact_score, contact_label, (0.01, 0.02, 0.05, 0.10, 0.20)),
            'calibration_bins': contact_bins,
            'positives': int(contact_label.sum()) if contact_label.size else 0,
            'negatives': int(contact_total - contact_label.sum()) if contact_label.size else 0,
        },
    }


def update_posterior_stat(
    stat: Dict[str, Any],
    outputs: Dict[str, torch.Tensor],
    batch: Stage1Batch,
    rotamer_subset_mask: np.ndarray,
    contact_subset_mask: np.ndarray,
    contact_mask: Optional[np.ndarray],
    threshold_deg: float,
) -> None:
    rotamer_probs_t = outputs.get('chi1_rotamer_probs')
    if rotamer_probs_t is None:
        rotamer_probs_t = outputs.get('candidate_chi1_probs')
    if rotamer_probs_t is not None:
        probs = rotamer_probs_t.detach().cpu().numpy()
        labels = _rotamer_labels_np(batch.chi_holo[..., 0].detach().cpu().numpy())
        pred = probs.argmax(axis=-1)
        conf = probs.max(axis=-1)
        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)
        holo = batch.torsion_holo[..., 3].detach().cpu().numpy()
        apo = batch.torsion_apo[..., 3].detach().cpu().numpy()
        pred_angles = angles_from_outputs(outputs)[..., 0].detach().cpu().numpy()
        pred_err = np.abs(wrap_angle_diff(pred_angles, holo)) * 180.0 / np.pi
        apo_err = np.abs(wrap_angle_diff(apo, holo)) * 180.0 / np.pi
        pred_correct_angle = pred_err < threshold_deg
        apo_correct = apo_err < threshold_deg
        mask = rotamer_subset_mask.astype(bool)
        _extend_stat_list(stat, 'rotamer_conf', conf[mask])
        _extend_stat_list(stat, 'rotamer_correct', (pred == labels)[mask])
        _extend_stat_list(stat, 'rotamer_entropy', entropy[mask])
        _extend_stat_list(stat, 'rotamer_rescue', ((~apo_correct) & pred_correct_angle)[mask])
        _extend_stat_list(stat, 'rotamer_harmful', (apo_correct & (~pred_correct_angle))[mask])

    contact_probs_t = outputs.get('contact_probs')
    if contact_probs_t is not None and contact_mask is not None:
        contact_probs = contact_probs_t.detach().cpu().numpy()
        mask = contact_subset_mask.astype(bool)
        _extend_stat_list(stat, 'contact_score', contact_probs[mask])
        _extend_stat_list(stat, 'contact_label', contact_mask.astype(bool)[mask])


def ca_ligand_contact_masks(batch: Stage1Batch, contact_threshold: float) -> np.ndarray:
    ca = batch.Ca_apo
    lig = batch.lig_points
    lig_mask = batch.lig_mask.bool()
    diff = ca[:, :, None, :] - lig[:, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)
    dist = dist.masked_fill(~lig_mask[:, None, :], float('inf'))
    return (dist.min(dim=-1).values < contact_threshold).detach().cpu().numpy()


def subset_masks(batch: Stage1Batch, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    chi_mask = batch.chi_mask[..., 0].detach().cpu().numpy().astype(bool)
    node_mask = batch.node_mask.detach().cpu().numpy().astype(bool)
    w_res = batch.w_res.detach().cpu().numpy()
    valid = chi_mask & node_mask
    masks = {
        'all_chi1': valid,
        'pocket': valid & (w_res > args.pocket_threshold),
    }
    contact_ca = ca_ligand_contact_masks(batch, args.contact_threshold)
    masks['ligand_facing_apo_ca'] = valid & contact_ca
    masks['contact'] = valid & contact_ca

    # Always add subgroup masks so the validation report can compare the same
    # contact/switch subsets whether or not decomposition mode is enabled.
    apo_chi = batch.torsion_apo[..., 3].detach().cpu().numpy()
    holo_chi = batch.chi_holo[..., 0].detach().cpu().numpy()
    apo_err_deg = np.abs(wrap_angle_diff(apo_chi, holo_chi)) * 180.0 / np.pi
    apo_correct = apo_err_deg < args.threshold_deg
    masks['apo_correct'] = valid & apo_correct
    masks['apo_wrong'] = valid & (~apo_correct)
    apo_bins = _rotamer_labels_np(apo_chi)
    holo_bins = _rotamer_labels_np(holo_chi)
    masks['switch'] = valid & (~apo_correct) & (apo_bins != holo_bins)
    masks['contact_switch'] = masks['contact'] & masks['switch']
    masks['pocket_switch'] = masks['pocket'] & masks['switch']
    ca_lig_dist = ca_ligand_contact_masks(batch, 8.0)
    masks['non_contact'] = valid & (~ca_lig_dist)
    return masks


def posterior_subset_masks(batch: Stage1Batch, args: argparse.Namespace) -> Dict[str, Dict[str, np.ndarray]]:
    chi_mask = batch.chi_mask[..., 0].detach().cpu().numpy().astype(bool)
    node_mask = batch.node_mask.detach().cpu().numpy().astype(bool)
    w_res = batch.w_res.detach().cpu().numpy()
    contact_ca = ca_ligand_contact_masks(batch, args.contact_threshold)
    return {
        'all_residues': {
            'rotamer': chi_mask & node_mask,
            'contact': node_mask,
        },
        'pocket_residues': {
            'rotamer': chi_mask & node_mask & (w_res > args.pocket_threshold),
            'contact': node_mask & (w_res > args.pocket_threshold),
        },
        'ligand_facing_apo_ca_residues': {
            'rotamer': chi_mask & node_mask & contact_ca,
            'contact': node_mask & contact_ca,
        },
    }


def run_diagnostics(trainer: Stage1Trainer, args: argparse.Namespace, current_step: int) -> Dict[str, Any]:
    model = trainer._model
    model.eval()
    device = torch.device(args.device)
    variants = ['correct_ligand', 'no_ligand', 'translated_away', 'scrambled_types', 'batch_shuffled_ligand']
    stats = {variant: {} for variant in ['apo_carryover', *variants]}
    posterior_stats = {variant: {} for variant in variants}
    candidate_rotamer_stats = {variant: {} for variant in variants}
    decomp_stats = {variant: {} for variant in ['base_only', *variants]} if getattr(args, 'decomposition', False) else {}
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    n_batches = 0
    n_samples = 0
    if trainer.val_loader is None:
        raise RuntimeError("Stage1Trainer did not create a validation loader")
    with torch.no_grad():
        for batch in trainer.val_loader:
            if batch is None:
                continue
            batch = batch_to_device(batch, device)
            n_batches += 1
            n_samples += len(batch.pdb_ids)
            masks = subset_masks(batch, args)
            posterior_masks = posterior_subset_masks(batch, args) if args.posterior_diagnostics else {}
            apo = batch.torsion_apo[..., 3].detach().cpu().numpy()
            holo = batch.torsion_holo[..., 3].detach().cpu().numpy()
            for name in masks:
                stats['apo_carryover'].setdefault(name, init_stat())
                update_stat(stats['apo_carryover'][name], apo, holo, apo, masks[name], args.threshold_deg)
            correct_pred = None
            # For decomposition: precompute holo/apo bins
            if getattr(args, 'decomposition', False):
                holo_bins_np = _rotamer_labels_np(holo)
                apo_bins_np = _rotamer_labels_np(apo)
                base_logits_np = None
                correct_outputs_cache = None
            else:
                holo_bins_np = _rotamer_labels_np(holo)
                apo_bins_np = _rotamer_labels_np(apo)
            correct_candidate_pred = None
            for variant in variants:
                variant_batch = make_variant(batch, variant, rng)
                outputs = outputs_from_model(model, variant_batch, current_step)
                pred = angles_from_outputs(outputs)[..., 0].detach().cpu().numpy()
                if variant == 'correct_ligand':
                    correct_pred = pred
                    if getattr(args, 'decomposition', False):
                        correct_outputs_cache = outputs
                        bl = outputs.get('geometry_chi1_base_logits')
                        if bl is not None:
                            base_logits_np = bl.detach().cpu().numpy()
                candidate_pred = None
                contact_mask = None
                if args.posterior_diagnostics and outputs.get('contact_probs') is not None:
                    target_atom14, target_atom14_mask = trainer._build_atom14_targets(batch)
                    if batch.node_mask is not None:
                        target_atom14_mask = target_atom14_mask & batch.node_mask.unsqueeze(-1)
                    contact_mask = trainer._compute_batch_ligand_contact_mask(
                        batch,
                        target_atom14,
                        target_atom14_mask,
                    ).detach().cpu().numpy()
                for name in masks:
                    stats[variant].setdefault(name, init_stat())
                    update_stat(
                        stats[variant][name],
                        pred,
                        holo,
                        apo,
                        masks[name],
                        args.threshold_deg,
                        reference_pred=None if variant == 'correct_ligand' else correct_pred,
                    )
                    candidate_rotamer_stats[variant].setdefault(name, init_rotamer_stat())
                    candidate_pred = update_rotamer_stat(
                        candidate_rotamer_stats[variant][name],
                        outputs,
                        holo_bins_np,
                        apo_bins_np,
                        masks[name],
                        correct_pred=None if variant == 'correct_ligand' else correct_candidate_pred,
                    )
                if variant == 'correct_ligand' and candidate_pred is not None:
                    correct_candidate_pred = candidate_pred
                if args.posterior_diagnostics:
                    for name, mask_pair in posterior_masks.items():
                        posterior_stats[variant].setdefault(name, init_posterior_stat())
                        update_posterior_stat(
                            posterior_stats[variant][name],
                            outputs,
                            batch,
                            mask_pair['rotamer'],
                            mask_pair['contact'],
                            contact_mask,
                            args.threshold_deg,
                        )
                # --- decomposition stats for this variant ---
                if getattr(args, 'decomposition', False) and base_logits_np is not None:
                    geom_probs = outputs.get('geometry_chi1_probs')
                    if geom_probs is not None:
                        pred_bins = geom_probs.detach().cpu().numpy().argmax(axis=-1).astype(np.int64)
                    else:
                        pred_bins = _rotamer_bins_from_logits(
                            outputs['geometry_chi1_logits'].detach().cpu().numpy()
                        ) if outputs.get('geometry_chi1_logits') is not None else None
                    if pred_bins is not None:
                        full_logits_np = outputs['geometry_chi1_logits'].detach().cpu().numpy() if outputs.get('geometry_chi1_logits') is not None else None
                        G_i_np = _compute_G_i(base_logits_np, full_logits_np) if full_logits_np is not None else None
                        gate_np = outputs.get('geometry_chi1_gate')
                        gate_np = gate_np.detach().cpu().numpy() if gate_np is not None else None
                        res_np = outputs.get('geometry_chi1_residual_logits')
                        res_np = res_np.detach().cpu().numpy() if res_np is not None else None
                        for name in masks:
                            decomp_stats[variant].setdefault(name, _init_decomp_stat())
                            _update_decomp_stat(
                                decomp_stats[variant][name],
                                pred_bins, holo_bins_np, apo_bins_np, masks[name],
                                G_i=G_i_np, holo_labels=holo_bins_np,
                                gate=gate_np, residual=res_np,
                            )
            # --- base_only decomposition stats ---
            if getattr(args, 'decomposition', False) and base_logits_np is not None:
                base_pred_bins = _rotamer_bins_from_logits(base_logits_np)
                for name in masks:
                    decomp_stats['base_only'].setdefault(name, _init_decomp_stat())
                    _update_decomp_stat(
                        decomp_stats['base_only'][name],
                        base_pred_bins, holo_bins_np, apo_bins_np, masks[name],
                    )
            if args.max_batches is not None and n_batches >= args.max_batches:
                break
    results = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'n_batches': n_batches,
        'n_samples': n_samples,
        'threshold_deg': args.threshold_deg,
        'pocket_threshold': args.pocket_threshold,
        'contact_threshold': args.contact_threshold,
        'metrics': {
            variant: {name: finalize_stat(stat) for name, stat in subset_stats.items()}
            for variant, subset_stats in stats.items()
        },
    }
    if args.posterior_diagnostics:
        results['posterior_metrics'] = {
            variant: {
                name: finalize_posterior_stat(stat, args.calibration_bins)
                for name, stat in subset_stats.items()
            }
            for variant, subset_stats in posterior_stats.items()
        }
    results['candidate_rotamer_metrics'] = {
        variant: {name: finalize_rotamer_stat(stat) for name, stat in subset_stats.items()}
        for variant, subset_stats in candidate_rotamer_stats.items()
    }
    if decomp_stats:
        results['decomposition_metrics'] = {
            variant: {name: _finalize_decomp_stat(stat) for name, stat in subset_stats.items()}
            for variant, subset_stats in decomp_stats.items()
        }
    return results


def run_forced_alpha_sweep(trainer: Stage1Trainer, args: argparse.Namespace, current_step: int) -> Dict[str, Any]:
    """Sweep forced alpha: logits(alpha) = base_logits + alpha * residual_logits.

    Single forward pass per batch (correct_ligand only), then evaluate multiple alpha values.
    Reports rotamer_acc, rescue, harmful, net_rescue, G(holo) per alpha × subgroup.
    """
    model = trainer._model
    model.eval()
    device = torch.device(args.device)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    alphas = [float(a) for a in args.forced_alpha.split(',')]
    subsets = ['all_chi1', 'pocket', 'ligand_facing_apo_ca', 'apo_wrong', 'switch', 'non_contact']

    # stats[alpha][subset] = {rotamer_hits, total, rescue_hits, harmful_flips, apo_wrong_total, apo_correct_total, G_holo_sum, n_G}
    stats = {
        alpha: {s: {
            'rotamer_hits': 0.0, 'total': 0.0,
            'rescue_hits': 0.0, 'harmful_flips': 0.0,
            'apo_wrong_total': 0.0, 'apo_correct_total': 0.0,
            'G_holo_sum': 0.0, 'n_G': 0.0,
        } for s in subsets} for alpha in alphas
    }

    n_batches = 0
    n_samples = 0
    if trainer.val_loader is None:
        raise RuntimeError("Stage1Trainer did not create a validation loader")

    with torch.no_grad():
        for batch in trainer.val_loader:
            if batch is None:
                continue
            batch = batch_to_device(batch, device)
            n_batches += 1
            n_samples += len(batch.pdb_ids)
            masks = subset_masks(batch, args)

            # Single forward pass with correct ligand
            outputs = outputs_from_model(model, batch, current_step)
            base_logits_t = outputs.get('geometry_chi1_base_logits')
            residual_logits_t = outputs.get('geometry_chi1_residual_logits')
            gate_t = outputs.get('geometry_chi1_gate')

            if base_logits_t is None:
                raise RuntimeError("Model did not return geometry_chi1_base_logits. Check --decomposition flag or model config.")

            base_np = base_logits_t.detach().cpu().numpy()  # [B, N, 3]
            res_np = residual_logits_t.detach().cpu().numpy() if residual_logits_t is not None else np.zeros_like(base_np)
            gate_np = gate_t.detach().cpu().numpy() if gate_t is not None else np.zeros(base_np.shape[:2] + (1,))

            holo = batch.torsion_holo[..., 3].detach().cpu().numpy()
            apo = batch.torsion_apo[..., 3].detach().cpu().numpy()
            holo_bins = _rotamer_labels_np(holo)
            apo_bins = _rotamer_labels_np(apo)

            for alpha in alphas:
                # forced_logits = base + alpha * residual (ignoring learned gate)
                forced_logits = base_np + alpha * res_np
                forced_probs = _softmax_np(forced_logits)
                pred_bins = np.argmax(forced_logits, axis=-1).astype(np.int64)

                # G_i for this alpha
                base_probs = _softmax_np(base_np)
                G_i = np.log(forced_probs + 1e-10) - np.log(base_probs + 1e-10)  # [B, N, 3]

                for subset in subsets:
                    mask = masks[subset]
                    if not mask.any():
                        continue
                    p = pred_bins[mask]
                    h = holo_bins[mask]
                    a = apo_bins[mask]
                    s = stats[alpha][subset]
                    s['total'] += float(mask.sum())
                    s['rotamer_hits'] += float((p == h).sum())
                    apo_correct = (a == h)
                    apo_wrong = (a != h)
                    s['apo_correct_total'] += float(apo_correct.sum())
                    s['apo_wrong_total'] += float(apo_wrong.sum())
                    s['rescue_hits'] += float((apo_wrong & (p == h)).sum())
                    s['harmful_flips'] += float((apo_correct & (p != h)).sum())
                    # G on holo bin
                    G_flat = G_i[mask]  # [n, 3]
                    h_flat = holo_bins[mask]  # [n]
                    G_on_holo = G_flat[np.arange(len(h_flat)), h_flat]
                    s['G_holo_sum'] += float(G_on_holo.sum())
                    s['n_G'] += float(len(h_flat))

            if args.max_batches is not None and n_batches >= args.max_batches:
                break

    # Finalize
    results = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'n_batches': n_batches,
        'n_samples': n_samples,
        'alphas': alphas,
        'gate_stats': {
            'mean_gate': float(gate_np.mean()) if gate_np is not None else None,
            'mean_residual_norm': float(np.linalg.norm(res_np, axis=-1).mean()),
        },
        'forced_alpha_metrics': {},
    }
    for alpha in alphas:
        results['forced_alpha_metrics'][str(alpha)] = {}
        for subset in subsets:
            s = stats[alpha][subset]
            total = max(s['total'], 1.0)
            apo_wrong = max(s['apo_wrong_total'], 1.0)
            apo_correct = max(s['apo_correct_total'], 1.0)
            n_G = max(s['n_G'], 1.0)
            results['forced_alpha_metrics'][str(alpha)][subset] = {
                'n': int(s['total']),
                'rotamer_acc': s['rotamer_hits'] / total,
                'rescue_rate': s['rescue_hits'] / apo_wrong,
                'harmful_flip_rate': s['harmful_flips'] / apo_correct,
                'net_rescue': (s['rescue_hits'] / apo_wrong) - (s['harmful_flips'] / apo_correct),
                'mean_G_holo': s['G_holo_sum'] / n_G if s['n_G'] > 0 else None,
            }
    return results


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax on last axis."""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def print_forced_alpha_summary(results: Dict[str, Any]) -> None:
    """Print forced alpha sweep table."""
    alphas = results['alphas']
    metrics = results['forced_alpha_metrics']
    subsets = ['all_chi1', 'pocket', 'ligand_facing_apo_ca', 'apo_wrong', 'switch', 'non_contact']

    gs = results.get('gate_stats', {})
    print(f"\nGate stats: mean_gate={gs.get('mean_gate')}, mean_residual_norm={gs.get('mean_residual_norm')}")

    header = f"{'alpha':>10s} | {'subset':18s} | {'n':>7s} | {'rot_acc':>7s} | {'rescue':>7s} | {'harmful':>7s} | {'net_res':>7s} | {'G(holo)':>8s}"
    sep = '-' * len(header)
    print(f"\n{'=' * len(header)}")
    print("FORCED RESIDUAL ALPHA SWEEP")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for alpha in alphas:
        key = str(alpha)
        for subset in subsets:
            s = metrics[key][subset]
            if s['n'] == 0:
                continue
            g_holo = f"{s['mean_G_holo']:+.4f}" if s['mean_G_holo'] is not None else '   -   '
            print(
                f"{alpha:10.6f} | {subset:18s} | {s['n']:7d} | {s['rotamer_acc']:.4f} | "
                f"{s['rescue_rate']:.4f} | {s['harmful_flip_rate']:.4f} | {s['net_rescue']:+.4f} | {g_holo}"
            )
        print(sep)

    # Summary: switch G(holo) vs alpha
    print("\nSUMMARY: switch G(holo) vs alpha")
    for alpha in alphas:
        key = str(alpha)
        s = metrics[key].get('switch', {})
        if s.get('n', 0) > 0 and s.get('mean_G_holo') is not None:
            print(f"  alpha={alpha:10.6f}  G(holo)={s['mean_G_holo']:+.6f}  rot_acc={s['rotamer_acc']:.4f}")

    print("\nSUMMARY: non_contact G(holo) vs alpha (should stay ~0)")
    for alpha in alphas:
        key = str(alpha)
        s = metrics[key].get('non_contact', {})
        if s.get('n', 0) > 0 and s.get('mean_G_holo') is not None:
            print(f"  alpha={alpha:10.6f}  G(holo)={s['mean_G_holo']:+.6f}  rot_acc={s['rotamer_acc']:.4f}")


def _print_decomposition_table(decomp_metrics: Dict[str, Any]) -> None:
    """Print the decomposition ablation table with rotamer acc, rescue, G_i, gate stats."""
    # Subsets to show (order matters)
    subsets = ['all_chi1', 'pocket', 'ligand_facing_apo_ca', 'apo_wrong', 'switch', 'non_contact']
    variants = ['apo_carryover', 'base_only', 'correct_ligand', 'no_ligand',
                'translated_away', 'scrambled_types', 'batch_shuffled_ligand']
    header = (
        f"{'variant':22s} | {'subset':18s} | {'n':>6s} | {'rot_acc':>7s} | "
        f"{'rescue':>7s} | {'harmful':>7s} | {'net_res':>7s} | "
        f"{'G(holo)':>8s} | {'G(max)':>8s} | {'gate':>6s} | {'res_norm':>8s}"
    )
    sep = '-' * len(header)
    print(f"\n{'=' * len(header)}")
    print("DECOMPOSITION ABLATION")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for variant in variants:
        if variant not in decomp_metrics:
            continue
        shown_any = False
        for subset in subsets:
            if subset not in decomp_metrics[variant]:
                continue
            s = decomp_metrics[variant][subset]
            if s['n'] == 0:
                continue
            shown_any = True
            g_holo = f"{s['mean_G_holo']:.4f}" if s['mean_G_holo'] is not None else '   -   '
            g_max = f"{s['mean_G_max']:.4f}" if s['mean_G_max'] is not None else '   -   '
            gate = f"{s['mean_gate']:.4f}" if s['mean_gate'] is not None else '  -  '
            res_n = f"{s['mean_residual_norm']:.4f}" if s['mean_residual_norm'] is not None else '    -    '
            print(
                f"  {variant:22s} | {subset:18s} | {s['n']:6d} | {s['rotamer_acc']:.4f} | "
                f"{s['rescue_rate']:.4f} | {s['harmful_flip_rate']:.4f} | {s['net_rescue']:.4f} | "
                f"{g_holo} | {g_max} | {gate} | {res_n}"
            )
        if shown_any:
            print(sep)
    # Summary: key comparisons
    print("\nKEY COMPARISONS (all_chi1 rotamer_acc):")
    for variant in variants:
        if variant in decomp_metrics and 'all_chi1' in decomp_metrics[variant]:
            s = decomp_metrics[variant]['all_chi1']
            print(f"  {variant:22s}: {s['rotamer_acc']:.4f}  (n={s['n']})")
    # G_i summary for switch residues
    has_switch = any(
        v in decomp_metrics and 'switch' in decomp_metrics[v]
        for v in variants
    )
    if has_switch:
        print("\nG_i ON SWITCH RESIDUES (should be positive if ligand helps):")
        for variant in variants:
            if variant in decomp_metrics and 'switch' in decomp_metrics[variant]:
                s = decomp_metrics[variant]['switch']
                if s['n'] > 0 and s['mean_G_holo'] is not None:
                    print(f"  {variant:22s}: G(holo)={s['mean_G_holo']:+.4f}  n={s['n']}")
    # G_i on non-contact (should be ~0)
    has_nc = any(
        v in decomp_metrics and 'non_contact' in decomp_metrics[v]
        for v in variants
    )
    if has_nc:
        print("\nG_i ON NON-CONTACT RESIDUES (should be ~0):")
        for variant in variants:
            if variant in decomp_metrics and 'non_contact' in decomp_metrics[variant]:
                s = decomp_metrics[variant]['non_contact']
                if s['n'] > 0 and s['mean_G_holo'] is not None:
                    print(f"  {variant:22s}: G(holo)={s['mean_G_holo']:+.4f}  n={s['n']}")


def print_summary(results: Dict[str, Any]) -> None:
    print(f"checkpoint: {results['checkpoint']}")
    print(f"samples: {results['n_samples']} batches: {results['n_batches']}")
    for subset in ('all_chi1', 'pocket', 'ligand_facing_apo_ca'):
        print(f"\n{subset}")
        for variant, subset_stats in results['metrics'].items():
            if subset not in subset_stats:
                continue
            stat = subset_stats[subset]
            print(
                f"  {variant:22s} n={stat['n']:6d} acc={stat['chi1_acc']:.4f} "
                f"rescue={stat['rescue_rate']:.4f} harmful={stat['harmful_flip_rate']:.4f} "
                f"changed={stat['prediction_changed_rate']:.4f}"
            )
    posterior_metrics = results.get('posterior_metrics')
    if posterior_metrics:
        print("\nposterior utility")
        for subset in ('all_residues', 'pocket_residues', 'ligand_facing_apo_ca_residues'):
            print(f"\n{subset}")
            for variant, subset_stats in posterior_metrics.items():
                if subset not in subset_stats:
                    continue
                stat = subset_stats[subset]
                rot = stat['rotamer']
                contact = stat['contact']
                top20 = next((x for x in rot['coverage_curve'] if x['coverage'] >= 0.2), None)
                top20_acc = float('nan') if top20 is None else top20['accuracy']
                print(
                    f"  {variant:22s} rot_n={rot['n']:6d} rot_acc={rot['accuracy']:.4f} "
                    f"top20_acc={top20_acc:.4f} rot_ece={rot['ece']:.4f} "
                    f"contact_ap={contact['average_precision']:.4f} contact_auc={contact['auroc']:.4f}"
                )
    candidate_metrics = results.get('candidate_rotamer_metrics')
    if candidate_metrics:
        print("\ncandidate/rotamer ligand sensitivity")
        for subset in ('all_chi1', 'pocket', 'ligand_facing_apo_ca', 'apo_wrong', 'switch', 'non_contact'):
            print(f"\n{subset}")
            for variant, subset_stats in candidate_metrics.items():
                if subset not in subset_stats:
                    continue
                stat = subset_stats[subset]
                if stat['n'] == 0:
                    continue
                print(
                    f"  {variant:22s} n={stat['n']:6d} rot_acc={stat['rotamer_acc']:.4f} "
                    f"rescue={stat['rescue_rate']:.4f} harmful={stat['harmful_flip_rate']:.4f} "
                    f"net={stat['net_rescue']:+.4f} changed={stat['changed_from_correct_rate']:.4f}"
                )
    decomp_metrics = results.get('decomposition_metrics')
    if decomp_metrics:
        _print_decomposition_table(decomp_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description='Diagnose Stage-1 prior utility and ligand causality')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--valid_samples_file', type=str, default=None)
    parser.add_argument('--val_samples_file', type=str, default=None)
    parser.add_argument('--sample_metadata_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_n_res', type=int, default=None)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='logs/stage1_diagnostics')
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--threshold_deg', type=float, default=20.0)
    parser.add_argument('--pocket_threshold', type=float, default=0.5)
    parser.add_argument('--contact_threshold', type=float, default=4.5)
    parser.add_argument('--current_step', type=int, default=None)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--posterior_diagnostics', action='store_true', help='Add rotamer confidence/coverage/calibration and contact ranking diagnostics')
    parser.add_argument('--calibration_bins', type=int, default=10, help='Number of bins for posterior calibration summaries')
    parser.add_argument('--decomposition', action='store_true', help='Add base-prior decomposition ablation: base_only mode, G_i likelihood-ratio, gate/residual stats')
    parser.add_argument('--forced_alpha', type=str, default=None,
                        help='Comma-separated alpha values for forced residual scale sweep (e.g. "0,1e-4,1e-3,0.01,0.1,0.3,1.0"). '
                             'Runs single forward pass and evaluates logits=base+alpha*residual.')
    parser.add_argument('--trust_checkpoint_pickle', action='store_true',
                        help='Allow unsafe pickle checkpoint loading for trusted BINDRAE legacy checkpoints only')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = load_trusted_checkpoint(args.checkpoint, args.device, args.trust_checkpoint_pickle)
    config = make_config(ckpt, args)
    trainer = Stage1Trainer(config)
    load_checkpoint_into_trainer(trainer, ckpt, args.checkpoint)
    if 'global_step' in ckpt:
        trainer.global_step = int(ckpt['global_step'])
    current_step = args.current_step if args.current_step is not None else trainer.global_step

    if args.forced_alpha:
        results = run_forced_alpha_sweep(trainer, args, current_step)
        output_json = Path(args.output_json) if args.output_json else Path(args.output_dir) / 'forced_alpha_sweep.json'
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print_forced_alpha_summary(results)
    else:
        results = run_diagnostics(trainer, args, current_step)
        output_json = Path(args.output_json) if args.output_json else Path(args.output_dir) / 'stage1_prior_diagnostics.json'
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print_summary(results)
    print(f"\nSaved diagnostics to: {output_json}")


if __name__ == '__main__':
    main()
