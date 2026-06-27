"""Losses and metrics for Stage-1-v2 posterior student."""

from typing import Collection, Dict, Mapping, Optional

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=eps)


def bce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return masked_mean(loss, mask)


def class_balanced_bce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    mask = mask.bool()
    pos = target.bool() & mask
    neg = (~target.bool()) & mask
    pos_count = pos.float().sum()
    neg_count = neg.float().sum()
    pos_loss = masked_mean(loss, pos) if bool(pos_count.detach().cpu().item() > 0) else logits.new_tensor(0.0)
    neg_loss = masked_mean(loss, neg) if bool(neg_count.detach().cpu().item() > 0) else logits.new_tensor(0.0)
    if bool((pos_count.detach().cpu().item() > 0) and (neg_count.detach().cpu().item() > 0)):
        return 0.5 * (pos_loss + neg_loss)
    return pos_loss + neg_loss


def mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target).abs(), mask)


def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum()


def compute_posterior_losses(
    pred: Dict[str, torch.Tensor],
    teacher_float: Dict[str, torch.Tensor],
    teacher_bool: Dict[str, torch.Tensor],
    node_mask: torch.Tensor,
    weights: Dict[str, float],
    class_balanced_heads: Optional[Collection[str]] = None,
) -> Dict[str, torch.Tensor]:
    valid = teacher_bool["valid_mask"].bool() & node_mask.bool()
    pocket = teacher_bool["pocket_mask"].bool() & valid
    active = teacher_bool["active"].bool() & valid
    contact = teacher_bool["contact_teacher"].float()
    approach = teacher_bool["approach"].float()
    release = teacher_bool["release"].float()
    switch = teacher_bool["active"].float()
    confidence = teacher_float["confidence"].float()

    balanced = set(class_balanced_heads or ())

    def cls_loss(name: str, logits_key: str, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if name in balanced:
            return class_balanced_bce_loss(pred[logits_key], target, mask)
        return bce_loss(pred[logits_key], target, mask)

    reg_mask = pocket | active
    losses = {
        "contact_bce": cls_loss("contact", "contact_logit", contact, valid),
        "approach_bce": cls_loss("approach", "approach_logit", approach, valid),
        "release_bce": cls_loss("release", "release_logit", release, valid),
        "switch_bce": cls_loss("switch", "switch_logit", switch, valid),
        "confidence_bce": bce_loss(pred["confidence_logit"], confidence, valid),
        "dist_mae": mae_loss(pred["teacher_min_dist"], teacher_float["teacher_min_dist"], reg_mask),
        "delta_mae": mae_loss(pred["signed_delta_dist"], teacher_float["signed_delta_dist"], reg_mask),
    }
    total = pred["contact_logit"].new_tensor(0.0)
    for key, loss in losses.items():
        total = total + float(weights.get(key, 1.0)) * loss
    losses["total"] = total
    return losses


def binary_count_stats(
    prob: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    mask = mask.bool()
    target_bool = target.bool()
    pred = prob >= threshold
    pos = target_bool & mask
    neg = (~target_bool) & mask
    stats = {
        f"{prefix}_tp": (pred & pos).float().sum(),
        f"{prefix}_tn": ((~pred) & neg).float().sum(),
        f"{prefix}_fp": (pred & neg).float().sum(),
        f"{prefix}_fn": ((~pred) & pos).float().sum(),
        f"{prefix}_count": mask.float().sum(),
        f"{prefix}_pos_count": pos.float().sum(),
        f"{prefix}_neg_count": neg.float().sum(),
    }
    return stats


def compute_posterior_metric_sums(
    pred: Dict[str, torch.Tensor],
    teacher_float: Dict[str, torch.Tensor],
    teacher_bool: Dict[str, torch.Tensor],
    node_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    valid = teacher_bool["valid_mask"].bool() & node_mask.bool()
    pocket = teacher_bool["pocket_mask"].bool() & valid
    active = teacher_bool["active"].bool() & valid
    reg_mask = pocket | active
    metrics: Dict[str, torch.Tensor] = {}
    metrics.update(binary_count_stats(pred["contact_prob"], teacher_bool["contact_teacher"], valid, "contact"))
    metrics.update(binary_count_stats(pred["switch_prob"], teacher_bool["active"], valid, "active"))
    metrics.update(binary_count_stats(pred["approach_prob"], teacher_bool["approach"], valid, "approach"))
    metrics.update(binary_count_stats(pred["release_prob"], teacher_bool["release"], valid, "release"))
    metrics.update(
        {
            "pocket_delta_abs_sum": masked_sum(
                (pred["signed_delta_dist"] - teacher_float["signed_delta_dist"]).abs(), reg_mask
            ),
            "pocket_dist_abs_sum": masked_sum(
                (pred["teacher_min_dist"] - teacher_float["teacher_min_dist"]).abs(), reg_mask
            ),
            "pocket_reg_count": reg_mask.float().sum(),
            "num_active_residues": active.float().sum(),
            "num_pocket_residues": pocket.float().sum(),
            "num_valid_residues": valid.float().sum(),
        }
    )
    return metrics


def finalize_posterior_metrics(metric_sums: Mapping[str, float], prefix: str = "") -> Dict[str, float]:
    eps = 1e-8
    out: Dict[str, float] = {}
    for name in ("contact", "active", "approach", "release"):
        tp = float(metric_sums.get(f"{name}_tp", 0.0))
        tn = float(metric_sums.get(f"{name}_tn", 0.0))
        fp = float(metric_sums.get(f"{name}_fp", 0.0))
        fn = float(metric_sums.get(f"{name}_fn", 0.0))
        count = float(metric_sums.get(f"{name}_count", 0.0))
        pos_count = float(metric_sums.get(f"{name}_pos_count", 0.0))
        neg_count = float(metric_sums.get(f"{name}_neg_count", 0.0))
        pos_recall = tp / max(tp + fn, eps)
        neg_recall = tn / max(tn + fp, eps)
        precision = tp / max(tp + fp, eps)
        f1 = 2.0 * precision * pos_recall / max(precision + pos_recall, eps)
        out.update(
            {
                f"{prefix}{name}_acc": (tp + tn) / max(count, eps),
                f"{prefix}{name}_balanced_acc": 0.5 * (pos_recall + neg_recall),
                f"{prefix}{name}_pos_recall": pos_recall,
                f"{prefix}{name}_neg_recall": neg_recall,
                f"{prefix}{name}_precision": precision,
                f"{prefix}{name}_f1": f1,
                f"{prefix}{name}_pos_rate": pos_count / max(count, eps),
                f"{prefix}{name}_pos_count": pos_count,
                f"{prefix}{name}_neg_count": neg_count,
            }
        )
    reg_count = float(metric_sums.get("pocket_reg_count", 0.0))
    out.update(
        {
            f"{prefix}pocket_delta_mae": float(metric_sums.get("pocket_delta_abs_sum", 0.0)) / max(reg_count, eps),
            f"{prefix}pocket_dist_mae": float(metric_sums.get("pocket_dist_abs_sum", 0.0)) / max(reg_count, eps),
            f"{prefix}pocket_reg_count": reg_count,
            f"{prefix}num_active_residues": float(metric_sums.get("num_active_residues", 0.0)),
            f"{prefix}num_pocket_residues": float(metric_sums.get("num_pocket_residues", 0.0)),
            f"{prefix}num_valid_residues": float(metric_sums.get("num_valid_residues", 0.0)),
        }
    )
    return out


def compute_posterior_metrics(
    pred: Dict[str, torch.Tensor],
    teacher_float: Dict[str, torch.Tensor],
    teacher_bool: Dict[str, torch.Tensor],
    node_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    metric_sums = compute_posterior_metric_sums(pred, teacher_float, teacher_bool, node_mask)
    finalized = finalize_posterior_metrics({k: float(v.detach().float().cpu().item()) for k, v in metric_sums.items()})
    return {k: pred["contact_prob"].new_tensor(v) for k, v in finalized.items()}
