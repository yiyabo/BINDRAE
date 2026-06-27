#!/usr/bin/env python3
"""Audit a trained Stage-1-v2 posterior student checkpoint."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from src.stage1.posterior_v2.dataset import (  # noqa: E402
    TeacherPosteriorDataset,
    collate_teacher_posterior_batch,
)
from src.stage1.posterior_v2.inference import (  # noqa: E402
    batch_to_device,
    load_stage1v2_posterior_checkpoint,
    posterior_selection_score,
    predict_posterior,
)
from src.stage1.posterior_v2.losses import (  # noqa: E402
    compute_posterior_metric_sums,
    finalize_posterior_metrics,
)


HEADS = {
    "contact": ("contact_prob", "contact_teacher"),
    "active": ("switch_prob", "active"),
    "approach": ("approach_prob", "approach"),
    "release": ("release_prob", "release"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage-1-v2 posterior student")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--output_dir", default="logs/stage1v2_audits")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--subset_seed", type=int, default=20260622)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--counterfactuals", default="nolig,shuffled,translated")
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_steps", type=int, default=19)
    parser.add_argument("--calibration_bins", type=int, default=10)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def _subset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    g = torch.Generator()
    g.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=g)[: int(max_samples)].tolist()
    return Subset(dataset, indices)


def _parse_modes(raw: str) -> List[str]:
    allowed = {"nolig", "shuffled", "translated"}
    modes: List[str] = []
    for item in str(raw or "").split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode not in allowed:
            raise ValueError(f"Unknown counterfactual mode {mode!r}; expected one of {sorted(allowed)}")
        if mode not in modes:
            modes.append(mode)
    return modes


def _add_metric_sums(dst: Dict[str, float], src: Dict[str, torch.Tensor]) -> None:
    for key, value in src.items():
        if torch.is_tensor(value):
            value = float(value.detach().float().cpu().item())
        dst[key] = dst.get(key, 0.0) + float(value)


def _binary_stats(prob: np.ndarray, target: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = prob >= float(threshold)
    target = target.astype(bool)
    tp = float(np.logical_and(pred, target).sum())
    tn = float(np.logical_and(~pred, ~target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    eps = 1e-8
    pos_recall = tp / max(tp + fn, eps)
    neg_recall = tn / max(tn + fp, eps)
    precision = tp / max(tp + fp, eps)
    f1 = 2.0 * precision * pos_recall / max(precision + pos_recall, eps)
    return {
        "threshold": float(threshold),
        "acc": (tp + tn) / max(tp + tn + fp + fn, eps),
        "balanced_acc": 0.5 * (pos_recall + neg_recall),
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
        "precision": precision,
        "f1": f1,
    }


def _ranking_metrics(prob: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    target = target.astype(bool)
    n_pos = int(target.sum())
    n_neg = int((~target).sum())
    if n_pos == 0 or n_neg == 0:
        return {"auroc": float("nan"), "auprc": float("nan")}

    order = np.argsort(-prob, kind="mergesort")
    y = target[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    recall = tp / max(float(n_pos), 1.0)
    fpr = fp / max(float(n_neg), 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)

    roc_x = np.concatenate(([0.0], fpr, [1.0]))
    roc_y = np.concatenate(([0.0], recall, [1.0]))
    pr_x = np.concatenate(([0.0], recall))
    pr_y = np.concatenate(([1.0], precision))
    return {
        "auroc": float(np.trapz(roc_y, roc_x)),
        "auprc": float(np.trapz(pr_y, pr_x)),
    }


def _calibration_metrics(prob: np.ndarray, target: np.ndarray, n_bins: int) -> Dict[str, float | List[Dict[str, float]]]:
    target_f = target.astype(np.float64)
    brier = float(np.mean((prob - target_f) ** 2)) if prob.size else float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = max(int(prob.size), 1)
    ece = 0.0
    records: List[Dict[str, float]] = []
    for i in range(int(n_bins)):
        lo = bins[i]
        hi = bins[i + 1]
        in_bin = (prob >= lo) & ((prob <= hi) if i == int(n_bins) - 1 else (prob < hi))
        count = int(in_bin.sum())
        if count == 0:
            records.append({"lo": float(lo), "hi": float(hi), "count": 0, "confidence": 0.0, "empirical": 0.0})
            continue
        conf = float(prob[in_bin].mean())
        empirical = float(target_f[in_bin].mean())
        ece += (count / total) * abs(conf - empirical)
        records.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "count": count,
                "confidence": conf,
                "empirical": empirical,
            }
        )
    return {"brier": brier, "ece": float(ece), "calibration_bins": records}


def _head_audit(prob_parts: List[np.ndarray], target_parts: List[np.ndarray], args: argparse.Namespace) -> Dict:
    prob = np.concatenate(prob_parts).astype(np.float64) if prob_parts else np.zeros((0,), dtype=np.float64)
    target = np.concatenate(target_parts).astype(bool) if target_parts else np.zeros((0,), dtype=bool)
    thresholds = np.linspace(float(args.threshold_min), float(args.threshold_max), int(args.threshold_steps))
    curve = [_binary_stats(prob, target, float(thr)) for thr in thresholds]
    fixed = _binary_stats(prob, target, 0.5)
    best_bal = max(curve, key=lambda row: row["balanced_acc"]) if curve else fixed
    best_f1 = max(curve, key=lambda row: row["f1"]) if curve else fixed
    out = {
        "count": int(prob.size),
        "pos_count": int(target.sum()),
        "neg_count": int((~target).sum()),
        "pos_rate": float(target.mean()) if target.size else 0.0,
        "fixed_threshold_0p5": fixed,
        "best_balanced_acc": best_bal,
        "best_f1": best_f1,
        "threshold_curve": curve,
    }
    out.update(_ranking_metrics(prob, target))
    out.update(_calibration_metrics(prob, target, args.calibration_bins))
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    tag = args.tag or f"{Path(args.checkpoint).parent.name}_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_stage1v2_posterior_checkpoint(args.checkpoint, device=device)
    dataset = TeacherPosteriorDataset(
        args.data_dir,
        args.label_dir,
        split=args.split,
        valid_samples_file=args.valid_samples_file,
    )
    dataset = _subset(dataset, args.max_samples, args.subset_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_teacher_posterior_batch,
        pin_memory=True,
    )

    counterfactuals = _parse_modes(args.counterfactuals)
    metric_sums: Dict[str, float] = {}
    cf_metric_sums: Dict[str, Dict[str, float]] = {mode: {} for mode in counterfactuals}
    prob_parts = {name: [] for name in HEADS}
    target_parts = {name: [] for name in HEADS}
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=(not args.no_amp) and device.type == "cuda", dtype=torch.bfloat16):
                pred = predict_posterior(model, batch, mode="real")
            _add_metric_sums(metric_sums, compute_posterior_metric_sums(pred, batch.teacher_float, batch.teacher_bool, batch.node_mask))

            valid = (batch.teacher_bool["valid_mask"].bool() & batch.node_mask.bool()).detach().cpu().numpy()
            for head_name, (prob_key, target_key) in HEADS.items():
                prob_np = pred[prob_key].detach().float().cpu().numpy()
                target_np = batch.teacher_bool[target_key].detach().cpu().numpy()
                prob_parts[head_name].append(prob_np[valid])
                target_parts[head_name].append(target_np[valid])

            for mode in counterfactuals:
                with torch.cuda.amp.autocast(enabled=(not args.no_amp) and device.type == "cuda", dtype=torch.bfloat16):
                    pred_cf = predict_posterior(model, batch, mode=mode)
                cf_sums = compute_posterior_metric_sums(pred_cf, batch.teacher_float, batch.teacher_bool, batch.node_mask)
                _add_metric_sums(cf_metric_sums[mode], cf_sums)
            n_samples += len(batch.pdb_ids)

    fixed_metrics = finalize_posterior_metrics(metric_sums)
    fixed_metrics["posterior_selection_score"] = posterior_selection_score(fixed_metrics)
    cf_metrics = {}
    for mode, sums in cf_metric_sums.items():
        metrics = finalize_posterior_metrics(sums)
        cf_metrics[mode] = metrics
        for metric in ("pocket_delta_mae", "pocket_dist_mae"):
            cf_metrics[mode][f"minus_real_{metric}"] = metrics[metric] - fixed_metrics[metric]
        for name in HEADS:
            cf_metrics[mode][f"real_minus_cf_{name}_balanced_acc"] = (
                fixed_metrics[f"{name}_balanced_acc"] - metrics[f"{name}_balanced_acc"]
            )

    result = {
        "schema_version": "bindrae_stage1v2_posterior_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
        "args": vars(args),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "split": args.split,
        "num_samples": n_samples,
        "fixed_metrics": fixed_metrics,
        "counterfactual_metrics": cf_metrics,
        "head_audits": {
            name: _head_audit(prob_parts[name], target_parts[name], args)
            for name in HEADS
        },
    }

    output_path = output_dir / f"{tag}.json"
    with output_path.open("w") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    print(f"Wrote audit: {output_path}")
    print(
        json.dumps(
            {
                "num_samples": n_samples,
                "posterior_selection_score": fixed_metrics["posterior_selection_score"],
                "contact_balanced_acc": fixed_metrics["contact_balanced_acc"],
                "active_balanced_acc": fixed_metrics["active_balanced_acc"],
                "approach_balanced_acc": fixed_metrics["approach_balanced_acc"],
                "release_balanced_acc": fixed_metrics["release_balanced_acc"],
                "contact_auprc": result["head_audits"]["contact"]["auprc"],
                "active_auprc": result["head_audits"]["active"]["auprc"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
