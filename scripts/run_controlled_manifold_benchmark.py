#!/usr/bin/env python3
"""Controlled product-manifold benchmark for phase/normal identifiability.

The benchmark generates paired endpoint-conditioned paths on
``SE(3) x T^2``.  Endpoint-derived features determine a monotone residue phase,
while a binary route variable determines an endpoint-zero normal detour.  In
the observed regime the route variable is supplied to the predictor; in the
hidden regime identical endpoint inputs are paired with opposite routes.  The
latter has an exactly zero deterministic conditional-mean residual.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage2.modules import so3_exp, so3_log, wrap_to_pi  # noqa: E402


VARIANTS = ("synchronous", "warp_only", "residual_only", "full")
REGIMES = ("route_observed", "route_hidden")


@dataclass
class ControlledDataset:
    features_observed: torch.Tensor
    features_hidden: torch.Tensor
    translation_apo: torch.Tensor
    translation_delta: torch.Tensor
    rotation_delta: torch.Tensor
    chi_apo: torch.Tensor
    chi_delta: torch.Tensor
    translation_normal: torch.Tensor
    rotation_normal: torch.Tensor
    chi_normal: torch.Tensor
    phase_coefficient: torch.Tensor
    residual_coefficient: torch.Tensor
    route_sign: torch.Tensor

    def features(self, regime: str) -> torch.Tensor:
        if regime == "route_observed":
            return self.features_observed
        if regime == "route_hidden":
            return self.features_hidden
        raise ValueError(f"Unsupported regime: {regime}")

    @property
    def systems(self) -> int:
        return int(self.phase_coefficient.shape[0])


class CoefficientPredictor(nn.Module):
    """Matched-capacity per-residue predictor with separate phase/route heads."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phase_head = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Linear(hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_context = features.mean(dim=1, keepdim=True).expand_as(features)
        hidden = self.trunk(torch.cat([features, global_context], dim=-1))
        return {
            "phase": 0.8 * torch.tanh(self.phase_head(hidden).squeeze(-1)),
            "residual": self.residual_head(hidden),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--model_seeds", default="7,42,137")
    parser.add_argument("--train_endpoint_pairs", type=int, default=256)
    parser.add_argument("--val_endpoint_pairs", type=int, default=64)
    parser.add_argument("--test_endpoint_pairs", type=int, default=128)
    parser.add_argument("--n_residues", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=40)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _orthogonal_unit(vector: torch.Tensor, fallback_axis: int = 2) -> torch.Tensor:
    axis = torch.zeros_like(vector)
    axis[..., fallback_axis] = 1.0
    parallel = (vector * axis).sum(dim=-1, keepdim=True)
    denominator = vector.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    normal = axis - parallel * vector / denominator
    weak = torch.linalg.norm(normal, dim=-1, keepdim=True) < 1e-3
    alternate = torch.zeros_like(vector)
    alternate[..., (fallback_axis + 1) % 3] = 1.0
    alternate = alternate - (
        (vector * alternate).sum(dim=-1, keepdim=True) * vector / denominator
    )
    normal = torch.where(weak, alternate, normal)
    return F.normalize(normal, dim=-1)


def make_dataset(
    endpoint_pairs: int,
    n_residues: int,
    *,
    seed: int,
) -> ControlledDataset:
    """Create exact opposite-route pairs for every endpoint-conditioned input."""
    if endpoint_pairs <= 0 or n_residues <= 1:
        raise ValueError("endpoint_pairs must be positive and n_residues must exceed one")
    generator = torch.Generator().manual_seed(int(seed))
    shape = (int(endpoint_pairs), int(n_residues))
    translation_apo = 0.25 * torch.randn(*shape, 3, generator=generator)
    translation_delta = torch.randn(*shape, 3, generator=generator)
    translation_delta[..., 0] = translation_delta[..., 0].abs() + 2.0
    translation_delta[..., 1:] *= 0.35
    rotation_delta = 0.65 * torch.randn(*shape, 3, generator=generator)
    rotation_delta = rotation_delta / torch.linalg.norm(
        rotation_delta, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    rotation_delta = rotation_delta * (
        0.35 + 0.55 * torch.rand(*shape, 1, generator=generator)
    )
    chi_apo = (2.0 * torch.rand(*shape, 2, generator=generator) - 1.0) * math.pi
    chi_delta = 1.2 * (2.0 * torch.rand(*shape, 2, generator=generator) - 1.0)
    weak_chi = torch.linalg.norm(chi_delta, dim=-1, keepdim=True) < 0.25
    chi_delta = torch.where(
        weak_chi,
        chi_delta + torch.tensor([0.45, -0.30]),
        chi_delta,
    )

    translation_normal = _orthogonal_unit(translation_delta, fallback_axis=2)
    rotation_normal = _orthogonal_unit(rotation_delta, fallback_axis=1)
    chi_normal = F.normalize(
        torch.stack([-chi_delta[..., 1], chi_delta[..., 0]], dim=-1), dim=-1
    )

    residue_position = torch.linspace(-1.0, 1.0, n_residues).view(1, n_residues)
    translation_norm = torch.linalg.norm(translation_delta, dim=-1)
    phase_coefficient = 0.65 * torch.tanh(
        0.55 * translation_delta[..., 1]
        - 0.45 * rotation_delta[..., 2]
        + 0.65 * residue_position
    )
    unsigned_residual = torch.stack(
        [
            0.72 + 0.28 * torch.sigmoid(translation_norm - 2.2),
            0.24 + 0.18 * torch.sigmoid(rotation_delta[..., 0]),
            0.38 + 0.20 * torch.sigmoid(chi_delta[..., 0]),
        ],
        dim=-1,
    )

    base_features = torch.cat(
        [
            translation_delta,
            rotation_delta,
            chi_delta,
            residue_position.expand(endpoint_pairs, -1).unsqueeze(-1),
            translation_norm.unsqueeze(-1),
        ],
        dim=-1,
    )
    route_sign = torch.tensor([-1.0, 1.0]).view(1, 2, 1, 1).expand(
        endpoint_pairs, 2, n_residues, 1
    )

    def duplicate(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(1).expand(-1, 2, *value.shape[1:]).reshape(
            endpoint_pairs * 2, *value.shape[1:]
        )

    route_sign = route_sign.reshape(endpoint_pairs * 2, n_residues, 1)
    duplicated_features = duplicate(base_features)
    features_observed = torch.cat([duplicated_features, route_sign], dim=-1)
    features_hidden = torch.cat(
        [duplicated_features, torch.zeros_like(route_sign)], dim=-1
    )
    residual_coefficient = duplicate(unsigned_residual) * route_sign
    return ControlledDataset(
        features_observed=features_observed,
        features_hidden=features_hidden,
        translation_apo=duplicate(translation_apo),
        translation_delta=duplicate(translation_delta),
        rotation_delta=duplicate(rotation_delta),
        chi_apo=duplicate(chi_apo),
        chi_delta=duplicate(chi_delta),
        translation_normal=duplicate(translation_normal),
        rotation_normal=duplicate(rotation_normal),
        chi_normal=duplicate(chi_normal),
        phase_coefficient=duplicate(phase_coefficient),
        residual_coefficient=residual_coefficient,
        route_sign=route_sign.squeeze(-1),
    )


def move_dataset(dataset: ControlledDataset, device: torch.device) -> ControlledDataset:
    return ControlledDataset(
        **{
            name: value.to(device)
            for name, value in dataset.__dict__.items()
        }
    )


def coefficient_loss(
    output: Mapping[str, torch.Tensor],
    dataset: ControlledDataset,
    variant: str,
) -> torch.Tensor:
    loss = output["phase"].new_tensor(0.0)
    if variant in {"warp_only", "full"}:
        loss = loss + F.mse_loss(output["phase"], dataset.phase_coefficient)
    if variant in {"residual_only", "full"}:
        scale = output["residual"].new_tensor([1.0, 0.5, 0.6])
        loss = loss + F.mse_loss(
            output["residual"] / scale,
            dataset.residual_coefficient / scale,
        )
    return loss


def train_variant(
    train: ControlledDataset,
    val: ControlledDataset,
    *,
    regime: str,
    variant: str,
    seed: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> CoefficientPredictor:
    if variant == "synchronous":
        raise ValueError("synchronous has no trainable predictor")
    torch.manual_seed(int(seed))
    model = CoefficientPredictor(train.features(regime).shape[-1], hidden_dim).to(
        train.phase_coefficient.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    for _ in range(int(epochs)):
        model.train()
        output = model(train.features(regime))
        loss = coefficient_loss(output, train, variant)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(
                coefficient_loss(model(val.features(regime)), val, variant).item()
            )
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model


def variant_coefficients(
    dataset: ControlledDataset,
    variant: str,
    model: CoefficientPredictor | None,
    regime: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = torch.zeros_like(dataset.phase_coefficient)
    residual = torch.zeros_like(dataset.residual_coefficient)
    if variant == "oracle_mode":
        return dataset.phase_coefficient, dataset.residual_coefficient
    if variant == "synchronous":
        return phase, residual
    if model is None:
        raise ValueError(f"{variant} requires a trained model")
    with torch.no_grad():
        output = model(dataset.features(regime))
    if variant in {"warp_only", "full"}:
        phase = output["phase"]
    if variant in {"residual_only", "full"}:
        residual = output["residual"]
    return phase, residual


def build_path(
    dataset: ControlledDataset,
    phase: torch.Tensor,
    residual: torch.Tensor,
    n_steps: int,
) -> Dict[str, torch.Tensor]:
    t = torch.linspace(
        0.0,
        1.0,
        int(n_steps) + 1,
        device=phase.device,
        dtype=phase.dtype,
    ).view(1, -1, 1)
    # |phase| < 1 keeps d tau / dt = 1 + phase * (1 - 2t) positive,
    # while the endpoint-zero quadratic term shifts midpoint event times.
    tau = t + phase.unsqueeze(1) * t * (1.0 - t)
    envelope = torch.sin(math.pi * t).square()
    translation = (
        dataset.translation_apo.unsqueeze(1)
        + tau.unsqueeze(-1) * dataset.translation_delta.unsqueeze(1)
        + envelope.unsqueeze(-1)
        * residual[..., 0].unsqueeze(1).unsqueeze(-1)
        * dataset.translation_normal.unsqueeze(1)
    )
    base_rotvec = tau.unsqueeze(-1) * dataset.rotation_delta.unsqueeze(1)
    residual_rotvec = (
        envelope.unsqueeze(-1)
        * residual[..., 1].unsqueeze(1).unsqueeze(-1)
        * dataset.rotation_normal.unsqueeze(1)
    )
    rotation = so3_exp(base_rotvec) @ so3_exp(residual_rotvec)
    chi = wrap_to_pi(
        dataset.chi_apo.unsqueeze(1)
        + tau.unsqueeze(-1) * dataset.chi_delta.unsqueeze(1)
        + envelope.unsqueeze(-1)
        * residual[..., 2].unsqueeze(1).unsqueeze(-1)
        * dataset.chi_normal.unsqueeze(1)
    )
    return {"tau": tau, "translation": translation, "rotation": rotation, "chi": chi}


def pair_order_accuracy(pred_tau: torch.Tensor, target_tau: torch.Tensor) -> float:
    midpoint = 0.5
    pred_events = (pred_tau >= midpoint).float().argmax(dim=1).float()
    target_events = (target_tau >= midpoint).float().argmax(dim=1).float()
    n_residues = int(pred_events.shape[-1])
    scores = []
    for left in range(n_residues):
        for right in range(left + 1, n_residues):
            target_delta = target_events[:, left] - target_events[:, right]
            valid = target_delta.abs() >= 1.0
            if not bool(valid.any()):
                continue
            pred_delta = pred_events[:, left] - pred_events[:, right]
            concordant = (pred_delta * target_delta > 0.0).float()
            tied = (pred_delta == 0.0).float()
            scores.append((concordant + 0.5 * tied)[valid])
    if not scores:
        return math.nan
    return float(torch.cat(scores).mean().item())


def path_metrics(
    dataset: ControlledDataset,
    predicted_phase: torch.Tensor,
    predicted_residual: torch.Tensor,
    n_steps: int,
) -> Dict[str, float]:
    target = build_path(
        dataset, dataset.phase_coefficient, dataset.residual_coefficient, n_steps
    )
    predicted = build_path(dataset, predicted_phase, predicted_residual, n_steps)
    translation_sq = (
        predicted["translation"] - target["translation"]
    ).square().sum(dim=-1)
    rotation_delta = predicted["rotation"].transpose(-2, -1) @ target["rotation"]
    rotation_sq = so3_log(rotation_delta).square().sum(dim=-1)
    chi_sq = wrap_to_pi(predicted["chi"] - target["chi"]).square().mean(dim=-1)
    product_rmse = torch.sqrt((translation_sq + rotation_sq + chi_sq).mean())
    center = dataset.translation_apo + 0.5 * dataset.translation_delta
    obstacle_distance = torch.linalg.norm(
        predicted["translation"] - center.unsqueeze(1), dim=-1
    )
    collision = obstacle_distance.min(dim=1).values < 0.35
    endpoint_translation = torch.maximum(
        torch.linalg.norm(
            predicted["translation"][:, 0] - dataset.translation_apo, dim=-1
        ).max(),
        torch.linalg.norm(
            predicted["translation"][:, -1]
            - (dataset.translation_apo + dataset.translation_delta),
            dim=-1,
        ).max(),
    )
    endpoint_rotation = torch.maximum(
        torch.linalg.norm(so3_log(predicted["rotation"][:, 0]), dim=-1).max(),
        torch.linalg.norm(
            so3_log(
                predicted["rotation"][:, -1].transpose(-2, -1)
                @ so3_exp(dataset.rotation_delta)
            ),
            dim=-1,
        ).max(),
    )
    endpoint_chi = torch.maximum(
        wrap_to_pi(predicted["chi"][:, 0] - dataset.chi_apo).abs().max(),
        wrap_to_pi(
            predicted["chi"][:, -1]
            - wrap_to_pi(dataset.chi_apo + dataset.chi_delta)
        ).abs().max(),
    )
    endpoint_max = torch.stack(
        [endpoint_translation, endpoint_rotation, endpoint_chi]
    ).max()
    phase_tangent_dot = (
        dataset.translation_delta * dataset.translation_normal
    ).sum(dim=-1).abs().max()
    rotation_tangent_dot = (
        dataset.rotation_delta * dataset.rotation_normal
    ).sum(dim=-1).abs().max()
    chi_tangent_dot = (dataset.chi_delta * dataset.chi_normal).sum(dim=-1).abs().max()
    return {
        "product_path_rmse": float(product_rmse.item()),
        "translation_path_rmse_a": float(torch.sqrt(translation_sq.mean()).item()),
        "rotation_path_rmse_rad": float(torch.sqrt(rotation_sq.mean()).item()),
        "chi_path_rmse_rad": float(torch.sqrt(chi_sq.mean()).item()),
        "phase_tau_mae": float(
            (predicted["tau"] - target["tau"]).abs().mean().item()
        ),
        "phase_coefficient_mae": float(
            (predicted_phase - dataset.phase_coefficient).abs().mean().item()
        ),
        "residual_coefficient_rmse": float(
            torch.sqrt(
                (predicted_residual - dataset.residual_coefficient).square().mean()
            ).item()
        ),
        "event_order_pair_accuracy": pair_order_accuracy(
            predicted["tau"], target["tau"]
        ),
        "obstacle_collision_fraction": float(collision.float().mean().item()),
        "endpoint_max_error": float(endpoint_max.item()),
        "predicted_residual_rms": float(
            torch.sqrt(predicted_residual.square().mean()).item()
        ),
        "normal_parallel_dot_max": float(
            torch.stack(
                [phase_tangent_dot, rotation_tangent_dot, chi_tangent_dot]
            ).max().item()
        ),
    }


def aggregate_seed_metrics(
    members: Sequence[Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    keys = sorted(set.intersection(*(set(member) for member in members)))
    return {
        key: {
            "mean": float(np.mean([member[key] for member in members])),
            "std": float(np.std([member[key] for member in members])),
        }
        for key in keys
        if all(math.isfinite(float(member[key])) for member in members)
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    if args.n_steps < 4:
        raise ValueError("n_steps must be at least four")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model_seeds = [int(item) for item in args.model_seeds.split(",") if item]
    train = move_dataset(
        make_dataset(
            args.train_endpoint_pairs,
            args.n_residues,
            seed=args.seed + 1,
        ),
        device,
    )
    val = move_dataset(
        make_dataset(
            args.val_endpoint_pairs,
            args.n_residues,
            seed=args.seed + 2,
        ),
        device,
    )
    test = move_dataset(
        make_dataset(
            args.test_endpoint_pairs,
            args.n_residues,
            seed=args.seed + 3,
        ),
        device,
    )

    result: Dict[str, object] = {
        "schema_version": "controlled_product_manifold_v1",
        "seed": args.seed,
        "model_seeds": model_seeds,
        "device": str(device),
        "train_systems": train.systems,
        "val_systems": val.systems,
        "test_systems": test.systems,
        "n_residues": args.n_residues,
        "n_steps": args.n_steps,
        "regimes": {},
    }
    for regime in REGIMES:
        regime_members: Dict[str, List[Dict[str, float]]] = {
            variant: [] for variant in (*VARIANTS, "oracle_mode")
        }
        synchronous_phase, synchronous_residual = variant_coefficients(
            test, "synchronous", None, regime
        )
        synchronous_metrics = path_metrics(
            test, synchronous_phase, synchronous_residual, args.n_steps
        )
        oracle_phase, oracle_residual = variant_coefficients(
            test, "oracle_mode", None, regime
        )
        oracle_metrics = path_metrics(test, oracle_phase, oracle_residual, args.n_steps)
        for seed in model_seeds:
            regime_members["synchronous"].append(synchronous_metrics)
            regime_members["oracle_mode"].append(oracle_metrics)
            for variant in VARIANTS[1:]:
                model = train_variant(
                    train,
                    val,
                    regime=regime,
                    variant=variant,
                    seed=seed,
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                )
                phase, residual = variant_coefficients(test, variant, model, regime)
                regime_members[variant].append(
                    path_metrics(test, phase, residual, args.n_steps)
                )
        result["regimes"][regime] = {
            "per_seed": regime_members,
            "aggregate": {
                variant: aggregate_seed_metrics(metrics)
                for variant, metrics in regime_members.items()
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
