"""Trainer for the compact Stage-1-v2 posterior student."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .dataset import TeacherPosteriorDataset, collate_teacher_posterior_batch
from .losses import compute_posterior_losses, compute_posterior_metric_sums, finalize_posterior_metrics
from .model import Stage1PosteriorV2, Stage1PosteriorV2Config


@dataclass
class PosteriorV2TrainingConfig:
    data_dir: str
    train_label_dir: str
    val_label_dir: str
    train_split: str = "train"
    val_split: str = "val"
    train_valid_samples_file: Optional[str] = None
    val_valid_samples_file: Optional[str] = None
    batch_size: int = 8
    num_workers: int = 4
    train_max_samples: int = 0
    val_max_samples: int = 0
    subset_seed: int = 20260622
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 20
    grad_clip: float = 1.0
    patience: int = 5
    save_dir: str = "checkpoints/stage1v2/posterior"
    log_dir: str = "logs/stage1v2/posterior"
    device: str = "cuda"
    distributed: bool = False
    c_s: int = 256
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    use_latent_head: bool = False
    amp: bool = True
    contact_bce_weight: float = 1.0
    approach_bce_weight: float = 0.5
    release_bce_weight: float = 0.5
    switch_bce_weight: float = 1.0
    confidence_bce_weight: float = 0.3
    dist_mae_weight: float = 0.1
    delta_mae_weight: float = 0.2
    eval_counterfactuals: str = "nolig,shuffled,translated"
    class_balanced_heads: str = "switch,approach,release"
    selection_metric: str = "posterior_score"


def _setup_distributed(enabled: bool):
    if not enabled:
        return False, 0, 1
    if "RANK" not in os.environ:
        return False, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return True, local_rank, world


def _subset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(len(dataset), generator=g)[: int(max_samples)].tolist()
    return Subset(dataset, idx)


def _to_device(batch, device: torch.device):
    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            setattr(batch, name, value.to(device))
    batch.teacher_float = {k: v.to(device) for k, v in batch.teacher_float.items()}
    batch.teacher_bool = {k: v.to(device) for k, v in batch.teacher_bool.items()}
    return batch


class PosteriorV2Trainer:
    def __init__(self, config: PosteriorV2TrainingConfig):
        self.config = config
        self.distributed, self.local_rank, self.world_size = _setup_distributed(config.distributed)
        self.is_main = (not self.distributed) or dist.get_rank() == 0
        self.device = torch.device(f"cuda:{self.local_rank}" if self.distributed else config.device)
        torch.manual_seed(config.subset_seed + self.local_rank)

        model_config = Stage1PosteriorV2Config(
            c_s=config.c_s,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_latent_head=config.use_latent_head,
        )
        self.model = Stage1PosteriorV2(model_config).to(self.device)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp and self.device.type == "cuda")
        self.loss_weights = {
            "contact_bce": config.contact_bce_weight,
            "approach_bce": config.approach_bce_weight,
            "release_bce": config.release_bce_weight,
            "switch_bce": config.switch_bce_weight,
            "confidence_bce": config.confidence_bce_weight,
            "dist_mae": config.dist_mae_weight,
            "delta_mae": config.delta_mae_weight,
        }
        self.counterfactual_modes = self._parse_counterfactual_modes(config.eval_counterfactuals)
        self.class_balanced_heads = self._parse_name_list(config.class_balanced_heads)

        train_ds = TeacherPosteriorDataset(
            config.data_dir,
            config.train_label_dir,
            split=config.train_split,
            valid_samples_file=config.train_valid_samples_file,
        )
        val_ds = TeacherPosteriorDataset(
            config.data_dir,
            config.val_label_dir,
            split=config.val_split,
            valid_samples_file=config.val_valid_samples_file,
        )
        train_ds = _subset(train_ds, config.train_max_samples, config.subset_seed)
        val_ds = _subset(val_ds, config.val_max_samples, config.subset_seed + 1)
        train_sampler = DistributedSampler(train_ds, shuffle=True) if self.distributed else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if self.distributed else None
        self.train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.num_workers,
            collate_fn=collate_teacher_posterior_batch,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.num_workers,
            collate_fn=collate_teacher_posterior_batch,
            pin_memory=True,
        )
        self.save_dir = Path(config.save_dir)
        self.log_dir = Path(config.log_dir)
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / "config.json").open("w") as f:
                json.dump(asdict(config), f, indent=2)
        self.best_metric = float("inf")
        self.bad_epochs = 0

    @staticmethod
    def _parse_name_list(raw: str):
        return tuple(item.strip().lower() for item in str(raw or "").split(",") if item.strip())

    @staticmethod
    def _parse_counterfactual_modes(raw: str):
        allowed = {"nolig", "shuffled", "translated"}
        modes = []
        for item in str(raw or "").split(","):
            mode = item.strip().lower()
            if not mode:
                continue
            if mode not in allowed:
                raise ValueError(f"Unknown eval counterfactual mode {mode!r}; expected one of {sorted(allowed)}")
            if mode not in modes:
                modes.append(mode)
        return modes

    def _ligand_inputs(self, batch, mode: str = "real"):
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
            perm = torch.roll(torch.arange(bsz, device=batch.lig_points.device), shifts=1) if bsz > 1 else torch.arange(bsz, device=batch.lig_points.device)
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
        raise ValueError(f"Unknown ligand control mode: {mode}")

    def _predict(self, batch, mode: str = "real"):
        lig_points, lig_types, lig_mask, w_res = self._ligand_inputs(batch, mode)
        return self.model(
            esm=batch.esm,
            Ca_apo=batch.Ca_apo,
            lig_points=lig_points,
            lig_types=lig_types,
            lig_mask=lig_mask,
            w_res=w_res,
            node_mask=batch.node_mask,
            torsion_apo=batch.torsion_apo,
        )

    def _forward_losses(self, batch):
        pred = self._predict(batch, mode="real")
        losses = compute_posterior_losses(
            pred,
            batch.teacher_float,
            batch.teacher_bool,
            batch.node_mask,
            self.loss_weights,
            class_balanced_heads=self.class_balanced_heads,
        )
        metric_sums = compute_posterior_metric_sums(pred, batch.teacher_float, batch.teacher_bool, batch.node_mask)
        return losses, metric_sums

    def _reduce_sum_dict(self, sums: Dict[str, float]) -> Dict[str, float]:
        if not self.distributed or not sums:
            return sums
        keys = sorted(sums)
        packed = torch.tensor([float(sums[k]) for k in keys], device=self.device, dtype=torch.float64)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        return {k: float(v.item()) for k, v in zip(keys, packed)}

    @staticmethod
    def _add_metric_sums(dst: Dict[str, float], src: Dict[str, torch.Tensor]):
        for key, value in src.items():
            if torch.is_tensor(value):
                value = float(value.detach().float().cpu().item())
            dst[key] = dst.get(key, 0.0) + float(value)

    def _add_counterfactual_lifts(self, result: Dict[str, float], metric_sums: Dict[str, float], cf_metric_sums: Dict[str, Dict[str, float]]):
        real = finalize_posterior_metrics(metric_sums)
        for mode, sums in cf_metric_sums.items():
            cf = finalize_posterior_metrics(sums, prefix=f"cf_{mode}_")
            result.update(cf)
            for metric in ("pocket_delta_mae", "pocket_dist_mae"):
                result[f"cf_{mode}_minus_real_{metric}"] = cf[f"cf_{mode}_{metric}"] - real[metric]
            for name in ("contact", "active", "approach", "release"):
                result[f"real_minus_cf_{mode}_{name}_balanced_acc"] = (
                    real[f"{name}_balanced_acc"] - cf[f"cf_{mode}_{name}_balanced_acc"]
                )

    def _run_epoch(self, loader, train: bool, epoch: int) -> Dict[str, float]:
        if train:
            self.model.train()
            if self.distributed and isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)
        else:
            self.model.eval()
        loss_sums: Dict[str, float] = {}
        metric_sums: Dict[str, float] = {}
        cf_metric_sums: Dict[str, Dict[str, float]] = {mode: {} for mode in self.counterfactual_modes} if not train else {}
        count = 0
        for batch in loader:
            batch = _to_device(batch, self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.config.amp and self.device.type == "cuda", dtype=torch.bfloat16):
                    losses, metrics = self._forward_losses(batch)
                    total = losses["total"]
                if train:
                    self.scaler.scale(total).backward()
                    if self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            bsz = len(batch.pdb_ids)
            for key, value in losses.items():
                if torch.is_tensor(value):
                    value = float(value.detach().float().cpu().item())
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value) * bsz
            self._add_metric_sums(metric_sums, metrics)
            if not train and self.counterfactual_modes:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.config.amp and self.device.type == "cuda", dtype=torch.bfloat16):
                        for mode in self.counterfactual_modes:
                            pred_cf = self._predict(batch, mode=mode)
                            cf_sums = compute_posterior_metric_sums(
                                pred_cf, batch.teacher_float, batch.teacher_bool, batch.node_mask
                            )
                            self._add_metric_sums(cf_metric_sums[mode], cf_sums)
            count += bsz
        if self.distributed:
            count_tensor = torch.tensor([count], device=self.device, dtype=torch.float64)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            count = int(count_tensor.item())
            loss_sums = self._reduce_sum_dict(loss_sums)
            metric_sums = self._reduce_sum_dict(metric_sums)
            cf_metric_sums = {mode: self._reduce_sum_dict(sums) for mode, sums in cf_metric_sums.items()}
        result = {key: value / max(count, 1) for key, value in loss_sums.items()}
        result.update(finalize_posterior_metrics(metric_sums))
        if cf_metric_sums:
            self._add_counterfactual_lifts(result, metric_sums, cf_metric_sums)
        return result

    def _save(self, name: str, epoch: int, metrics: Dict[str, float]):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "config": asdict(self.config),
                "metrics": metrics,
            },
            self.save_dir / name,
        )

    def _selection_value(self, val_metrics: Dict[str, float]) -> float:
        metric = str(self.config.selection_metric or "posterior_score")
        if metric == "posterior_score":
            score = (
                0.45 * float(val_metrics.get("active_balanced_acc", 0.0))
                + 0.25 * float(val_metrics.get("contact_balanced_acc", 0.0))
                + 0.15 * float(val_metrics.get("approach_balanced_acc", 0.0))
                + 0.15 * float(val_metrics.get("release_balanced_acc", 0.0))
                - 0.02 * min(float(val_metrics.get("pocket_delta_mae", 0.0)), 20.0)
            )
            val_metrics["posterior_selection_score"] = score
            return -score
        if metric.endswith("_acc") or metric.endswith("_f1") or metric.endswith("_score"):
            return -float(val_metrics.get(metric, 0.0))
        return float(val_metrics.get(metric, val_metrics.get("total", 1e9)))

    def train(self):
        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            with torch.no_grad():
                val_metrics = self._run_epoch(self.val_loader, train=False, epoch=epoch)
            select = self._selection_value(val_metrics)
            if self.is_main:
                record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                with (self.log_dir / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"epoch {epoch:03d} "
                    f"train_total={train_metrics.get('total', 0):.4f} "
                    f"val_total={val_metrics.get('total', 0):.4f} "
                    f"val_delta_mae={val_metrics.get('pocket_delta_mae', 0):.4f} "
                    f"val_active_bal={val_metrics.get('active_balanced_acc', 0):.4f} "
                    f"val_contact_bal={val_metrics.get('contact_balanced_acc', 0):.4f} "
                    f"val_select={val_metrics.get('posterior_selection_score', -select):.4f}"
                )
                self._save("latest_model.pt", epoch, val_metrics)
                if select < self.best_metric:
                    self.best_metric = select
                    self.bad_epochs = 0
                    self._save("best_model.pt", epoch, val_metrics)
                else:
                    self.bad_epochs += 1
            if self.distributed:
                state = torch.tensor([self.bad_epochs], device=self.device)
                dist.broadcast(state, src=0)
                self.bad_epochs = int(state.item())
            if self.bad_epochs >= self.config.patience:
                break
        if self.distributed:
            dist.destroy_process_group()
