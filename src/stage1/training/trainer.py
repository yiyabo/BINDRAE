"""
训练器

Author: BINDRAE Team
Date: 2025-10-28
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional
import time
from tqdm import tqdm

from .config import TrainingConfig
from ..models import Stage1Model, Stage1ModelConfig
from ..datasets import create_ipa_dataloader
from ..modules.losses import fape_loss, torsion_loss, distance_loss, clash_penalty
from utils.metrics import compute_pocket_irmsd, compute_chi1_accuracy, compute_fape


class Stage1Trainer:
    """
    Stage-1 训练器
    
    功能：
    - 训练循环
    - 验证循环  
    - 早停
    - Checkpoint管理
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # 创建模型
        print("创建模型...")
        model_config = Stage1ModelConfig()
        self.model = Stage1Model(model_config).to(self.device)
        
        # 创建数据加载器
        print("创建数据加载器...")
        self.train_loader = create_ipa_dataloader(
            config.data_dir,
            split='train',
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        
        self.val_loader = create_ipa_dataloader(
            config.data_dir,
            split='val',
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        # 创建优化器
        print("创建优化器...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # LR Scheduler (Cosine with warmup)
        total_steps = len(self.train_loader) * config.max_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - config.warmup_steps,
            eta_min=config.lr * 0.01
        )
        
        # 混合精度
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0
        
        # 创建保存目录
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ 训练器初始化完成")
        print(f"  - 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - 训练样本: {len(self.train_loader.dataset)}")
        print(f"  - 验证样本: {len(self.val_loader.dataset)}")
        print(f"  - 总训练步数: {total_steps:,}")
    
    def compute_pocket_warmup(self, step: int) -> float:
        """计算口袋权重warmup系数"""
        if step >= self.config.pocket_warmup_steps:
            return 1.0
        return step / self.config.pocket_warmup_steps
    
    def compute_loss(self, outputs: Dict, batch, step: int) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        
        Args:
            outputs: 模型输出
            batch: IPABatch数据
            step: 当前步数
            
        Returns:
            损失字典
        """
        pred_torsions_sincos = outputs['pred_torsions']  # [B, N, 7, 2]
        
        # 转换为角度
        pred_torsions = torch.atan2(
            pred_torsions_sincos[..., 0],
            pred_torsions_sincos[..., 1]
        )  # [B, N, 7]
        
        # 口袋权重warmup
        kappa = self.compute_pocket_warmup(step)
        w_res_warmed = batch.w_res * kappa + (1 - kappa) * 0.1  # 从0.1→1
        
        # 1. 扭转角损失
        loss_tor = torsion_loss(
            pred_torsions,
            batch.torsion_angles,
            batch.torsion_mask,
            w_res_warmed
        )
        
        # 2. Cα距离损失（用真实Ca坐标）
        # 简化版：暂时用真实Ca计算距离图，验证损失函数工作
        loss_dist = distance_loss(batch.Ca, batch.Ca, w_res_warmed)
        
        # 3. FAPE损失（用rigids_final）
        rigids_final = outputs['rigids_final']
        
        # 提取预测的帧（R, t）
        pred_R = rigids_final.get_rots().get_rot_mats()  # [B, N, 3, 3]
        pred_t = rigids_final.get_trans()  # [B, N, 3]
        
        # 真实帧（简化：用单位旋转+Ca坐标）
        B, N = batch.Ca.shape[:2]
        true_R = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        true_t = batch.Ca
        
        loss_fape = fape_loss(
            batch.Ca,  # 预测坐标（暂时用真实Ca）
            batch.Ca,  # 真实坐标
            (pred_R, pred_t),  # 预测帧
            (true_R, true_t),  # 真实帧
            w_res_warmed
        )
        
        # 4. Clash惩罚（需要全原子坐标）
        # TODO: FK还原后计算，暂时禁用
        loss_clash = torch.tensor(0.0, device=self.device)
        
        # 组合损失
        total_loss = (
            self.config.w_torsion * loss_tor +
            self.config.w_dist * loss_dist +
            self.config.w_clash * loss_clash +
            self.config.w_fape * loss_fape
        )
        
        return {
            'total': total_loss,
            'torsion': loss_tor,
            'distance': loss_dist,
            'clash': loss_clash,
            'fape': loss_fape,
        }
    
    def train_step(self, batch) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 移动到GPU
        batch = self._batch_to_device(batch)
        
        # 前向传播
        if self.scaler is not None:
            with autocast():
                outputs = self.model(batch, self.global_step)
                losses = self.compute_loss(outputs, batch, self.global_step)
                loss = losses['total']
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(batch, self.global_step)
            losses = self.compute_loss(outputs, batch, self.global_step)
            loss = losses['total']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        
        # Warmup后更新LR
        if self.global_step >= self.config.warmup_steps:
            self.scheduler.step()
        
        self.global_step += 1
        
        # 返回标量损失
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()}
    
    def _batch_to_device(self, batch):
        """将batch移动到GPU"""
        batch.esm = batch.esm.to(self.device)
        batch.N = batch.N.to(self.device)
        batch.Ca = batch.Ca.to(self.device)
        batch.C = batch.C.to(self.device)
        batch.node_mask = batch.node_mask.to(self.device)
        batch.lig_points = batch.lig_points.to(self.device)
        batch.lig_types = batch.lig_types.to(self.device)
        batch.lig_mask = batch.lig_mask.to(self.device)
        batch.torsion_angles = batch.torsion_angles.to(self.device)
        batch.torsion_mask = batch.torsion_mask.to(self.device)
        batch.w_res = batch.w_res.to(self.device)
        return batch
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'torsion': 0.0,
            'distance': 0.0,
            'clash': 0.0,
            'fape': 0.0,
        }
        
        pbar = tqdm(self.train_loader, 
                   desc=f'Epoch {self.current_epoch:3d}',
                   ncols=100,  # 固定宽度，避免换行
                   leave=True)  # 保留进度条
        
        for batch_idx, batch in enumerate(pbar):
            # 训练步
            step_losses = self.train_step(batch)
            
            # 累积损失
            for k in epoch_losses:
                epoch_losses[k] += step_losses[k]
            
            # 实时更新进度条（每步都更新）
            pbar.set_postfix({
                'loss': f"{step_losses['total']:.3f}",
                'tor': f"{step_losses['torsion']:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}",
            }, refresh=True)
        
        # 平均损失
        n_batches = len(self.train_loader)
        epoch_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'torsion': 0.0,
        }
        
        val_metrics = {
            'chi1_acc': 0.0,
        }
        
        pbar = tqdm(self.val_loader, 
                   desc='  Validating',
                   ncols=100,
                   leave=False)  # 验证完成后清除进度条
        
        for batch in pbar:
            batch = self._batch_to_device(batch)
            
            outputs = self.model(batch, self.global_step)
            losses = self.compute_loss(outputs, batch, self.global_step)
            
            # 累积损失
            val_losses['total'] += losses['total'].item()
            val_losses['torsion'] += losses['torsion'].item()
            
            # 计算指标（chi1准确率）
            pred_torsions = torch.atan2(
                outputs['pred_torsions'][..., 0],
                outputs['pred_torsions'][..., 1]
            )
            # 只取chi1 (索引3)
            from utils.metrics import compute_chi1_accuracy
            import numpy as np
            chi1_acc = compute_chi1_accuracy(
                pred_torsions[:, :, 3].cpu().numpy().flatten(),
                batch.torsion_angles[:, :, 3].cpu().numpy().flatten(),
                batch.torsion_mask[:, :, 3].cpu().numpy().flatten()
            )
            if not np.isnan(chi1_acc):
                val_metrics['chi1_acc'] += chi1_acc
        
        # 平均
        n_batches = len(self.val_loader)
        val_losses = {k: v / n_batches for k, v in val_losses.items()}
        val_metrics = {k: v / n_batches for k, v in val_metrics.items()}
        
        # 合并返回
        results = {**val_losses, **val_metrics}
        
        return results
    
    def save_checkpoint(self, filepath: str, verbose: bool = True):
        """保存checkpoint"""
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config,
        }, filepath)
        if verbose:
            print(f"  ✓ Saved: {Path(filepath).name}")
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*80}")
        print(f"开始训练 - BINDRAE Stage-1")
        print(f"{'='*80}\n")
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch()
            
            # 单行输出训练结果
            train_info = (f"Epoch {epoch:3d} | "
                         f"Train Loss: {train_losses['total']:.4f} "
                         f"(Tor:{train_losses['torsion']:.3f})")
            
            # 验证
            if epoch % self.config.val_interval == 0:
                val_results = self.validate()
                
                val_info = (f" | Val Loss: {val_results['total']:.4f} "
                           f"χ1:{val_results['chi1_acc']:5.1%}")
                
                # 早停检查
                current_metric = val_results['total']
                if current_metric < self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.patience_counter = 0
                    save_path = Path(self.config.save_dir) / 'best_model.pt'
                    self.save_checkpoint(str(save_path), verbose=False)
                    val_info += " | ⭐ Best"
                else:
                    self.patience_counter += 1
                    val_info += f" | ↑{self.patience_counter}"
                    
                    if self.patience_counter >= self.config.early_stop_patience:
                        print(f"\n{'='*80}")
                        print(f"早停！验证损失{self.config.early_stop_patience}个epoch未改善")
                        print(f"{'='*80}\n")
                        break
                
                print(train_info + val_info)
            else:
                print(train_info)
            
            # 定期保存（静默）
            if epoch % 10 == 0 and epoch > 0:
                save_path = Path(self.config.save_dir) / f'epoch_{epoch}.pt'
                self.save_checkpoint(str(save_path), verbose=False)
        
        print(f"\n{'='*80}")
        print(f"训练完成！")
        print(f"  最佳验证损失: {self.best_val_metric:.4f}")
        print(f"  Checkpoint: {self.config.save_dir}/best_model.pt")
        print(f"{'='*80}\n")


