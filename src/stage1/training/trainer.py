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
import math
from tqdm import tqdm

from .config import TrainingConfig
from ..models import Stage1Model, Stage1ModelConfig
from ..datasets import create_ipa_dataloader
from ..modules.losses import fape_loss, torsion_loss, distance_loss, clash_penalty, chi1_rotamer_loss
from utils.metrics import compute_pocket_irmsd, compute_chi1_accuracy, compute_fape, compute_clash_percentage


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

        # 1b. χ1 rotamer 辅助损失
        chi1_logits = outputs.get('chi1_logits', None)
        if chi1_logits is not None and self.config.w_rotamer > 0:
            loss_rot = chi1_rotamer_loss(
                chi1_logits,                      # [B, N, 3]
                batch.torsion_angles[:, :, 3],    # [B, N]
                batch.torsion_mask[:, :, 3],      # [B, N]
                w_res_warmed
            )
        else:
            loss_rot = torch.tensor(0.0, device=self.device)
        
        # 2. 提取FK重建的坐标
        atom14_pos = outputs['atom14_pos']  # [B, N, 14, 3]
        atom14_mask = outputs['atom14_mask']  # [B, N, 14]
        
        # 提取主链4原子（N, CA, C, O）
        pred_N = atom14_pos[:, :, 0]   # [B, N, 3]
        pred_CA = atom14_pos[:, :, 1]
        pred_C = atom14_pos[:, :, 2]
        pred_O = atom14_pos[:, :, 3]
        
        # 真实坐标
        true_N = batch.N
        true_CA = batch.Ca
        true_C = batch.C
        
        # 3. Cα距离损失（使用FK重建的CA）
        loss_dist = distance_loss(pred_CA, true_CA, w_res_warmed)
        
        # 4. FAPE损失（使用主链3原子：N, CA, C）
        # 预测和真实都用3个原子，维度匹配
        pred_backbone = torch.stack([pred_N, pred_CA, pred_C], dim=2)  # [B, N, 3, 3]
        true_backbone = torch.stack([true_N, true_CA, true_C], dim=2)  # [B, N, 3, 3]
        
        # 提取帧
        rigids_final = outputs['rigids_final']
        pred_R = rigids_final.get_rots().get_rot_mats()  # [B, N, 3, 3]
        pred_t = pred_CA
        
        # 构建真实帧（从真实N, CA, C）- 带强数值保护
        B, N = batch.Ca.shape[:2]
        eps = 1e-6  # 更大的epsilon保护
        
        # 向量化构建所有帧（避免循环）
        # x轴: CA → C
        x_axis = true_C - true_CA  # [B, N, 3]
        x_norm = torch.norm(x_axis, dim=-1, keepdim=True).clamp(min=eps)
        x_axis = x_axis / x_norm
        
        # y轴: 垂直于(CA-N)和x_axis
        v = true_N - true_CA
        v_proj = v - torch.sum(v * x_axis, dim=-1, keepdim=True) * x_axis
        y_norm = torch.norm(v_proj, dim=-1, keepdim=True).clamp(min=eps)
        y_axis = v_proj / y_norm
        
        # z轴: x × y
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        z_norm = torch.norm(z_axis, dim=-1, keepdim=True).clamp(min=eps)
        z_axis = z_axis / z_norm  # 额外归一化保护
        
        # 旋转矩阵: [B, N, 3, 3]
        true_R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        true_t = true_CA
        
        # NaN检测（训练时检查）
        if torch.isnan(true_R).any():
            # 降级为单位矩阵（保护措施）
            true_R = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
            print("  ⚠️ FAPE帧出现NaN，使用单位矩阵")
        
        loss_fape = fape_loss(
            pred_backbone,  # [B, N, 3, 3] - fape_loss会自动处理4维输入
            true_backbone,  # [B, N, 3, 3]
            (pred_R, pred_t),
            (true_R, true_t),
            w_res_warmed
        )
        
        # 5. Clash惩罚（使用所有atom14原子，符合理论）
        # 理论：对非键合原子对施加最小距离惩罚（docs/理论 第116-120行）
        # 提取所有有效原子
        atom14_pos = outputs['atom14_pos']  # [B, N, 14, 3]
        atom14_mask = outputs['atom14_mask']  # [B, N, 14]
        
        # 展平为[B, N*14, 3]，但只计算有效原子
        # 简化：先展平，clash_penalty内部会处理所有原子对
        pred_all_atoms = atom14_pos.reshape(B, -1, 3)  # [B, N*14, 3]
        
        # Clash阈值：主链~2.0Å，侧链稍大~2.5Å，折中用2.2Å
        loss_clash = clash_penalty(pred_all_atoms, clash_threshold=2.2)
        
        # 组合损失
        total_loss = (
            self.config.w_torsion * loss_tor +
            self.config.w_rotamer * loss_rot +
            self.config.w_dist * loss_dist +
            self.config.w_clash * loss_clash +
            self.config.w_fape * loss_fape
        )
        
        return {
            'total': total_loss,
            'torsion': loss_tor,
            'rotamer': loss_rot,
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
            'rotamer': 0.0,
            'distance': 0.0,
            'clash': 0.0,
            'fape': 0.0,
        }
        
        pbar = tqdm(self.train_loader, 
                   desc=f'Epoch {self.current_epoch:3d}',
                   ncols=120,
                   leave=True)
        
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
                'fape': f"{step_losses['fape']:.3f}",
                'dist': f"{step_losses['distance']:.3f}",
                'rot': f"{step_losses['rotamer']:.3f}",
                'clash': f"{step_losses['clash']:.3f}",
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
            'rotamer': 0.0,
            'distance': 0.0,
            'clash': 0.0,
            'fape': 0.0,
        }
        
        val_metrics = {
            'chi1_acc': 0.0,
            'chi1_rot_acc': 0.0,
        }
        pocket_irmsd_sum = 0.0
        clash_pct_sum = 0.0
        n_structures = 0
        
        pbar = tqdm(self.val_loader, 
                   desc='  Validating',
                   ncols=120,
                   leave=False)
        
        for batch in pbar:
            batch = self._batch_to_device(batch)
            
            outputs = self.model(batch, self.global_step)
            losses = self.compute_loss(outputs, batch, self.global_step)
            
            # 累积损失
            val_losses['total'] += losses['total'].item()
            val_losses['torsion'] += losses['torsion'].item()
            val_losses['rotamer'] += losses['rotamer'].item()
            val_losses['distance'] += losses['distance'].item()
            val_losses['clash'] += losses['clash'].item()
            val_losses['fape'] += losses['fape'].item()
            
            # 计算指标（chi1准确率）
            pred_torsions = torch.atan2(
                outputs['pred_torsions'][..., 0],
                outputs['pred_torsions'][..., 1]
            )
            # 只取chi1 (索引3)
            chi1_acc = compute_chi1_accuracy(
                pred_torsions[:, :, 3].flatten(),
                batch.torsion_angles[:, :, 3].flatten(),
                batch.torsion_mask[:, :, 3].flatten()
            )
            val_metrics['chi1_acc'] += chi1_acc

            chi1_logits = outputs.get('chi1_logits', None)
            if chi1_logits is not None:
                true_chi1 = batch.torsion_angles[:, :, 3]
                chi1_mask = batch.torsion_mask[:, :, 3].bool()
                if chi1_mask.any():
                    angle = ((true_chi1 + math.pi) % (2 * math.pi)) - math.pi
                    labels = torch.zeros_like(true_chi1, dtype=torch.long)
                    labels[chi1_mask & (angle < -math.pi / 3)] = 0
                    t_region = chi1_mask & (angle >= -math.pi / 3) & (angle < math.pi / 3)
                    labels[t_region] = 1
                    labels[chi1_mask & (angle >= math.pi / 3)] = 2
                    preds = chi1_logits.argmax(dim=-1)
                    correct = (preds == labels) & chi1_mask
                    acc_rot = correct.sum().float() / chi1_mask.sum().float()
                    val_metrics['chi1_rot_acc'] += acc_rot.item()

            B = batch.Ca.shape[0]
            for i in range(B):
                pocket_mask = (batch.w_res[i] > 0.5)
                if pocket_mask.any():
                    irmsd = compute_pocket_irmsd(
                        outputs['atom14_pos'][i, :, 1],
                        batch.Ca[i],
                        pocket_mask
                    )
                    if not math.isnan(irmsd):
                        pocket_irmsd_sum += irmsd
                valid_atom_mask = outputs['atom14_mask'][i].bool().view(-1)
                coords_i = outputs['atom14_pos'][i].view(-1, 3)[valid_atom_mask]
                if coords_i.shape[0] > 1:
                    clash_pct = compute_clash_percentage(coords_i)
                    clash_pct_sum += clash_pct
                n_structures += 1

            pbar.set_postfix({
                'v_loss': f"{losses['total'].item():.3f}",
                'chi1': f"{chi1_acc:5.1%}",
            }, refresh=False)
        
        # 平均
        n_batches = len(self.val_loader)
        val_losses = {k: v / n_batches for k, v in val_losses.items()}
        val_metrics['chi1_acc'] = val_metrics['chi1_acc'] / n_batches
        val_metrics['chi1_rot_acc'] = val_metrics['chi1_rot_acc'] / max(n_batches, 1)
        if n_structures > 0:
            val_metrics['pocket_irmsd'] = pocket_irmsd_sum / n_structures
            val_metrics['clash_pct'] = clash_pct_sum / n_structures
        else:
            val_metrics['pocket_irmsd'] = float('nan')
            val_metrics['clash_pct'] = float('nan')
        
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
            
            # 单行输出训练结果（显示所有分项）
            train_info = (f"Epoch {epoch:3d} | "
                         f"Loss: {train_losses['total']:.4f} "
                         f"(T:{train_losses['torsion']:.2f} "
                         f"F:{train_losses['fape']:.2f} "
                         f"D:{train_losses['distance']:.2f} "
                         f"C:{train_losses['clash']:.2f} "
                         f"R:{train_losses['rotamer']:.2f})")
            
            # 验证
            if epoch % self.config.val_interval == 0:
                val_results = self.validate()
                
                val_info = (f" | Val Loss: {val_results['total']:.4f} "
                           f"χ1:{val_results['chi1_acc']:5.1%} "
                           f"Rot:{val_results['chi1_rot_acc']:5.1%} "
                           f"FAPE:{val_results['fape']:.3f} "
                           f"iRMSD:{val_results['pocket_irmsd']:.2f} "
                           f"Clash:{val_results['clash_pct']*100:.1f}%")
                
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


