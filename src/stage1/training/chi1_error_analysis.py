import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .config import TrainingConfig
from .trainer import Stage1Trainer


def wrap_angle_diff(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """wrap(pred-true) to [-pi, pi]."""
    diff = pred - true
    diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
    return diff


def analyze_chi1_errors(trainer: Stage1Trainer,
                        pocket_threshold: float = 0.5,
                        device: str = "cuda:0") -> None:
    model = trainer.model.to(device)
    model.eval()

    val_loader = trainer.val_loader

    # 累积结构化结果，方便保存 npz
    records: List[Dict] = []

    # 全局误差统计
    global_err_deg: List[float] = []

    # 按氨基酸类型
    aa_err: Dict[str, List[float]] = {}

    # pocket vs non-pocket
    pocket_err_deg: List[float] = []
    non_pocket_err_deg: List[float] = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Chi1 Error Analysis", ncols=120)
        for batch in pbar:
            # 与 trainer._batch_to_device 保持一致
            batch = trainer._batch_to_device(batch)

            outputs = model(batch, trainer.global_step)

            # 预测和真实 chi1（弧度）
            pred_chi = torch.atan2(
                outputs['pred_chi'][..., 0],
                outputs['pred_chi'][..., 1]
            )  # [B, N, 4]

            pred_chi1 = pred_chi[:, :, 0]
            true_chi1 = batch.chi_holo[:, :, 0]
            chi1_mask = batch.chi_mask[:, :, 0].bool()

            # 误差（度）
            diff = wrap_angle_diff(pred_chi1, true_chi1)
            err_deg = diff.abs() * 180.0 / torch.pi

            B, N = pred_chi1.shape

            for b in range(B):
                pdb_id = batch.pdb_ids[b]
                seq = batch.sequences[b]
                n_res = batch.n_residues[b]

                for i in range(n_res):
                    if not chi1_mask[b, i]:
                        continue

                    aa = seq[i] if i < len(seq) else 'X'
                    e = float(err_deg[b, i].item())
                    w = float(batch.w_res[b, i].item())
                    is_pocket = w > pocket_threshold

                    # 记录到全局
                    global_err_deg.append(e)

                    # 按 aa
                    aa_err.setdefault(aa, []).append(e)

                    # pocket vs non-pocket
                    if is_pocket:
                        pocket_err_deg.append(e)
                    else:
                        non_pocket_err_deg.append(e)

                    # 记录逐残基信息
                    records.append({
                        'pdb_id': pdb_id,
                        'res_idx': i,
                        'aa': aa,
                        'w_res': w,
                        'is_pocket': is_pocket,
                        'pred_chi1': float(pred_chi1[b, i].item()),
                        'true_chi1': float(true_chi1[b, i].item()),
                        'err_deg': e,
                    })

    def summarize_errors(name: str, errors: List[float]) -> None:
        if not errors:
            print(f"{name}: no data")
            return
        arr = np.array(errors, dtype=np.float32)
        mean = float(arr.mean())
        median = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))

        # 简单 hist 区间
        bins = [0, 20, 40, 60, 90, 180]
        hist, _ = np.histogram(arr, bins=bins)
        hist = hist.astype(np.float32) / max(len(arr), 1)

        print(f"\n[{name}] N={len(arr)}")
        print(f"  mean={mean:.2f}°, median={median:.2f}°, p90={p90:.2f}°, p95={p95:.2f}°")
        print("  hist (fraction):")
        print("    [0,20):   {:.3f}".format(hist[0]))
        print("    [20,40):  {:.3f}".format(hist[1]))
        print("    [40,60):  {:.3f}".format(hist[2]))
        print("    [60,90):  {:.3f}".format(hist[3]))
        print("    [90,180]: {:.3f}".format(hist[4]))

    # 全局
    summarize_errors("Global", global_err_deg)

    # pocket / non-pocket
    summarize_errors("Pocket (w_res>{:.2f})".format(pocket_threshold), pocket_err_deg)
    summarize_errors("Non-pocket (w_res<={:.2f})".format(pocket_threshold), non_pocket_err_deg)

    # 按 aa 类型
    print("\n[Per-amino acid χ1 error]")
    for aa in sorted(aa_err.keys()):
        errors = aa_err[aa]
        arr = np.array(errors, dtype=np.float32)
        mean = float(arr.mean())
        p90 = float(np.percentile(arr, 90))
        n = len(arr)
        print(f"  {aa}: N={n:5d}, mean={mean:6.2f}°, p90={p90:6.2f}°")

    # 保存 npz
    out_dir = Path(trainer.config.log_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chi1_errors_val.npz"

    # 把 records 拆成 numpy 数组，方便后续使用
    pdb_ids = np.array([r['pdb_id'] for r in records], dtype=object)
    res_idx = np.array([r['res_idx'] for r in records], dtype=np.int32)
    aa_arr = np.array([r['aa'] for r in records], dtype=object)
    w_res_arr = np.array([r['w_res'] for r in records], dtype=np.float32)
    is_pocket_arr = np.array([r['is_pocket'] for r in records], dtype=bool)
    pred_chi1_arr = np.array([r['pred_chi1'] for r in records], dtype=np.float32)
    true_chi1_arr = np.array([r['true_chi1'] for r in records], dtype=np.float32)
    err_deg_arr = np.array([r['err_deg'] for r in records], dtype=np.float32)

    np.savez_compressed(
        out_path,
        pdb_id=pdb_ids,
        res_idx=res_idx,
        aa=aa_arr,
        w_res=w_res_arr,
        is_pocket=is_pocket_arr,
        pred_chi1=pred_chi1_arr,
        true_chi1=true_chi1_arr,
        err_deg=err_deg_arr,
    )

    print(f"\nSaved per-residue χ1 error records to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage-1 χ1 error analysis on validation set")
    parser.add_argument("--data_dir", type=str, default="data/casf2016")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pocket_threshold", type=float, default=0.5)

    args = parser.parse_args()

    # 构造与训练一致的配置
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 初始化 trainer（会创建模型和 dataloader）
    trainer = Stage1Trainer(config)

    # 加载 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    trainer.model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 运行误差分析
    analyze_chi1_errors(trainer, pocket_threshold=args.pocket_threshold, device=args.device)


if __name__ == "__main__":
    main()
