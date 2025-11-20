import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_chi1_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required_keys = [
        "pdb_id",
        "res_idx",
        "aa",
        "w_res",
        "is_pocket",
        "pred_chi1",
        "true_chi1",
        "err_deg",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in npz: {missing}")
    return {k: data[k] for k in required_keys}


def summarize_global_high_errors(err_deg: np.ndarray,
                                 is_pocket: np.ndarray,
                                 t1: float = 60.0,
                                 t2: float = 90.0) -> None:
    N = err_deg.shape[0]
    mask_high = err_deg > t1
    mask_extreme = err_deg > t2

    frac_high = float(mask_high.mean()) if N > 0 else float("nan")
    frac_extreme = float(mask_extreme.mean()) if N > 0 else float("nan")

    if mask_high.any():
        frac_high_pocket = float(is_pocket[mask_high].mean())
    else:
        frac_high_pocket = float("nan")

    if mask_extreme.any():
        frac_extreme_pocket = float(is_pocket[mask_extreme].mean())
    else:
        frac_extreme_pocket = float("nan")

    print("[Global high-error χ1]")
    print(f"  N={N}")
    print(f"  frac(|Δχ1|>{t1:.0f}°)   = {frac_high:6.3f}")
    print(f"  frac(|Δχ1|>{t2:.0f}°)   = {frac_extreme:6.3f}")
    print(f"  pocket fraction among |Δχ1|>{t1:.0f}°   = {frac_high_pocket:6.3f}")
    print(f"  pocket fraction among |Δχ1|>{t2:.0f}°   = {frac_extreme_pocket:6.3f}")


def per_aa_stats_all(err_deg: np.ndarray,
                     aa: np.ndarray,
                     is_pocket: np.ndarray,
                     t1: float = 60.0,
                     t2: float = 90.0) -> None:
    aa_list = [str(x) for x in aa]
    unique_aas = sorted(set(aa_list))

    rows: List[Dict] = []

    for aa_code in unique_aas:
        mask_aa = np.array([x == aa_code for x in aa_list], dtype=bool)
        if not mask_aa.any():
            continue
        errs = err_deg[mask_aa]
        pockets = is_pocket[mask_aa]

        N = errs.shape[0]
        mean = float(errs.mean())
        p90 = float(np.percentile(errs, 90))

        high = errs > t1
        extreme = errs > t2
        frac_high = float(high.mean()) if N > 0 else float("nan")
        frac_extreme = float(extreme.mean()) if N > 0 else float("nan")

        if high.any():
            pocket_frac_high = float(pockets[high].mean())
        else:
            pocket_frac_high = float("nan")

        rows.append({
            "aa": aa_code,
            "N": N,
            "mean": mean,
            "p90": p90,
            "frac_high": frac_high,
            "frac_extreme": frac_extreme,
            "pocket_frac_high": pocket_frac_high,
        })

    # 按高误差比例排序，便于看到最难的残基类型
    rows.sort(key=lambda r: r["frac_high"], reverse=True)

    print("\n[Per-aa χ1 error (all residues), sorted by frac(|Δχ1|>60°)]")
    header = (
        "  AA   N      mean°    p90°   frac>60°   frac>90°   pocket_frac_high(|Δχ1|>60°)"
    )
    print(header)
    for r in rows:
        print(
            f"  {r['aa']:>2}  {r['N']:5d}  "
            f"{r['mean']:8.2f}  {r['p90']:6.2f}  "
            f"{r['frac_high']:8.3f}  {r['frac_extreme']:8.3f}  "
            f"{r['pocket_frac_high']:10.3f}"
        )


def per_aa_stats_pocket_only(err_deg: np.ndarray,
                             aa: np.ndarray,
                             is_pocket: np.ndarray,
                             t1: float = 60.0,
                             t2: float = 90.0) -> None:
    aa_list = [str(x) for x in aa]
    unique_aas = sorted(set(aa_list))

    rows: List[Dict] = []

    for aa_code in unique_aas:
        mask = np.array([x == aa_code for x in aa_list], dtype=bool) & is_pocket
        if not mask.any():
            continue
        errs = err_deg[mask]

        N = errs.shape[0]
        mean = float(errs.mean())
        p90 = float(np.percentile(errs, 90))

        high = errs > t1
        extreme = errs > t2
        frac_high = float(high.mean()) if N > 0 else float("nan")
        frac_extreme = float(extreme.mean()) if N > 0 else float("nan")

        rows.append({
            "aa": aa_code,
            "N": N,
            "mean": mean,
            "p90": p90,
            "frac_high": frac_high,
            "frac_extreme": frac_extreme,
        })

    # 同样按高误差比例排序
    rows.sort(key=lambda r: r["frac_high"], reverse=True)

    print("\n[Per-aa χ1 error (pocket only), sorted by frac(|Δχ1|>60°)]")
    header = "  AA   N      mean°    p90°   frac>60°   frac>90°"
    print(header)
    for r in rows:
        print(
            f"  {r['aa']:>2}  {r['N']:5d}  "
            f"{r['mean']:8.2f}  {r['p90']:6.2f}  "
            f"{r['frac_high']:8.3f}  {r['frac_extreme']:8.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc χ1 error analysis based on chi1_errors_val.npz",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="logs/stage1/analysis/chi1_errors_val.npz",
        help="Path to chi1_errors_val.npz produced by chi1_error_analysis.py",
    )
    parser.add_argument("--t1", type=float, default=60.0, help="High-error threshold in degrees")
    parser.add_argument("--t2", type=float, default=90.0, help="Extreme-error threshold in degrees")

    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"npz file not found: {npz_path}")

    data = load_chi1_npz(npz_path)
    err_deg = data["err_deg"].astype(np.float32)
    aa = data["aa"]
    is_pocket = data["is_pocket"].astype(bool)

    summarize_global_high_errors(err_deg, is_pocket, t1=args.t1, t2=args.t2)
    per_aa_stats_all(err_deg, aa, is_pocket, t1=args.t1, t2=args.t2)
    per_aa_stats_pocket_only(err_deg, aa, is_pocket, t1=args.t1, t2=args.t2)


if __name__ == "__main__":
    main()
