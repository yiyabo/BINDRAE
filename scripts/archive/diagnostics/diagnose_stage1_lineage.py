#!/usr/bin/env python3
"""Run Stage-1 ligand-causality diagnostics across checkpoint lineage."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_CHECKPOINTS = [
    ("m3_fixbin", "checkpoints/stage1/geom_m3_fixbin_20260430_011149/best_model.pt"),
    ("m3_unfreeze", "checkpoints/stage1/geom_m3_unfreeze_20260501_000058/best_model.pt"),
    ("baseprior", "checkpoints/stage1/geom_baseprior_20260501_170858/best_model.pt"),
    ("phase1_residual", "checkpoints/stage1/phase1_residual_20260502_181331/best_model.pt"),
]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def parse_checkpoint_arg(items: Optional[Iterable[str]]) -> List[tuple[str, str]]:
    if not items:
        return list(DEFAULT_CHECKPOINTS)
    parsed: List[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Checkpoint must be name=path, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Checkpoint must be name=path, got: {item}")
        parsed.append((name, path))
    return parsed


def make_command(args: argparse.Namespace, checkpoint: Path, output_json: Path) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/diagnose_stage1_prior.py",
        "--checkpoint", str(checkpoint),
        "--data_dir", args.data_dir,
        "--val_samples_file", args.val_samples_file,
        "--sample_metadata_file", args.sample_metadata_file,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--output_dir", str(output_json.parent),
        "--output_json", str(output_json),
        "--threshold_deg", str(args.threshold_deg),
        "--pocket_threshold", str(args.pocket_threshold),
        "--contact_threshold", str(args.contact_threshold),
        "--seed", str(args.seed),
        "--decomposition",
    ]
    if args.valid_samples_file:
        cmd.extend(["--valid_samples_file", args.valid_samples_file])
    if args.max_n_res is not None:
        cmd.extend(["--max_n_res", str(args.max_n_res)])
    if args.max_batches is not None:
        cmd.extend(["--max_batches", str(args.max_batches)])
    if args.posterior_diagnostics:
        cmd.append("--posterior_diagnostics")
        cmd.extend(["--calibration_bins", str(args.calibration_bins)])
    return cmd


def summarize_one(name: str, checkpoint: str, result: Dict[str, Any]) -> Dict[str, Any]:
    decomp = result.get("decomposition_metrics", {})
    regular = result.get("metrics", {})

    def rot_acc(variant: str, subset: str) -> float:
        return _as_float(_safe_get(decomp, variant, subset, "rotamer_acc"))

    def g_holo(variant: str, subset: str) -> float:
        return _as_float(_safe_get(decomp, variant, subset, "mean_G_holo"))

    def rescue(variant: str, subset: str) -> float:
        return _as_float(_safe_get(decomp, variant, subset, "rescue_rate"))

    def harm(variant: str, subset: str) -> float:
        return _as_float(_safe_get(decomp, variant, subset, "harmful_flip_rate"))

    contact_subset = "ligand_facing_apo_ca"
    pocket_subset = "pocket"
    switch_subset = "switch"
    all_subset = "all_chi1"

    correct_contact = rot_acc("correct_ligand", contact_subset)
    base_contact = rot_acc("base_only", contact_subset)
    translated_contact = rot_acc("translated_away", contact_subset)
    shuffled_contact = rot_acc("batch_shuffled_ligand", contact_subset)
    no_lig_contact = rot_acc("no_ligand", contact_subset)

    correct_switch = rot_acc("correct_ligand", switch_subset)
    base_switch = rot_acc("base_only", switch_subset)
    translated_switch = rot_acc("translated_away", switch_subset)
    shuffled_switch = rot_acc("batch_shuffled_ligand", switch_subset)

    return {
        "name": name,
        "checkpoint": checkpoint,
        "n_samples": result.get("n_samples"),
        "n_batches": result.get("n_batches"),
        "all_chi1": {
            "base_only_acc": rot_acc("base_only", all_subset),
            "correct_ligand_acc": rot_acc("correct_ligand", all_subset),
            "no_ligand_acc": rot_acc("no_ligand", all_subset),
            "translated_away_acc": rot_acc("translated_away", all_subset),
            "batch_shuffled_ligand_acc": rot_acc("batch_shuffled_ligand", all_subset),
        },
        "contact": {
            "base_only_acc": base_contact,
            "correct_ligand_acc": correct_contact,
            "no_ligand_acc": no_lig_contact,
            "translated_away_acc": translated_contact,
            "batch_shuffled_ligand_acc": shuffled_contact,
            "lift_over_base": correct_contact - base_contact,
            "lift_over_translated": correct_contact - translated_contact,
            "lift_over_shuffled": correct_contact - shuffled_contact,
            "G_holo_correct": g_holo("correct_ligand", contact_subset),
            "G_holo_translated": g_holo("translated_away", contact_subset),
        },
        "switch": {
            "base_only_acc": base_switch,
            "correct_ligand_acc": correct_switch,
            "translated_away_acc": translated_switch,
            "batch_shuffled_ligand_acc": shuffled_switch,
            "lift_over_base": correct_switch - base_switch,
            "lift_over_translated": correct_switch - translated_switch,
            "lift_over_shuffled": correct_switch - shuffled_switch,
            "rescue_rate": rescue("correct_ligand", switch_subset),
            "harmful_flip_rate": harm("correct_ligand", switch_subset),
            "G_holo_correct": g_holo("correct_ligand", switch_subset),
            "G_holo_translated": g_holo("translated_away", switch_subset),
        },
        "pocket": {
            "base_only_acc": rot_acc("base_only", pocket_subset),
            "correct_ligand_acc": rot_acc("correct_ligand", pocket_subset),
            "lift_over_base": rot_acc("correct_ligand", pocket_subset) - rot_acc("base_only", pocket_subset),
        },
        "regular_angle_metrics": {
            "correct_ligand_contact_chi1_acc": _as_float(_safe_get(regular, "correct_ligand", "ligand_facing_apo_ca_residues", "chi1_acc")),
        },
    }


def write_summary_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    rows = summary["checkpoints"]
    lines = [
        "# Stage-1 Lineage Ligand-Causality Diagnostic",
        "",
        "## Purpose",
        "Compare M3, base-prior, and phase-1 residual checkpoints under the same ligand ablation protocol to decide whether any lineage contains stable ligand-causal signal.",
        "",
        "## Checkpoint comparison",
        "",
        "| checkpoint | all correct | all base | contact lift vs base | contact lift vs translated | switch lift vs base | switch lift vs translated | switch G(holo) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {all_c:.4f} | {all_b:.4f} | {clb:+.4f} | {clt:+.4f} | {slb:+.4f} | {slt:+.4f} | {g:+.4f} |".format(
                name=row["name"],
                all_c=row["all_chi1"]["correct_ligand_acc"],
                all_b=row["all_chi1"]["base_only_acc"],
                clb=row["contact"]["lift_over_base"],
                clt=row["contact"]["lift_over_translated"],
                slb=row["switch"]["lift_over_base"],
                slt=row["switch"]["lift_over_translated"],
                g=row["switch"]["G_holo_correct"],
            )
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "A checkpoint is considered ligand-causal only if correct-ligand accuracy or G(holo) improves over base-only and decoy ligands on contact/switch subsets. Raw global chi1 accuracy alone is not sufficient.",
        "",
        "## Files",
        f"- JSON summary: `{summary['summary_json']}`",
        "- Per-checkpoint raw diagnostics are stored beside this report.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage-1 lineage ligand-causality diagnostics")
    parser.add_argument("--checkpoint", action="append", help="Checkpoint as name=path. Defaults to M3/baseprior/phase1 lineage.")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--val_samples_file", default="processed_data/triplets/val_valid.txt")
    parser.add_argument("--sample_metadata_file", default="sample_metadata.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_n_res", type=int, default=900)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="logs/stage1_diagnostics/lineage_ligand_causality")
    parser.add_argument("--threshold_deg", type=float, default=20.0)
    parser.add_argument("--pocket_threshold", type=float, default=0.5)
    parser.add_argument("--contact_threshold", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--posterior_diagnostics", action="store_true")
    parser.add_argument("--calibration_bins", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = parse_checkpoint_arg(args.checkpoint)
    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for name, ckpt_str in checkpoints:
        checkpoint = Path(ckpt_str)
        if not checkpoint.exists():
            failures.append({"name": name, "checkpoint": ckpt_str, "error": "checkpoint not found"})
            print(f"[WARN] {name}: checkpoint not found: {checkpoint}", flush=True)
            continue
        ckpt_dir = output_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        output_json = ckpt_dir / "diagnostics.json"
        cmd = make_command(args, checkpoint, output_json)
        print(f"\n=== Running {name}: {checkpoint} ===", flush=True)
        print(" ".join(cmd), flush=True)
        completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
        if completed.returncode != 0:
            failures.append({"name": name, "checkpoint": ckpt_str, "error": f"exit {completed.returncode}"})
            continue
        result = json.loads(output_json.read_text(encoding="utf-8"))
        summaries.append(summarize_one(name, ckpt_str, result))

    summary_json = output_dir / "lineage_summary.json"
    report_md = output_dir / "lineage_summary.md"
    summary = {
        "checkpoints": summaries,
        "failures": failures,
        "summary_json": str(summary_json),
        "report_md": str(report_md),
        "settings": {
            "val_samples_file": args.val_samples_file,
            "max_batches": args.max_batches,
            "threshold_deg": args.threshold_deg,
            "pocket_threshold": args.pocket_threshold,
            "contact_threshold": args.contact_threshold,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_markdown(summary, report_md)

    print(f"\nSaved lineage summary JSON: {summary_json}")
    print(f"Saved lineage summary report: {report_md}")
    if failures:
        print(f"[WARN] {len(failures)} checkpoint(s) failed or were missing")


if __name__ == "__main__":
    main()
