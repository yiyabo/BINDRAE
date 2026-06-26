#!/usr/bin/env python3
"""Generate apo-to-holo Stage-2 trajectory examples.

This is an inference/export entry point, not a metric-only evaluator.  It
reuses the Stage-2 transition evaluator's model loading, prior-feature, FK, and
Heun integration conventions, then writes per-sample trajectory artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from evaluate_stage2_transition_paths import (  # noqa: E402
    batch_to_device,
    build_model_config_for_checkpoint,
    build_rigids_from_backbone,
    combined_prior_features,
    compute_interaction_prior_feature,
    integrate_path,
    min_sc_ligand_dist,
    resolve_integration_clips,
    torsions_to_atom14,
)
from src.stage1.data.residue_constants import (  # noqa: E402
    restype_1to3,
    restype_name_to_atom14_names,
    restypes,
)
from src.stage1.models.fk_openfold import create_openfold_fk  # noqa: E402
from src.stage2.datasets import create_stage2_dataloader  # noqa: E402
from src.stage2.models import TorsionFlowNet, TorsionFlowNetConfig  # noqa: E402


LIGAND_ELEMENTS = ["C", "N", "O", "S", "P", "F", "CL", "BR", "I"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-2 generated apo-to-holo trajectories")
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint path")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--valid_samples_file", default=None, help="sample filter file relative to data_dir")
    parser.add_argument("--output_dir", required=True, help="directory for exported trajectory artifacts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--sample_ids", nargs="*", default=None, help="optional exact sample IDs to export")
    parser.add_argument("--n_integration_steps", type=int, default=None)
    parser.add_argument("--integration_chi_clip", type=float, default=None)
    parser.add_argument("--integration_rot_clip", type=float, default=None)
    parser.add_argument("--integration_trans_clip", type=float, default=None)
    parser.add_argument("--interaction_prior_ckpt", default=None)
    parser.add_argument(
        "--interaction_prior_feature_mode",
        default=None,
        choices=["none", "prior", "prior_shuffled", "zero", "oracle_contact"],
        help="override checkpoint interaction prior feature mode",
    )
    parser.add_argument("--interaction_prior_temperature", type=float, default=None)
    parser.add_argument("--interaction_prior_feature_scale", type=float, default=None)
    parser.add_argument(
        "--stage1v2_posterior_feature_mode",
        default=None,
        choices=[
            "none",
            "zero",
            "student",
            "student_shuffled",
            "oracle_holo_truth",
            "external_teacher_cached",
            "oracle_motion",
            "oracle_motion_residue_shuffled",
            "oracle_motion_sample_shuffled",
        ],
        help="override checkpoint Stage-1-v2/oracle-motion feature mode",
    )
    parser.add_argument(
        "--stage1v2_posterior_cache_dir",
        default=None,
        help="override Stage-1-v2/oracle-motion feature cache/label directory for this split",
    )
    parser.add_argument(
        "--stage1v2_posterior_feature_names",
        default=None,
        help="override comma-separated Stage-1-v2/oracle-motion feature names",
    )
    parser.add_argument("--stage1v2_posterior_feature_scale", type=float, default=None)
    parser.add_argument("--contact_dist", type=float, default=4.5)
    parser.add_argument("--active_delta", type=float, default=0.75)
    parser.add_argument("--pocket_threshold", type=float, default=0.3)
    parser.add_argument("--path_dist_cap", type=float, default=20.0)
    parser.add_argument("--no_pdb", action="store_true", help="skip PDB export; always writes NPZ/JSON")
    return parser.parse_args()


def load_model_and_fk(args: argparse.Namespace, device: torch.device):
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", SimpleNamespace())
    args.config = config
    model_config, interaction_settings, stage1v2_settings = build_model_config_for_checkpoint(
        args, config, args.split
    )
    model = TorsionFlowNet(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    fk_module = create_openfold_fk().to(device)
    fk_module.eval()
    n_steps = int(args.n_integration_steps or getattr(config, "n_integration_steps", 5))
    integration_clips = resolve_integration_clips(args, config)
    return model, fk_module, config, interaction_settings, stage1v2_settings, n_steps, integration_clips


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def residue_name(aatype_idx: int) -> str:
    if 0 <= int(aatype_idx) < len(restypes):
        return restype_1to3[restypes[int(aatype_idx)]]
    return "UNK"


def pdb_element(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if not letters:
        return "X"
    if letters[:2].upper() in {"CL", "BR"}:
        return letters[:2].upper()
    return letters[0].upper()


def write_atom14_multimodel_pdb(
    path: Path,
    atom14_path: np.ndarray,
    atom14_mask: np.ndarray,
    aatype: np.ndarray,
    node_mask: np.ndarray,
    model_labels: Optional[List[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_models, n_res, _, _ = atom14_path.shape
    lines: List[str] = []
    for model_idx in range(n_models):
        lines.append(f"MODEL     {model_idx + 1:4d}")
        if model_labels and model_idx < len(model_labels):
            lines.append(f"REMARK BINDRAE_FRAME {model_labels[model_idx]}")
        serial = 1
        for res_idx in range(n_res):
            if not bool(node_mask[res_idx]):
                continue
            resname = residue_name(int(aatype[res_idx]))
            atom_names = restype_name_to_atom14_names.get(resname, [""] * 14)
            for atom_idx, atom_name in enumerate(atom_names):
                if not atom_name or not bool(atom14_mask[res_idx, atom_idx]):
                    continue
                x, y, z = atom14_path[model_idx, res_idx, atom_idx]
                elem = pdb_element(atom_name)
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} A{res_idx + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
                )
                serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ligand_element_from_type(type_vec: np.ndarray) -> str:
    if type_vec.size >= len(LIGAND_ELEMENTS):
        idx = int(np.argmax(type_vec[: len(LIGAND_ELEMENTS)]))
        if type_vec[idx] > 0:
            return LIGAND_ELEMENTS[idx]
    return "X"


def write_ligand_tokens_pdb(path: Path, lig_points: np.ndarray, lig_types: np.ndarray, lig_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["REMARK BINDRAE ligand token coordinates; probe tokens may be included"]
    serial = 1
    for idx, ok in enumerate(lig_mask.astype(bool)):
        if not ok:
            continue
        x, y, z = lig_points[idx]
        elem = ligand_element_from_type(lig_types[idx])
        atom_name = f"{elem[:2]}{(idx % 100):02d}"[:4]
        lines.append(
            f"HETATM{serial:5d} {atom_name:^4s} LIG L   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
        )
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_or_none(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    mask = mask.bool()
    if not bool(mask.any()):
        return None
    return float(values[mask].detach().float().mean().item())


def summarize_sample(args, d_apo, d_holo, d_path, node_mask, w_res) -> Dict[str, object]:
    cap = float(args.path_dist_cap)
    delta_target = d_holo - d_apo
    delta_actual = d_path[-1] - d_apo
    pocket = node_mask.bool() & (w_res > float(args.pocket_threshold))
    finite = pocket & torch.isfinite(d_apo) & torch.isfinite(d_holo) & (d_apo < 49.0) & (d_holo < 49.0)
    active = finite & (delta_target.abs() >= float(args.active_delta))
    approach = active & (delta_target < 0)
    release = active & (delta_target > 0)
    endpoint_abs = (d_path[-1].clamp(max=cap) - d_holo.clamp(max=cap)).abs()
    path_mae = d_path.new_zeros(d_apo.shape)
    for idx in range(d_path.shape[0]):
        t_val = idx / max(d_path.shape[0] - 1, 1)
        target_t = d_apo + float(t_val) * delta_target
        path_mae = path_mae + (d_path[idx].clamp(max=cap) - target_t.clamp(max=cap)).abs()
    path_mae = path_mae / max(int(d_path.shape[0]), 1)
    direction = ((delta_actual * delta_target) > 0).float()
    classes = {"active": active, "approach": approach, "release": release}
    out: Dict[str, object] = {"counts": {name: int(mask.sum().item()) for name, mask in classes.items()}}
    for name, mask in classes.items():
        out[f"{name}/endpoint_abs_dist"] = mean_or_none(endpoint_abs, mask)
        out[f"{name}/path_mae_dist"] = mean_or_none(path_mae, mask)
        out[f"{name}/direction_acc"] = mean_or_none(direction, mask)
    return out


def export_batch(
    args,
    model,
    fk_module,
    batch,
    n_steps: int,
    output_dir: Path,
    wanted_ids: Optional[set],
    limit_left: int,
    interaction_settings: Dict[str, object],
    stage1v2_settings: Dict[str, object],
    integration_clips: Dict[str, float],
):
    batch = batch_to_device(batch, torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"))
    rigids_apo = build_rigids_from_backbone(batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask)
    rigids_holo = build_rigids_from_backbone(batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask)
    interaction_prior_feature = compute_interaction_prior_feature(
        args, args.config, batch, fk_module, rigids_apo, rigids_holo
    )
    prior_features = combined_prior_features(
        args,
        args.config,
        batch,
        interaction_prior_feature,
        stage1v2_settings,
        interaction_settings,
    )
    rigids_list, chi_list, t_list = integrate_path(
        model,
        batch,
        rigids_apo,
        batch.torsion_apo[..., 3:7],
        n_steps=n_steps,
        interaction_prior=prior_features,
        chi_clip=integration_clips["chi"],
        rot_clip=integration_clips["rot"],
        trans_clip=integration_clips["trans"],
    )

    atom14_apo = torsions_to_atom14(
        fk_module, batch.torsion_apo[..., :3], batch.torsion_apo[..., 3:7], rigids_apo, batch.aatype
    )
    atom14_holo = torsions_to_atom14(
        fk_module, batch.torsion_holo[..., :3], batch.torsion_holo[..., 3:7], rigids_holo, batch.aatype
    )
    atom14_path_tensors = [
        torsions_to_atom14(fk_module, batch.torsion_apo[..., :3], chi_t, rigids_t, batch.aatype)
        for rigids_t, chi_t in zip(rigids_list, chi_list)
    ]

    d_apo = min_sc_ligand_dist(
        atom14_apo["atom14_pos"].float(),
        atom14_apo["atom14_mask"].bool(),
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
    )
    d_holo = min_sc_ligand_dist(
        atom14_holo["atom14_pos"].float(),
        atom14_holo["atom14_mask"].bool(),
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
    )
    d_path = torch.stack(
        [
            min_sc_ligand_dist(
                atom14_t["atom14_pos"].float(),
                atom14_t["atom14_mask"].bool(),
                batch.lig_points.float(),
                batch.lig_mask.bool(),
                batch.node_mask.bool(),
            )
            for atom14_t in atom14_path_tensors
        ],
        dim=0,
    )

    manifest_rows = []
    exported = 0
    for b, sample_id in enumerate(batch.pdb_ids):
        if wanted_ids is not None and sample_id not in wanted_ids:
            continue
        if exported >= limit_left:
            break
        n_res = int(batch.n_residues[b])
        lig_n = int(batch.lig_mask[b].sum().item())
        sample_dir = output_dir / safe_name(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        atom14_path = np.stack(
            [tensor_to_np(atom14_t["atom14_pos"][b, :n_res]) for atom14_t in atom14_path_tensors],
            axis=0,
        )
        atom14_mask = tensor_to_np(atom14_path_tensors[-1]["atom14_mask"][b, :n_res]).astype(bool)
        atom14_apo_np = tensor_to_np(atom14_apo["atom14_pos"][b, :n_res])
        atom14_holo_np = tensor_to_np(atom14_holo["atom14_pos"][b, :n_res])
        atom14_apo_mask = tensor_to_np(atom14_apo["atom14_mask"][b, :n_res]).astype(bool)
        atom14_holo_mask = tensor_to_np(atom14_holo["atom14_mask"][b, :n_res]).astype(bool)
        node_mask_np = tensor_to_np(batch.node_mask[b, :n_res]).astype(bool)
        aatype_np = tensor_to_np(batch.aatype[b, :n_res]).astype(np.int64)
        lig_points_np = tensor_to_np(batch.lig_points[b, :lig_n])
        lig_types_np = tensor_to_np(batch.lig_types[b, :lig_n])
        lig_mask_np = tensor_to_np(batch.lig_mask[b, :lig_n]).astype(bool)
        chi_path_np = np.stack([tensor_to_np(chi_t[b, :n_res]) for chi_t in chi_list], axis=0)
        ca_path_np = np.stack([tensor_to_np(rigids_t.get_trans()[b, :n_res]) for rigids_t in rigids_list], axis=0)
        prior_np = (
            tensor_to_np(prior_features[b, :n_res])
            if prior_features is not None
            else np.zeros((n_res, 0), dtype=np.float32)
        )
        stage1v2_np = (
            tensor_to_np(batch.stage1v2_posterior_features[b, :n_res])
            if batch.stage1v2_posterior_features is not None
            else np.zeros((n_res, 0), dtype=np.float32)
        )
        sample_summary = summarize_sample(
            args,
            d_apo[b, :n_res],
            d_holo[b, :n_res],
            d_path[:, b, :n_res],
            batch.node_mask[b, :n_res],
            batch.w_res[b, :n_res],
        )
        sample_summary.update(
            {
                "sample_id": sample_id,
                "n_residues": n_res,
                "n_ligand_tokens": lig_n,
                "n_frames": len(t_list),
                "npz": str(sample_dir / "trajectory.npz"),
            }
        )

        np.savez_compressed(
            sample_dir / "trajectory.npz",
            t=np.asarray(t_list, dtype=np.float32),
            atom14_path=atom14_path.astype(np.float32),
            atom14_mask=atom14_mask,
            atom14_apo=atom14_apo_np.astype(np.float32),
            atom14_apo_mask=atom14_apo_mask,
            atom14_holo=atom14_holo_np.astype(np.float32),
            atom14_holo_mask=atom14_holo_mask,
            chi_path=chi_path_np.astype(np.float32),
            ca_path=ca_path_np.astype(np.float32),
            aatype=aatype_np,
            node_mask=node_mask_np,
            ligand_points=lig_points_np.astype(np.float32),
            ligand_types=lig_types_np.astype(np.float32),
            ligand_mask=lig_mask_np,
            w_res=tensor_to_np(batch.w_res[b, :n_res]).astype(np.float32),
            interaction_prior=prior_np.astype(np.float32),
            stage1v2_posterior_features=stage1v2_np.astype(np.float32),
            d_apo=tensor_to_np(d_apo[b, :n_res]).astype(np.float32),
            d_holo=tensor_to_np(d_holo[b, :n_res]).astype(np.float32),
            d_path=tensor_to_np(d_path[:, b, :n_res]).astype(np.float32),
        )
        (sample_dir / "summary.json").write_text(
            json.dumps(sample_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not args.no_pdb:
            labels = [f"t={t_val:.4f}" for t_val in t_list]
            write_atom14_multimodel_pdb(
                sample_dir / "trajectory_atom14.pdb",
                atom14_path,
                atom14_mask,
                aatype_np,
                node_mask_np,
                labels,
            )
            write_atom14_multimodel_pdb(
                sample_dir / "apo_atom14.pdb",
                atom14_apo_np[None, ...],
                atom14_apo_mask,
                aatype_np,
                node_mask_np,
                ["apo_reference"],
            )
            write_atom14_multimodel_pdb(
                sample_dir / "holo_atom14.pdb",
                atom14_holo_np[None, ...],
                atom14_holo_mask,
                aatype_np,
                node_mask_np,
                ["holo_reference"],
            )
            write_ligand_tokens_pdb(sample_dir / "ligand_tokens.pdb", lig_points_np, lig_types_np, lig_mask_np)
            sample_summary["pdb"] = str(sample_dir / "trajectory_atom14.pdb")
        manifest_rows.append(sample_summary)
        exported += 1
    return manifest_rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.device = str(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        model,
        fk_module,
        config,
        interaction_settings,
        stage1v2_settings,
        n_steps,
        integration_clips,
    ) = load_model_and_fk(args, device)
    loader = create_stage2_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        require_nma=bool(getattr(config, "use_nma", False)),
        valid_samples_file=args.valid_samples_file,
        stage1v2_posterior_cache_dir=stage1v2_settings["cache_dir"],
        stage1v2_posterior_feature_mode=stage1v2_settings["mode"],
        stage1v2_posterior_feature_names=stage1v2_settings["names_raw"],
    )

    wanted_ids = set(args.sample_ids) if args.sample_ids else None
    max_samples = int(args.max_samples) if args.max_samples is not None else 10**9
    manifest: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "valid_samples_file": args.valid_samples_file,
        "interaction_prior_feature_mode": interaction_settings["mode"],
        "stage1v2_posterior_feature_mode": stage1v2_settings["mode"],
        "stage1v2_posterior_cache_dir": stage1v2_settings["cache_dir"],
        "stage1v2_posterior_feature_names": list(stage1v2_settings["names"]),
        "stage1v2_posterior_feature_scale": stage1v2_settings["scale"],
        "n_integration_steps": n_steps,
        "integration_clips": integration_clips,
        "output_dir": str(output_dir),
        "samples": [],
    }
    exported = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Generate trajectories", ncols=120)):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            rows = export_batch(
                args,
                model,
                fk_module,
                batch,
                n_steps=n_steps,
                output_dir=output_dir,
                wanted_ids=wanted_ids,
                limit_left=max_samples - exported,
                interaction_settings=interaction_settings,
                stage1v2_settings=stage1v2_settings,
                integration_clips=integration_clips,
            )
            manifest["samples"].extend(rows)
            exported += len(rows)
            if exported >= max_samples:
                break

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Exported {exported} trajectory sample(s) to {output_dir}")


if __name__ == "__main__":
    main()
