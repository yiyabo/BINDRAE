#!/usr/bin/env python3
"""Compare per-residue SE(3) and chain-internal backbone bridges."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FLASH_IPA = PROJECT_ROOT / "vendor" / "flash_ipa" / "src"
if str(FLASH_IPA) not in sys.path:
    sys.path.insert(0, str(FLASH_IPA))

from src.stage1.models.fk_openfold import create_openfold_fk
from src.stage2.datasets import create_stage2_dataloader
from src.stage2.modules.chain_internal import (
    interpolate_backbone_internal,
    project_anchored_pose_graph,
    project_peptide_frame_translations,
)
from src.stage2.modules.geometry import compute_peptide_loss
from scripts import evaluate_stage2_transition_paths as path_eval


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid_samples_file", required=True)
    parser.add_argument("--n_path_steps", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--projection_iterations", type=int, default=12)
    parser.add_argument("--projection_relaxation", type=float, default=0.75)
    parser.add_argument("--projection_anchor_strength", type=float, default=0.02)
    parser.add_argument("--projection_max_translation", type=float, default=2.0)
    parser.add_argument("--projection_activation_loss_threshold", type=float, default=0.0)
    parser.add_argument("--pose_graph_iterations", type=int, default=20)
    parser.add_argument("--pose_graph_learning_rate", type=float, default=0.05)
    parser.add_argument("--pose_graph_edge_weight", type=float, default=1.0)
    parser.add_argument("--pose_graph_anchor_weight", type=float, default=0.1)
    parser.add_argument("--pose_graph_rotation_metric_scale", type=float, default=1.5)
    parser.add_argument("--pose_graph_max_rotation", type=float, default=0.5)
    parser.add_argument("--pose_graph_max_translation", type=float, default=2.0)
    return parser.parse_args()


def peptide_loss(fk_module, batch, rigids, chi, t_value):
    atom14 = path_eval.torsions_to_atom14(
        fk_module,
        path_eval.interpolate_backbone_torsions(batch, t_value),
        chi,
        rigids,
        batch.aatype,
    )
    return compute_peptide_loss(
        atom14["atom14_pos"].float(),
        atom14["atom14_mask"].bool(),
        batch.node_mask.bool(),
        peptide_bond_mask=batch.peptide_bond_mask.bool(),
    )


def summarize(records, key):
    values = np.asarray([record[key] for record in records], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main():
    args = parse_args()
    if args.n_path_steps < 2:
        raise ValueError("--n_path_steps must be >= 2")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", SimpleNamespace())
    cache_dir = getattr(config, "stage1v2_val_cache_dir", None)
    loader = create_stage2_dataloader(
        "processed_data/triplets",
        split="val",
        batch_size=1,
        shuffle=False,
        num_workers=0,
        valid_samples_file=args.valid_samples_file,
        trust_prechecked_samples=True,
        stage1v2_posterior_cache_dir=cache_dir,
        stage1v2_posterior_feature_mode=getattr(
            config, "stage1v2_posterior_feature_mode", "none"
        ),
        stage1v2_posterior_feature_names=getattr(
            config, "stage1v2_posterior_feature_names", ""
        ),
        esm_num_layers=int(getattr(config, "esm_num_layers", 1)),
    )
    fk_module = create_openfold_fk().to(device)
    fk_module.eval()
    interior_times = [step / args.n_path_steps for step in range(1, args.n_path_steps)]
    records = []

    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(
            tqdm(loader, desc="Chain-internal bridge", ncols=120)
        ):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            internal_backbones = [
                interpolate_backbone_internal(
                    cpu_batch.N_apo,
                    cpu_batch.Ca_apo,
                    cpu_batch.C_apo,
                    cpu_batch.N_holo,
                    cpu_batch.Ca_holo,
                    cpu_batch.C_holo,
                    cpu_batch.node_mask,
                    cpu_batch.peptide_bond_mask,
                    torch.tensor(time, dtype=cpu_batch.N_apo.dtype),
                )
                for time in interior_times
            ]
            start = interpolate_backbone_internal(
                cpu_batch.N_apo,
                cpu_batch.Ca_apo,
                cpu_batch.C_apo,
                cpu_batch.N_holo,
                cpu_batch.Ca_holo,
                cpu_batch.C_holo,
                cpu_batch.node_mask,
                cpu_batch.peptide_bond_mask,
                torch.tensor(0.0, dtype=cpu_batch.N_apo.dtype),
            )
            end = interpolate_backbone_internal(
                cpu_batch.N_apo,
                cpu_batch.Ca_apo,
                cpu_batch.C_apo,
                cpu_batch.N_holo,
                cpu_batch.Ca_holo,
                cpu_batch.C_holo,
                cpu_batch.node_mask,
                cpu_batch.peptide_bond_mask,
                torch.tensor(1.0, dtype=cpu_batch.N_apo.dtype),
            )
            endpoint_error = max(
                (start[0] - cpu_batch.N_apo).abs().max().item(),
                (start[1] - cpu_batch.Ca_apo).abs().max().item(),
                (start[2] - cpu_batch.C_apo).abs().max().item(),
                (end[0] - cpu_batch.N_holo).abs().max().item(),
                (end[1] - cpu_batch.Ca_holo).abs().max().item(),
                (end[2] - cpu_batch.C_holo).abs().max().item(),
            )

            batch = path_eval.batch_to_device(cpu_batch, device)
            rigids_apo = path_eval.build_rigids_from_backbone(
                batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
            )
            rigids_holo = path_eval.build_rigids_from_backbone(
                batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
            )
            pure_rigids, pure_chi, pure_times = path_eval.pure_bridge_path(
                batch, rigids_apo, rigids_holo, args.n_path_steps
            )
            apo_rotation, apo_translation = path_eval.rigid_to_rt(rigids_apo)
            holo_rotation, holo_translation = path_eval.rigid_to_rt(rigids_holo)
            pure_losses = []
            cartesian_losses = []
            cartesian_projected_losses = []
            cartesian_projection_rms = []
            cartesian_projection_max = []
            projected_losses = []
            projection_rms = []
            projection_max = []
            pose_graph_losses = []
            pose_graph_rotation_rms = []
            pose_graph_translation_rms = []
            pose_graph_rotation_max = []
            pose_graph_translation_max = []
            internal_losses = []
            internal_cn_errors = []
            for index, time in enumerate(interior_times):
                pure_atom14 = path_eval.torsions_to_atom14(
                    fk_module,
                    path_eval.interpolate_backbone_torsions(
                        batch, pure_times[index + 1]
                    ),
                    pure_chi[index + 1],
                    pure_rigids[index + 1],
                    batch.aatype,
                )
                pure_losses.append(
                    compute_peptide_loss(
                        pure_atom14["atom14_pos"].float(),
                        pure_atom14["atom14_mask"].bool(),
                        batch.node_mask.bool(),
                        peptide_bond_mask=batch.peptide_bond_mask.bool(),
                    )
                )
                t_value = pure_times[index + 1]
                gamma = 3.0 * t_value * t_value - 2.0 * t_value * t_value * t_value
                cartesian_rigid = path_eval.build_rigids_from_backbone(
                    (1.0 - gamma) * batch.N_apo + gamma * batch.N_holo,
                    (1.0 - gamma) * batch.Ca_apo + gamma * batch.Ca_holo,
                    (1.0 - gamma) * batch.C_apo + gamma * batch.C_holo,
                    batch.node_mask,
                )
                cartesian_losses.append(
                    peptide_loss(
                        fk_module,
                        batch,
                        cartesian_rigid,
                        pure_chi[index + 1],
                        t_value,
                    )
                )
                target_length, target_cacn, target_cnca = (
                    path_eval.interpolate_peptide_geometry_targets(
                        batch, pure_times[index + 1]
                    )
                )
                cartesian_atom14 = path_eval.torsions_to_atom14(
                    fk_module,
                    path_eval.interpolate_backbone_torsions(batch, t_value),
                    pure_chi[index + 1],
                    cartesian_rigid,
                    batch.aatype,
                )
                cartesian_translation = project_peptide_frame_translations(
                    cartesian_atom14["atom14_pos"].float(),
                    cartesian_atom14["atom14_mask"].bool(),
                    batch.node_mask.bool(),
                    batch.peptide_bond_mask.bool(),
                    n_iterations=args.projection_iterations,
                    relaxation=args.projection_relaxation,
                    anchor_strength=args.projection_anchor_strength,
                    bond_length=target_length,
                    angle_cacn=target_cacn,
                    angle_cnca=target_cnca,
                    max_translation=args.projection_max_translation,
                    activation_loss_threshold=(
                        args.projection_activation_loss_threshold
                    ),
                )
                cart_rotation, cart_guide_translation = path_eval.rigid_to_rt(
                    cartesian_rigid
                )
                cartesian_projected_losses.append(
                    peptide_loss(
                        fk_module,
                        batch,
                        path_eval.rt_to_rigid(
                            cart_rotation,
                            cart_guide_translation + cartesian_translation,
                        ),
                        pure_chi[index + 1],
                        t_value,
                    )
                )
                valid_residue = batch.node_mask.bool()
                cartesian_projection_norm = torch.linalg.norm(
                    cartesian_translation, dim=-1
                )[valid_residue]
                cartesian_projection_rms.append(
                    torch.sqrt(cartesian_projection_norm.square().mean())
                )
                cartesian_projection_max.append(cartesian_projection_norm.max())
                frame_translation = project_peptide_frame_translations(
                    pure_atom14["atom14_pos"].float(),
                    pure_atom14["atom14_mask"].bool(),
                    batch.node_mask.bool(),
                    batch.peptide_bond_mask.bool(),
                    n_iterations=args.projection_iterations,
                    relaxation=args.projection_relaxation,
                    anchor_strength=args.projection_anchor_strength,
                    bond_length=target_length,
                    angle_cacn=target_cacn,
                    angle_cnca=target_cnca,
                    max_translation=args.projection_max_translation,
                    activation_loss_threshold=(
                        args.projection_activation_loss_threshold
                    ),
                )
                pure_rotation, pure_translation = path_eval.rigid_to_rt(
                    pure_rigids[index + 1]
                )
                projected_rigid = path_eval.rt_to_rigid(
                    pure_rotation, pure_translation + frame_translation
                )
                projected_losses.append(
                    peptide_loss(
                        fk_module,
                        batch,
                        projected_rigid,
                        pure_chi[index + 1],
                        pure_times[index + 1],
                    )
                )
                projection_norm = torch.linalg.norm(frame_translation, dim=-1)
                projection_rms.append(
                    torch.sqrt(
                        frame_translation.square().sum(dim=-1)[valid_residue].mean()
                    )
                )
                projection_max.append(projection_norm[valid_residue].max())
                guide_rotation, guide_translation = path_eval.rigid_to_rt(
                    pure_rigids[index + 1]
                )
                gamma = 3.0 * t_value * t_value - 2.0 * t_value * t_value * t_value
                pose_rotation, pose_translation, pose_delta = (
                    project_anchored_pose_graph(
                        guide_rotation,
                        guide_translation,
                        apo_rotation,
                        apo_translation,
                        holo_rotation,
                        holo_translation,
                        batch.node_mask,
                        batch.peptide_bond_mask,
                        progress=gamma,
                        n_iterations=args.pose_graph_iterations,
                        learning_rate=args.pose_graph_learning_rate,
                        edge_weight=args.pose_graph_edge_weight,
                        anchor_weight=args.pose_graph_anchor_weight,
                        rotation_metric_scale=args.pose_graph_rotation_metric_scale,
                        max_rotation=args.pose_graph_max_rotation,
                        max_translation=args.pose_graph_max_translation,
                    )
                )
                pose_graph_losses.append(
                    peptide_loss(
                        fk_module,
                        batch,
                        path_eval.rt_to_rigid(pose_rotation, pose_translation),
                        pure_chi[index + 1],
                        t_value,
                    )
                )
                pose_rotation_norm = torch.linalg.norm(
                    pose_delta[..., :3], dim=-1
                )[valid_residue]
                pose_translation_norm = torch.linalg.norm(
                    pose_delta[..., 3:], dim=-1
                )[valid_residue]
                pose_graph_rotation_rms.append(
                    torch.sqrt(pose_rotation_norm.square().mean())
                )
                pose_graph_translation_rms.append(
                    torch.sqrt(pose_translation_norm.square().mean())
                )
                pose_graph_rotation_max.append(pose_rotation_norm.max())
                pose_graph_translation_max.append(pose_translation_norm.max())
                n_coord, ca_coord, c_coord = (
                    tensor.to(device) for tensor in internal_backbones[index]
                )
                internal_rigids = path_eval.build_rigids_from_backbone(
                    n_coord, ca_coord, c_coord, batch.node_mask
                )
                chi = path_eval.interpolate_endpoints(
                    batch, rigids_apo, rigids_holo, time
                )[1]
                internal_losses.append(
                    peptide_loss(fk_module, batch, internal_rigids, chi, time)
                )
                cn_distance = torch.linalg.norm(c_coord[:, :-1] - n_coord[:, 1:], dim=-1)
                cn_apo = torch.linalg.norm(batch.C_apo[:, :-1] - batch.N_apo[:, 1:], dim=-1)
                cn_holo = torch.linalg.norm(batch.C_holo[:, :-1] - batch.N_holo[:, 1:], dim=-1)
                cn_target = (1.0 - time) * cn_apo + time * cn_holo
                mask = batch.peptide_bond_mask.bool()
                if mask.any():
                    internal_cn_errors.append(
                        (cn_distance[mask] - cn_target[mask]).abs().max()
                    )

            pure_value = torch.stack(pure_losses).mean().item()
            cartesian_value = torch.stack(cartesian_losses).mean().item()
            cartesian_projected_value = torch.stack(
                cartesian_projected_losses
            ).mean().item()
            projected_value = torch.stack(projected_losses).mean().item()
            pose_graph_value = torch.stack(pose_graph_losses).mean().item()
            internal_value = torch.stack(internal_losses).mean().item()
            records.append({
                "sample_id": str(batch.pdb_ids[0]),
                "n_residues": int(batch.n_residues[0]),
                "endpoint_max_abs_error": float(endpoint_error),
                "pure_pep_interior": float(pure_value),
                "cartesian_pep_interior": float(cartesian_value),
                "cartesian_delta_vs_pure": float(cartesian_value - pure_value),
                "cartesian_projected_pep_interior": float(
                    cartesian_projected_value
                ),
                "cartesian_projected_delta_vs_cartesian": float(
                    cartesian_projected_value - cartesian_value
                ),
                "cartesian_projection_rms_translation": float(
                    torch.stack(cartesian_projection_rms).mean().item()
                ),
                "cartesian_projection_max_translation": float(
                    torch.stack(cartesian_projection_max).max().item()
                ),
                "projected_pep_interior": float(projected_value),
                "projected_delta_vs_pure": float(projected_value - pure_value),
                "projection_rms_translation": float(
                    torch.stack(projection_rms).mean().item()
                ),
                "projection_max_translation": float(
                    torch.stack(projection_max).max().item()
                ),
                "pose_graph_pep_interior": float(pose_graph_value),
                "pose_graph_delta_vs_pure": float(pose_graph_value - pure_value),
                "pose_graph_rotation_rms": float(
                    torch.stack(pose_graph_rotation_rms).mean().item()
                ),
                "pose_graph_translation_rms": float(
                    torch.stack(pose_graph_translation_rms).mean().item()
                ),
                "pose_graph_rotation_max": float(
                    torch.stack(pose_graph_rotation_max).max().item()
                ),
                "pose_graph_translation_max": float(
                    torch.stack(pose_graph_translation_max).max().item()
                ),
                "chain_internal_pep_interior": float(internal_value),
                "chain_internal_delta_vs_pure": float(internal_value - pure_value),
                "chain_internal_cn_max_error": float(
                    torch.stack(internal_cn_errors).max().item()
                    if internal_cn_errors
                    else 0.0
                ),
            })

    output = {
        "checkpoint": args.checkpoint,
        "valid_samples_file": args.valid_samples_file,
        "n_path_steps": args.n_path_steps,
        "samples": len(records),
        "summary": {
            key: summarize(records, key)
            for key in (
                "endpoint_max_abs_error",
                "pure_pep_interior",
                "cartesian_pep_interior",
                "cartesian_delta_vs_pure",
                "cartesian_projected_pep_interior",
                "cartesian_projected_delta_vs_cartesian",
                "cartesian_projection_rms_translation",
                "cartesian_projection_max_translation",
                "projected_pep_interior",
                "projected_delta_vs_pure",
                "projection_rms_translation",
                "projection_max_translation",
                "pose_graph_pep_interior",
                "pose_graph_delta_vs_pure",
                "pose_graph_rotation_rms",
                "pose_graph_translation_rms",
                "pose_graph_rotation_max",
                "pose_graph_translation_max",
                "chain_internal_pep_interior",
                "chain_internal_delta_vs_pure",
                "chain_internal_cn_max_error",
            )
        },
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
