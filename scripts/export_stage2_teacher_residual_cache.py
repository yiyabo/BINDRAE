#!/usr/bin/env python3
"""Export free-flow teacher residual targets for boundary-residual students.

Each output npz stores the teacher path's interior deviation from the analytic
apo-holo bridge. Stage-2 boundary-residual training can then learn this path
shape while keeping endpoints exact through the residual envelope.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import scripts.evaluate_stage2_transition_paths as eval_paths  # noqa: E402
from src.stage1.models.fk_openfold import create_openfold_fk  # noqa: E402
from src.stage2.datasets import create_stage2_dataloader  # noqa: E402
from src.stage2.models import TorsionFlowNet  # noqa: E402
from src.stage2.modules import rigid_compose, rigid_inverse, se3_log, wrap_to_pi  # noqa: E402


SCHEMA_VERSION = "stage2_teacher_residual_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Free-flow teacher checkpoint")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--trust_prechecked_samples", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")

    parser.add_argument(
        "--path_parameterization",
        default="flow",
        choices=["checkpoint", "flow", "projected_flow", "boundary_residual_v1", "boundary_residual"],
        help="Teacher path construction; use flow for raw free-flow teacher",
    )
    parser.add_argument("--n_integration_steps", type=int, default=None)
    parser.add_argument("--integration_chi_clip", type=float, default=None)
    parser.add_argument("--integration_rot_clip", type=float, default=None)
    parser.add_argument("--integration_trans_clip", type=float, default=None)
    parser.add_argument("--boundary_residual_envelope", default=None, choices=["sin2", "poly"])
    parser.add_argument("--boundary_residual_scale", type=float, default=None)
    parser.add_argument(
        "--terminal_projection_schedule",
        default=None,
        choices=["smoothstep", "smootherstep", "late_smoother", "quadratic"],
    )
    parser.add_argument("--t_min", type=float, default=0.08)
    parser.add_argument("--t_max", type=float, default=0.92)
    parser.add_argument("--motion_eps", type=float, default=1e-4)
    parser.add_argument("--pocket_threshold", type=float, default=0.3)
    parser.add_argument(
        "--ligand_clash_dist",
        type=float,
        default=2.2,
        help="Distance cutoff used to define clash severity for local teacher weighting",
    )
    parser.add_argument(
        "--clash_focus_dist",
        type=float,
        default=3.0,
        help="Only bridge residues closer than this distance receive near-ligand gain weight",
    )
    parser.add_argument(
        "--clash_relief_gain_weight",
        type=float,
        default=0.25,
        help="Weight for teacher-vs-bridge ligand-distance gain near the ligand",
    )
    parser.add_argument(
        "--clash_relief_weight_cap",
        type=float,
        default=5.0,
        help="Upper cap for cached clash-relief weights",
    )

    parser.add_argument("--interaction_prior_ckpt", default=None)
    parser.add_argument(
        "--interaction_prior_feature_mode",
        default=None,
        choices=["none", "prior", "prior_shuffled", "zero", "oracle_contact"],
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
    )
    parser.add_argument("--stage1v2_posterior_cache_dir", default=None)
    parser.add_argument("--stage1v2_posterior_feature_names", default=None)
    parser.add_argument("--stage1v2_posterior_feature_scale", type=float, default=None)
    return parser.parse_args()


def safe_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def residual_to_bridge(teacher_rigids, teacher_chi, bridge_rigids, bridge_chi, chi_mask):
    residual_rot: List[torch.Tensor] = []
    residual_trans: List[torch.Tensor] = []
    residual_chi: List[torch.Tensor] = []
    t_indices: List[int] = []
    for idx, (rigids_t, chi_t, rigids_b, chi_b) in enumerate(
        zip(teacher_rigids, teacher_chi, bridge_rigids, bridge_chi)
    ):
        R_b = rigids_b.get_rots().get_rot_mats()
        t_b = rigids_b.get_trans()
        R_t = rigids_t.get_rots().get_rot_mats()
        trans_t = rigids_t.get_trans()
        R_inv, t_inv = rigid_inverse(R_b, t_b)
        R_delta, t_delta = rigid_compose(R_inv, t_inv, R_t, trans_t)
        xi = se3_log(R_delta, t_delta)
        residual_rot.append(xi[..., :3])
        residual_trans.append(xi[..., 3:])
        residual_chi.append(wrap_to_pi(chi_t - chi_b) * chi_mask.float())
        t_indices.append(idx)
    return (
        torch.stack(residual_rot, dim=0),
        torch.stack(residual_trans, dim=0),
        torch.stack(residual_chi, dim=0),
        t_indices,
    )


def min_sc_ligand_dist_for_state(fk_module, batch, rigids_t, chi_t) -> torch.Tensor:
    atom14_t = eval_paths.torsions_to_atom14(
        fk_module,
        batch.torsion_apo[..., :3],
        chi_t,
        rigids_t,
        batch.aatype,
    )
    return eval_paths.min_sc_ligand_dist(
        atom14_t["atom14_pos"].float(),
        atom14_t["atom14_mask"].bool(),
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
    )


def export_batch(
    args: argparse.Namespace,
    model: TorsionFlowNet,
    fk_module,
    batch,
    n_steps: int,
    interaction_settings: Dict[str, object],
    stage1v2_settings: Dict[str, object],
    integration_clips: Dict[str, float],
    output_dir: Path,
    saved_so_far: int,
) -> List[Dict[str, object]]:
    rigids_apo = eval_paths.build_rigids_from_backbone(
        batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
    )
    rigids_holo = eval_paths.build_rigids_from_backbone(
        batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
    )
    interaction_prior_feature = eval_paths.compute_interaction_prior_feature(
        args, args.config, batch, fk_module, rigids_apo, rigids_holo
    )
    prior_features = eval_paths.combined_prior_features(
        args,
        args.config,
        batch,
        interaction_prior_feature,
        stage1v2_settings,
        interaction_settings,
    )
    gate_context = eval_paths.esm_gate_context(args.config, batch, stage1v2_settings)
    teacher_rigids, teacher_chi, t_list, _ = eval_paths.construct_path(
        args,
        args.config,
        model,
        batch,
        rigids_apo,
        rigids_holo,
        n_steps,
        prior_features,
        gate_context,
        integration_clips,
    )
    bridge_rigids, bridge_chi, bridge_t = eval_paths.pure_bridge_path(
        batch, rigids_apo, rigids_holo, n_steps
    )

    keep = [
        idx for idx, t_val in enumerate(t_list)
        if float(args.t_min) <= float(t_val) <= float(args.t_max)
    ]
    if not keep:
        raise ValueError(f"No path time points in requested t range {args.t_min}-{args.t_max}")

    residual_rot, residual_trans, residual_chi, _ = residual_to_bridge(
        [teacher_rigids[idx] for idx in keep],
        [teacher_chi[idx] for idx in keep],
        [bridge_rigids[idx] for idx in keep],
        [bridge_chi[idx] for idx in keep],
        batch.chi_mask,
    )
    teacher_dist = torch.stack(
        [
            min_sc_ligand_dist_for_state(fk_module, batch, teacher_rigids[idx], teacher_chi[idx])
            for idx in keep
        ],
        dim=0,
    )
    bridge_dist = torch.stack(
        [
            min_sc_ligand_dist_for_state(fk_module, batch, bridge_rigids[idx], bridge_chi[idx])
            for idx in keep
        ],
        dim=0,
    )
    clash_dist = teacher_dist.new_tensor(float(args.ligand_clash_dist))
    focus_dist = teacher_dist.new_tensor(float(args.clash_focus_dist))
    bridge_severity = torch.relu(clash_dist - bridge_dist)
    teacher_severity = torch.relu(clash_dist - teacher_dist)
    severity_relief = torch.relu(bridge_severity - teacher_severity)
    near_ligand_focus = torch.relu(focus_dist - bridge_dist) / focus_dist.clamp(min=1e-6)
    distance_gain = torch.relu(teacher_dist - bridge_dist) * near_ligand_focus
    clash_relief_weight = (
        severity_relief + float(args.clash_relief_gain_weight) * distance_gain
    ).clamp(min=0.0, max=float(args.clash_relief_weight_cap))
    clash_relief_weight = clash_relief_weight * batch.node_mask.unsqueeze(0).float()
    t_values = np.asarray([float(t_list[idx]) for idx in keep], dtype=np.float32)

    records: List[Dict[str, object]] = []
    for b, sample_id in enumerate(batch.pdb_ids):
        if args.max_samples is not None and saved_so_far + len(records) >= int(args.max_samples):
            break
        n_res = int(batch.n_residues[b])
        out_path = output_dir / f"{safe_name(sample_id)}.npz"
        if args.skip_existing and out_path.is_file():
            records.append(
                {
                    "sample_id": sample_id,
                    "path": str(out_path),
                    "relative_path": out_path.name,
                    "n_residues": n_res,
                    "status": "exists",
                }
            )
            continue

        rot_np = tensor_to_np(residual_rot[:, b, :n_res]).astype(np.float32)
        trans_np = tensor_to_np(residual_trans[:, b, :n_res]).astype(np.float32)
        chi_np = tensor_to_np(residual_chi[:, b, :n_res]).astype(np.float32)
        teacher_dist_np = tensor_to_np(teacher_dist[:, b, :n_res]).astype(np.float32)
        bridge_dist_np = tensor_to_np(bridge_dist[:, b, :n_res]).astype(np.float32)
        clash_relief_np = tensor_to_np(clash_relief_weight[:, b, :n_res]).astype(np.float32)
        chi_mask_np = tensor_to_np(batch.chi_mask[b, :n_res]).astype(np.bool_)
        node_mask_np = tensor_to_np(batch.node_mask[b, :n_res]).astype(np.bool_)
        w_res_np = tensor_to_np(batch.w_res[b, :n_res]).astype(np.float32)
        residual_mag = (
            np.linalg.norm(rot_np, axis=-1)
            + np.linalg.norm(trans_np, axis=-1)
            + np.sum(np.abs(chi_np) * chi_mask_np[None, :, :], axis=-1)
        )
        motion_active = np.max(residual_mag, axis=0) > float(args.motion_eps)
        pocket_mask = w_res_np > float(args.pocket_threshold)

        np.savez_compressed(
            out_path,
            schema_version=np.array(SCHEMA_VERSION),
            source=np.array("free_flow_teacher"),
            teacher_checkpoint=np.array(str(args.checkpoint)),
            teacher_path_parameterization=np.array(str(args.path_parameterization)),
            sample_id=np.array(str(sample_id)),
            n_residues=np.array(n_res, dtype=np.int32),
            t_values=t_values,
            residual_rot=rot_np,
            residual_trans=trans_np,
            residual_chi=chi_np,
            teacher_min_ligand_dist=teacher_dist_np,
            bridge_min_ligand_dist=bridge_dist_np,
            clash_relief_weight=clash_relief_np,
            node_mask=node_mask_np,
            chi_mask=chi_mask_np,
            w_res=w_res_np,
            motion_active=motion_active.astype(np.bool_),
            pocket_mask=pocket_mask.astype(np.bool_),
        )
        valid_mag = residual_mag[:, node_mask_np]
        valid_relief = clash_relief_np[:, node_mask_np]
        records.append(
            {
                "sample_id": sample_id,
                "path": str(out_path),
                "relative_path": out_path.name,
                "n_residues": n_res,
                "n_t": int(t_values.size),
                "mean_residual_norm": float(np.mean(valid_mag)) if valid_mag.size else 0.0,
                "mean_clash_relief_weight": float(np.mean(valid_relief)) if valid_relief.size else 0.0,
                "clash_relief_residues": int((clash_relief_np.max(axis=0) > 1e-4).sum()),
                "motion_active_residues": int(motion_active.sum()),
                "status": "written",
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.t_min < args.t_max <= 1.0):
        raise ValueError("--t_min/--t_max must satisfy 0 <= min < max <= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", argparse.Namespace())
    args.config = config
    model_config, interaction_settings, stage1v2_settings = eval_paths.build_model_config_for_checkpoint(
        args, config, args.split
    )
    integration_clips = eval_paths.resolve_integration_clips(args, config)
    n_steps = int(args.n_integration_steps or getattr(config, "n_integration_steps", 5))
    if n_steps <= 0:
        raise ValueError(f"n_integration_steps must be > 0, got {n_steps}")

    model = TorsionFlowNet(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    fk_module = create_openfold_fk().to(device)
    fk_module.eval()

    loader = create_stage2_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        require_nma=bool(getattr(config, "use_nma", False)),
        valid_samples_file=args.valid_samples_file,
        trust_prechecked_samples=args.trust_prechecked_samples,
        stage1v2_posterior_cache_dir=stage1v2_settings["cache_dir"],
        stage1v2_posterior_feature_mode=stage1v2_settings["mode"],
        stage1v2_posterior_feature_names=stage1v2_settings["names_raw"],
        esm_num_layers=int(getattr(config, "esm_num_layers", 1)),
    )

    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Export teacher residuals", ncols=120)):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            if args.max_samples is not None and len(records) >= int(args.max_samples):
                break
            batch = eval_paths.batch_to_device(batch, device)
            new_records = export_batch(
                args,
                model,
                fk_module,
                batch,
                n_steps,
                interaction_settings,
                stage1v2_settings,
                integration_clips,
                output_dir,
                len(records),
            )
            records.extend(new_records)

    manifest_t_values: List[float] = []
    if records:
        with np.load(records[0]["path"], allow_pickle=False) as data:
            manifest_t_values = [float(v) for v in data["t_values"]]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split": str(args.split),
        "valid_samples_file": args.valid_samples_file,
        "path_parameterization": str(args.path_parameterization),
        "n_integration_steps": n_steps,
        "t_min": float(args.t_min),
        "t_max": float(args.t_max),
        "t_values": manifest_t_values,
        "ligand_clash_dist": float(args.ligand_clash_dist),
        "clash_focus_dist": float(args.clash_focus_dist),
        "clash_relief_gain_weight": float(args.clash_relief_gain_weight),
        "clash_relief_weight_cap": float(args.clash_relief_weight_cap),
        "samples": len(records),
        "output_dir": str(output_dir),
        "records": records,
    }
    text = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
