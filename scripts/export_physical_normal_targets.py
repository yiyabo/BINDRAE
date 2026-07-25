#!/usr/bin/env python3
"""Export deterministic physical-normal pseudo-targets for Path-4 distillation.

The frozen learned warp supplies residue phases. A small endpoint-exact,
block-normal physical optimization supplies translation-only corrections. The
cache stores the projected correction before the endpoint-zero envelope so the
student preserves the same boundary contract at inference time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

import scripts.evaluate_stage2_transition_paths as eval_paths  # noqa: E402
from src.stage1.models.fk_openfold import create_openfold_fk  # noqa: E402
from src.stage2.datasets import create_stage2_dataloader  # noqa: E402
from src.stage2.models import TorsionFlowNet  # noqa: E402
from src.stage2.modules import PhysicalPathOptimizationResult  # noqa: E402


SCHEMA_VERSION = "md_phase_normal_v1"
SOURCE_VERSION = "physical_normal_teacher_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = eval_paths.build_arg_parser()
    parser.description = __doc__
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.set_defaults(
        split="train",
        batch_size=1,
        num_workers=0,
        path_parameterization="phase_physical_normal_v1",
        n_integration_steps=20,
        physical_normal_iterations=8,
        physical_normal_learning_rate=0.05,
        physical_normal_envelope="poly",
        physical_normal_projection_mode="block",
        physical_normal_components="translation",
        physical_normal_max_metric_norm=1.0,
        physical_normal_max_clash_atoms=256,
        physical_normal_weight_peptide=1.0,
        physical_normal_weight_protein_clash=1.0,
        physical_normal_weight_ligand_clash=1.0,
        physical_normal_weight_contact_anchor=0.25,
        physical_normal_weight_distance_anchor=1.0,
        physical_normal_weight_residual=20.0,
        physical_normal_weight_temporal=0.2,
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def resolve_checkpoint_value(args, config, name: str, default):
    value = getattr(args, name, None)
    return value if value is not None else getattr(config, name, default)


def physical_config_dict(args) -> Dict[str, object]:
    return {
        "iterations": int(args.physical_normal_iterations),
        "learning_rate": float(args.physical_normal_learning_rate),
        "envelope": str(args.physical_normal_envelope),
        "projection_mode": str(args.physical_normal_projection_mode),
        "components": str(args.physical_normal_components),
        "max_metric_norm": float(args.physical_normal_max_metric_norm),
        "gradient_clip": float(args.physical_normal_gradient_clip),
        "protein_clash_distance": float(args.physical_normal_protein_clash_dist),
        "ligand_clash_distance": float(args.physical_normal_ligand_clash_dist),
        "max_clash_atoms": int(args.physical_normal_max_clash_atoms),
        "weight_peptide": float(args.physical_normal_weight_peptide),
        "weight_protein_clash": float(args.physical_normal_weight_protein_clash),
        "weight_ligand_clash": float(args.physical_normal_weight_ligand_clash),
        "weight_contact_anchor": float(args.physical_normal_weight_contact_anchor),
        "weight_distance_anchor": float(args.physical_normal_weight_distance_anchor),
        "weight_residual": float(args.physical_normal_weight_residual),
        "weight_temporal": float(args.physical_normal_weight_temporal),
        "optimizer": str(getattr(args, "physical_normal_optimizer", "adam")),
        "num_starts": int(getattr(args, "physical_normal_num_starts", 1)),
        "route_seed_scale": float(
            getattr(args, "physical_normal_route_seed_scale", 0.0)
        ),
        "route_seed_rank": int(
            getattr(args, "physical_normal_route_seed_rank", 2)
        ),
        "route_seed_smoothing_steps": int(
            getattr(args, "physical_normal_route_seed_smoothing_steps", 2)
        ),
        "route_seed": int(getattr(args, "physical_normal_route_seed", 20260720)),
        "frame_aggregation": str(
            getattr(args, "physical_normal_frame_aggregation", "mean")
        ),
        "frame_softmax_beta": float(
            getattr(args, "physical_normal_frame_softmax_beta", 10.0)
        ),
        "line_search_steps": int(
            getattr(args, "physical_normal_line_search_steps", 8)
        ),
        "line_search_shrink": float(
            getattr(args, "physical_normal_line_search_shrink", 0.5)
        ),
        "acceptance_tolerance": float(
            getattr(args, "physical_normal_acceptance_tolerance", 1e-8)
        ),
    }


def build_cache_payload(
    args,
    checkpoint_config,
    sample_id: str,
    n_residues: int,
    batch,
    result: PhysicalPathOptimizationResult,
    tau_values: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Create one trainer-compatible physical teacher cache payload."""
    n_interior = len(result.times) - 2
    if n_interior <= 0:
        raise ValueError("physical teacher path has no interior frames")
    if result.projected_rigid.shape[0] != n_interior:
        raise ValueError("projected residual/time grid mismatch")
    if tau_values.shape[0] != len(result.times):
        raise ValueError("phase/time grid mismatch")

    node_mask = tensor_to_numpy(batch.node_mask[0, :n_residues]).astype(np.bool_)
    chi_mask = tensor_to_numpy(batch.chi_mask[0, :n_residues]).astype(np.bool_)
    residual_rigid = tensor_to_numpy(
        result.projected_rigid[:, 0, :n_residues]
    ).astype(np.float32)
    residual_chi = tensor_to_numpy(
        result.projected_chi[:, 0, :n_residues]
    ).astype(np.float32)
    applied_rigid = tensor_to_numpy(
        result.applied_rigid[:, 0, :n_residues]
    ).astype(np.float32)
    tau_target = tensor_to_numpy(
        tau_values[1:-1, 0, :n_residues]
    ).astype(np.float32)
    valid = np.broadcast_to(node_mask[None, :], (n_interior, n_residues)).copy()
    confidence = valid.astype(np.float32)

    rotation_scale = float(
        resolve_checkpoint_value(
            args, checkpoint_config, "phase_residual_rotation_metric_scale", 1.0
        )
    )
    translation_scale = float(
        resolve_checkpoint_value(
            args,
            checkpoint_config,
            "phase_residual_translation_metric_scale",
            1.0,
        )
    )
    chi_scale = float(
        resolve_checkpoint_value(
            args, checkpoint_config, "phase_residual_chi_metric_scale", 1.0
        )
    )
    return {
        "schema_version": np.array(SCHEMA_VERSION),
        "source": np.array(SOURCE_VERSION),
        "teacher_checkpoint": np.array(str(args.checkpoint)),
        "sample_id": np.array(str(sample_id)),
        "n_residues": np.array(n_residues, dtype=np.int32),
        "bridge_mode": np.array(
            str(
                resolve_checkpoint_value(
                    args,
                    checkpoint_config,
                    "phase_residual_bridge_mode",
                    "cartesian_backbone",
                )
            )
        ),
        "residual_envelope": np.array(str(args.physical_normal_envelope)),
        "phase_target_mode": np.array("learned_teacher"),
        "normal_projection_mode": np.array(
            str(args.physical_normal_projection_mode)
        ),
        "rotation_metric_scale": np.array(rotation_scale, dtype=np.float32),
        "translation_metric_scale": np.array(
            translation_scale, dtype=np.float32
        ),
        "chi_metric_scale": np.array(chi_scale, dtype=np.float32),
        "t_values": np.asarray(result.times[1:-1], dtype=np.float32),
        "tau_target": tau_target,
        "residual_rot": residual_rigid[..., :3],
        "residual_trans": residual_rigid[..., 3:],
        "residual_chi": residual_chi,
        "applied_residual_trans": applied_rigid[..., 3:],
        "residual_valid_mask": valid,
        "residual_confidence": confidence,
        "node_mask": node_mask,
        "chi_mask": chi_mask,
        "physical_config_json": np.array(
            json.dumps(physical_config_dict(args), sort_keys=True)
        ),
        "teacher_diagnostics_json": np.array(
            json.dumps(result.diagnostics, sort_keys=True)
        ),
    }


def export_sample(
    args,
    checkpoint_config,
    model,
    fk_module,
    batch,
    n_steps: int,
    interaction_settings: Dict[str, object],
    stage1v2_settings: Dict[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    sample_id = str(batch.pdb_ids[0])
    n_residues = int(batch.n_residues[0])
    output_path = output_dir / f"{safe_sample_id(sample_id)}.npz"
    if args.skip_existing and output_path.is_file():
        return {
            "sample_id": sample_id,
            "path": str(output_path),
            "relative_path": output_path.name,
            "n_residues": n_residues,
            "status": "exists",
        }

    rigids_apo = eval_paths.build_rigids_from_backbone(
        batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
    )
    rigids_holo = eval_paths.build_rigids_from_backbone(
        batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
    )
    interaction_feature = eval_paths.compute_interaction_prior_feature(
        args, checkpoint_config, batch, fk_module, rigids_apo, rigids_holo
    )
    prior_features = eval_paths.combined_prior_features(
        args,
        checkpoint_config,
        batch,
        interaction_feature,
        stage1v2_settings,
        interaction_settings,
    )
    gate_context = eval_paths.esm_gate_context(
        checkpoint_config, batch, stage1v2_settings
    )
    result, tau_values = eval_paths.phase_physical_normal_result(
        args,
        checkpoint_config,
        model,
        batch,
        rigids_apo,
        rigids_holo,
        n_steps,
        prior_features,
        gate_context,
        fk_module,
    )
    payload = build_cache_payload(
        args,
        checkpoint_config,
        sample_id,
        n_residues,
        batch,
        result,
        tau_values,
    )
    np.savez_compressed(output_path, **payload)
    valid = payload["residual_valid_mask"]
    raw_norm = np.linalg.norm(payload["residual_trans"], axis=-1)[valid]
    applied_norm = np.linalg.norm(payload["applied_residual_trans"], axis=-1)[valid]
    return {
        "sample_id": sample_id,
        "path": str(output_path),
        "relative_path": output_path.name,
        "n_residues": n_residues,
        "n_t": int(payload["t_values"].size),
        "raw_translation_rms": float(np.sqrt(np.mean(raw_norm**2))),
        "applied_translation_rms": float(np.sqrt(np.mean(applied_norm**2))),
        "objective_improvement": float(
            result.diagnostics["objective_improvement"]
        ),
        "normal_parallel_cos_abs": float(
            result.diagnostics["normal_parallel_cos_abs"]
        ),
        "status": "written",
    }


def main() -> None:
    args = parse_args()
    if int(args.batch_size) != 1:
        raise ValueError("physical target export requires --batch_size 1")
    if args.path_parameterization != "phase_physical_normal_v1":
        raise ValueError("physical target export requires phase_physical_normal_v1")
    if args.physical_normal_projection_mode != "block":
        raise ValueError("canonical physical targets require block projection")
    if args.physical_normal_components != "translation":
        raise ValueError("canonical physical targets require translation-only correction")
    if int(args.num_shards) <= 0:
        raise ValueError("--num_shards must be positive")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_dir / f"manifest_shard{args.shard_index:03d}.json"
    )
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", SimpleNamespace())
    args.config = checkpoint_config
    model_config, interaction_settings, stage1v2_settings = (
        eval_paths.build_model_config_for_checkpoint(
            args, checkpoint_config, args.split
        )
    )
    model = TorsionFlowNet(model_config).to(device)
    eval_paths.load_model_state_allow_timewarp_head(
        model, checkpoint["model_state_dict"]
    )
    model.eval()
    fk_module = create_openfold_fk().to(device)
    fk_module.eval()
    n_steps = int(
        args.n_integration_steps
        or getattr(checkpoint_config, "n_integration_steps", 5)
    )
    loader = create_stage2_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        require_nma=bool(getattr(checkpoint_config, "use_nma", False)),
        valid_samples_file=args.valid_samples_file,
        trust_prechecked_samples=args.trust_prechecked_samples,
        stage1v2_posterior_cache_dir=stage1v2_settings["cache_dir"],
        stage1v2_posterior_feature_mode=stage1v2_settings["mode"],
        stage1v2_posterior_feature_names=stage1v2_settings["names_raw"],
        esm_num_layers=int(getattr(checkpoint_config, "esm_num_layers", 1)),
    )
    if len(loader.dataset) == 0:
        raise ValueError("physical target export dataset is empty")

    records: List[Dict[str, object]] = []
    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"Physical targets {args.shard_index}/{args.num_shards}", ncols=120)
    ):
        if args.max_batches is not None and batch_index >= int(args.max_batches):
            break
        if batch_index % int(args.num_shards) != int(args.shard_index):
            continue
        if args.max_samples is not None and len(records) >= int(args.max_samples):
            break
        batch = eval_paths.batch_to_device(batch, device)
        records.append(
            export_sample(
                args,
                checkpoint_config,
                model,
                fk_module,
                batch,
                n_steps,
                interaction_settings,
                stage1v2_settings,
                output_dir,
            )
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_VERSION,
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split": str(args.split),
        "valid_samples_file": args.valid_samples_file,
        "n_integration_steps": n_steps,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "physical_config": physical_config_dict(args),
        "samples": len(records),
        "output_dir": str(output_dir),
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "samples": len(records),
                "shard": f"{args.shard_index}/{args.num_shards}",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
