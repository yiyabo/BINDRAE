#!/usr/bin/env python3
"""Export versioned atom14 path candidates for independent physical scoring.

This script is the boundary between the BINDRAE inference environment and the
separate BINDRAE-MD/OpenMM environment.  It deliberately exports coordinates
and immutable identity metadata instead of importing OpenMM into model
inference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FLASH_IPA_PATH = PROJECT_ROOT / "vendor" / "flash_ipa" / "src"
if FLASH_IPA_PATH.exists() and str(FLASH_IPA_PATH) not in sys.path:
    sys.path.insert(0, str(FLASH_IPA_PATH))

import scripts.evaluate_stage2_transition_paths as eval_paths  # noqa: E402
from src.data.residue_identity import (  # noqa: E402
    load_residue_keys,
    residue_identity_hash,
    residue_keys_to_array,
)
from src.data.openmm_gate0 import (  # noqa: E402
    PATH_CANDIDATE_SCHEMA_VERSION,
    reconstruct_peptide_carbonyl_oxygen,
)
from src.stage1.models.fk_openfold import create_openfold_fk  # noqa: E402
from src.stage2.datasets import create_stage2_dataloader  # noqa: E402
from src.stage2.models import TorsionFlowNet  # noqa: E402


SCHEMA_VERSION = PATH_CANDIDATE_SCHEMA_VERSION


def build_arg_parser() -> argparse.ArgumentParser:
    parser = eval_paths.build_arg_parser()
    parser.description = __doc__
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--candidate_label", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.set_defaults(batch_size=1, num_workers=0)
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError("Only scalar tensors may be serialized as diagnostics")
        return value.detach().cpu().item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_residue_axis(data_dir: Path, sample_id: str):
    path = data_dir / "samples" / sample_id / "torsion_apo.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical torsion cache: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        residue_keys = load_residue_keys(loaded)
    if residue_keys is None:
        raise ValueError(f"{path} has no canonical residue_keys")
    return residue_keys


def _decode_path(fk_module, batch, rigids_list, chi_list, times, n_residues: int):
    positions: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for rigid, chi, t_value in zip(rigids_list, chi_list, times):
        atom14 = eval_paths.torsions_to_atom14(
            fk_module,
            eval_paths.interpolate_backbone_torsions(batch, float(t_value)),
            chi,
            rigid,
            batch.aatype,
        )
        frame_mask = (
            atom14["atom14_mask"].bool() & batch.node_mask.unsqueeze(-1).bool()
        )
        positions.append(
            atom14["atom14_pos"][0, :n_residues].detach().cpu().numpy().astype(np.float32)
        )
        masks.append(
            frame_mask[0, :n_residues].detach().cpu().numpy().astype(np.bool_)
        )
    return np.stack(positions, axis=0), np.stack(masks, axis=0)


def _assert_endpoint_contract(
    atom14_pos: np.ndarray,
    batch,
    n_residues: int,
) -> Tuple[float, float]:
    """Validate exact frame translation and bound FK idealization error.

    Stage-2 endpoints are exact in the frame-plus-chi state.  OpenFold FK uses
    ideal residue geometry, so its reconstructed N/C atoms need not reproduce
    the observed PDB bond geometry exactly even when the endpoint frame is
    unchanged.  CA is the frame translation and is therefore the strict
    coordinate-level endpoint contract; the full N/CA/C discrepancy is kept as
    a reported representation diagnostic.
    """
    predicted = atom14_pos[[0, -1], :n_residues, :3]
    expected = np.stack(
        [
            np.stack(
                [
                    batch.N_apo[0, :n_residues].detach().cpu().numpy(),
                    batch.Ca_apo[0, :n_residues].detach().cpu().numpy(),
                    batch.C_apo[0, :n_residues].detach().cpu().numpy(),
                ],
                axis=1,
            ),
            np.stack(
                [
                    batch.N_holo[0, :n_residues].detach().cpu().numpy(),
                    batch.Ca_holo[0, :n_residues].detach().cpu().numpy(),
                    batch.C_holo[0, :n_residues].detach().cpu().numpy(),
                ],
                axis=1,
            ),
        ],
        axis=0,
    )
    valid = (
        batch.node_mask[0, :n_residues].detach().cpu().numpy().astype(np.bool_)
        & batch.bb_mask[0, :n_residues].all(dim=-1).detach().cpu().numpy().astype(np.bool_)
    )
    if not valid.any():
        raise ValueError("Path candidate has no valid endpoint backbone residues")
    error = np.linalg.norm(predicted[:, valid] - expected[:, valid], axis=-1)
    ca_maximum = float(error[..., 1].max())
    backbone_maximum = float(error.max())
    if ca_maximum > 1e-4:
        raise ValueError(
            "Endpoint frame-translation contract failed: "
            f"max CA error={ca_maximum:.6g} A"
        )
    if backbone_maximum > 0.5:
        raise ValueError(
            "Endpoint FK representation is inconsistent with the observed backbone: "
            f"max N/CA/C error={backbone_maximum:.6g} A"
        )
    return ca_maximum, backbone_maximum


def path_product_state_arrays(
    rigids_list,
    chi_list,
    n_residues: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serialize the exact per-frame product-manifold state used by Stage-2."""
    if len(rigids_list) != len(chi_list) or not rigids_list:
        raise ValueError("Rigid and chi path frame counts must agree and be non-empty")
    rotations = np.stack(
        [
            rigid.get_rots()
            .get_rot_mats()[0, :n_residues]
            .detach()
            .cpu()
            .numpy()
            for rigid in rigids_list
        ],
        axis=0,
    ).astype(np.float32)
    translations = np.stack(
        [
            rigid.get_trans()[0, :n_residues].detach().cpu().numpy()
            for rigid in rigids_list
        ],
        axis=0,
    ).astype(np.float32)
    chi = np.stack(
        [value[0, :n_residues].detach().cpu().numpy() for value in chi_list],
        axis=0,
    ).astype(np.float32)
    expected_frames = len(rigids_list)
    if rotations.shape != (expected_frames, n_residues, 3, 3):
        raise ValueError(f"Unexpected rigid rotation path shape: {rotations.shape}")
    if translations.shape != (expected_frames, n_residues, 3):
        raise ValueError(f"Unexpected rigid translation path shape: {translations.shape}")
    if chi.shape != (expected_frames, n_residues, 4):
        raise ValueError(f"Unexpected chi path shape: {chi.shape}")
    return rotations, translations, chi


def build_candidate_payload(
    *,
    args,
    checkpoint_config,
    batch,
    rigids_list,
    chi_list,
    times,
    correction,
    fk_module,
) -> Dict[str, np.ndarray]:
    sample_id = str(batch.pdb_ids[0])
    n_residues = int(batch.n_residues[0])
    residue_keys = _canonical_residue_axis(Path(args.data_dir), sample_id)
    if len(residue_keys) != n_residues:
        raise ValueError(
            f"{sample_id} residue key count {len(residue_keys)} != batch count {n_residues}"
        )
    expected_hash = str(batch.residue_identity_hashes[0])
    actual_hash = residue_identity_hash(residue_keys)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{sample_id} residue identity hash mismatch: {actual_hash} != {expected_hash}"
        )

    atom14_pos, atom14_mask = _decode_path(
        fk_module, batch, rigids_list, chi_list, times, n_residues
    )
    atom14_pos, rebuilt_carbonyl_oxygen_count = reconstruct_peptide_carbonyl_oxygen(
        atom14_pos,
        atom14_mask,
        batch.node_mask[0, :n_residues].detach().cpu().numpy(),
        batch.peptide_bond_mask[0, : max(n_residues - 1, 0)]
        .detach()
        .cpu()
        .numpy(),
    )
    endpoint_ca_max_error, endpoint_backbone_representation_error = (
        _assert_endpoint_contract(atom14_pos, batch, n_residues)
    )
    rigid_rotation, rigid_translation, chi_radians = path_product_state_arrays(
        rigids_list, chi_list, n_residues
    )
    times_array = np.asarray(times, dtype=np.float32)
    if times_array.ndim != 1 or times_array.size != atom14_pos.shape[0]:
        raise ValueError("Path time and coordinate frame counts disagree")
    if not np.isclose(times_array[0], 0.0) or not np.isclose(times_array[-1], 1.0):
        raise ValueError("Path candidate must contain exact t=0 and t=1 endpoints")
    if np.any(np.diff(times_array) <= 0.0):
        raise ValueError("Path candidate times must be strictly increasing")

    path_mode = eval_paths.resolve_path_parameterization(args, checkpoint_config)
    candidate_label = str(args.candidate_label or path_mode)
    ligand_mask = batch.lig_mask[0].detach().cpu().numpy().astype(np.bool_)
    payload = {
        "schema_version": np.array(SCHEMA_VERSION),
        "sample_id": np.array(sample_id),
        "candidate_label": np.array(candidate_label),
        "path_parameterization": np.array(path_mode),
        "checkpoint": np.array(str(args.checkpoint)),
        "checkpoint_config_json": np.array(
            json.dumps(_jsonable(vars(checkpoint_config)), sort_keys=True)
        ),
        "correction_diagnostics_json": np.array(
            json.dumps(_jsonable(correction), sort_keys=True)
        ),
        "coordinate_decoder": np.array("openfold_fk_plus_peptide_oxygen_v1"),
        "rebuilt_carbonyl_oxygen_count": np.array(
            rebuilt_carbonyl_oxygen_count, dtype=np.int32
        ),
        "n_residues": np.array(n_residues, dtype=np.int32),
        "n_frames": np.array(times_array.size, dtype=np.int32),
        "times": times_array,
        "atom14_pos_angstrom": atom14_pos,
        "atom14_mask": atom14_mask,
        "rigid_rotation_matrix": rigid_rotation,
        "rigid_translation_angstrom": rigid_translation,
        "chi_radians": chi_radians,
        "aatype": batch.aatype[0, :n_residues].detach().cpu().numpy().astype(np.int16),
        "node_mask": batch.node_mask[0, :n_residues].detach().cpu().numpy().astype(np.bool_),
        "residue_keys": residue_keys_to_array(residue_keys),
        "residue_identity_hash": np.array(actual_hash),
        "sequence": np.array(str(batch.sequences[0])),
        "ligand_pos_angstrom": batch.lig_points[0][ligand_mask]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "endpoint_ca_max_error_angstrom": np.array(
            endpoint_ca_max_error, dtype=np.float32
        ),
        "endpoint_backbone_representation_max_error_angstrom": np.array(
            endpoint_backbone_representation_error, dtype=np.float32
        ),
    }
    return payload


def export_sample(
    args,
    checkpoint_config,
    model,
    fk_module,
    batch,
    n_steps: int,
    interaction_settings: Dict[str, object],
    stage1v2_settings: Dict[str, object],
    integration_clips: Dict[str, float],
    output_dir: Path,
) -> Dict[str, object]:
    sample_id = str(batch.pdb_ids[0])
    output_path = output_dir / f"{safe_sample_id(sample_id)}.npz"
    if args.skip_existing and output_path.is_file():
        with np.load(output_path, allow_pickle=False) as loaded:
            if str(np.asarray(loaded["schema_version"]).item()) != SCHEMA_VERSION:
                raise ValueError(f"Existing path candidate has the wrong schema: {output_path}")
            if str(np.asarray(loaded["sample_id"]).item()) != sample_id:
                raise ValueError(f"Existing path candidate has the wrong sample ID: {output_path}")
        return {"sample_id": sample_id, "path": str(output_path), "status": "exists"}

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
    rigids_list, chi_list, times, correction = eval_paths.construct_path(
        args,
        checkpoint_config,
        model,
        batch,
        rigids_apo,
        rigids_holo,
        n_steps,
        prior_features,
        gate_context,
        integration_clips,
        fk_module=fk_module,
    )
    payload = build_candidate_payload(
        args=args,
        checkpoint_config=checkpoint_config,
        batch=batch,
        rigids_list=rigids_list,
        chi_list=chi_list,
        times=times,
        correction=correction,
        fk_module=fk_module,
    )
    np.savez_compressed(output_path, **payload)
    return {
        "sample_id": sample_id,
        "path": str(output_path),
        "status": "written",
        "n_residues": int(payload["n_residues"].item()),
        "n_frames": int(payload["n_frames"].item()),
        "endpoint_ca_max_error_angstrom": float(
            payload["endpoint_ca_max_error_angstrom"].item()
        ),
        "endpoint_backbone_representation_max_error_angstrom": float(
            payload["endpoint_backbone_representation_max_error_angstrom"].item()
        ),
    }


def main() -> None:
    args = parse_args()
    if int(args.batch_size) != 1:
        raise ValueError("Path candidate export requires --batch_size 1")
    if int(args.num_shards) <= 0:
        raise ValueError("--num_shards must be positive")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else (
        output_dir / f"manifest_shard{int(args.shard_index):03d}.json"
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
    integration_clips = eval_paths.resolve_integration_clips(args, checkpoint_config)
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
    if n_steps <= 0:
        raise ValueError("n_integration_steps must be positive")

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
        raise ValueError("No samples remain after path-export filtering")

    records: List[Dict[str, object]] = []
    selected = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="Path export", ncols=120)):
            if batch_index % int(args.num_shards) != int(args.shard_index):
                continue
            if args.max_samples is not None and selected >= int(args.max_samples):
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
                    integration_clips,
                    output_dir,
                )
            )
            selected += 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint),
        "candidate_label": str(
            args.candidate_label
            or eval_paths.resolve_path_parameterization(args, checkpoint_config)
        ),
        "path_parameterization": eval_paths.resolve_path_parameterization(
            args, checkpoint_config
        ),
        "valid_samples_file": args.valid_samples_file,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**summary, "records": len(records)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
