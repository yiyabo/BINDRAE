#!/usr/bin/env python3
"""Reconstruct atom14 Path-3 or surrogate Path-4 candidates from an audited cache.

This engineering utility avoids a redundant FlashIPA forward pass when a
versioned physical-teacher cache already stores the frozen Path-3 ``tau``
field.  The optional paired mode composes the cache's already-applied
translation correction; it never re-optimizes or consumes raw residual labels.
"""

from __future__ import annotations

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
FLASH_IPA_PATH = PROJECT_ROOT / "vendor" / "flash_ipa" / "src"
if FLASH_IPA_PATH.exists() and str(FLASH_IPA_PATH) not in sys.path:
    sys.path.insert(0, str(FLASH_IPA_PATH))

import scripts.evaluate_stage2_transition_paths as eval_paths  # noqa: E402
from flash_ipa.rigid import Rigid, Rotation  # noqa: E402
from scripts.export_stage2_path_candidates import (  # noqa: E402
    SCHEMA_VERSION,
    build_candidate_payload,
    safe_sample_id,
)
from src.stage1.models.fk_openfold import create_openfold_fk  # noqa: E402
from src.stage2.datasets import create_stage2_dataloader  # noqa: E402
from src.stage2.modules.se3 import rigid_compose, se3_exp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="processed_data/triplets")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--valid-samples-file", required=True)
    parser.add_argument("--phase-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-label", default="cached_frozen_path3")
    parser.add_argument(
        "--candidate-mode",
        choices=["path3", "cached_applied_translation"],
        default="path3",
    )
    parser.add_argument("--phase-tau-postprocess", choices=["none", "cummax"], default="cummax")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"Expected scalar cache metadata, got {array.shape}")
    return str(array.item())


def load_cached_tau(
    path: Path,
    *,
    sample_id: str,
    n_residues: int,
    device: torch.device,
    candidate_mode: str,
):
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached Path-3 phase: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "sample_id",
            "n_residues",
            "t_values",
            "tau_target",
            "bridge_mode",
            "phase_target_mode",
        }
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"Cached Path-3 phase misses fields: {missing}")
        cached_sample = _scalar_text(loaded["sample_id"])
        cached_n = int(np.asarray(loaded["n_residues"]).item())
        if cached_sample != sample_id or cached_n != n_residues:
            raise ValueError(
                f"Cached phase identity mismatch: ({cached_sample}, {cached_n}) "
                f"!= ({sample_id}, {n_residues})"
            )
        t_values = np.asarray(loaded["t_values"], dtype=np.float64)
        tau = np.asarray(loaded["tau_target"], dtype=np.float32)
        bridge_mode = _scalar_text(loaded["bridge_mode"])
        phase_target_mode = _scalar_text(loaded["phase_target_mode"])
        applied_translation = None
        physical_config_json = None
        teacher_diagnostics_json = None
        if candidate_mode == "cached_applied_translation":
            residual_required = {
                "applied_residual_trans",
                "physical_config_json",
                "teacher_diagnostics_json",
            }
            residual_missing = sorted(residual_required - set(loaded.files))
            if residual_missing:
                raise ValueError(
                    f"Cached physical correction misses fields: {residual_missing}"
                )
            applied_translation = np.asarray(
                loaded["applied_residual_trans"], dtype=np.float32
            )
            physical_config_json = _scalar_text(loaded["physical_config_json"])
            teacher_diagnostics_json = _scalar_text(
                loaded["teacher_diagnostics_json"]
            )
    if t_values.ndim != 1 or tau.shape != (t_values.size, n_residues):
        raise ValueError(
            f"Cached phase grid mismatch: t={t_values.shape}, tau={tau.shape}"
        )
    if np.any(t_values <= 0.0) or np.any(t_values >= 1.0) or np.any(np.diff(t_values) <= 0.0):
        raise ValueError("Cached phase interior times must be strictly increasing in (0, 1)")
    if applied_translation is not None and applied_translation.shape != (
        t_values.size,
        n_residues,
        3,
    ):
        raise ValueError(
            "Cached applied translation grid mismatch: "
            f"{applied_translation.shape} != {(t_values.size, n_residues, 3)}"
        )
    tau_values = [torch.zeros((1, n_residues), device=device)]
    tau_values.extend(
        torch.as_tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
        for row in tau
    )
    tau_values.append(torch.ones((1, n_residues), device=device))
    applied_values = None
    if applied_translation is not None:
        applied_values = [
            torch.as_tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
            for row in applied_translation
        ]
    return (
        [0.0, *t_values.tolist(), 1.0],
        tau_values,
        bridge_mode,
        phase_target_mode,
        applied_values,
        physical_config_json,
        teacher_diagnostics_json,
    )


def _apply_cached_translation_residual(
    base_rigid: Rigid, applied_translation: torch.Tensor
) -> Rigid:
    rigid_residual = torch.cat(
        [torch.zeros_like(applied_translation), applied_translation], dim=-1
    )
    residual_rotation, residual_translation = se3_exp(rigid_residual)
    path_rotation, path_translation = rigid_compose(
        base_rigid.get_rots().get_rot_mats(),
        base_rigid.get_trans(),
        residual_rotation,
        residual_translation,
    )
    return Rigid(rots=Rotation(rot_mats=path_rotation), trans=path_translation)


def reconstruct_path(
    batch,
    times,
    tau_values,
    bridge_mode: str,
    applied_translation_values=None,
):
    rigids_apo = eval_paths.build_rigids_from_backbone(
        batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
    )
    rigids_holo = eval_paths.build_rigids_from_backbone(
        batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
    )
    rigids = [rigids_apo]
    chi = [batch.torsion_apo[..., 3:7]]
    node_mask = batch.node_mask.bool()
    for interior_index, (t_value, tau) in enumerate(
        zip(times[1:-1], tau_values[1:-1])
    ):
        effective_tau = torch.where(
            node_mask,
            tau,
            torch.full_like(tau, float(t_value)),
        )
        rigid_t, chi_t = eval_paths.phase_interpolate_endpoints_tensor(
            batch,
            rigids_apo,
            rigids_holo,
            effective_tau,
            bridge_mode,
        )
        if applied_translation_values is not None:
            rigid_t = _apply_cached_translation_residual(
                rigid_t, applied_translation_values[interior_index]
            )
        rigids.append(rigid_t)
        chi.append(chi_t)
    rigids.append(rigids_holo)
    chi.append(batch.torsion_holo[..., 3:7])
    return rigids, chi


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", SimpleNamespace())
    loader = create_stage2_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        require_nma=False,
        valid_samples_file=args.valid_samples_file,
        trust_prechecked_samples=True,
        stage1v2_posterior_cache_dir=None,
        stage1v2_posterior_feature_mode="none",
        esm_num_layers=int(getattr(checkpoint_config, "esm_num_layers", 1)),
    )
    if len(loader.dataset) == 0:
        raise ValueError("No samples remain after cached-phase export filtering")
    fk_module = create_openfold_fk().to(device)
    fk_module.eval()
    args.data_dir = str(args.data_dir)
    args.candidate_label = str(args.candidate_label)
    args.path_parameterization = (
        "phase_physical_normal_v1"
        if args.candidate_mode == "cached_applied_translation"
        else "phase_block_orthogonal_residual_v2"
    )
    args.checkpoint = str(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Cached path export", ncols=120):
            if len(records) >= args.max_samples:
                break
            batch = eval_paths.batch_to_device(batch, device)
            sample_id = str(batch.pdb_ids[0])
            n_residues = int(batch.n_residues[0])
            output_path = args.output_dir / f"{safe_sample_id(sample_id)}.npz"
            if args.skip_existing and output_path.is_file():
                records.append(
                    {"sample_id": sample_id, "path": str(output_path), "status": "exists"}
                )
                continue
            (
                times,
                tau_values,
                bridge_mode,
                phase_target_mode,
                applied_translation_values,
                physical_config_json,
                teacher_diagnostics_json,
            ) = load_cached_tau(
                args.phase_cache_dir / f"{safe_sample_id(sample_id)}.npz",
                sample_id=sample_id,
                n_residues=n_residues,
                device=device,
                candidate_mode=args.candidate_mode,
            )
            tau_values = eval_paths.postprocess_phase_tau_values(
                tau_values, args.phase_tau_postprocess
            )
            rigids, chi = reconstruct_path(
                batch,
                times,
                tau_values,
                bridge_mode,
                applied_translation_values,
            )
            consumes_physical_residual = applied_translation_values is not None
            payload = build_candidate_payload(
                args=args,
                checkpoint_config=checkpoint_config,
                batch=batch,
                rigids_list=rigids,
                chi_list=chi,
                times=times,
                correction={
                    "source": "cached_phase_only",
                    "phase_cache": str(args.phase_cache_dir),
                    "phase_target_mode": phase_target_mode,
                    "phase_tau_postprocess": args.phase_tau_postprocess,
                    "physical_residual_consumed": consumes_physical_residual,
                    "physical_config_json": physical_config_json,
                    "teacher_diagnostics_json": teacher_diagnostics_json,
                },
                fk_module=fk_module,
            )
            np.savez_compressed(output_path, **payload)
            records.append(
                {
                    "sample_id": sample_id,
                    "path": str(output_path),
                    "status": "written",
                    "n_frames": int(payload["n_frames"].item()),
                    "endpoint_ca_max_error_angstrom": float(
                        payload["endpoint_ca_max_error_angstrom"].item()
                    ),
                    "endpoint_backbone_representation_max_error_angstrom": float(
                        payload[
                            "endpoint_backbone_representation_max_error_angstrom"
                        ].item()
                    ),
                }
            )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source": "audited_cached_path3_phase",
        "checkpoint": str(args.checkpoint),
        "phase_cache_dir": str(args.phase_cache_dir),
        "candidate_mode": args.candidate_mode,
        "phase_tau_postprocess": args.phase_tau_postprocess,
        "physical_residual_consumed": (
            args.candidate_mode == "cached_applied_translation"
        ),
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**summary, "records": len(records)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
