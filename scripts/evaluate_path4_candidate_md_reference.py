#!/usr/bin/env python3
"""Evaluate one exported Path-4 candidate against audited MD references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FLASH_IPA_PATH = PROJECT_ROOT / "vendor" / "flash_ipa" / "src"
if FLASH_IPA_PATH.exists() and str(FLASH_IPA_PATH) not in sys.path:
    sys.path.insert(0, str(FLASH_IPA_PATH))

import scripts.evaluate_stage2_md_reference_paths as md_eval  # noqa: E402
import scripts.evaluate_stage2_transition_paths as base  # noqa: E402
from src.data.openmm_gate0 import load_path_candidate  # noqa: E402
from src.stage2.modules import so3_log, wrap_to_pi  # noqa: E402


SCHEMA_VERSION = "bindrae_path4_candidate_md_reference_eval_v1"
METRICS = (
    "md_path_product_rmse",
    "md_path_translation_mae_a",
    "md_path_rotation_mae_rad",
    "md_path_chi_mae_rad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--valid-samples-file", type=Path, required=True)
    parser.add_argument("--md-reference-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--endpoint-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def candidate_state_tensors(candidate, device: torch.device):
    if not candidate.has_product_state:
        raise ValueError(
            "MD Product-RMSE evaluation requires a v3 candidate with explicit "
            "rigid_rotation_matrix, rigid_translation_angstrom, and chi_radians"
        )
    rotation = torch.as_tensor(
        candidate.rigid_rotation_matrix, dtype=torch.float32, device=device
    )
    translation = torch.as_tensor(
        candidate.rigid_translation_angstrom, dtype=torch.float32, device=device
    )
    chi = torch.as_tensor(candidate.chi_radians, dtype=torch.float32, device=device)
    rigids = [
        base.rt_to_rigid(rotation[index].unsqueeze(0), translation[index].unsqueeze(0))
        for index in range(candidate.n_frames)
    ]
    chi_path = [chi[index].unsqueeze(0) for index in range(candidate.n_frames)]
    return rigids, chi_path


def validate_candidate_endpoints(
    candidate,
    predicted_rigids,
    predicted_chi,
    batch,
    rigids_apo,
    rigids_holo,
    *,
    tolerance: float,
) -> Dict[str, float]:
    if tolerance <= 0.0:
        raise ValueError("endpoint tolerance must be positive")
    node_mask = batch.node_mask[0].bool()
    chi_mask = batch.chi_mask[0].bool() & node_mask[:, None]
    max_rotation = 0.0
    max_translation = 0.0
    max_chi = 0.0
    for predicted, target, predicted_angle, target_angle in (
        (
            predicted_rigids[0],
            rigids_apo,
            predicted_chi[0],
            batch.torsion_apo[..., 3:7],
        ),
        (
            predicted_rigids[-1],
            rigids_holo,
            predicted_chi[-1],
            batch.torsion_holo[..., 3:7],
        ),
    ):
        predicted_rotation, predicted_translation = base.rigid_to_rt(predicted)
        target_rotation, target_translation = base.rigid_to_rt(target)
        relative = target_rotation.transpose(-2, -1) @ predicted_rotation
        rotation_error = torch.linalg.vector_norm(so3_log(relative), dim=-1)[0]
        translation_error = torch.linalg.vector_norm(
            predicted_translation - target_translation, dim=-1
        )[0]
        chi_error = wrap_to_pi(predicted_angle - target_angle).abs()[0]
        max_rotation = max(max_rotation, float(rotation_error[node_mask].max().item()))
        max_translation = max(
            max_translation, float(translation_error[node_mask].max().item())
        )
        if chi_mask.any():
            max_chi = max(max_chi, float(chi_error[chi_mask].max().item()))
    maximum = max(max_rotation, max_translation, max_chi)
    if maximum > tolerance:
        raise ValueError(
            f"Candidate endpoint state mismatch for {candidate.sample_id}: "
            f"rotation={max_rotation:.6g}, translation={max_translation:.6g} A, "
            f"chi={max_chi:.6g}, tolerance={tolerance:.6g}"
        )
    return {
        "rotation_max_rad": max_rotation,
        "translation_max_angstrom": max_translation,
        "chi_max_rad": max_chi,
    }


def validate_md_reference(data, path: Path, batch, sample_id: str) -> None:
    if str(data["schema_version"].item()) != "md_phase_normal_v1":
        raise ValueError(f"Unsupported MD cache schema in {path}")
    phase_target_mode = (
        str(data["phase_target_mode"].item())
        if "phase_target_mode" in data
        else "inferred"
    )
    if phase_target_mode != "inferred":
        raise ValueError(f"MD benchmark requires inferred phase targets: {path}")
    cached_sample = str(data["sample_id"].item())
    if not (
        cached_sample == sample_id
        or cached_sample.startswith(f"{sample_id}__silver_r")
    ):
        raise ValueError(
            f"MD reference sample mismatch in {path}: {cached_sample!r} != "
            f"{sample_id!r}"
        )
    expected_n = int(batch.n_residues[0])
    if int(data["n_residues"].item()) != expected_n:
        raise ValueError(f"Residue count mismatch for {path}")
    expected_hash = str(batch.residue_identity_hashes[0])
    cached_hash = (
        str(data["residue_identity_hash"].item())
        if "residue_identity_hash" in data
        else ""
    )
    if cached_hash != expected_hash:
        raise ValueError(
            f"Residue identity mismatch for {path}: "
            f"cache={cached_hash!r}, batch={expected_hash!r}"
        )


def aggregate_records(records: List[Mapping[str, object]]) -> Dict[str, object]:
    replica_macro = {
        metric: md_eval.finite_mean(
            float(record["metrics"][metric]) for record in records
        )
        for metric in METRICS
    }
    return {
        "replica_macro": replica_macro,
        "system_macro": dict(replica_macro),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    candidate = load_path_candidate(args.candidate)
    if not candidate.has_product_state:
        raise ValueError(
            f"Candidate {args.candidate} uses {candidate.schema_version}; "
            "MD Product-RMSE evaluation requires the v3 product-state contract"
        )
    loader = base.create_stage2_dataloader(
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
        esm_num_layers=1,
    )
    if len(loader.dataset) != 1:
        raise ValueError(
            "Candidate MD-reference evaluation requires exactly one production sample; "
            f"loader returned {len(loader.dataset)}"
        )

    records: List[Dict[str, object]] = []
    endpoint_diagnostics: Dict[str, float] | None = None
    with torch.no_grad():
        for batch in loader:
            batch = base.batch_to_device(batch, device)
            sample_id = str(batch.pdb_ids[0])
            if sample_id != candidate.sample_id:
                raise ValueError(
                    f"Candidate/loader sample mismatch: {candidate.sample_id} != {sample_id}"
                )
            if int(batch.n_residues[0]) != candidate.n_residues:
                raise ValueError("Candidate/loader residue count mismatch")
            if str(batch.residue_identity_hashes[0]) != candidate.residue_identity_hash:
                raise ValueError("Candidate/loader residue identity mismatch")
            rigids_apo = base.build_rigids_from_backbone(
                batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
            )
            rigids_holo = base.build_rigids_from_backbone(
                batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
            )
            predicted_rigids, predicted_chi = candidate_state_tensors(
                candidate, device
            )
            endpoint_diagnostics = validate_candidate_endpoints(
                candidate,
                predicted_rigids,
                predicted_chi,
                batch,
                rigids_apo,
                rigids_holo,
                tolerance=float(args.endpoint_tolerance),
            )
            reference_paths = md_eval.replica_paths(
                args.md_reference_cache_dir, sample_id
            )
            if not reference_paths:
                raise FileNotFoundError(
                    f"No MD reference replicas for {sample_id} under "
                    f"{args.md_reference_cache_dir}"
                )
            for path in reference_paths:
                with np.load(path, allow_pickle=False) as data:
                    validate_md_reference(data, path, batch, sample_id)
                    target_rigids, target_chi, arrays = md_eval.reconstruct_md_reference(
                        batch,
                        rigids_apo,
                        rigids_holo,
                        data,
                        candidate.times,
                    )
                    metrics = md_eval.path_error_metrics(
                        predicted_rigids,
                        predicted_chi,
                        target_rigids,
                        target_chi,
                        arrays,
                        data,
                    )
                    records.append(
                        {
                            "sample_id": sample_id,
                            "reference_id": str(data["sample_id"].item()),
                            "reference_path": str(path),
                            "metrics": metrics,
                        }
                    )
    if not records or endpoint_diagnostics is None:
        raise RuntimeError("Candidate MD-reference evaluation produced no records")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "candidate": str(args.candidate),
        "candidate_schema_version": candidate.schema_version,
        "candidate_label": candidate.candidate_label,
        "path_parameterization": candidate.path_parameterization,
        "sample_id": candidate.sample_id,
        "md_reference_cache_dir": str(args.md_reference_cache_dir),
        "valid_samples_file": str(args.valid_samples_file),
        "systems": 1,
        "replicas": len(records),
        "endpoint_diagnostics": endpoint_diagnostics,
        "aggregate": aggregate_records(records),
        "records": records,
    }
    text = json.dumps(
        md_eval.json_safe(result), indent=2, sort_keys=True, allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
