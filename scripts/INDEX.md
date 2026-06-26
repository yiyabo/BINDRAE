# Scripts Index

Top-level `scripts/` keeps active entrypoints and data-prep utilities. Older
helpers and one-off diagnostics are archived under `scripts/archive/`.

## Active Entry Points

- `train_stage1.py` - main Stage-1 training CLI.
- `train_stage2.py` - main Stage-2 training CLI; currently supports OracleMotion features, ESM last-K fusion, and optional REPA-style hidden-state alignment.
- `train_interaction_prior.py` - current LC-PGBF v1 local interaction-prior trainer.
- `train_stage1v2_posterior.py` - trains the Stage-1-v2 teacher-distilled posterior student.
- `audit_stage1v2_posterior.py` - audits a trained Stage-1-v2 posterior checkpoint with threshold, ranking, calibration, and ligand-control metrics.
- `export_stage1v2_posterior_cache.py` - exports per-sample Stage-1-v2 student posterior `.npz` caches for Stage-2 consumption.
- `export_oracle_motion_features.py` - exports OracleMotion-UB apo-to-holo motion features and direct oracle-apply audits.
- `validate_triplets_data.py` - validates Stage-1 triplet datasets.
- `audit_ligand_sensitive_dataset.py` - ligand-sensitive dataset audit.
- `diagnose_stage1_prior.py` - Stage-1 diagnostic entrypoint.
- `summarize_stage1_ligand_causality_validation.py` - validation report summarizer.
- `build_teacher_posterior_labels.py` - Stage-1-v2 teacher-posterior label exporter.

## Active Stage-2 Evaluation / Demo

- `evaluate_stage2_transition_paths.py` - residue-level transition path evaluator.
- `evaluate_stage2_trajectory_reliability.py` - internal trajectory reliability benchmark against cubic SE(3)+chi interpolation.
- `evaluate_stage2_path_critic.py` - Stage-2 path critic/evaluator helper.
- `generate_stage2_trajectories.py` - exports generated Stage-2 trajectory files for visualization.

## Current Stage-2 Run Path

Use the runbook before launching full-scale jobs:

- `../docs/FULL_SCALE_TRAINING_AND_EVALUATION_RUNBOOK_20260626.md`

Current primary launcher:

- `slurm/train_stage2_oracle_motion_ablation_4gpu.sh` - despite the historical filename, this is the active OracleMotion / ESM last-K / REPA ablation launcher and defaults to 2xA100.

## Transitional Experiment Entrypoints

These are useful for reproducing recent pivots, but should be archived once the
Stage-1-v2 posterior encoder is stable.

- `train_change_prediction.py`
- `train_change_prediction_fast.py`
- `train_chi_head.py`
- `train_chi_head_simple.py`
- `train_ligand_chi_probe.py`
- `train_pocket_relax_v2.py`
- `diagnose_delta_z_predictor.py`
- `diagnose_stage2_gradients.py`
- `audit_apo_holo_ligand_geometry.py`
- `audit_rotamer_oracle_signal.py`
- `filter_stage1_strict_samples.py`
- `precompute_latents.py`
- `create_full_model_from_fast.py`

## Data Preparation

- `prepare_casf2016.py`
- `prepare_ligands.py`
- `extract_pockets.py`
- `extract_torsions.py`
- `cache_esm2.py`
- `split_dataset.py`
- `verify_casf2016.py`
- `verify_ligand_consistency.py`

## Alternate Dataset Line

- `download_ahojdb_pdbs.py`
- `extract_ahojdb_torsions.py`
- `prepare_ahojdb_triplets.py`
- `cache_ahojdb_esm2.py`

## Archive

- `archive/legacy_tools/` - older setup helpers and test wrappers.
- `archive/diagnostics/` - one-off audits and lineage/debug scripts.
- `slurm/` - current cluster launchers; see `scripts/slurm/INDEX.md`.
