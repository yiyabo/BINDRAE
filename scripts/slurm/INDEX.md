# Slurm Launcher Index

Top-level launchers are the currently supported entrypoints. Superseded
experiment variants are archived under `scripts/slurm/archive/`. For new runs,
prefer the launchers listed here and treat archived files as reproducibility
records, not templates.

## Stage-1 Diagnostics

- `audit_ligand_sensitive_dataset.sh` - ligand-sensitive dataset and contact-label audit.
- `diagnose_stage1_prior.sh` - Stage-1 prior diagnostics.
- `build_teacher_posterior_labels_1gpu.sh` - Stage-1-v2 teacher-posterior label export smoke/full wrapper.
- `audit_stage1v2_posterior_1gpu.sh` - Stage-1-v2 best-checkpoint posterior audit wrapper.
- `export_stage1v2_posterior_cache_1gpu.sh` - Stage-1-v2 student posterior cache export wrapper.

## Stage-1 Training

- `train_stage1v2_posterior_4gpu.sh` - trains the teacher-distilled Stage-1-v2 posterior student.
- `cache_esm_lastk_1gpu.sh` - exports last-K ESM feature caches needed by ESM fusion runs.

## Stage-2

- `train_stage2_smoke.sh` - Stage-2 smoke.
- `train_stage2_ddp.sh` - Stage-2 DDP training.
- `export_oracle_motion_features_1gpu.sh` - OracleMotion-UB feature export and direct oracle-apply audit wrapper.
- `train_stage2_oracle_motion_smoke_1gpu.sh` - Stage-2 smoke with OracleMotion-UB matched/zero/shuffled controls.
- `train_stage2_oracle_motion_ablation_4gpu.sh` - active Stage-2 OracleMotion / ESM last-K / REPA ablation launcher; legacy filename, defaults to 2xA100 on the current cluster and supports matched plus REPA-target-shuffled controls.
- `evaluate_stage2_transition_paths_1gpu.sh` - residue-level transition evaluator.
- `evaluate_stage2_trajectory_reliability_1gpu.sh` - internal physical/contact trajectory reliability benchmark against cubic interpolation.
- `generate_stage2_trajectories_1gpu.sh` - trajectory export wrapper for visualization.

## Full-Scale Submission Notes

- The active OracleMotion launcher defaults to 2 A100 through its SBATCH header.
  If requesting 4-6 GPUs, pass a matching Slurm resource request and set
  `NPROC_PER_NODE` to the same value.
- Run `PRECHECK_ONLY=1` before a long submission to validate cache shape,
  subset files, amino-acid typing, node masks, and ESM last-K availability.
- For REPA comparisons, compare `val_total_no_repa` plus endpoint/contact/path
  metrics; `val_repa` alone is only a target-fitting diagnostic.

## Archive

- `archive/legacy_stage1/` - older Stage-1 launchers kept for reproducibility, not active use.
- `archive/20260627_pre_fullscale_stage2/` - older exploratory launchers moved
  out of the main path before 60k/full-scale Stage-2 training.
