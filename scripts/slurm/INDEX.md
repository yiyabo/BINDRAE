# Slurm Launcher Index

Top-level launchers are the currently supported entrypoints. Superseded
experiment variants are archived under `scripts/slurm/archive/`.

## Stage-1 Diagnostics

- `audit_ligand_sensitive_dataset.sh` - ligand-sensitive dataset and contact-label audit.
- `diagnose_stage1_prior.sh` - Stage-1 prior diagnostics.
- `diagnose_stage1_postcand4safe.sh` - posterior/candidate diagnostic with ligand controls.
- `build_teacher_posterior_labels_1gpu.sh` - Stage-1-v2 teacher-posterior label export smoke/full wrapper.
- `audit_stage1v2_posterior_1gpu.sh` - Stage-1-v2 best-checkpoint posterior audit wrapper.
- `export_stage1v2_posterior_cache_1gpu.sh` - Stage-1-v2 student posterior cache export wrapper.

## Stage-1 Training

- `train_stage1_geom_baseprior.sh` - base geometry prior anchor.
- `train_stage1_posterior_candidate_screen4_safe.sh` - retained posterior/candidate screening baseline.
- `train_stage1_typed_candidate_energy_smoke.sh` - fast typed-energy smoke test.
- `train_stage1_typed_candidate_energy_screen8.sh` - active 8-GPU typed-energy screening run.
- `train_stage1v2_posterior_4gpu.sh` - trains the teacher-distilled Stage-1-v2 posterior student.

## Stage-2

- `train_stage2_smoke.sh` - Stage-2 smoke.
- `train_stage2_stage1v2_posterior_smoke_1gpu.sh` - Stage-2 smoke with Stage-1-v2 posterior scalar features.
- `train_stage2_stage1v2_posterior_ablation_4gpu.sh` - Stage-2 multi-GPU ablation with Stage-1-v2 posterior scalar features.
- `train_stage2_ddp.sh` - Stage-2 DDP training.
- `train_stage2_lc_pgbf_confirm_4gpu.sh` - current LC-PGBF v1 ablation/confirmation launcher.
- `export_oracle_motion_features_1gpu.sh` - OracleMotion-UB feature export and direct oracle-apply audit wrapper.
- `train_stage2_oracle_motion_smoke_1gpu.sh` - Stage-2 smoke with OracleMotion-UB matched/zero/shuffled controls.
- `train_stage2_oracle_motion_ablation_4gpu.sh` - active Stage-2 OracleMotion / ESM last-K / REPA ablation launcher; legacy filename, defaults to 2xA100 on the current cluster and supports matched plus REPA-target-shuffled controls.
- `evaluate_stage2_transition_paths_1gpu.sh` - residue-level transition evaluator.
- `evaluate_stage2_trajectory_reliability_1gpu.sh` - internal physical/contact trajectory reliability benchmark against cubic interpolation.
- `generate_stage2_trajectories_1gpu.sh` - trajectory export wrapper for visualization.

## Transitional Experiment Launchers

These remain for reproducibility but are not the preferred starting point for
new Stage-1-v2 work:

- `train_change_prediction_*.sh`
- `train_chi_head_1gpu.sh`
- `train_interaction_prior_4gpu.sh`
- `train_ligand_chi_probe_4gpu.sh`
- `train_pocket_relax_v2_4gpu.sh`
- `train_stage1_contrastive*.sh`
- `train_stage1_ligand_*`
- `train_stage1_typed_*`
- `diagnose_delta_z_predictor.sh`
- `diagnose_stage2_gradients_1gpu.sh`

## Archive

- `archive/legacy_stage1/` - older Stage-1 launchers kept for reproducibility, not active use.
