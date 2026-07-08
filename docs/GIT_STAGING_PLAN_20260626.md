# Git Staging Plan

Date: 2026-06-26

This plan splits the current dirty worktree into reviewable commits. It is a
staging guide only; do not use `git add .` for this branch.

## 2026-07-08 Update: Boundary/Endpoint Worktree Split

The worktree now contains Stage-2 exact-endpoint boundary work, external path
baselines, evaluator updates, documentation, and sampler/dataset reliability
fixes. Keep these split. Do not fold all current dirty files into the older
OracleMotion/ESM/REPA commit.

Recommended current split:

1. Stage-2 boundary and path-evaluator implementation:
   - `src/stage2/training/config.py`
   - `src/stage2/training/trainer.py`
   - `scripts/train_stage2.py`
   - `scripts/evaluate_stage2_transition_paths.py`
   - `scripts/evaluate_stage2_trajectory_reliability.py`
   - matching Stage-2 Slurm launchers
2. External baseline implementation:
   - `scripts/run_anm_baseline.py`
   - `scripts/run_adaptive_anm_baseline.py`
   - `scripts/run_ca_morph_baseline.py`
   - `scripts/run_ebdims2_baseline.py`
   - `scripts/evaluate_ebdims2_ca_paths.py`
   - matching CPU Slurm launchers
3. Documentation and experiment ledgers:
   - `docs/STAGE2_ENDPOINT_BOUNDARY_STATUS_20260708.md`
   - `docs/STAGE2_COMPARISON_METRICS_AND_BASELINES_20260701.md`
   - `docs/PATH_BASELINE_LITERATURE_SCAN_20260630.md`
   - `docs/BINDRAE_POSITIONING_AND_BENCHMARK_STRATEGY_20260629.md`
   - `docs/CODEX_HANDOFF_FULLSCALE_20260630.md`
   - `docs/INDEX.md`
4. Dataset/sampler reliability and feature-cache fixes:
   - `src/stage1/datasets/samplers.py`
   - `src/stage2/datasets/dataset_stage2.py`
   - `src/stage2/datasets/features.py`
   - `scripts/export_oracle_motion_features.py`
   - matching export Slurm launcher

Current decision from the 2026-07-08 endpoint work:

- keep free-flow as learned-path teacher / upper-bound evidence;
- keep `pure_bridge` as the clean exact-endpoint baseline;
- keep ligand-clearance and hard-negative clearance as negative ablations;
- stop expanding clearance/anchor sweeps unless a new model target is defined.

## Commit 1: Local Hygiene And Agent Guides

Purpose: keep local generated state out of git and make the active operating
rules discoverable.

Files:

- `.gitignore`
- `AGENTS.md`
- `scripts/AGENTS.md`
- `src/stage1/AGENTS.md`
- `src/stage2/AGENTS.md`
- `docs/PROJECT_CLEANUP_LOG_20260622.md`
- `docs/REPOSITORY_CLEANUP_POLICY_20260626.md`

Validation:

- `git status --ignored --short`
- confirm `.omo/`, `.playwright-mcp/`, caches, logs, checkpoints, and datasets
  remain untracked or ignored.

2026-06-27 update:

- `AGENTS.md`, `scripts/AGENTS.md`, `scripts/INDEX.md`,
  `scripts/slurm/INDEX.md`, `docs/REPOSITORY_CLEANUP_POLICY_20260626.md`, and
  `docs/FULL_SCALE_TRAINING_AND_EVALUATION_RUNBOOK_20260626.md` now describe the
  active OracleMotion/ESM/REPA full-scale path.
- Exploratory Python and Slurm entrypoints were moved to
  `scripts/archive/20260627_pre_fullscale_stage2/` and
  `scripts/slurm/archive/20260627_pre_fullscale_stage2/`.
- Stage-2 metric hygiene added `val_total_no_repa`; place that trainer change
  with the Stage-2 OracleMotion/ESM/REPA commit, not with documentation-only
  cleanup.

## Commit 2: Current Documentation And Archive Move

Purpose: preserve the Stage-1-v2 pivot and Stage-2 OracleMotion baseline
evidence without keeping obsolete docs in the first reading path.

Files:

- `docs/INDEX.md`
- `docs/archive/README.md`
- `docs/CURRENT_PROJECT_STATUS_20260622.md`
- `docs/STAGE1V2_TEACHER_POSTERIOR_PLAN_20260622.md`
- `docs/TODO_STAGE1V2_POSTERIOR.md`
- `docs/STAGE1_INTERACTION_PRIOR_PIVOT_20260617.md`
- `docs/LC_PGBF_ARCHITECTURE_PLAN_20260618.md`
- `docs/LC_PGBF_STAGE2_EXPERIMENT_RECORD_20260620.md`
- `docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md`
- `docs/RAEV2_REPA_STAGE2_ENHANCEMENT_PLAN_20260625.md`
- `docs/FULL_SCALE_TRAINING_AND_EVALUATION_RUNBOOK_20260626.md`
- `docs/COMPETITOR_DATASETS_METRICS_20260624.md`
- `docs/archive/20260622_pre_stage1v2_pivot/`
- deleted top-level docs that were moved into the archive:
  - `docs/STAGE1_CANDIDATE_RERANKING_FINAL_DECISION_20260511.md`
  - `docs/STAGE1_LIGAND_CAUSALITY_VALIDATION_DECISION_20260511.md`
  - `docs/STAGE1_SCREENING_WORKFLOW.md`
  - `docs/STAGE1_TYPED_CANDIDATE_INTERACTION_ENERGY_PLAN_20260511.md`

Validation:

- inspect `docs/INDEX.md`;
- verify every moved document appears under
  `docs/archive/20260622_pre_stage1v2_pivot/`.

## Commit 3: Stage-2 OracleMotion, ESM Last-K, And REPA

Purpose: capture the current active Stage-2 implementation used by the
OracleMotion baseline and ESM/REPA ablations.

Files:

- `scripts/train_stage2.py`
- `src/stage2/`
- `src/stage2/evaluation/`
- `scripts/export_oracle_motion_features.py`
- `scripts/build_oracle_motion_remainder.py`
- `scripts/evaluate_stage2_transition_paths.py`
- `scripts/evaluate_stage2_trajectory_reliability.py`
- `scripts/evaluate_stage2_path_critic.py`
- `scripts/generate_stage2_trajectories.py`
- `scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh`
- `scripts/slurm/train_stage2_oracle_motion_smoke_1gpu.sh`
- `scripts/slurm/train_stage2_esm_fusion_smoke_1gpu.sh`
- `scripts/slurm/export_oracle_motion_features_1gpu.sh`
- `scripts/slurm/evaluate_stage2_transition_paths_1gpu.sh`
- `scripts/slurm/evaluate_stage2_trajectory_reliability_1gpu.sh`
- `scripts/slurm/generate_stage2_trajectories_1gpu.sh`
- `src/stage2/training/trainer.py` metric hygiene, including validation
  `force_geom=True` and `val_total_no_repa`
- `ablation_subsets/`
- `viewer/stage2_trajectory_viewer/`
- `scripts/INDEX.md`
- `scripts/slurm/INDEX.md`

Validation:

- `python -m py_compile scripts/train_stage2.py`
- `python -m py_compile scripts/export_oracle_motion_features.py`
- `python -m py_compile scripts/evaluate_stage2_transition_paths.py`
- `python -m py_compile scripts/evaluate_stage2_trajectory_reliability.py`
- `python -m py_compile scripts/generate_stage2_trajectories.py`
- `python -m py_compile src/stage2/models/torsion_flow.py`
- `python -m py_compile src/stage2/training/trainer.py`
- `python -m py_compile src/stage2/training/config.py`
- `bash -n` for each staged Stage-2 Slurm launcher.

## Commit 4: Stage-1-v2 Posterior And Historical Diagnostics

Purpose: keep the teacher-distilled posterior path reproducible, separate from
the Stage-2 mainline commit.

Files:

- `src/stage1/posterior_v2/`
- `scripts/train_stage1v2_posterior.py`
- `scripts/build_teacher_posterior_labels.py`
- `scripts/export_stage1v2_posterior_cache.py`
- `scripts/audit_stage1v2_posterior.py`
- `scripts/slurm/train_stage1v2_posterior_4gpu.sh`
- `scripts/slurm/build_teacher_posterior_labels_1gpu.sh`
- `scripts/slurm/export_stage1v2_posterior_cache_1gpu.sh`
- `scripts/slurm/audit_stage1v2_posterior_1gpu.sh`
- Stage-1 source changes that are required by this path:
  - `scripts/train_stage1.py`
  - `src/stage1/data/residue_constants.py`
  - `src/stage1/datasets/`
  - `src/stage1/models/`
  - `src/stage1/modules/`
  - `src/stage1/training/`

Validation:

- `python -m py_compile scripts/train_stage1v2_posterior.py`
- `python -m py_compile scripts/build_teacher_posterior_labels.py`
- `python -m py_compile scripts/export_stage1v2_posterior_cache.py`
- `python -m py_compile scripts/audit_stage1v2_posterior.py`
- `python -m py_compile src/stage1/posterior_v2/model.py`
- `python -m py_compile src/stage1/posterior_v2/trainer.py`
- `bash -n` for each staged Stage-1-v2 Slurm launcher.

## Commit 5: Transitional Stage-1 Experiments

Purpose: preserve reproducibility for earlier failed or exploratory tracks
without mixing them into the current Stage-2 paper path.

Files:

- `scripts/train_change_prediction.py`
- `scripts/train_change_prediction_fast.py`
- `scripts/train_chi_head.py`
- `scripts/train_chi_head_simple.py`
- `scripts/train_interaction_prior.py`
- `scripts/train_ligand_chi_probe.py`
- `scripts/train_pocket_relax_v2.py`
- `scripts/diagnose_delta_z_predictor.py`
- `scripts/diagnose_stage1_prior.py`
- `scripts/audit_apo_holo_ligand_geometry.py`
- `scripts/audit_rotamer_oracle_signal.py`
- `scripts/filter_stage1_strict_samples.py`
- `scripts/precompute_latents.py`
- `scripts/create_full_model_from_fast.py`
- matching `scripts/slurm/` launchers for these experiments.

Recommendation:

- stage this commit only after Commit 3 and Commit 4 are reviewed;
- consider archiving these files under
  `scripts/archive/20260626_pre_fullscale_stage2/` and
  `scripts/slurm/archive/20260626_pre_fullscale_stage2/` before committing if
  they are not expected to be run again.

## Keep Ignored / Do Not Stage

- `logs/`
- `checkpoints/`
- `processed_data/`
- `data/`
- `outputs/`
- `runs/`
- `wandb/`
- `RP/`
- `proposal/`
- `.omo/`
- `.playwright-mcp/`
- Python caches and local tool state.

## Notes

- Prefer `git add <explicit files>` or `git add -p`.
- Do not use `git add .` while transitional Stage-1 experiments remain mixed
  with the active Stage-2 mainline.
- If a Slurm launcher is actively queued or running, avoid editing it in the
  same commit that records its submitted configuration.
