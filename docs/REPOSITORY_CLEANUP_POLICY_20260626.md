# Repository Cleanup Policy

Date: 2026-06-26

The repository is now in a transition from exploratory validation to full-scale
Stage-2 training and downstream comparison. Cleanup should reduce navigation
noise without deleting experiment evidence.

## Cleanup Principle

Prefer these actions in order:

1. Update indexes and runbooks.
2. Archive superseded documents and launchers.
3. Remove generated local noise.
4. Delete source files only after a replacement path is documented and tested.

Do not remove code just because it is not part of the current mainline. Many
Stage-1 scripts are reproducibility anchors for previous negative controls.

## Current Mainline

Docs:

- `docs/CURRENT_PROJECT_STATUS_20260622.md`
- `docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md`
- `docs/RAEV2_REPA_STAGE2_ENHANCEMENT_PLAN_20260625.md`
- `docs/STAGE2_ENDPOINT_BOUNDARY_STATUS_20260708.md`
- `docs/FULL_SCALE_TRAINING_AND_EVALUATION_RUNBOOK_20260626.md`
- `docs/COMPETITOR_DATASETS_METRICS_20260624.md`
- `docs/STAGE1V2_TEACHER_POSTERIOR_PLAN_20260622.md`
- `docs/STAGE1_STAGE2_PRIOR_INTERFACE_DECISION.md`

Stage-2 code:

- `src/stage2/models/torsion_flow.py`
- `src/stage2/training/config.py`
- `src/stage2/training/trainer.py`
- `src/stage2/datasets/dataset_stage2.py`
- `src/stage2/modules/`
- `src/stage2/evaluation/`

Stage-1-v2 code:

- `src/stage1/posterior_v2/`
- `scripts/train_stage1v2_posterior.py`
- `scripts/build_teacher_posterior_labels.py`
- `scripts/export_stage1v2_posterior_cache.py`
- `scripts/audit_stage1v2_posterior.py`

Stage-2 scripts:

- `scripts/train_stage2.py`
- `scripts/export_oracle_motion_features.py`
- `scripts/evaluate_stage2_transition_paths.py`
- `scripts/evaluate_stage2_trajectory_reliability.py`
- `scripts/generate_stage2_trajectories.py`

Slurm launchers:

- `scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh`
- `scripts/slurm/export_oracle_motion_features_1gpu.sh`
- `scripts/slurm/evaluate_stage2_transition_paths_1gpu.sh`
- `scripts/slurm/evaluate_stage2_trajectory_reliability_1gpu.sh`
- `scripts/slurm/generate_stage2_trajectories_1gpu.sh`
- `scripts/slurm/train_stage1v2_posterior_4gpu.sh`

## Archive Candidates

Archive only after the current REPA comparison and full-scale launcher are
stable:

- change-prediction RAE scripts;
- chi-head-only scripts;
- ligand discriminator/guidance launchers;
- typed candidate energy launchers;
- old contrastive Stage-1 launchers;
- one-off gradient and geometry diagnostics.

Recommended archive destination:

```text
scripts/archive/20260626_pre_fullscale_stage2/
scripts/slurm/archive/20260626_pre_fullscale_stage2/
docs/archive/20260626_pre_fullscale_stage2/
```

Keep a short `README.md` in each archive folder explaining why the files moved.

2026-06-27 cleanup note: exploratory Python and Slurm entrypoints were moved
non-destructively to:

```text
scripts/archive/20260627_pre_fullscale_stage2/
scripts/slurm/archive/20260627_pre_fullscale_stage2/
```

The top-level script surface should now be treated as the supported entrypoint
set for multi-agent work. Archived scripts are reproducibility records and may
contain stale defaults.

## Safe Immediate Deletions

These are generated local noise and can be removed whenever they appear:

- `__pycache__/`
- `.pytest_cache/`
- `.DS_Store`
- `*.pyc`
- temporary `*.tmp`, `*.bak`, `*~`

They are already covered by `.gitignore`.

## Do Not Commit By Default

These should remain local/generated unless explicitly needed:

- `logs/`
- `checkpoints/`
- `processed_data/`
- `data/`
- `viewer/` outputs
- `.omo/`
- presentation binaries under `docs/RP/`

## Commit Boundary Recommendation

Before the next clean commit, split changes into at least two commits:

1. Stage-2 ESM/REPA implementation and validation launchers.
2. Documentation and cleanup indexes/runbooks.

Avoid mixing large Stage-1 historical rewrites with the Stage-2 full-scale
training baseline.
