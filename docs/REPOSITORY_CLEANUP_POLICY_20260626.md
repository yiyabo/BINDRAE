# Repository Cleanup Policy

Date: 2026-06-26; updated 2026-07-10

The repository is in a transition from exploratory Stage-2 variants to the
endpoint-exact asynchronous phase-normal bridge. Cleanup should keep this
method easy to find without deleting experiment evidence.

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

- `docs/BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md`
- `docs/CURRENT_PROJECT_STATUS_20260710.md`
- `docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md`
- `docs/COMPETITOR_DATASETS_METRICS_20260624.md`
- `docs/PATH_BASELINE_LITERATURE_SCAN_20260630.md`
- `docs/STAGE2_COMPARISON_METRICS_AND_BASELINES_20260701.md`
- `docs/STAGE1_STAGE2_PRIOR_INTERFACE_DECISION.md`

Stage-2 code:

- `src/stage2/models/torsion_flow.py`
- `src/stage2/training/config.py`
- `src/stage2/training/trainer.py`
- `src/stage2/datasets/dataset_stage2.py`
- `src/stage2/modules/`
- `src/stage2/modules/phase_residual.py`
- `src/stage2/evaluation/`

Future Stage-1 deployment code:

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

## Archive Posture

The 2026-07-10 method cleanup moved superseded top-level documents and the old
proposal to:

```text
docs/archive/20260710_pre_phase_normal_bridge/
```

Continue to archive, rather than delete, reproducibility anchors such as:

- change-prediction RAE scripts;
- chi-head-only scripts;
- ligand discriminator/guidance launchers;
- typed candidate energy launchers;
- old contrastive Stage-1 launchers;
- one-off gradient and geometry diagnostics.

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

Before the next clean commit, split changes into at least three commits:

1. Canonical residue/cache/export reliability fixes.
2. Phase-normal Stage-2 implementation, evaluator, and tests.
3. Documentation and archive moves.

Avoid mixing future stochastic or Stage-1 work into the deterministic APNB
implementation commit.
