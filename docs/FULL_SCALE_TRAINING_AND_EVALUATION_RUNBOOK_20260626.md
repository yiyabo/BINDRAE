# Full-Scale Training And Evaluation Runbook

Date: 2026-06-26

This runbook defines the current operational path after the OracleMotion
baseline and the first ESM last-K / REPA enhancement checks.

## Current Stable Baseline

The stable Stage-2 starting point is:

```text
OracleMotion conditioning
+ ESM last-7 softmax fusion
+ no REPA
```

Known comparable baseline:

```text
train4096 / val512 / e5 / batch_per_gpu=4 / 2xA100
cache: oracle_motion_train4096_esm7sync_20260625_scale
val:   oracle_motion_val512_esm7sync_20260625_scale
```

The active REPA same-budget comparison adds:

```text
matched REPA target
target-residue-shuffled REPA control
```

Treat REPA as a hypothesis until the matched run improves over both no-REPA and
target-shuffled REPA under the same budget.

## Full-Scale Training Gate

Do not start full-scale training until the current same-budget comparison has
at least one clean epoch and no startup/runtime contract failure.

Proceed if:

- DDP starts cleanly on 2 A100.
- `metrics.jsonl` contains finite `train_total`, `val_total`, and `val_repa`.
- matched REPA is not clearly worse than no-REPA.
- target-shuffled REPA does not reproduce any matched gain.

If matched REPA is neutral or harmful, keep full-scale training on the no-REPA
last-K baseline and move REPA to a separate ablation branch.

## DDP Graph Contract

Stage-2 training builds one loss from several model forwards: the reference CFM
forward plus differentiable path-integration forwards. In DDP mode, optional or
discarded heads must still remain connected to the autograd graph through
zero-valued dependencies.

Current examples:

- REPA projection is returned only for the reference forward, but is kept as a
  zero-valued dependency on integration forwards.
- the final FlashIPA backbone-update head predicts an updated rigid that
  Stage-2 does not consume directly, so its update is also kept as a zero-valued
  dependency.

Do not remove these zero-valued dependencies as cosmetic cleanup unless the
trainer is refactored to use a different DDP strategy. The REPA DDP smoke
`s2repa_smoke`, job `137985`, completed successfully with this contract.

## Recommended Full-Scale Budget

Default resource request:

```text
2-3 x A100
batch_per_gpu=4
bf16
geom_every=1
n_integration_steps=3
n_geom_steps=4
```

Use 4 A100 only when the cluster has contiguous free GPUs or when explicitly
running a formal long training job. The current launcher filename may say
`4gpu`, but its default request is intentionally 2 A100.

For formal 60k-scale runs, 4-6 A100 is acceptable when resources are available.
Keep Slurm `--gres` and `NPROC_PER_NODE` consistent, and run the identical cache
configuration once with `PRECHECK_ONLY=1` before queueing the long job.

Initial scale-up ladder:

```text
train12000 / val512 / e10
train24000 / val512 / e10
full valid train cache / val512 / e10+
```

Keep `save_dir` and `log_dir` unique. Do not overwrite the frozen OracleMotion
baseline or the same-budget REPA comparison directories.

## Primary Metrics

Use validation metrics for model selection and comparison:

- `val_total_no_repa`: broad task objective excluding auxiliary REPA loss; use
  this for checkpoint selection and no-REPA vs REPA comparisons.
- `val_total`: actual training objective including auxiliary terms; useful for
  checking optimization health within one run.
- `val_end`, `val_end_rigid`, `val_end_chi`: endpoint quality.
- `val_contact_score_gain`: whether generated paths move contact in the useful
  direction.
- `val_contact_score_holo_gap_abs`: how far final contact behavior is from holo.
- `val_contact_score_direction_acc`: contact direction consistency.
- `val_contact_score_sidechain_formed_recall`: formed contact recovery.
- `val_repa`: only diagnostic for REPA target fitting; not sufficient evidence
  by itself.

Older runs before 2026-06-27 may not have `val_total_no_repa`; for those, use
`val_total` only within the same REPA setting and rely on endpoint/contact/path
metrics for cross-setting comparisons.

For physical reliability, run the trajectory reliability evaluator after
selecting checkpoints, not during every training iteration.

## Downstream Comparison Plan

Minimum internal comparisons:

- apo/static baseline.
- cubic SE(3)+chi interpolation baseline.
- OracleMotion direct upper-bound application.
- Stage-2 no-REPA last-K baseline.
- Stage-2 matched REPA, if validated.
- Stage-2 target-shuffled REPA control.

Paper-facing comparisons should separate:

- endpoint accuracy;
- trajectory reliability;
- ligand-contact formation;
- clash/peptide geometry;
- sample-level helps/hurts analysis.

The strongest current claim should remain known-pose induced-fit trajectory
generation, not docking.

## Active Entry Points

Training:

```text
scripts/train_stage2.py
scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
```

OracleMotion cache:

```text
scripts/export_oracle_motion_features.py
scripts/slurm/export_oracle_motion_features_1gpu.sh
```

Evaluation:

```text
scripts/evaluate_stage2_transition_paths.py
scripts/evaluate_stage2_trajectory_reliability.py
scripts/generate_stage2_trajectories.py
```

## Operational Checks

Before submission:

```bash
python -m py_compile src/stage2/models/torsion_flow.py src/stage2/training/config.py src/stage2/training/trainer.py scripts/train_stage2.py
bash -n scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
```

After submission:

```bash
squeue -u zhaozc_criait
tail -f logs/slurm/<job>.out
tail -f logs/slurm/<job>.err
```

Confirm the log prints:

- expected train/val cache paths;
- expected train/val subset sizes;
- ESM fusion settings;
- REPA settings, including target shuffle mode;
- DDP `world_size`;
- finite first epoch metrics.
