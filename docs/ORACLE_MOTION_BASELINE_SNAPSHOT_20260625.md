# OracleMotion Stage-2 Baseline Snapshot

Date: 2026-06-25

> Status update (2026-07-10): this is a frozen historical baseline. The active
> architecture is the endpoint-exact asynchronous phase-normal bridge described
> in `BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md`.

## Status

This document freezes the then-best BINDRAE baseline before starting the
RAEv2/REPA-style representation enhancement track.

The frozen claim is intentionally narrow:

```text
Given oracle per-residue apo->holo motion features, Stage-2 can exploit the
Stage-1-style conditioning interface to generate better induced-fit endpoints
and contact rearrangements than zero or residue-shuffled controls.
```

This is an upper-bound result. It does not claim that the current learned
Stage-1-v2 scalar/contact posterior already matches oracle motion, and it does
not replace future physical validation by MD or external trajectory benchmarks.

## Frozen Code Path

Core Stage-2 implementation:

- `src/stage2/datasets/dataset_stage2.py`
  - loads Stage-1-v2 and OracleMotion feature caches;
  - validates `sample_id`, `n_residues`, `aatype`, and cache masks before
    feature consumption;
  - supports `zero`, `student`, `student_shuffled`, `oracle_holo_truth`,
    `external_teacher_cached`, `oracle_motion`,
    `oracle_motion_residue_shuffled`, and `oracle_motion_sample_shuffled`.
- `src/stage2/models/torsion_flow.py`
  - `TorsionFlowNet` consumes ESM, ligand tokens, current frames/chi, time,
    optional Stage-1/OracleMotion residue features, and optional endpoint
    delta features;
  - the current ESM path is a simple last-layer `ESMAdapter`, not last-K layer
    fusion.
- `src/stage2/training/trainer.py`
  - integrates the bridge flow;
  - computes endpoint, path, contact, and guidance metrics;
  - exposes explicit integration clamps.
- `scripts/export_oracle_motion_features.py`
  - exports `bindrae_oracle_motion_v1` feature caches and direct oracle-apply
    audits.
- `scripts/evaluate_stage2_trajectory_reliability.py`
  - evaluates learned trajectories against endpoint truth and cubic SE(3)+chi
    interpolation.
- `scripts/generate_stage2_trajectories.py`
  - exports generated trajectories for visualization.

Cluster wrappers:

- `scripts/slurm/export_oracle_motion_features_1gpu.sh`
- `scripts/slurm/train_stage2_oracle_motion_smoke_1gpu.sh`
- `scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh`
- `scripts/slurm/evaluate_stage2_trajectory_reliability_1gpu.sh`
- `scripts/slurm/generate_stage2_trajectories_1gpu.sh`

The `train_stage2_oracle_motion_ablation_4gpu.sh` filename is historical; its
current cluster default should favor 2-3 A100 GPUs unless a larger allocation
is explicitly requested.

## Frozen Artifacts

Remote project root:

```text
/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
```

Train12000 OracleMotion cache:

```text
logs/stage2_oracle_motion/oracle_motion_train12000_direct_20260624_scaleup/
```

The direct oracle-apply audit wrote 12000 `.npz` caches and reached numerical
ceiling. Manifest:

```text
logs/stage2_oracle_motion/oracle_motion_train12000_direct_20260624_scaleup/manifest.json
```

Train12000/e10 checkpoints:

```text
checkpoints/stage2/stage2_omotion_scale12000_relaxed_zero_train12000_val512_e10_bs2x2_20260624_234241/best_model.pt
checkpoints/stage2/stage2_omotion_scale12000_relaxed_res_shuffle_train12000_val512_e10_bs2x2_20260625_001608/best_model.pt
checkpoints/stage2/stage2_omotion_scale12000_relaxed_matched_train12000_val512_e10_bs2x2_20260625_003756/best_model.pt
```

Train12000/e10 reliability outputs, val512, steps=12:

```text
logs/stage2/trajectory_reliability/omotion_scale12000_zero_best_val512_reliability_steps12_retry_20260625.json
logs/stage2/trajectory_reliability/omotion_scale12000_res_shuffle_best_val512_reliability_steps12_retry_20260625.json
logs/stage2/trajectory_reliability/omotion_scale12000_matched_best_val512_reliability_steps12_retry_20260625.json
```

## Training Result: Train12000/e10

Final epoch 9 validation:

| Mode | val_total | val_end | val_end_chi_uw | formed recall | direction acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero | 12.95957 | 186.86199 | 2.19651 | 0.117088 | 0.498047 |
| residue-shuffled | 13.15280 | 179.51842 | 2.15857 | 0.126773 | 0.517578 |
| oracle matched | 3.70048 | 99.81124 | 1.80751 | 0.646893 | 0.892578 |

Best by `val_total`:

| Mode | best epoch | best val_total | val_end | formed recall |
| --- | ---: | ---: | ---: | ---: |
| zero | 6 | 12.9192 | 186.99 | 0.1133 |
| residue-shuffled | 2 | 13.0616 | 183.19 | 0.0943 |
| oracle matched | 7 | 3.6993 | 99.38 | 0.6617 |

Interpretation:

- matched OracleMotion separates very strongly from both zero and
  residue-shuffled controls;
- residue-shuffled remains close to zero on the main path/contact metrics,
  meaning the effect is not simply extra feature capacity;
- the matched signal persists when scaling from train512/train4096 to
  train12000.

## Reliability Result: Val512, Steps=12

| Mode | active endpoint | path MAE | direction acc | chi err | formed recall | peptide loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero | 1.917827 | 1.015995 | 0.588632 | 1.428105 | 0.144022 | 0.009040 |
| residue-shuffled | 1.938949 | 1.017304 | 0.566474 | 1.457668 | 0.125000 | 0.018542 |
| oracle matched | 0.776324 | 0.628676 | 0.946692 | 0.265712 | 0.885870 | 0.048089 |

Cubic reference on the same samples:

```text
cubic active path MAE: 0.502479
cubic peptide loss:   17.436876
```

Interpretation:

- cubic interpolation can be endpoint-close by construction, but its peptide
  geometry is unreliable;
- matched OracleMotion is not just endpoint interpolation: it substantially
  improves contact recovery and movement direction while keeping learned-path
  peptide loss far below cubic interpolation;
- zero/residue-shuffled do not recover the same contact rearrangements.

## What This Proves

The current code proves an upper-bound interface result:

1. Stage-2 has enough conditioning capacity to use aligned motion guidance.
2. The useful signal is residue-aligned local motion information, not generic
   extra channels.
3. Endpoint/contact gains can be obtained while learned paths remain much more
   geometrically sane than naive cubic interpolation.

## What This Does Not Prove Yet

This snapshot does not prove:

- that a learned Stage-1 model can already replace oracle motion;
- that the generated intermediate structures are physically valid under MD;
- that BINDRAE is superior to DynamicFlow, AlphaFlow, DynamicBind, FlowDock, or
  other external systems under a shared benchmark;
- that all reliability/ranking/visualization audits are complete.

Those are later validation tasks. For now, the experimental baseline is strong
enough to justify architecture work on representation enhancement.

## Current Decision

Do not automatically launch additional `steps=6/24` reliability, sample-level
ranking, or trajectory visualization jobs from this snapshot. They remain
available for paper evidence and debugging, but they are not the next default
step.

The next default engineering track is:

```text
freeze current OracleMotion Stage-2 baseline
-> add RAEv2-style ESM last-K fusion
-> add REPA-style representation alignment
-> rerun small controlled matched/zero/shuffled ablations
```
