# Stage-2 Endpoint Boundary Status

Date: 2026-07-08

This note freezes the current Stage-2 endpoint/path evidence after the
free-flow, exact-endpoint bridge, teacher residual, ligand-clearance, and
hard-negative clearance experiments. It is intended as the handoff document
before the next model-design pass.

## Executive Summary

The original free-flow Stage-2 model remains the strongest learned path signal
seen so far: it gave useful path-level improvements and was strong against
external path baselines. Its central weakness is that the final integrated
state is not guaranteed to equal the known holo endpoint. Since the current
task provides both apo and holo endpoints, this is a paper-facing risk.

The exact-endpoint boundary formulation fixes that weakness. The most reliable
current exact-endpoint result is the analytic `pure_bridge`: it reaches the
holo endpoint exactly and gives the cleanest 512-sample path MAE among the
tested exact-endpoint variants. However, learned residual variants have not yet
improved over `pure_bridge`.

The recent ligand-clearance and hard-negative clearance experiments are valid
negative results. They slightly reduce ligand clash proxies in some settings,
but they consistently worsen path MAE. This indicates that local repulsion-style
losses are not the right main route for improving exact-endpoint paths.

## Current Decision

Do not continue tuning ligand-clearance, hard-negative clearance, or bridge
anchor weights as a mainline experiment.

Keep:

- free-flow as a strong learned-path teacher / upper-bound reference;
- `pure_bridge` as the clean exact-endpoint lower-bound baseline;
- the clearance variants as negative ablations;
- exact endpoint as the default paper-facing boundary condition for the current
  apo-to-holo setting.

Next model work should focus on learning endpoint-conditioned path curvature or
teacher-derived intermediate deviations, not on adding stronger ligand
repulsion losses.

## Model Lineage

### 1. Free-Flow Stage-2

Purpose:

- Learn an apo-to-holo path by integrating predicted motion through time.

Observed strength:

- Strong path-level behavior.
- Previously competitive or better than path baselines such as ANM/morphing and
  eBDIMS2-style endpoint-conditioned paths.

Observed weakness:

- The terminal state is not exactly holo after integration.
- This is hard to defend because the current setup already conditions on the
  holo endpoint.

Interpretation:

- Useful as a teacher or learned-path reference.
- Risky as the final paper method unless endpoint error is explicitly framed,
  bounded, and shown not to harm downstream use.

### 2. Exact-Endpoint Boundary Formulation

The current exact-endpoint family uses:

```text
path(t) = analytic_bridge(apo, holo, t) + eta(t) * learned_residual(t)
```

where `eta(t)` is zero at both endpoints. This makes apo and holo exact by
construction.

Main variants:

- `pure_bridge`: no learned residual; analytic endpoint interpolation only.
- teacher-residual variants: residual learned from the free-flow path.
- ligand-clearance variants: residual shaped by ligand clearance / clash losses.
- hard-negative clearance + bridge-anchor variants: clearance restricted to
  current clash-like residues, with non-clash residues anchored to the bridge.

Current result:

- `pure_bridge` is the strongest exact-endpoint result in the 512-sample lane.
- Learned residuals have not yet produced a better path than `pure_bridge`.

## Recent 512-Sample Evidence

All metrics below are from the 512-sample validation lane with transition path
evaluation. `MAE` and clash metrics are lower-is-better. Direction accuracy was
1.0 for all rows in this exact-endpoint comparison and is omitted for brevity.

| Run | Active MAE (lower better) | Formed-contact MAE (lower better) | All-pocket MAE (lower better) | Active clash (lower better) | Clash severity (lower better) | Endpoint error (lower better) |
|---|---:|---:|---:|---:|---:|---:|
| `pure_bridge` | **0.3892** | **0.3010** | **0.2078** | 0.3356 | 0.2893 | 1.14e-04 |
| `ligclear_w10` | 0.3997 | 0.3243 | 0.2336 | 0.3337 | **0.2839** | 1.14e-04 |
| `hard_c1a01` | 0.4002 | 0.3176 | 0.2427 | **0.3330** | 0.2857 | 1.14e-04 |
| `hard_c1a03` | 0.4186 | 0.3412 | 0.2569 | 0.3369 | 0.2892 | 1.14e-04 |
| `hard_c2a03` | 0.4218 | 0.3444 | 0.2633 | 0.3376 | 0.2883 | 1.14e-04 |

Interpretation:

- `ligclear_w10` slightly improves clash severity but worsens all MAE metrics.
- `hard_c1a01` gives the lowest active clash proxy, but the gain is small and
  it still worsens MAE.
- Stronger anchor (`hard_c1a03`) and stronger clearance (`hard_c2a03`) both
  degrade path quality further.
- This family does not justify further sweep expansion.

## Completed Jobs And Artifacts

Recent hard-negative clearance jobs:

```text
140608  s2hn16sm      COMPLETED  smoke, hard-negative + anchor
140610  s2hnc1a01     COMPLETED  train, clearance=1.0 anchor=0.1
140611  ev_s2hnc1a01  COMPLETED  eval
140612  s2hnc1a03     COMPLETED  train, clearance=1.0 anchor=0.3
140613  ev_s2hnc1a03  COMPLETED  eval
140614  s2hnc2a03     COMPLETED  train, clearance=2.0 anchor=0.3
140615  ev_s2hnc2a03  COMPLETED  eval
```

Key eval outputs:

```text
logs/stage2/transition_eval_v2/teacher512_pure_bridge_val512_s3.json
logs/stage2/transition_eval_v2/stage2_ligclear_teacher512_e10_w10_lr5e5_bs2x2_20260708_val512_s3.json
logs/stage2/transition_eval_v2/stage2_hardneg_teacher512_e10_c1_a01_lr5e5_bs2x2_20260708_val512_s3.json
logs/stage2/transition_eval_v2/stage2_hardneg_teacher512_e10_c1_a03_lr5e5_bs2x2_20260708_val512_s3.json
logs/stage2/transition_eval_v2/stage2_hardneg_teacher512_e10_c2_a03_lr5e5_bs2x2_20260708_val512_s3.json
```

As of the last check on 2026-07-08, Slurm had no active BINDRAE jobs from this
experiment group.

## Code State Relevant To This Note

The current code supports:

- `PATH_PARAMETERIZATION=boundary_residual_v1`;
- exact endpoint path generation through analytic bridge plus endpoint-zero
  residual envelope;
- `projected_flow` and `pure_bridge` as ablations;
- teacher residual loss controls;
- ligand clearance loss controls;
- hard-negative clearance mode;
- bridge-anchor controls and metrics;
- transition evaluator support for path parameterization and endpoint metrics.

Recent local validation:

```text
python3 -m py_compile src/stage2/training/config.py src/stage2/training/trainer.py scripts/train_stage2.py
bash -n scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
git diff --check -- src/stage2/training/config.py src/stage2/training/trainer.py scripts/train_stage2.py scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
```

Remote validation matched the local compile and Slurm syntax checks before the
hard-negative smoke and 512-lane jobs were submitted.

## Next Recommended Direction

The next experiment should be designed around one of these two routes:

### Route A: Clean Paper Baseline Track

Use the exact-endpoint `pure_bridge` as the defensible endpoint-conditioned
baseline, keep free-flow as an upper-bound learned-path comparison, and focus on
external baselines and path metrics.

This is the safest route if the next priority is paper evidence organization.

### Route B: New Learned Boundary Model

Design a learned residual model whose target is path curvature or teacher
intermediate deviation, with strict gates:

- endpoint error must remain exact or near-zero by construction;
- active/path/contact MAE must beat `pure_bridge`;
- clash proxy must not improve by sacrificing MAE;
- improvements must hold on at least a 512-sample validation lane before any
  full-scale run.

Potential formulations:

```text
path(t) = bridge(apo, holo, t) + t(1-t) * residual_teacher_or_dynamics(t)
```

or a conditional bridge dynamics model that naturally contracts toward the holo
endpoint near `t=1`.

Do not restart another broad loss-weight sweep until this model target is
specified more sharply.

## Repository Hygiene Recommendation

Keep the dirty worktree split by topic:

1. Stage-2 boundary/evaluator/training implementation.
2. External baseline scripts and evaluators.
3. Documentation and experiment ledgers.
4. Dataset/sampler reliability fixes.

Do not stage logs, checkpoints, processed datasets, or Slurm output. Do not use
`git add .` in this repository.
