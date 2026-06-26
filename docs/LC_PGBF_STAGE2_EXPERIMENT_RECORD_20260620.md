# LC-PGBF Stage-2 Experiment Record

Date: 2026-06-20

## Question

Does learned ligand-causal Stage-1 guidance improve Stage-2 apo-to-holo path
generation on ligand-active residues?

The endpoint-only question is secondary. The main metric is residue-level
transition quality along the generated path.

## Evaluator

Script:

```text
scripts/evaluate_stage2_transition_paths.py
```

Validation lane:

```text
ablation_subsets/stage2_lc_pgbf_val_1200_seed20260618.txt
```

Main settings:

```text
max_batches=1200
batch_size=1
n_integration_steps=3
active_delta=0.75
contact_dist=4.5
pocket_threshold=0.3
```

Residue counts:

| class | count |
|---|---:|
| active | 7443 |
| approach | 2974 |
| release | 4469 |
| formed_contact | 908 |
| released_contact | 1563 |
| stable_contact | 10620 |
| stable_noncontact | 7533 |
| all_pocket | 21213 |

## Stage-1 Prior

Stronger interaction prior:

```text
checkpoints/stage1/interaction_prior_stronger8g_label_enriched_bs64_local256_tc4.5_20260619_080838/best_model.pt
```

Training job:

```text
134799
```

Key validation signal:

| metric | value |
|---|---:|
| best interaction lift | 3.639661 |
| final val_contact_auroc | 0.9812 |
| final val_contact_ap | 0.8576 |
| final val_contact_f1 | 0.7117 |
| val_pocket_switch_lift_shuffled_logit | 7.7288 |
| val_contact_switch_lift_shuffled_logit | 7.3359 |

Interpretation: this prior has strong correct-ligand versus shuffled-ligand
separation and is suitable as soft Stage-2 guidance.

## Stage-2 Retraining Gate

All runs used the 12k/1.2k lane, 5 epochs, 4 A100 GPUs, and `w_contact=2`.

| job | setup | best val_end | best contact gap | best contact direction |
|---:|---|---:|---:|---:|
| 134965 | new prior, scale=1, feature-only | 169.696235 | 0.050791 | 0.561667 |
| 134966 | new prior, scale=2, feature-only | 169.640118 | 0.050669 | 0.565000 |
| 134967 | new prior, scale=1, `w_interaction_prior=0.1` | 169.961389 | 0.050686 | 0.552500 |

Training metric interpretation:

- scale=2 looked best by `val_end`;
- `w_interaction_prior=0.1` did not help endpoint quality;
- residue-level transition evaluation was required before choosing the main run.

## Trained-Checkpoint Transition Evaluation

| job | setup | active endpoint | active path MAE | active direction |
|---:|---|---:|---:|---:|
| 135020 | 134965, scale=1 feature-only | 2.147848 | 1.112842 | 0.544135 |
| 135021 | 134966, scale=2 feature-only | 2.162751 | 1.119825 | 0.539299 |
| 135022 | 134967, scale=1 + `w_interaction_prior=0.1` | 2.157719 | 1.117282 | 0.547494 |

Decision: choose `134965` as the current main checkpoint. It is not the best
endpoint-only training run, but it is best on active-residue path quality.

Main Stage-2 checkpoint:

```text
checkpoints/stage2/stage2_lc_pgbf_newp_s1_wc2_featonly_train12000_val1200_e5_bs2x4_20260619_174421/best_model.pt
```

## Same-Checkpoint Inference Ablation

All rows below use the same Stage-2 checkpoint (`134965`). Only the inference
guidance channel changes.

| Slurm job | inference mode | active endpoint | active path MAE | active direction |
|---:|---|---:|---:|---:|
| 135020 | correct prior | 2.147848 | 1.112842 | 0.544135 |
| 135059 | zero channel | 2.149285 | 1.113536 | 0.542523 |
| 135057 | prior shuffled | 2.148589 | 1.113200 | 0.543464 |
| 135058 | oracle contact | 2.148339 | 1.113123 | 0.542792 |

Detailed transition metrics:

| mode | approach path | release path | formed dir | released dir |
|---|---:|---:|---:|---:|
| correct prior | 0.778460 | 1.335365 | 0.528634 | 0.530390 |
| zero channel | 0.778701 | 1.336360 | 0.524229 | 0.524632 |
| prior shuffled | 0.778155 | 1.336163 | 0.527533 | 0.530390 |
| oracle contact | 0.778516 | 1.335795 | 0.525330 | 0.524632 |

## Strict Interpretation

This is a positive but modest result.

What is supported:

- correct Stage-1 prior improves active endpoint, active path MAE, and active
  direction over zero-channel and shuffled-prior controls;
- the effect survives a same-checkpoint inference ablation, so it is not simply
  checkpoint randomness;
- shuffled prior is weaker than correct prior, supporting ligand-specificity;
- `oracle_contact` does not dominate, so scalar contact truth is not a perfect
  upper-bound guidance channel.

What is not yet proven:

- seed-level robustness;
- large-budget scaling;
- full path-ensemble quality;
- superiority of a multi-channel posterior over the current scalar prior.

## Current Mainline

Use:

```text
Stage-1: interaction_prior_stronger8g_label_enriched_bs64_local256_tc4.5_20260619_080838
Stage-2: stage2_lc_pgbf_newp_s1_wc2_featonly_train12000_val1200_e5_bs2x4_20260619_174421
interaction_prior_feature_mode=prior
interaction_prior_feature_scale=1
w_contact=2
w_interaction_prior=0
```

Next experiments:

1. Repeat the main setup with 2-3 seeds.
2. Run a larger train subset after the seed check.
3. Keep reporting residue-level transition metrics as the main Stage-2 evidence.
4. Defer multi-channel posterior expansion until the scalar-prior result is
   stable across seeds.

## Seed Robustness

| job | seed | subset seed | setup |
|---:|---:|---:|---|
| 134965 | 42 | 20260618 | main setup, scale=1, feature-only |
| 135063 | 43 | 20260618 | main setup, scale=1, feature-only |
| 135064 | 44 | 20260618 | main setup, scale=1, feature-only |
| 135065 | 45 | 20260618 | main setup, scale=1, feature-only |

These runs keep the same 12k/1.2k data subset and vary only the training seed.

Evaluator jobs:

| eval job | depends on | output |
|---:|---:|---|
| 135020 | 134965 | `logs/stage2/transition_eval/trans_retrain_newp_s1_featonly_prior_s1_maxb1200.json` |
| 135067 | 135063 | `logs/stage2/transition_eval/trans_seed43_main_s1_maxb1200.json` |
| 135068 | 135064 | `logs/stage2/transition_eval/trans_seed44_main_s1_maxb1200.json` |
| 135069 | 135065 | `logs/stage2/transition_eval/trans_seed45_main_s1_maxb1200.json` |

All seed robustness training and evaluator jobs completed with exit code `0:0`.

| seed | active endpoint | active path MAE | active direction | approach path | release path |
|---:|---:|---:|---:|---:|---:|
| 42 | 2.147848 | 1.112842 | 0.544135 | 0.778460 | 1.335365 |
| 43 | 2.170156 | 1.123065 | 0.545210 | 0.782747 | 1.349537 |
| 44 | 2.131096 | 1.106365 | 0.552868 | 0.770804 | 1.329673 |
| 45 | 2.168748 | 1.121690 | 0.543329 | 0.787053 | 1.344381 |
| mean | 2.154462 | 1.115990 | 0.546386 | 0.779766 | 1.339739 |
| sd | 0.018620 | 0.007855 | 0.004390 | 0.006929 | 0.008907 |

Interpretation: the main setup is reasonably stable across training seeds on the
same data subset. Seed 44 is best, seed 43/45 are weaker than seed 42, and the
spread is small. This supports robustness of the Stage-2 training lane, but it
does not by itself prove that the correct-prior advantage over zero/shuffled is
seed-stable. The next strict test is to run same-checkpoint inference ablations
for the best additional seed, starting with seed 44.

Seed 44 same-checkpoint inference ablations:

| job | mode | active endpoint | active path MAE | active direction |
|---:|---|---:|---:|---:|
| 135068 | correct prior | 2.131096 | 1.106365 | 0.552868 |
| 135134 | zero channel | 2.133536 | 1.107346 | 0.550853 |
| 135135 | prior shuffled | 2.132014 | 1.106775 | 0.554346 |
| 135136 | oracle contact | 2.131763 | 1.106649 | 0.555018 |

Seed 44 interpretation: the correct prior again gives the best active endpoint
and active path MAE, so the path-quality advantage is not unique to seed 42.
However, direction accuracy is slightly higher for shuffled/oracle guidance on
this seed. The robust claim should therefore emphasize endpoint/path quality on
active residues, not direction accuracy alone.

## In-Flight Train-Level Baselines

The current missing experiment is not another same-checkpoint inference ablation;
it is a train-level control where Stage-2 is trained from scratch with no Stage-1
feature signal.

Submitted on 2026-06-20:

| job | seed | mode | best val_end | best direction | setup |
|---:|---:|---|---:|---:|---|
| 135141 | 42 | `none` | 169.920105 | 0.526667 | no interaction-prior feature channel, `w_contact=2` |
| 135142 | 42 | `zero` | 169.900374 | 0.550833 | same one-channel architecture, zero-filled guidance, `w_contact=2` |
| 135143 | 44 | `none` | 169.680276 | 0.574167 | no interaction-prior feature channel, `w_contact=2` |
| 135144 | 44 | `zero` | 169.503991 | 0.560833 | same one-channel architecture, zero-filled guidance, `w_contact=2` |

These should be compared against the existing correct-prior seeds 42 and 44:

| seed | correct-prior train job | transition eval |
|---:|---:|---:|
| 42 | 134965 | 135020 |
| 44 | 135064 | 135068 |

All four training jobs completed with exit code `0:0`. Training endpoint metrics
alone are mixed, especially because seed44-zero has the best `val_end`; use the
residue-level transition evaluator before drawing the main conclusion.

Evaluator jobs:

| eval job | train job | output |
|---:|---:|---|
| 135214 | 135141 | `logs/stage2/transition_eval/trans_ctrl_none_seed42_maxb1200.json` |
| 135215 | 135142 | `logs/stage2/transition_eval/trans_ctrl_zero_seed42_maxb1200.json` |
| 135298 | 135143 | `logs/stage2/transition_eval/trans_ctrl_none_seed44_maxb1200.json` |
| 135300 | 135144 | `logs/stage2/transition_eval/trans_ctrl_zero_seed44_maxb1200.json` |

All evaluator jobs above completed with exit code `0:0`. The original seed44
evaluator attempts (`135216`, `135217`, `135218`, `135219`) were cancelled by
Slurm before execution and produced no logs; the successful runs are `135298`
and `135300`.

Train-level transition results:

| seed | mode | active endpoint | active path MAE | active direction | approach path | release path |
|---:|---|---:|---:|---:|---:|---:|
| 42 | correct prior | 2.147848 | 1.112842 | 0.544135 | 0.778460 | 1.335365 |
| 42 | none | 2.168541 | 1.121862 | 0.549107 | 0.783891 | 1.346772 |
| 42 | zero | 2.154719 | 1.115763 | 0.551794 | 0.779251 | 1.339703 |
| 44 | correct prior | 2.131096 | 1.106365 | 0.552868 | 0.770804 | 1.329673 |
| 44 | none | 2.123278 | 1.102445 | 0.565095 | 0.768745 | 1.324514 |
| 44 | zero | 2.140827 | 1.110101 | 0.560661 | 0.773367 | 1.334189 |

Two-seed means:

| mode | active endpoint | active path MAE | active direction | approach path | release path |
|---|---:|---:|---:|---:|---:|
| correct prior | 2.139472 | 1.109604 | 0.548502 | 0.774632 | 1.332519 |
| none | 2.145909 | 1.112154 | 0.557101 | 0.776318 | 1.335643 |
| zero | 2.147773 | 1.112932 | 0.556227 | 0.776309 | 1.336946 |

Train-level baseline interpretation: correct-prior training has the best
two-seed average endpoint/path quality on active residues, approach residues,
and release residues. The effect is modest and not monotonic per seed: seed42
supports the prior clearly, while seed44 `none` is best on endpoint/path. Direction
accuracy still favors no-prior/zero baselines. The current strict claim is
therefore an average endpoint/path quality advantage over train-level baselines,
not a universal per-seed win and not a direction-accuracy win.

## In-Flight Four-Seed Baseline Completion

To close the paired-seed baseline gap, seed43 and seed45 train-level controls
were submitted on 2026-06-21. The first batch did not execute successfully:

| job | seed | mode | setup |
|---:|---:|---|---|
| 135307 | 43 | `none` | no interaction-prior feature channel, `w_contact=2`; afterany dependency on `135305` |
| 135308 | 43 | `zero` | same one-channel architecture, zero-filled guidance, `w_contact=2`; afterany dependency on `135306` |
| 135305 | 45 | `none` | no interaction-prior feature channel, `w_contact=2` |
| 135306 | 45 | `zero` | same one-channel architecture, zero-filled guidance, `w_contact=2` |

Initial seed43 submissions (`135303`, `135304`) were cancelled by Slurm before
execution and produced no logs, so they were resubmitted as dependency-gated jobs
`135307` and `135308`. Jobs `135305` and `135306` were also cancelled before
execution; the dependent `135307` and `135308` were then cancelled due to the
failed dependency.

A single-job retry for seed45/none (`135319`) was allocated to `gpu53` and
cancelled after one second. `sinfo -R` reported `gpu53` drained because
`SlurmdSpoolDir is full`, so this was a node-state failure rather than a model
or script failure.

The seed43/45 baseline controls were therefore resubmitted with `--exclude=gpu53`:

| job | seed | mode | setup |
|---:|---:|---|---|
| 135320 | 43 | `none` | no interaction-prior feature channel, `w_contact=2`; excludes `gpu53` |
| 135321 | 43 | `zero` | failed before training: submitted with invalid CLI mode `zero_channel` instead of `zero` |
| 135322 | 45 | `none` | no interaction-prior feature channel, `w_contact=2`; excludes `gpu53` |
| 135323 | 45 | `zero` | failed before training: submitted with invalid CLI mode `zero_channel` instead of `zero` |

Corrected zero-channel controls were resubmitted using the accepted CLI mode
`zero`:

| job | seed | mode | setup |
|---:|---:|---|---|
| 135336 | 43 | `zero` | same one-channel architecture, zero-filled guidance, `w_contact=2`; excludes `gpu53` |
| 135337 | 45 | `zero` | same one-channel architecture, zero-filled guidance, `w_contact=2`; excludes `gpu53` |

Live update on 2026-06-22:

| job | seed | mode | status | notes |
|---:|---:|---|---|---|
| 135320 | 43 | `none` | completed `0:0` | best checkpoint: `stage2_lc_pgbf_ctrl_none_wc2_seed43_retry_exgpu53_train12000_val1200_e5_bs2x4_20260621_190418/best_model.pt` |
| 135322 | 45 | `none` | completed `0:0` | best checkpoint: `stage2_lc_pgbf_ctrl_none_wc2_seed45_retry_exgpu53_train12000_val1200_e5_bs2x4_20260621_193621/best_model.pt` |
| 135336 | 43 | `zero` | completed `0:0` | best checkpoint: `stage2_lc_pgbf_ctrl_zero_wc2_seed43_retry2_exgpu53_train12000_val1200_e5_bs2x4_20260621_215517/best_model.pt` |
| 135337 | 45 | `zero` | completed `0:0` | best checkpoint: `stage2_lc_pgbf_ctrl_zero_wc2_seed45_retry2_exgpu53_train12000_val1200_e5_bs2x4_20260621_220101/best_model.pt` |

Transition evaluators for the completed controls:

| eval job | train job | output |
|---:|---:|---|
| 135433 | 135320 | completed `0:0`; `logs/stage2/transition_eval/trans_ctrl_none_seed43_maxb1200.json` |
| 135434 | 135322 | completed `0:0`; `logs/stage2/transition_eval/trans_ctrl_none_seed45_maxb1200.json` |
| 135443 | 135336 | completed `0:0`; `logs/stage2/transition_eval/trans_ctrl_zero_seed43_maxb1200.json` |
| 135444 | 135337 | completed `0:0`; `logs/stage2/transition_eval/trans_ctrl_zero_seed45_maxb1200.json` |

Final four-seed train-level transition comparison:

| mode | active endpoint | active path MAE | active direction | approach path | release path |
|---|---:|---:|---:|---:|---:|
| correct prior | 2.155545 | 1.116365 | 0.549107 | 0.780139 | 1.340115 |
| none | 2.157417 | 1.116925 | 0.547931 | 0.779678 | 1.341354 |
| zero | 2.160387 | 1.118291 | 0.550047 | 0.779892 | 1.343487 |

Final interpretation: correct-prior training has the best four-seed mean active
endpoint error, active path MAE, and release path MAE. It is also better than
`none` on active direction, but slightly worse than `zero` on active direction.
Approach path does not support the prior claim. The strict conclusion is a small
but reproducible active/release endpoint-path advantage, not a broad win across
all transition metrics.

## Trajectory Export Smoke

After closing the train-level transition ablation, the next practical milestone
is to export actual apo-to-holo trajectory examples from a trained Stage-2
checkpoint. This checks the product deliverable directly: not just scalar
metrics, but generated intermediate structures that can be inspected in PyMOL or
ChimeraX.

New exporter:

```bash
scripts/generate_stage2_trajectories.py
scripts/slurm/generate_stage2_trajectories_1gpu.sh
```

Initial smoke submitted on 2026-06-22:

| job | checkpoint | setup | output |
|---:|---|---|---|
| 135499 | `stage2_lc_pgbf_newp_s1_wc2_featonly_seed44.../best_model.pt` | completed `0:0`; correct Stage-1 prior, 4 validation samples, 12 Heun steps | `logs/stage2/generated_paths/lc_pgbf_seed44_correct_steps12_val4_20260622/` |

The exporter writes per-sample `trajectory.npz`, `summary.json`,
`trajectory_atom14.pdb`, `apo_atom14.pdb`, `holo_atom14.pdb`, and
`ligand_tokens.pdb`. This is a smoke/demo artifact, not yet a polished production
inference interface.

Smoke output summary:

| samples | frames per sample | active endpoint | active path MAE | active direction | release path MAE |
|---:|---:|---:|---:|---:|---:|
| 4 | 13 | 1.280768 | 0.646975 | 0.775379 | 0.878836 |

Exported sample IDs: `8hnm-R-4IE-603`, `5rkb-A-UWG-1501`, `5rk2-A-UUY-1501`,
and `5rko-A-GX4-1501`.
