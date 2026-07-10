# Stage-2 Gated Representation Experiments

Date: 2026-06-29

## Executive Summary

The current best comparable Stage-2 12k/e10 result is the OracleMotion-conditioned
no-REPA model with ESM last-7 `gated_residual` fusion and `gate_bias=-2.0`:

```text
job:                 138217 / s2gate_bm2
best epoch:          6
val_total_no_repa:   5.162529
val_fm_chi+rigid:    3.315601
gate_mean:           0.118681
```

This improves over the previous single-layer ESM baseline (`5.1827`) and the
old ESM last-7 softmax fusion (`5.2296`) under the same 12k/e10, 4xA100,
batch/GPU=4, global-batch=16 screening setting. The result indicates that ESM
last-K information is useful when treated as a small gated residual on top of
the last ESM layer, while naive global averaging dilutes the last-layer signal.

## Experimental Context

All runs in this note use the same Stage-2 screening lane:

```text
task:             known-pose apo-to-holo induced-fit path generation
conditioning:     OracleMotion residue features, 28 dimensions
train samples:    12000
validation:       512
epochs:           10
hardware:         4xA100
batch/GPU:        4
global batch:     16
AMP:              bf16
REPA:             off unless explicitly stated
cache train:      oracle_motion_train12000_esm7sync_20260627
cache val:        oracle_motion_val512_esm7sync_20260625_scale
subset tag:       esm7sync20260627
```

The validation set is intentionally small for fast screening. Differences below
approximately 0.01 should be interpreted cautiously until repeated on a larger
validation lane or by the trajectory reliability evaluator.

## Method: ESM7 Gated Residual Fusion

The old ESM last-7 `softmax_weighted` path computed one global layer-weighted
average before projecting ESM to Stage-2 scalar features. In practice, layer
weights remained close to uniform, making the adapter behave like a mean over
the last seven ESM layers. This diluted the strongest last-layer structural
signal and underperformed the single-layer baseline.

The `gated_residual` adapter instead preserves the last layer as the base and
adds earlier ESM layers only as a gated residual:

```text
s_base   = Proj_last(ESM_last)
s_k      = Proj_k(ESM_k) for earlier layers k
residual = sum_k gate_k * weight_k * (s_k - s_base)
s_out    = s_base + residual
```

`gate_bias` controls the initial residual amplitude. The tested values map to
rough initial gates:

```text
gate_bias=-3.0 -> sigmoid ~= 0.047
gate_bias=-2.0 -> sigmoid ~= 0.119
gate_bias=-1.5 -> sigmoid ~= 0.182
gate_bias=-1.0 -> sigmoid ~= 0.269
```

An optional `pocket_motion` context concatenates `[w_res, motion_active]` to the
gate input. This context was tested as an explicit pocket/motion-aware routing
signal but did not improve the screening metric.

## Results

### Main Comparisons

| Job | Fusion | gate_bias | gate_context | REPA | Best epoch | val_total_no_repa | FM total | gate_mean | Notes |
|---:|---|---:|---|---|---:|---:|---:|---:|---|
| 138217 | gated_residual | -2.0 | none | off | 6 | **5.162529** | **3.315601** | 0.118681 | current best |
| 138229 | gated_residual | -1.5 | none | off | 6 | 5.166038 | 3.318596 | 0.173684 | close second |
| 138150 | gated_residual | -3.0 | none | off | 6 | 5.175738 | 3.321694 | ~0.05 | original gated v0 |
| 138159 | gated_residual | -3.0 | none | off | 6 | 5.182008 | 3.329775 | 0.061988 | diagnostic repeat |
| 138105 | single ESM | n/a | n/a | off | 6 | 5.1827 | 3.3396 | n/a | strong baseline |
| 138231 | gated_residual | -1.0 | none | off | 6 | 5.181356 | 3.331679 | 0.254217 | gate too open |
| 138109 | soft7 + mcont REPA | n/a | n/a | matched | 6 | 5.2152 | 3.3773 | n/a | REPA signal positive but not best |
| 138106 | old soft7 | n/a | n/a | off | 6 | 5.2296 | 3.3889 | n/a | old fusion underperforms |

### Gate Bias Sweep

| gate_bias | context | Best val_total_no_repa | Interpretation |
|---:|---|---:|---|
| -2.0 | none | **5.162529** | best; earlier-layer residual useful at ~0.12 gate |
| -1.5 | none | 5.166038 | close; ~0.17 gate still useful |
| -1.0 | none | 5.181356 | residual too strong; returns to single-level |

The best range is currently `gate_bias=-2.0` to `-1.5`, with `-2.0` selected
as the main candidate because it is best and more conservative.

### Pocket/Motion Context Sweep

| gate_bias | context | Best val_total_no_repa | gate_mean | pocket_gate | nonpocket_gate |
|---:|---|---:|---:|---:|---:|
| -2.0 | none | **5.162529** | 0.118681 | 0.118364 | 0.118693 |
| -2.0 | pocket_motion | 5.194462 | 0.117520 | 0.117063 | 0.117538 |
| -1.5 | none | 5.166038 | 0.173684 | 0.172170 | 0.173747 |
| -1.5 | pocket_motion | 5.258227 | 0.172077 | 0.170346 | 0.172149 |
| -1.0 | none | 5.181356 | 0.254217 | 0.251815 | 0.254317 |
| -1.0 | pocket_motion | 5.202464 | 0.252861 | 0.250234 | 0.252971 |

Directly concatenating `[w_res, motion_active]` into the gate input was harmful
in all tested settings. It also did not produce higher pocket gates; pocket gate
means were slightly lower than non-pocket gate means. This suggests that
context-aware routing remains promising but should not be implemented as a naive
concat at this stage.

### REPA Status

The same 12k/e10 matrix showed that REPA targets are not noise:

```text
motion_continuous matched:  val_total_no_repa ~= 5.2152
motion_continuous shuffled: val_total_no_repa ~= 5.2407
full28 matched:             val_total_no_repa ~= 5.2339
full28 shuffled:            val_total_no_repa ~= 5.3197
```

Matched REPA consistently beat shuffled controls, indicating that the teacher
motion signal is sample/residue aligned. However, current hidden-state alignment
does not beat the best no-REPA gated model. REPA should remain a mechanism branch
for 24k-scale testing, preferably attached to the best gated interface rather
than to the old softmax fusion.

## Interpretation

The experiments support three conclusions:

1. **The old soft7 failure was an interface problem, not evidence that ESM
   last-K features are useless.** Global averaging over seven layers diluted the
   last-layer representation.
2. **Earlier ESM layers are useful as low-amplitude residual corrections.** A
   gate around 0.12 to 0.17 improves the screening loss; a gate around 0.25 is
   too strong.
3. **Naive pocket/motion gate context is not yet useful.** The Stage-2 gate does
   not become pocket-selective by directly concatenating `w_res` and
   `motion_active`. Future context-aware routing should use a better mechanism,
   such as FiLM, a separate supervised gate head, or a Stage-1-v2 posterior
   routing signal.

The current best method is therefore:

```text
OracleMotion conditioning
+ ESM last-7 gated_residual fusion
+ gate_bias=-2.0
+ gate_context=none
+ no REPA
```

## Next Experimental Gate

The next stage should not rely only on `val_total_no_repa`. The recommended
selection path is:

1. Run trajectory reliability evaluation on the current best checkpoints:
   - single baseline;
   - old soft7;
   - gated v0;
   - gated `gate_bias=-2.0`;
   - gated `gate_bias=-1.5`;
   - motion-continuous REPA matched and shuffled.
2. Move to an 8-card 24k matrix only after checking active/path/contact metrics.
3. Increase validation size for 24k-scale model selection, preferably to 2048
   examples if the cache/split is available.

Suggested 24k matrix:

```text
1. single_noREPA
2. gated_residual_bias_m2_noREPA
3. gated_residual_bias_m15_noREPA
4. gated_residual_bias_m2 + motion_continuous_REPA matched
5. gated_residual_bias_m2 + motion_continuous_REPA shuffled
```

If resources are constrained, keep only `single_noREPA`,
`gated_bias_m2_noREPA`, and the matched/shuffled REPA pair.

## Benchmark Implication

For paper-facing evaluation, this internal loss table is only a candidate
selector. Final claims should be based on endpoint, active-motion, contact, path
reliability, and external endpoint/path baselines as defined in
`COMPETITOR_DATASETS_METRICS_20260624.md`.
