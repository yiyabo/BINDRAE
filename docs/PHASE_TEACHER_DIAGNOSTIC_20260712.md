# Phase Teacher Diagnostic

Date: 2026-07-12

## Question

Can a validated free-flow model provide enough pseudo-timing supervision to
prevent the APNB residue phase from collapsing to the synchronous bridge?

This is a pseudo-teacher experiment. It is not MD trajectory supervision and
does not validate physical time or kinetics.

## Cache And Coverage

Free-flow states were projected onto the Cartesian endpoint bridge with a
monotone dynamic program and transition regularization `0.5`. The
`phase_teacher_v1` cache stores projected `tau`, projection confidence, and
contact-event masks.

- validation export: job `141435`, 128 systems;
- training export: job `141436`, 512 systems;
- enriched train: 49 systems, 432 supervised residue-time points;
- enriched validation: 14 systems, 133 supervised residue-time points.

The enrichment is a learnability diagnostic and cannot replace evaluation on
the original validation distribution.

## Learnability

The first ordinary fine-tune (`141440`) used `LR=2e-5`, phase loss weight `0.1`,
and eight integration steps. It did not learn: validation phase-target MAE
remained approximately `0.1324` (lower is better).

Head-only job `141444` froze the shared trunk and residual heads, used
`LR=1e-3`, phase loss weight `1.0`, and time-warp logit scale `5`. Validation
phase-target MAE improved from `0.1324` to `0.1087` (17.9%, lower is better),
and mean phase deviation from identity reached `0.0884`. Therefore the target
was learnable and the original failure was partly an optimization-scale issue.

## Joint Training

Full-model sequential job `141446` preserved some timing but moved back toward
identity as the residual branch grew. Phase/residual-head-only job `141452`
reduced shared-trunk drift, but its best trade-off occurred very early:

| Variant | Epoch | Phase-target MAE | Phase deviation | Residual norm |
|---|---:|---:|---:|---:|
| Head-only | 9 | 0.1087 | 0.0884 | approximately 0 |
| Sequential full | 0 | 0.1093 | 0.0796 | 0.0032 |
| Heads-only joint | 1 | 0.1098 | 0.0762 | 0.0291 |
| Heads-only final | 9 | 0.1139 | 0.0483 | 0.0407 |

All MAE values are lower-is-better. Phase deviation is a diagnostic and is not
monotonically better.

## Gradient Audit

Job `141451` measured objective gradients on the time-warp head over four
batches. Cosine with the phase-teacher gradient was:

| Objective | Gradient cosine | Interpretation |
|---|---:|---|
| Contact | +0.924 | strongly aligned |
| Peptide | -0.379 | principal conflict |
| Clash | -0.237 | secondary conflict |
| Residual regularizers | approximately 0 to +0.58 | negligible norms |

Raw gradient norms were `0.519` for phase teacher, `0.123` for peptide, `0.103`
for contact, and `0.0045` for clash. The phase signal was not absent; the joint
objective and shared representation did not preserve it cleanly.

## Matched Path Evaluation

Jobs `141453`-`141456` evaluated 128 original validation systems with eight
path steps. Lower is better except where stated otherwise.

| Metric | Sync Cartesian | Head-only | Heads best | Heads final |
|---|---:|---:|---:|---:|
| All-pocket path MAE | **0.1439** | 0.1596 | 0.1583 | 0.1539 |
| Active path MAE | **0.4743** | 0.5394 | 0.5316 | 0.4979 |
| Formed-contact path MAE | **0.5931** | 0.6206 | 0.6222 | 0.6183 |
| Clash severity | 0.1294 | 0.1276 | **0.1270** | 0.1294 |
| Path action | **11.016** | 12.806 | 12.531 | 11.844 |
| Velocity spike ratio | **1.483** | 1.769 | 1.707 | 1.509 |

Jobs `141458`-`141461` repeated the comparison on the 14-system enriched
validation subset. Synchronous Cartesian remained better on path MAE and path
smoothness; learned phase only produced a small clash-severity improvement.

## Decision

The pseudo-teacher lane is retained as tested infrastructure and a negative
result. It is not promoted to the paper method because fitting its phase target
did not improve matched path quality, even on the enriched subset.

Do not run further pseudo-teacher hyperparameter sweeps without new evidence.
The next phase claim requires one of:

1. controlled paths with known asynchronous ground truth;
2. independent MD transition paths with progress-aligned event supervision;
3. a downstream sampling result that validates the learned schedule externally.

Until then, the Cartesian synchronous bridge and controlled normal residual
are the honest endpoint-corpus baselines.
