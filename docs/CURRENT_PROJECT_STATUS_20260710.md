# Current Project Status

Date: 2026-07-20

This is the operational source of truth for the current BINDRAE research track.
For the method and manuscript logic, read
`BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md` first.
For trajectory acquisition, evidence tiers, manifest fields, and the cluster MD
pilot, read `MD_TRANSITION_CORPUS_PLAN_20260713.md`.

## Current Objective

Develop and validate an endpoint-exact, ligand-conditioned protein
conformational path model whose promoted contribution is a learned,
chain-coupled residue progress field over an analytic endpoint bridge.

The current paper setting is:

```text
known apo + known holo + aligned ligand pose
    -> endpoint-conditioned path reconstruction
```

It is not docking, apo-only holo prediction, or physical MD generation.

## Active Method

Frozen deterministic conference candidate:

```text
PATH_PARAMETERIZATION=phase_block_orthogonal_residual_v2
PHASE_WARP_VARIANT=chain_nonmonotone
PHASE_CHAIN_RESIDUAL_SCALE=1.0
PHASE_CHAIN_SMOOTHING_STEPS=1
PHASE_RESIDUAL_BRIDGE_MODE=cartesian_backbone
```

This candidate was selected on validation and evaluated once on the audited
strict30 test. The deterministic normal-residual branch remains implemented as
a matched negative result; it is not part of the promoted model.

The preferred validation-time inference stabilization is an endpoint-fixed
`cummax` projection of the learned phase. Across three seeds on the frozen
30-system validation split it removes all predicted phase backtracking and all
non-monotone residue traces. Product RMSE changes by only `-0.000089` in the
lower-is-better convention (95% CI `[-0.000654, 0.001173]`), while phase tau MAE
worsens by `0.000204`. This is a stability constraint, not evidence for physical
kinetics. It was selected after the one-time strict30 evaluation and therefore
must not be used to replace the already reported strict30 headline numbers; it
requires a fresh independent holdout before promotion to the final test protocol.

The path is

\[
X_i(t)=
\operatorname{Exp}_{\gamma_i(\tau_i(t))}
\left[b(t)\Delta_i^\perp(t)\right].
\]

For the promoted phase-only candidate, `Delta_i^perp(t)=0`; the full expression
is retained to define the residual-only and full negative ablations.

Implemented properties:

- exact apo/holo endpoints;
- endpoint-fixed learned residue phase with peptide-chain coupling;
- tangent-normal residual projection on `SE(3) x T^k`;
- explicit rotation/translation/chi metric scales;
- dedicated zero-initialized residual heads;
- residual magnitude, temporal, neighbor, and background controls;
- atom14 endpoint-consistent FK decoding;
- evaluator support for deterministic component ablations;
- Cartesian `N/CA/C` endpoint bridge with locally differentiated tangents;
- reproducible `se3_geodesic` reference-bridge ablation.
- `sin^2(pi t)` residual envelope as the main contract, giving both zero
  displacement and zero residual slope at the two endpoints;
- immutable MD-target cache contracts that record phase mode, residual
  envelope, and rotation/translation/chi metric scales.

### Controlled identifiability benchmark (2026-07-20)

`scripts/run_controlled_manifold_benchmark.py` instantiates the four-model
decomposition on synthetic `SE(3) x T^2` paths with exact endpoints, known
monotone residue phase, metric-normal detours, and known obstacle geometry.
Every endpoint-conditioned input has two exactly opposite route modes. The
benchmark compares a route-observed condition, where a one-bit route cue is
available, with a route-hidden condition, where both modes have identical
inputs and the deterministic conditional-mean residual is exactly zero.

The three-seed run (`147041`; 256 endpoint pairs / 512 paths for training,
64 / 128 for validation, and 128 / 256 for test) gives:

| Route condition | Method | Product RMSE ↓ | Event-order accuracy ↑ | Predicted residual RMS | Collision fraction ↓ |
|---|---|---:|---:|---:|---:|
| observed | synchronous | 0.636607 | 0.5000 | 0.0000 | 1.0000 |
| observed | warp-only | 0.616695 | 0.9962 | 0.0000 | 1.0000 |
| observed | residual-only | 0.158134 | 0.5000 | 0.6207 | 0.0000 |
| observed | full phase + normal | **0.007543** | **0.9963** | 0.6208 | **0.0000** |
| hidden | warp-only | **0.616695** | **0.9962** | 0.0000 | 1.0000 |
| hidden | residual-only | 0.636608 | 0.5000 | 0.0011 | 1.0000 |
| hidden | full phase + normal | 0.616697 | 0.9961 | 0.0014 | 1.0000 |

Endpoint maximum error is `2.38e-7` and the maximum tangent-normal dot product
is `4.58e-7` in every arm. This result proves the parameterization is expressive
and the two components are complementary when route-identifying information is
present. It also reproduces the protein-corpus failure mechanism under exact
control: when route information is hidden, the deterministic residual collapses
to zero and the full model becomes warp-only. This is an identifiability result,
not evidence that the protein model has recovered physical transition routes.

### Inference-time peptide retraction screen (2026-07-20)

The existing endpoint-preserving translation retraction was exposed as an
explicit inference override and tested without retraining. Its default settings
reduce the single-system maximum peptide-bond error from `0.989 A` to `0.532 A`,
but worsen bond violation fraction from `0.0343` to `0.2468` and Product RMSE
from `0.8657` to `0.9246`; they are rejected.

On six validation systems, a conservative `2 iterations / relaxation 0.25 /
anchor 0.10 / cap 0.25 A / activation threshold 0.01` setting is the only
configuration that jointly lowers mean and maximum bond error, bond/angle
violation fractions, and clash summaries. Its matched check on six systems and
24 MD replicas also passed the predeclared 1% Product-RMSE non-inferiority
margin. Product RMSE changed from `0.993737` to `0.993170` (lower is better;
candidate improvement `+0.000567`, 95% CI `[+0.000024, +0.001194]`), while
translation MAE changed from `0.693550 A` to `0.694028 A` (candidate improvement
`-0.000478 A`, 95% CI `[-0.001567, +0.000252]`) and remained non-inferior. The
other evaluated phase and event metrics were unchanged. This is a small
validation-only result selected after the one-time strict30 evaluation; the
setting remains an optional inference-time validity layer until it passes a
fresh independent holdout.

### Path-4 optimizer pivot (2026-07-20)

The long-term Path-4 branch is no longer defined as direct deterministic
regression from endpoint-derived features to one MD normal residual. A prior
same-replica OpenMM-force upper-bound diagnostic found no transferable alignment
between instantaneous generalized force and the silver normal residual: the
30-system rigid relative-MSE reduction was `-0.000731`, mean cosine was
`-0.01910`, and positive alignment was `0.4765`. This closes direct
force-to-residual regression, not path-level physical optimization.

The existing surrogate physical-normal optimizer has been extended with paired
global route seeds, global-frame-consistent chain smoothing, tail-aware frame
aggregation, backtracking descent acceptance, strict Path-3 fallback, and
objective-call diagnostics. Five focused physical-path tests and 22 related
evaluator/export tests pass remotely. Surrogate CUDA smoke job `147108` was
cancelled before allocation and provides no evidence. CPU engineering smoke
job `147139` completed the single-training-system Path-3 export and OpenMM
scorer path, but did not exercise the OpenMM optimizer, CUDA, or Gate-0 cohort.

The subsequent one-system optimizer/scorer audit found and closed a paired
evaluation defect: candidate-specific warm starts changed endpoint energies,
while direct per-frame resets created pathological hidden-atom nonbonded
states. The active scorer now requires a frozen Path-3 all-atom frame-reference
cache with per-frame force preflight and exact cache/endpoint contract checks.
The direct CUDA engineering run on `3zw1-E-FUC-3` completed with zero paired
endpoint-energy differences. The optimized candidate improved raw p95/max
excess energy by `26.43%`/`37.03%` across all 19 interior frames. Its original
relaxed `81.37%`/`96.92%` headline was dominated by one non-valid Path-3 frame
at `t=0.25` (`18588.31 kJ/mol/nm` maximum residue-net force). Scorer v5 now
freezes a `500 kJ/mol/nm` relaxed-frame threshold, and paired summary v2 uses
only the intersection of valid frames while reporting failure incidence. A
direct CUDA v5/v2 rescore
(`path4_gate0_relaxed_validity_3zw1-E-FUC-3_direct_gpu33_20260721_183700_v1`)
completed with `18/19` paired-valid frames: relaxed p95 is `4.26%` worse,
relaxed maximum is `4.45%` better, invalid incidence is `1/19 -> 0/19`, and
valid-frame residue-force p95 is `14.20%` worse. Raw p95/max improvements remain
`26.43%`/`37.03%`. This validates the scorer contract on one old training
system, not Gate-0 efficacy; the next step is a small diverse development panel
before freezing the 40-60-system cohort.

This scaffold remains a surrogate screen, not a validated OpenMM optimizer or a
learned Path-4 result. The next scientific gate is a non-learned, multi-start
physical necessity audit on 40-60 fresh systems before any 1,000-system data
campaign or unrolled neural optimizer. See `PATH4_OPTIMIZER_GATE0_20260720.md`.

## What Has Been Validated

### Unit and model checks

- Projection/path tests passed.
- True FlashIPA CUDA forward/backward passed.
- Every trainable parameter receives a gradient tensor under the DDP contract.
- Evaluator preserves exact endpoints and applies the projected residual.

### End-to-end smoke

Job `140896` completed a 64-train/64-validation one-epoch smoke.

Key diagnostics:

- `val_end_rigid=0` (lower is better);
- `val_end_chi=0` (lower is better);
- `val_end_fape=0.001` (lower is better; numerical floor);
- projected parallel cosine approximately `2.1e-8` (lower is better);
- non-zero learned residual norm after one epoch.

Job `140898` completed the matching transition-path evaluator smoke.

These results establish implementation correctness only. They do not establish
scientific superiority.

### Backbone bridge correction (2026-07-12)

The original independent per-residue SE(3) bridge had a severe heavy-tailed
peptide-geometry failure. On a matched random-128 diagnostic, replacing it with
the Cartesian backbone-triplet bridge improved pocket path MAE from 0.2467 to
0.2107 and active path MAE from 0.8037 to 0.6898, while reducing maximum
interior peptide loss from 1344.20 to 0.423.

Four-GPU smoke job `141403` then validated the bridge inside the trainable APNB
model. Relative to the matched old-bridge smoke, validation interior peptide
mean fell from 31.41 to 0.0147, p95 from 0.348 to 0.0247, and maximum from
1972.00 to 0.378. Contact direction accuracy remained 0.8125.

### Matched four-model result (2026-07-12)

Jobs `141404`, `141405`, and `141406` completed the matched warp-only,
residual-only, and full APNB three-epoch screen; the synchronous Cartesian
bridge required no training. The result falsified phase identifiability under
endpoint-only aggregate losses:

- warp-only phase deviation was `1.68e-5` (identity collapse; larger is not
  inherently better, but this value shows the head was unused);
- full APNB phase deviation was `6.99e-6`;
- residual-only and full APNB had effectively identical validation objectives
  (`0.004347` versus `0.004346`, lower is better);
- full APNB did not improve path MAE or clash metrics over residual-only.

### MD-supervised capacity and geometry diagnostic (2026-07-15)

The first audited MD cache contained 29 accepted paths from seven endpoint
systems. Those same-system runs were capacity and optimization diagnostics
only, not generalization evidence. The silver corpus has since expanded to 82
accepted paths from 23 endpoint systems, enabling a fixed 17-system train / six-
system held-out pilot split.

Isolated 100-epoch tests showed that both supervised components are learnable:

- phase-only tau MAE decreased from 0.1905 to 0.1146;
- residual-only normal MAE decreased from 0.5166 to 0.2428.

The initial mixed run undertrained both heads. Increasing the peptide loss
reduced geometry violations but suppressed asynchronous phase. A differentiable
violation-gated peptide retraction and stronger graph-neighbor penalties were
then tested. The table reports validation means over the last ten epochs of
matched 60-epoch runs; lower is better for every error/loss column.

| Variant | Phase tau MAE | Normal MAE | Peptide interior | Smoothness | Contact loss | Mean retraction |
|---|---:|---:|---:|---:|---:|---:|
| Base: neighbor 0.01, no retraction | **0.1193** | **0.3426** | 0.01192 | **0.9011** | **0.00292** | 0 |
| Neighbor 0.1 | 0.1081 | 0.3634 | 0.01236 | 0.9009 | 0.00337 | 0 |
| Neighbor 1.0 | **0.1049** | 0.4228 | 0.01083 | 0.9119 | 0.00376 | 0 |
| Retraction gate 0.005 | 0.1178 | 0.3679 | 0.00686 | 0.9693 | 0.00307 | 0.117 A |
| Retraction gate 0.01 | 0.1235 | 0.3687 | **0.00565** | 0.9616 | 0.00334 | 0.090 A |

Decision:

- keep neighbor weight 0.01 and soft peptide weight 0.1 as the balanced main
  training configuration;
- keep gated peptide retraction as an optional validity-layer ablation, off by
  default, because it improves peptide geometry but harms normal fitting,
  temporal smoothness, and contact loss;
- do not promote stronger neighbor smoothing: it improves phase supervision by
  shrinking spatial residuals, but sacrifices the complementary normal branch;
- repeat selection on a family/scaffold-separated MD set after data expansion.

The Cartesian bridge correction remains validated, but asynchronous phase is
not a supported claim from endpoint-only training. The next phase experiment
therefore uses explicit supervision rather than another loss-weight sweep.

### Free-flow pseudo-teacher diagnostic (2026-07-12)

Job `141422` projected a previously validated free-flow model onto the
Cartesian bridge for 128 validation systems with transition regularization
`0.5`. Relative to the synchronous bridge, the projected schedule improved
formed-contact path MAE from `0.6342` to `0.5334` (lower is better) and approach
path MAE from `0.4284` to `0.3864`, but worsened active and release path MAE.

This establishes a nontrivial candidate contact-formation schedule, not true
dynamics. The reported path MAE uses a linear apo-to-holo ligand-distance
schedule as its reference. Consequently, free-flow phase labels are explicitly
marked as pseudo-teacher targets and restricted to confidence-weighted contact
events. Independent MD paths remain mandatory for scientific timing claims.

The subsequent learnability and matched-path experiments are complete; see
`PHASE_TEACHER_DIAGNOSTIC_20260712.md`. The phase target was learnable in a
head-only setting (`0.1324` to `0.1087` validation tau MAE, lower is better),
but did not improve matched path MAE over the synchronous Cartesian bridge on
either the original 128-system validation set or the 14-system enriched set.
The pseudo-teacher lane is therefore closed as a negative result rather than
promoted to the active paper method.

## Deterministic Experiment Gate

Before full training, run the following on exactly matched data and compute:

| Variant | Phase mode | Residual mode |
|---|---|---|
| Synchronous bridge | identity | off |
| Warp-only | learned | off |
| Residual-only | identity | normal residual |
| Full APNB | learned | normal residual |

All four variants now use `phase_residual_bridge_mode=cartesian_backbone`.
`se3_geodesic` is retained as a separate reference-bridge ablation.

The first endpoint-only screen did not pass this gate: full APNB matched
residual-only and phase collapsed to identity. Phase development continues only
through explicit pseudo-teacher or MD supervision; residual-only remains the
honest endpoint-corpus baseline.

## Canonical Data State

The current structural manifest audit produced:

```text
requested: 64,724
structurally valid: 64,606
structurally invalid: 118
validation: 2,007
```

The first four-way OracleMotion export revealed an additional ESM/canonical-
residue mismatch (`2fgu-A-ASN-505`, 94 canonical residues versus ESM length
188). It also revealed that strict export mode did not apply shard indices.

The exporter has been fixed so sharding is independent of bad-sample policy.
Retry exports use true four-way sharding and record/skip load failures:

```text
140933  train shard 0/4
140934  train shard 1/4
140935  train shard 2/4
140936  train shard 3/4
140937  dependent CPU merge
```

The completed 2,007-sample validation OracleMotion cache from job `140888` is
reusable.

## Scientific Position

### Current paper

- Endpoint-conditioned, holo-informed path reconstruction.
- OracleMotion is privileged endpoint-derived conditioning.
- AHoJ-DB supplies broad paired endpoint coverage.
- Independent MD paths supply held-out intermediate/path evidence.
- External physics and downstream sampling supply orthogonal validation.

### Next method extension

Introduce a global stochastic path latent `z` after deterministic APNB is
identified. Every sampled path remains endpoint exact:

\[
X_i(t,z)=
\operatorname{Exp}_{\gamma_i(\tau_i(t,z))}
\left[b(t)\Delta_i^\perp(t,z)\right].
\]

This extension requires MD path-ensemble supervision or a comparably defensible
set objective. Noise injection alone is not sufficient.

### Future deployment paper

Replace OracleMotion with a Stage-1 apo-and-ligand-conditioned posterior and a
predicted endpoint distribution. That is a separate scientific question and
must not dilute the current endpoint-conditioned method paper.

## Claims And Non-Claims

Primary claims:

- exact endpoint-constrained path learning;
- asynchronous residue event ordering;
- off-bridge correction on a product manifold;
- fast proposals for interpretation and physical-sampling initialization.

Non-claims:

- exact MD trajectories;
- physical time or kinetics;
- free-energy barriers without explicit physical sampling;
- apo-only unknown-holo prediction in the current model.

## Immediate Work Queue

### 2026-07-15 Matched MD Diagnostic

The first 10-epoch matched screen was invalid as evidence of learning for two
reasons:

1. its synchronous baseline used the legacy SE(3) bridge while learned variants
   used the Cartesian-backbone bridge;
2. the training phase head was evaluated on the legacy bridge even when
   `phase_residual_bridge_mode=cartesian_backbone`.

Training now evaluates the phase head on the configured bridge. Validation also
uses a fixed MD replica instead of rotating replicas by epoch, so checkpoint
selection is comparable. Normal-residual supervision is evaluated at the MD
target phase, matching the tangent-space coordinates in which the cached target
was defined. These fixes are in commit `f435c1f7`.

The corrected pilot uses a strict 17-system train / 6-system held-out split.
The held-out side contains 21 independent silver MD replicas. All rows below
use 20 path steps and the same Cartesian-backbone bridge:

| Variant | Product RMSE ↓ | Translation MAE Å ↓ | Rotation MAE rad ↓ | Chi MAE rad ↓ | Phase tau MAE ↓ | Phase Spearman ↑ | Pair accuracy ↑ | Contact timing MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Synchronous Cartesian | 1.475828 | 0.928127 | 0.333600 | 0.317907 | 0.210338 | n/a | 0.0000 | 0.398950 |
| Phase-only, best epoch 17 | **1.286644** | **0.820931** | **0.307076** | 0.281851 | **0.189226** | **0.2660** | **0.6151** | **0.319078** |
| Normal-only capacity diagnostic | 1.479336 | 0.940285 | 0.339049 | 0.313659 | 0.210338 | n/a | 0.0000 | 0.399315 |
| Joint phase + normal, best epoch 21 | 1.301359 | 0.852455 | 0.310407 | **0.277420** | 0.197640 | 0.0184 | 0.5040 | 0.329653 |

Relative to synchronous Cartesian, phase-only improves product RMSE by 12.82%,
translation by 11.55%, rotation by 7.95%, chi by 11.34%, phase tau MAE by
10.04%, and contact timing MAE by 20.02%. It wins 20/21 replicas on product
RMSE and 21/21 on chi MAE. A six-system paired bootstrap remains positive for
all four path-error components, but six systems are not enough for final paper
statistics.

Contact-event coverage (0.3325) and transient-contact recall (0.2367) are
unchanged. This is expected for a time warp that changes event timing without
creating a new spatial route. The normal-only branch does not generalize, and
the joint branch does not yet beat phase-only overall. Therefore the current
evidence supports learned asynchronous phase, not a successful off-bridge
normal residual.

The normal-only run above is a capacity diagnostic, not the final four-way
ablation: its cached normal target is defined at `tau_target`, while a proper
residual-only ablation fixes `tau=t` and requires an identity-phase residual
target. That target must be exported explicitly before the final matched table.

Updated queue:

1. Treat phase-only as the current best validated deterministic model.
2. Expand MD supervision from 17 training systems toward at least 100 systems,
   preserving protein-family and ligand-scaffold separation.
3. Export identity-phase normal targets for the scientifically correct
   residual-only ablation.
4. Re-test the joint model with more systems, controlled trunk unfreezing, and
   residual magnitude/smoothness regularization; do not claim residual success
   before it beats phase-only on held-out paths.
5. Repeat the four-way matched screen and report system-level confidence
   intervals, failure rates, and tail geometry metrics.
6. Reserve gold atomistic transitions for final external validation and keep
   the stochastic multipath extension after the deterministic decomposition is
   identified.

### Smooth-envelope target contract and matched screen (2026-07-16)

The main APNB residual envelope is now `sin2 = sin^2(pi t)`. Historical `poly`
targets and checkpoints remain loadable and must continue to use `poly`; new
targets record the envelope explicitly and incompatible cache contracts cannot
be merged silently.

The 82-path collection was re-exported under both phase references:

- inferred phase: 81/82 paths passed; one path was conservatively rejected at
  4.918% residual-supervision coverage against the fixed 5% gate;
- identity phase: 82/82 paths passed;
- both caches use metric scales `(rotation, translation, chi) = (1, 1, 1)`.

The first `sin2` screen used a learning rate and supervision weight that were
too small for component identification (`2e-5` and `0.1`, respectively), while
training the entire 15.3M-parameter model under competing auxiliary losses.
All three learned rows stayed within 0.03% of the synchronous bridge. This was
an optimization collapse, not evidence against the target or evaluator: the
previous strong phase checkpoint improved product RMSE by 14.38% when
re-evaluated against the exact new `sin2` reference.

The corrected capacity screen trained only the relevant phase/residual heads
for 100 epochs with `lr=3e-4`, unit MD supervision, and zero auxiliary or
residual-regularization weights. It used the same strict 17-system train / six-
system held-out split, with 21 held-out replicas and 20 generated path steps.
The table reports system-macro means:

| Variant | Product RMSE ↓ | Translation MAE Å ↓ | Rotation MAE rad ↓ | Chi MAE rad ↓ | Phase tau MAE ↓ | Phase Spearman ↑ | Pair accuracy ↑ | Contact timing MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Synchronous Cartesian | 1.554051 | 0.986568 | 0.383072 | 0.340054 | 0.225814 | n/a | 0.0000 | 0.423953 |
| Phase-only, best epoch 15 | **1.317409** | **0.856073** | **0.331144** | 0.291845 | **0.207280** | **0.0904** | **0.5412** | **0.344684** |
| Residual-only, best epoch 6 | 1.554039 | 0.990337 | 0.383358 | 0.337615 | 0.225814 | n/a | 0.0000 | 0.414969 |
| Full APNB, best epoch 21 | 1.322098 | 0.869276 | 0.338023 | **0.289076** | 0.207327 | -0.0311 | 0.4817 | 0.345138 |

Phase-only improves product RMSE by 15.23%, translation by 13.23%, rotation by
13.56%, chi by 14.18%, phase tau MAE by 8.21%, and contact timing MAE by
18.70% relative to the synchronous bridge. It wins all six systems on product,
rotation, and chi error. Six-system paired bootstrap intervals are positive for
all six lower-is-better metrics, although this sample is still too small for
paper-level uncertainty estimates.

Residual-only is statistically indistinguishable from the synchronous bridge.
Its training normal MAE falls from 0.726 to 0.608 while held-out normal MAE
worsens from 0.793 to 0.842, which is direct evidence of overfitting. Full APNB
preserves most phase gains but does not beat phase-only: translation is worse
on five of six systems, product and rotation are slightly worse, and its phase-
order metrics are lower. Its held-out interior peptide loss at the selected
checkpoint is 0.0302 versus 0.00774 for phase-only. Contact-event coverage and
transient-contact recall are unchanged across all four rows.

Therefore this component-identification experiment succeeds for asynchronous
phase and fails the promotion gate for the normal residual. Phase-only is the
current deterministic anchor. Full APNB remains an implemented hypothesis and
must not be presented as validated until more MD training systems and a
controlled residual retest make it beat phase-only on independent path and
geometry metrics.

### Frozen-corpus deterministic residual decision (2026-07-19)

The silver corpus now contains 326 systems and 1,340 accepted replicas. Its
family/scaffold-disjoint consensus split has 241 train, 30 validation, and 32
untouched test systems. The from-scratch ten-epoch screen confirms warp-only as
the deterministic anchor: system-macro Product RMSE is 1.305897 versus 1.405388
for the synchronous Cartesian bridge (lower is better).

A same-system leave-one-replica-out consensus oracle reduces residual MSE by
43.29%, proving that a shared deterministic residual exists within systems.
However, a route-mode oracle is worse than consensus. The consensus target is
also strongly low-rank: rank four explains 92.49% of median training residual
energy and 93.67% on validation. These findings justified one graph-coupled
rank-four residual test, not an open-ended architecture sweep.

The rank-four implementation passed CPU, CUDA, and end-to-end trainer smokes,
but did not generalize. In residual-only training, validation loss worsened
from 0.3865 to 0.4394 while training loss fell. Rank-four full reaches Product
RMSE 1.310484, which is worse than warp-only; its paired improvement is
`-0.00459` with 95% CI `[-0.03484, 0.02522]`. Rotation and chi are
significantly worse. A small contact-event coverage increase does not transfer
to aggregate path quality.

The deterministic normal residual is therefore frozen as a reproducible
negative result. Do not sweep residual rank, gate bias, learning rate, or trunk
initialization on this split. The controlled observed/hidden-route benchmark
confirms that inference-time route information can recover the intended full
decomposition, while hidden routes collapse to the deterministic mean. Reopen
Path-4 only with such route-identifying conditioning or a substantially larger
disjoint corpus that establishes a positive endpoint-conditioned learning
curve. A stochastic latent remains future work and requires new evidence
that route modes beat deterministic consensus. At this decision point the
32-system test split was still untouched; the audited one-time strict30
evaluation is recorded below.

### Phase-specificity control and frozen strict test (2026-07-20)

The matched phase screen is complete. It compared global monotone,
residue-wise monotone, residue-wise endpoint-fixed non-monotone, and a
chain-coupled endpoint-fixed non-monotone warp with the same trunk, head budget,
MD supervision, split, and optimization protocol. Validation selected the
chain-coupled non-monotone variant with unit scale and one graph-smoothing step.
This is the frozen deterministic candidate; it should be described as a
chain-coupled residue phase field, not as recovered physical kinetics.

The original 32-system test manifest was audited before headline reporting.
Two systems had incomplete production coordinate masks despite matching
residue-identity hashes: `6y6n-A-ODQ-501` had no valid production nodes and
`7k8h-C-9F2-302` had 261/265 valid nodes. The immutable benchmark is therefore
the remaining 30 systems, for which production-loader and endpoint-cache
coordinates agree exactly. The exclusions and generated manifest are recorded
by `scripts/audit_stage2_md_reference_subset.py`.

On this strict 30-system test, the three-seed chain candidate has system-macro
Product RMSE `0.956221`, translation MAE `0.518536 A`, phase tau MAE `0.255128`,
and pair-order accuracy `0.573426`. The global-monotone row has Product RMSE
`1.046872` and translation MAE `0.524586 A`; the independent non-monotone row
has `0.991824` and `0.535634 A`. The chain candidate has the best point
estimates for path geometry, but its paired confidence intervals against both
learned controls cross zero. It is also worse than the independent
non-monotone control on tau MAE and pair ordering. Therefore the test supports
learned endpoint-fixed temporal correction, but does not establish physical
timing recovery or a statistically decisive chain-coupling gain.

The external strict-test benchmark uses the same MD references and reports
C-alpha/frame-origin translation error as the primary common metric. Analytic
and ANM baselines now consume the canonical Stage-2 endpoint caches rather than
raw PDB coordinates, and missing-system coverage is explicit. The complete
table and paired intervals are tracked in
`MD_REFERENCE_STRICT30_BENCHMARK_20260720.md`.

Against endpoint-completed external paths, the chain model significantly beats
smoothstep, ANM20, AdaptiveANM50, and eBDIMS2 on translation path error. The
eBDIMS2 comparison is `+1.937439 A` improvement on 29 shared systems with 95%
paired CI `[0.154900, 5.396881]`. The linear-morph mean is worse, but its
interval crosses zero because of system-level heterogeneity. Pair-order
accuracy is significantly better than all five external baselines.

TPS-Flow reproduction is tracked separately in
`docs/TPS_FLOW_REPRODUCTION_PLAN_20260719.md`. Its released system-specific
checkpoints are eligible for a public-system case-study table, not automatic
inference-only evaluation on the frozen 326-system corpus.

## Repository Posture

The local branch is `codex/stage2-esm-repa-enhance`. Recent method-contract
commits are:

- `f435c1f7`: align phase-normal supervision coordinates;
- `42123470`: use the smooth endpoint envelope and immutable cache contracts;
- `fa35a554`: add the matched APNB screen submitter;
- `63020fc6`: record audit-rejected targets during cache assembly.
- `3947239c`: evaluate the MD holdout from the correct master manifest.

Do not overwrite historical checkpoints, logs, or failed export directories.
Use unique tags and output paths for every retry.
