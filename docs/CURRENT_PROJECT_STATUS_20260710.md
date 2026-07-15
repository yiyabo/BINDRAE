# Current Project Status

Date: 2026-07-16

This is the operational source of truth for the current BINDRAE research track.
For the method and manuscript logic, read
`BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md` first.
For trajectory acquisition, evidence tiers, manifest fields, and the cluster MD
pilot, read `MD_TRANSITION_CORPUS_PLAN_20260713.md`.

## Current Objective

Develop and validate an endpoint-exact, ligand-conditioned protein
conformational path model whose main novelty is an identifiable decomposition
between residue-wise asynchronous progress and off-bridge spatial correction.

The current paper setting is:

```text
known apo + known holo + aligned ligand pose
    -> endpoint-conditioned path reconstruction
```

It is not docking, apo-only holo prediction, or physical MD generation.

## Active Method

Validated endpoint-corpus baseline:

```text
PATH_PARAMETERIZATION=phase_orthogonal_residual_v1
PHASE_RESIDUAL_TAU_MODE=identity
PHASE_RESIDUAL_BRIDGE_MODE=cartesian_backbone
```

Research candidate retained for ground-truth path supervision:

```text
PHASE_RESIDUAL_TAU_MODE=learned
Asynchronous Phase-Normal Bridge (APNB)
```

The learned-phase claim is currently unvalidated and must not be presented as
an endpoint-only result. The normal-residual decomposition remains implemented;
its scientific value must be judged against the synchronous Cartesian bridge.

The path is

\[
X_i(t)=
\operatorname{Exp}_{\gamma_i(\tau_i(t))}
\left[b(t)\Delta_i^\perp(t)\right].
\]

Implemented properties:

- exact apo/holo endpoints;
- monotone learned residue phase;
- tangent-normal residual projection on `SE(3) x T^k`;
- explicit rotation/translation/chi metric scales;
- dedicated zero-initialized residual heads;
- residual magnitude, temporal, neighbor, and background controls;
- atom14 endpoint-consistent FK decoding;
- evaluator support for deterministic component ablations.
- Cartesian `N/CA/C` endpoint bridge with locally differentiated tangents;
- reproducible `se3_geodesic` reference-bridge ablation.
- `sin^2(pi t)` residual envelope as the main contract, giving both zero
  displacement and zero residual slope at the two endpoints;
- immutable MD-target cache contracts that record phase mode, residual
  envelope, and rotation/translation/chi metric scales.

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

The next matched screen uses the same 17/6 system split, trunk, objective,
two-A100 budget, batch size, and 40-epoch ceiling for every trainable row:

```text
143174  phase-only     pending
143175  residual-only  pending
143176  full APNB      pending
```

This screen is a deterministic component-identification experiment, not a
final paper result. Full APNB is promoted only if it beats phase-only and
residual-only on held-out path metrics while retaining controlled residual and
tail-geometry behavior.

## Repository Posture

The local branch is `codex/stage2-esm-repa-enhance`. Recent method-contract
commits are:

- `f435c1f7`: align phase-normal supervision coordinates;
- `42123470`: use the smooth endpoint envelope and immutable cache contracts;
- `fa35a554`: add the matched APNB screen submitter;
- `63020fc6`: record audit-rejected targets during cache assembly.

Do not overwrite historical checkpoints, logs, or failed export directories.
Use unique tags and output paths for every retry.
