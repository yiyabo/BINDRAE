# Current Project Status

Date: 2026-07-10

This is the operational source of truth for the current BINDRAE research track.
For the method and manuscript logic, read
`BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md` first.

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

Code parameterization:

```text
PATH_PARAMETERIZATION=phase_orthogonal_residual_v1
```

Manuscript working name:

```text
Asynchronous Phase-Normal Bridge (APNB)
```

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

## Deterministic Experiment Gate

Before full training, run the following on exactly matched data and compute:

| Variant | Phase mode | Residual mode |
|---|---|---|
| Synchronous bridge | identity | off |
| Warp-only | learned | off |
| Residual-only | identity | normal residual |
| Full APNB | learned | normal residual |

The full method advances only if it beats both learned single-component models
and preserves controlled residual magnitude.

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

1. Complete and verify the canonical OracleMotion retry shards.
2. Merge the full training cache.
3. Run 5-10 epoch deterministic four-model screening.
4. Evaluate phase use, residual use, contact order, path error, and external
   geometry.
5. Build a controlled manifold benchmark with known path truth.
6. Freeze the deterministic method before implementing stochastic multipath.

## Repository Posture

The local branch is `codex/stage2-esm-repa-enhance` and remains intentionally
dirty while canonical-residue fixes, APNB implementation, tests, and launch
scripts are reviewed as separate commit units.

Do not overwrite historical checkpoints, logs, or failed export directories.
Use unique tags and output paths for every retry.
