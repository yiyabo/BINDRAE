# BINDRAE

BINDRAE studies ligand-conditioned protein conformational paths between known
apo and holo endpoint structures.

## Active Research Question

```text
Given:
  - an apo endpoint,
  - a holo endpoint,
  - an aligned ligand pose,
  - protein sequence features,

generate:
  - an ordered, endpoint-exact atom14 conformational path.
```

The active task is endpoint-conditioned path reconstruction. It is not docking,
apo-only holo prediction, or molecular-dynamics simulation.

## Current Method

The implemented Stage-2 method is an **Asynchronous Phase-Normal Bridge** over
per-residue backbone frames and side-chain torsions:

\[
X_i(t)=
\operatorname{Exp}_{\gamma_i(\tau_i(t))}
\left[b(t)\Delta_i^\perp(t)\right].
\]

- `gamma_i` is an analytic apo-to-holo bridge.
- `tau_i(t)` is a learned monotone residue-wise phase.
- `Delta_i^perp(t)` is a learned rigid/chi residual projected normal to the
  endpoint-bridge tangent under a product metric.
- `b(0)=b(1)=0` guarantees exact endpoints.

The protein state lies on

\[
\prod_i\left(\mathrm{SE}(3)\times\mathbb T^{k_i}\right),
\]

and intermediate frames are decoded to atom14 coordinates with differentiable
forward kinematics.

## Why This Parameterization

Earlier free-flow models learned useful motion but did not reliably close at the
known holo endpoint. Warp-only bridges closed exactly but could not leave the
fixed endpoint arc. The current method separates:

- **when residues move** through the phase field;
- **how they detour** through the normal-space residual.

This decomposition is the central methodological claim.

## Deterministic And Stochastic Tracks

The deterministic phase-normal bridge is implemented and has passed unit,
CUDA-gradient, training-smoke, and evaluator-smoke checks.

A stochastic multipath extension is planned:

\[
X_i(t,z)=
\operatorname{Exp}_{\gamma_i(\tau_i(t,z))}
\left[b(t)\Delta_i^\perp(t,z)\right],
\qquad z\sim p(z\mid x_0,x_1,c).
\]

The first version will use a global path latent so sampled transition channels
remain residue-coherent. This extension will be implemented only after the
deterministic phase and residual components are independently validated.

## OracleMotion

The current upper-bound lane uses OracleMotion, a residue-aligned conditioning
field computed from apo/holo endpoints. It summarizes rigid/chi endpoint motion
and contact changes. It contains no MD intermediate frames, velocities, forces,
or physical time.

A future paper will replace OracleMotion with an apo-and-ligand-conditioned
Stage-1 posterior and predicted endpoint distribution.

## Validation Program

The method is evaluated through complementary evidence:

1. controlled manifold path tasks with known ground truth;
2. large paired apo/holo evaluation on AHoJ-DB;
3. held-out MD transition paths, used only for evaluation or explicitly
   separated fine-tuning data;
4. external OpenMM/Rosetta-style physical checks;
5. downstream acceleration of string, targeted-MD, or ensemble sampling.

The model generates path proposals. It does not claim to reproduce an exact MD
trajectory or physical timescale.

## Start Here

- [Conference method blueprint](docs/BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md)
- [Current project status](docs/CURRENT_PROJECT_STATUS_20260710.md)
- [Metrics and baseline plan](docs/STAGE2_COMPARISON_METRICS_AND_BASELINES_20260701.md)
- [Path-baseline literature scan](docs/PATH_BASELINE_LITERATURE_SCAN_20260630.md)
- [Documentation index](docs/INDEX.md)

## Code Layout

- `src/stage2/models/torsion_flow.py`: ligand-conditioned geometry trunk and
  phase/residual heads.
- `src/stage2/modules/phase_residual.py`: endpoint envelope and product-metric
  tangent-normal projection.
- `src/stage2/training/trainer.py`: deterministic phase-normal path,
  regularization, training, and checkpoint handling.
- `scripts/train_stage2.py`: Stage-2 training CLI.
- `scripts/evaluate_stage2_transition_paths.py`: transition-path evaluator.
- `scripts/slurm/`: active cluster launchers.
- `docs/archive/`: superseded plans and experiment provenance.

## Current Execution Order

1. Finish the canonical OracleMotion cache export.
2. Run synchronous, warp-only, residual-only, and full-model screening.
3. Validate the deterministic model on controlled and MD path benchmarks.
4. Add and validate the stochastic global path latent.
5. Run full-scale benchmarks and prepare a method-first AI conference paper.
