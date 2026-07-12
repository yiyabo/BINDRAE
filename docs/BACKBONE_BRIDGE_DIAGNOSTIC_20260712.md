# Backbone Bridge Diagnostic

## Decision

Use a Cartesian backbone-triplet bridge as the next Stage-2 reference bridge:

```text
N_i(t)  = lerp(N_i^apo,  N_i^holo,  gamma(tau_i(t)))
CA_i(t) = lerp(CA_i^apo, CA_i^holo, gamma(tau_i(t)))
C_i(t)  = lerp(C_i^apo,  C_i^holo,  gamma(tau_i(t)))
F_i(t)  = Frame(N_i(t), CA_i(t), C_i(t))
```

The endpoint-zero phase-normal residual remains the learned off-bridge term.
The old independent per-residue SE(3) geodesic remains a matched ablation.
Bounded violation-gated peptide projection remains an optional validity layer,
not the main learned mechanism.

Do not use the single-root chain-internal NeRF bridge or the diagnostic
pose-graph optimizer as the training reference.

## Evidence

All values below use the same 128 validation systems and four path steps.
Lower is better for every listed metric.

| Path | Pocket path MAE | Active path MAE | Path action | Peptide mean | Peptide p95 | Peptide max |
|---|---:|---:|---:|---:|---:|---:|
| Per-residue SE(3) bridge | 0.2467 | 0.8037 | 31.04 | 30.1885 | 1.7924 | 1344.20 |
| Single-root chain NeRF | 3.7618 | 6.3515 | 1535.55 | 0.000925 | 0.00182 | 0.00188 |
| Cartesian backbone bridge | **0.2107** | **0.6898** | **22.01** | 0.03189 | 0.2199 | 0.4228 |
| Cartesian + gated projection | 0.2133 | 0.6977 | 22.18 | **0.01664** | **0.1042** | **0.2281** |

The Cartesian bridge also improved formed-contact path MAE from 0.9681 to
0.8395 and release path MAE from 0.8687 to 0.7357. Its ligand-clash proxy was
slightly worse, 0.1692 to 0.1760, so clash handling remains an explicit loss
and evaluation target.

## Failure Analysis

### Independent SE(3) geodesics

Interpolating every residue rotation independently can make neighboring
residue frames mutually inconsistent. The failure is highly heavy-tailed:
the top 20 systems contributed 93.65% of the validation objective, and the
worst interior peptide losses exceeded 1000. Removing the learned residual
made all top-20 peptide cases worse, so the residual was not the root cause.

### Single-root chain NeRF

Internal-coordinate reconstruction enforces peptide geometry almost exactly,
but local torsion interpolation accumulates along the entire chain. It caused
large downstream displacement, path action, contact error, and ligand clash.
This is a correctness reference for hard connectivity, not a useful global
path prior.

### Translation-only projection

Projecting complete residue-frame translations can repair severe peptide
outliers, but it lacks rotational degrees of freedom. Unbounded projection
required corrections as large as 88 Angstrom. Capping corrections at one or
two Angstrom preserved the path but left substantial peptide error.

### Anchored SE(3) pose graph

The pose-graph prototype balanced adjacent-frame transforms against the
absolute bridge, but it only modestly improved peptide loss and introduced
time-inconsistent corrections. On the 16-system smoke it increased path
length to 13.3, so it is not a training candidate in its current form.

## Why Cartesian Works

Interpolating endpoint backbone atoms before rebuilding frames avoids the
independent-rotation pathology without propagating a single local torsion
choice through the full chain. It is endpoint exact by explicit insertion,
cheap, differentiable, and compatible with per-residue phase progress.

It is not a hard kinematic guarantee. Different residue phases can still
stretch an adjacent bond, and a Cartesian interpolation can temporarily
shrink a bond when endpoint directions differ strongly. These residual errors
must be measured and controlled rather than hidden.

## Next Training Screen

1. Add `cartesian_backbone` as an explicit reference-bridge mode in the
   Stage-2 trainer and evaluator.
2. Keep `se3_geodesic` reproducible as the bridge ablation.
3. Run matched synchronous, phase-only, residual-only, and phase-normal
   smokes on the same subset and seed.
4. Log peptide mean, p95, p99, maximum, outlier share, and per-path correction
   norms in addition to the optimized scalar objective.
5. Use a robust or capped peptide training term so a few systems cannot
   determine checkpoint selection, while retaining uncapped physical metrics
   for reporting.
6. Do not start the 64k run until Cartesian phase-normal training is stable and
   improves contact/event-order metrics over the synchronous Cartesian bridge.

## Scope Of The Evidence

The current contact-distance target is derived from apo and holo endpoints.
It is useful for matched internal diagnostics but is not a true transition
trajectory. A separate MD-transition benchmark is still required to support
claims about intermediate ordering or path realism. The active paper task
remains endpoint-conditioned path reconstruction with privileged OracleMotion,
not apo-only holo prediction or physical MD generation.
