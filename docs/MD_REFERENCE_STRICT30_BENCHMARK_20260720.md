# Strict30 MD-Reference Path Benchmark

## Decision

The frozen deterministic candidate is the from-scratch, chain-coupled,
endpoint-fixed non-monotone phase model with phase scale `1.0` and one graph
smoothing step. It generates an endpoint-exact path from known apo/holo
structures and an aligned ligand pose. It is a path-reconstruction model, not a
physical MD trajectory generator and not an apo-only holo predictor.

This document supersedes preliminary results on the original 32-system test
manifest.

## Strict Test Contract

Before headline evaluation, the 32-system manifest was audited against the
production Stage-2 loader, MD-reference cache, residue identity, and valid-node
masks. Two systems failed the coordinate-availability contract:

| System | Reason |
|---|---|
| `6y6n-A-ODQ-501` | Production `node_mask` was 0/116. |
| `7k8h-C-9F2-302` | Production `node_mask` was 261/265 while the MD cache exposed 264 valid nodes. |

The frozen strict set contains the other 30 systems and 126 MD replicas. On all
30 systems, the production DataLoader and canonical endpoint caches have exact
coordinate agreement. The audit artifact is
`logs/stage2/md_reference_eval/test32_strict_subset_audit_20260720.json`; the
immutable manifest is
`processed_data/triplets/ablation_subsets/stage2_oracle_motion_mdphase_consensus303_fam30scaf_blockv2_test_strict30_20260720.txt`.

## Metrics

- **MD translation MAE (A), lower is better:** progress-aligned C-alpha/frame-
  origin path error against held-out MD references. This is the primary common
  metric for BINDRAE and C-alpha-only external baselines.
- **Product RMSE, lower is better:** BINDRAE full-state path error combining
  translation, rotation, and chi terms. It is not reported for C-alpha-only
  baselines.
- **Phase tau MAE, lower is better:** error of inferred residue progress against
  the MD silver phase.
- **Pair-order accuracy, higher is better:** agreement of residue lead-lag order
  with the MD silver reference.
- **Endpoint C-alpha error (A), lower is better:** final-frame error to the
  canonical holo endpoint. BINDRAE is exact by construction.

All headline means are system-macro averages. Pairwise claims use 100,000-draw
system-level paired bootstrap intervals; incomplete baselines are compared only
on their shared systems and report coverage explicitly.

## Internal Phase Controls

| Method | Product RMSE down | Translation MAE A down | Tau MAE down | Pair order up |
|---|---:|---:|---:|---:|
| Global monotone, seed 42 | 1.046872 | 0.524586 | 0.256574 | 0.000000 |
| Independent non-monotone, seed 42 | 0.991824 | 0.535634 | **0.241670** | **0.653054** |
| Chain non-monotone, three-seed mean | **0.956221** | **0.518536** | 0.255128 | 0.573426 |

The chain model has the best point estimates for product and translation path
error. Its translation and product intervals versus global monotone and
independent non-monotone both cross zero, so the strict test does not establish
a statistically decisive chain-coupling gain. The independent non-monotone
control is significantly better on tau MAE and pair ordering. The defensible
claim is learned endpoint-fixed temporal correction with controlled spatial
coupling, not recovery of physical residue timing.

## External C-Alpha Baselines

| Method | Coverage | Translation MAE A down | Tau MAE down | Pair order up | Endpoint error A down | Final step A down |
|---|---:|---:|---:|---:|---:|---:|
| BINDRAE chain, three-seed mean | 30/30 | **0.518536** | **0.255128** | **0.573426** | exact by construction | n/a |
| Linear morph | 30/30 | 0.855984 | 0.271831 | 0.490342 | 0.000002 | 0.346637 |
| Smoothstep morph | 30/30 | 0.761172 | 0.302770 | 0.490130 | 0.000002 | 0.050267 |
| ANM20, endpoint-completed | 30/30 | 2.345441 | 0.431445 | 0.437726 | 0.000000 | 5.993986 |
| AdaptiveANM50, endpoint-completed | 29/30 | 0.672966 | 0.357402 | 0.455365 | 0.000000 | 2.750951 |
| eBDIMS2, endpoint-completed | 29/30 | 2.460712 | 0.385203 | 0.530414 | 0.000000 | 3.605608 |

AdaptiveANM50 could not generate nonzero modes for `1tjw-C-AS1-1001` and is
therefore reported at 29/30 coverage. eBDIMS2 timed out before producing any
frame for `2ajs-H-P33-701`, also giving 29/30 coverage. The common coordinate
contract uses the same aligned `apo_backbone.npz` and `holo_backbone.npz`
C-alpha axes consumed by Stage-2; raw-PDB global-frame artifacts are not
accepted.

ANM and eBDIMS2 are reported in an endpoint-completed variant that appends the
known canonical holo structure when the native path does not reach it. This is
favorable to the baselines on endpoint error but exposes a large final jump.
The native eBDIMS2 result is similar: translation MAE `2.469728 A`, tau MAE
`0.394413`, pair-order accuracy `0.531087`, and endpoint error `1.995128 A`.

## Paired Results

For translation MAE, positive improvement means the BINDRAE chain mean is
better:

| Comparator | Improvement A | 95% paired CI A | Shared systems |
|---|---:|---:|---:|
| Global monotone | +0.006051 | [-0.046945, 0.077951] | 30 |
| Independent non-monotone | +0.017098 | [-0.039056, 0.099112] | 30 |
| Linear morph | +0.337449 | [-0.051715, 1.052893] | 30 |
| Smoothstep morph | +0.242636 | [0.002482, 0.664701] | 30 |
| ANM20, endpoint-completed | +1.826905 | [0.196904, 4.973314] | 30 |
| AdaptiveANM50, endpoint-completed | +0.177119 | [0.086972, 0.281268] | 29 |
| eBDIMS2, endpoint-completed | +1.937439 | [0.154900, 5.396881] | 29 |

The chain model significantly improves translation path error over smoothstep,
ANM20, AdaptiveANM50, and eBDIMS2. The linear-morph mean is worse, but the
interval crosses zero because the per-system effect is heterogeneous.
Pair-order accuracy is significantly higher than all five external baselines.

## Claim Boundary

The result supports an endpoint-exact learned conformational path proposal that
outperforms several analytic and elastic-network baselines on held-out
MD-reference geometry and event order. It does not prove that one generated
path is the unique physical transition mechanism, recover kinetics, or replace
MD. Rotation, chi, atom14 contact, and full-atom physical-validity comparisons
must not be inferred from C-alpha-only baseline rows.
