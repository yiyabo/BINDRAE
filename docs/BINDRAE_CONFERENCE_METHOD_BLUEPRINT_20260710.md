# BINDRAE Conference Method Blueprint

Date: 2026-07-10

Status: canonical method document for the current conference-oriented research
track. It separates implemented components, near-term experiments, and proposed
extensions. Historical plans must not override this document.

## Working Title

**Endpoint-Exact Asynchronous Phase-Normal Bridges for Protein Conformational
Path Learning**

Working method name: **BINDRAE-APNB** (Asynchronous Phase-Normal Bridge).

The name is provisional. The method definition, rather than the acronym, is the
stable contract.

## One-Sentence Method

Given known apo and holo endpoints and an aligned ligand pose, BINDRAE learns an
endpoint-exact conformational path on a product manifold by decomposing motion
into a monotone residue-wise phase field and a controlled normal-space spatial
residual.

## Scope And Claim Boundary

### Main setting

The current paper studies **endpoint-conditioned path reconstruction**:

```text
input  = apo endpoint + holo endpoint + aligned ligand pose + sequence features
output = an ordered atom14 conformational path connecting apo to holo
```

The structural state for residue `i` is

\[
x_i=(F_i,\chi_i)\in \mathrm{SE}(3)\times\mathbb T^{k_i},
\]

and the complete protein state lies on

\[
\mathcal M=\prod_{i=1}^{N}
\left(\mathrm{SE}(3)\times\mathbb T^{k_i}\right).
\]

The ligand pose is assumed to be known and aligned consistently with the apo
frame. This is not a docking or ligand-pose search task.

### OracleMotion posture

OracleMotion is a privileged endpoint-derived conditioning field. It contains
local apo-to-holo rigid/chi displacement summaries, motion masks, and contact
formation/release indicators. It contains no MD intermediate frames, forces,
velocities, or physical time information.

For the current study, OracleMotion is an upper-bound conditioning lane that
tests whether the path architecture can exploit exact local endpoint evidence.
It must never be described as deployable apo-only inference.

### Claims we may make

- Endpoint-exact conformational path reconstruction.
- Learned asynchronous residue progress.
- Learned off-bridge geometric correction.
- Fast path proposals that can be compared with MD ensembles and used to
  initialize downstream physical sampling.
- Ligand-aware pocket/contact transition modeling.

### Claims we must not make

- Generation of an exact MD trajectory.
- Prediction of physical time, kinetics, `kon/koff`, or free-energy barriers.
- Apo-only prediction of an unknown holo state in the current Oracle setting.
- Replacement of molecular dynamics.
- A unique ground-truth transition path.

The model time variable is a normalized **path progress coordinate**, not
physical time.

## Why Earlier Routes Were Insufficient

### Free-flow integration

The original learned flow produced useful path motion and performed strongly
against several geometric baselines, but numerical integration did not
guarantee arrival at the known holo endpoint. That is structurally awkward for
an endpoint-conditioned task.

### Endpoint-zero residual without an identifiable decomposition

An analytic bridge plus an unconstrained endpoint-zero residual guaranteed the
boundary, but the residual could duplicate along-bridge progress. The model had
no clear separation between *when to move* and *how to deviate from the bridge*.

### Warp-only bridge

Residue-wise time warping represented asynchronous progress while preserving
the endpoints, but every residue remained on its fixed endpoint arc. It could
not represent clash avoidance, rotamer detours, or transient off-bridge states.

### Loss-only clearance and anchor sweeps

Stronger ligand-clearance or bridge-anchor penalties produced small proxy gains
but degraded path error. The central limitation was representational, not a
missing repulsion coefficient.

### Representation-only enhancements

ESM last-layer fusion and REPA-style alignment remain useful representation
experiments, but they are not the main methodological novelty. The conference
paper must center the path parameterization, not feature fusion.

## Deterministic Asynchronous Phase-Normal Bridge

### Reference endpoint bridge

For each residue, let

\[
\gamma_i:[0,1]\rightarrow
\mathrm{SE}(3)\times\mathbb T^{k_i}
\]

be an analytic endpoint bridge from apo to holo. Rigid frames use Lie-group
interpolation and chi angles use wrapped shortest-arc interpolation.

The synchronous baseline is simply `x_i(t)=gamma_i(t)`.

### Monotone residue-wise phase

The model predicts positive phase rates

\[
r_i(t)=\operatorname{softplus}(a_i(t))+\epsilon,
\]

which are normalized to construct

\[
\tau_i(t)=
\frac{\int_0^t r_i(s)\,ds}
     {\int_0^1 r_i(s)\,ds}.
\]

This guarantees

\[
\tau_i(0)=0,\qquad
\tau_i(1)=1,\qquad
\partial_t\tau_i(t)>0.
\]

Different residues may therefore move at different stages while all residues
still complete the endpoint transition.

### Product-manifold tangent-normal decomposition

Let `v_i` be the endpoint-bridge tangent in the local product tangent space

\[
T_{\gamma_i}\mathcal M_i
\simeq \mathfrak{se}(3)\times\mathbb R^{k_i}.
\]

The network predicts a raw rigid/chi residual `Delta_i`. Under a product metric
that assigns explicit characteristic scales to rotation, translation, and chi,
the along-bridge component is removed:

\[
\Delta_i^\perp
=\Delta_i-
\frac{\langle\Delta_i,v_i\rangle_g}
     {\langle v_i,v_i\rangle_g}v_i.
\]

Residues with near-zero endpoint tangent are disabled because an off-bridge
direction is not identifiable for a stationary endpoint pair.

### Endpoint-exact spatial path

The final deterministic path is

\[
X_i(t)=
\operatorname{Exp}_{\gamma_i(\tau_i(t))}
\left[b(t)\Delta_i^\perp(t)\right],
\]

where `b(0)=b(1)=0`, for example

\[
b(t)=4t(1-t)
\quad\text{or}\quad
b(t)=\sin^2(\pi t).
\]

Rigid residuals are applied through the `SE(3)` exponential map and chi
residuals are wrapped on the torus. Apo and holo are inserted explicitly as the
first and final states.

### Functional interpretation

- `tau_i(t)` explains **when** residue `i` progresses along its endpoint arc.
- `Delta_i^perp(t)` explains **how** it leaves that arc to avoid obstacles or
  express transient states.
- The endpoint envelope controls **where in path time** spatial freedom is
  available.

This is the main identifiable decomposition that must be validated by the
paper's four-model experiment.

## Protein-Specific Network Instantiation

### Inputs

- Apo and holo residue frames and chi angles.
- Frozen ESM last-K residue embeddings.
- Aligned ligand atom/probe tokens.
- Pocket weights and node masks.
- Optional endpoint-derived OracleMotion features.

### Geometry trunk

The current implementation uses:

1. gated ESM feature fusion;
2. ligand cross-conditioning;
3. pair/edge geometric features;
4. a FlashIPA geometry trunk;
5. residue-level phase, residual-gate, rigid-residual, and chi-residual heads.

The residual output heads are zero-initialized, so the initial model is exactly
the phase bridge and can only depart from it when the data provide a useful
gradient.

### Structural decoding

Each intermediate state is decoded with differentiable OpenFold-style forward
kinematics to atom14 coordinates. Decode-only wrapped interpolation of
apo/holo backbone torsions preserves endpoint atom geometry without introducing
redundant learned backbone state.

## Stochastic Multi-Path Extension

### Motivation

Apo-to-holo transitions are not unique. Different initial velocities, solvent
fluctuations, force fields, and rare events can produce multiple transition
channels. A deterministic model should be presented as a primary path proposal,
not a complete transition ensemble.

### Proposed formulation

Introduce a **global path latent**

\[
z\sim p_\theta(z\mid x_0,x_1,c),
\]

where `c` contains ligand, sequence, and optional endpoint-derived conditioning.
The same latent conditions every residue through the geometry trunk, ensuring
globally coherent pathway modes. The stochastic path is

\[
X_i(t,z)=
\operatorname{Exp}_{\gamma_i(\tau_i(t,z))}
\left[b(t)\Delta_i^\perp(t,z)\right].
\]

For every sample `z`, monotone rate normalization and the endpoint envelope
preserve exact apo/holo boundaries.

### Why a global latent comes first

Independent residue noise would create incoherent local motion. A global path
latent should first choose a transition channel, while graph-coupled residue
heads translate that channel into coordinated phase and spatial corrections.
Local stochastic latents may be added later only under strong coupling and
regularization.

### Training requirements

Simply injecting Gaussian noise is not evidence of multi-path learning. A
stochastic model requires one or more of:

- multiple MD transition trajectories for the same or homologous endpoint
  system;
- path clustering or metastable-channel labels;
- a conditional latent-variable objective with trajectory reconstruction;
- best-of-K or set-likelihood supervision;
- an energy/validity filter that prevents diversity from becoming arbitrary
  distortion.

The recommended schedule is:

1. deterministic endpoint-only pretraining on the large AHoJ-DB lane;
2. deterministic APNB validation and component identification;
3. stochastic latent fine-tuning on a smaller MD path-ensemble corpus;
4. optional distillation back to a fast multi-path generator.

### Stochastic objectives

A conference-grade stochastic objective should combine path fidelity, ensemble
coverage, and physical validity:

\[
\mathcal L_{\mathrm{stoch}}
=\mathcal L_{\mathrm{set\mbox{-}path}}
+\lambda_{\mathrm{KL}}\mathcal L_{\mathrm{latent}}
+\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{phys}}
+\lambda_{\mathrm{collapse}}\mathcal L_{\mathrm{anti\mbox{-}collapse}}.
\]

Pairwise diversity alone is unsafe because it rewards nonphysical deviations.
Diversity must be measured jointly with coverage/precision against an MD
ensemble and external physical filters.

### Multi-path evaluation

- Endpoint error for every sampled path: lower is better and should be at the
  numerical floor.
- MD-ensemble coverage or recall: higher is better.
- MD-ensemble precision: higher is better.
- Best-of-K path error: lower is better.
- Contact-order mode coverage: higher is better.
- Pairwise diversity after alignment: higher is useful only at matched
  validity.
- Energy/clash/peptide violation distributions: lower is better.
- Latent-to-path mutual information or mode utilization: higher is better.

## Theoretical Properties To State And Prove

### Proposition 1: endpoint exactness

Because `tau_i(0)=0`, `tau_i(1)=1`, and `b(0)=b(1)=0`, every deterministic or
stochastic path satisfies

\[
X_i(0)=x_i^{\mathrm{apo}},\qquad
X_i(1)=x_i^{\mathrm{holo}}.
\]

### Proposition 2: phase monotonicity

Positive normalized rates imply nondecreasing residue progress and rule out
temporal reversal along the reference bridge.

### Proposition 3: tangent-normal identifiability

Under the chosen product metric,

\[
\langle\Delta_i^\perp,v_i\rangle_g=0.
\]

Thus phase and spatial residual cannot represent the same first-order
along-bridge motion.

### Proposition 4: global rigid-motion consistency

When the encoder and local frame operations are globally `SE(3)` equivariant,
applying a common rigid transform to apo, holo, and ligand transforms the
generated path by the same global rigid transform.

### Proposition 5: local representation in a tubular neighborhood

For a smooth embedded reference bridge with a valid normal neighborhood, a
nearby endpoint-fixed path can be represented locally by a phase coordinate and
a normal displacement field. This proposition supplies the geometric rationale
for APNB beyond an ad hoc architectural combination.

The final manuscript must state the assumptions clearly. Global uniqueness is
not guaranteed when the reference bridge self-intersects or leaves its normal
neighborhood.

## Learning Objective

### Implemented deterministic objective

The current code combines:

- frame and chi flow/path losses;
- peptide and atom14 geometry losses;
- clash and contact-transition losses;
- background stability and path smoothness;
- residual magnitude regularization;
- temporal residual smoothness;
- neighboring-residue residual smoothness.

The endpoint is enforced by parameterization and should not be optimized as a
soft terminal penalty in the main APNB mode.

### Conference-oriented unified interpretation

Rather than presenting a bag of losses, frame training as **amortized
constrained path optimization**. For a path `X`, define an action-like objective

\[
\mathcal J[X]
=\int_0^1
\left(
E_{\mathrm{geom}}(X_t)
+\lambda_v\|\dot X_t\|_g^2
+\lambda_c E_{\mathrm{contact}}(X_t)
\right)dt
+\lambda_{\mathrm{data}}D(X,\mathcal Y),
\]

where `Y` is an optional reference path or path ensemble. The network amortizes
this constrained optimization across proteins instead of optimizing a separate
spline or string for every endpoint pair.

### Two-level supervision strategy

1. **Large endpoint corpus:** learn endpoint-conditioned geometry, stable
   residue coordination, and physically filtered path proposals.
2. **MD path corpus:** supervise intermediate geometry, event ordering, and
   stochastic channel coverage without exposing MD frames at test time.

Arc-length or event-aligned progress should be used when MD trajectories have
different physical durations. Absolute MD time is not a valid target unless the
simulation protocol makes it identifiable.

## Required Four-Model Identification Experiment

All four variants must use the same trunk, data, objective, and compute budget.

| Variant | Phase | Spatial residual | Scientific question |
|---|---|---|---|
| Synchronous bridge | `tau_i(t)=t` | none | What does endpoint interpolation provide? |
| Warp-only | learned | none | Does phase recover asynchronous event order? |
| Residual-only | identity | normal residual | Does off-bridge motion improve geometry? |
| Full APNB | learned | normal residual | Are the two components complementary? |

The desired result is not merely that the full model wins. It should show:

- warp-only improves event ordering;
- residual-only improves off-bridge geometry and clash/energy;
- full APNB improves both;
- full APNB uses a smaller residual than residual-only;
- projected parallel cosine remains near zero;
- phase does not collapse to identity on every sample.

## Experimental Program

### Track A: controlled manifold paths

Construct synthetic `SO(3)`, `SE(3)`, and multi-rigid-body path tasks with known
obstacles and ground-truth asynchronous order. These experiments isolate the
method contribution from protein-specific encoders.

Primary metrics:

- path error: lower is better;
- obstacle collision: lower is better;
- event-order correlation: higher is better;
- endpoint error: lower is better;
- decomposition recovery: higher is better.

### Track B: AHoJ-DB endpoint-conditioned benchmark

Use strict train/validation/test separation, with protein-family and ligand-
scaffold controls where possible. The main purpose is broad structural
generalization, not physical trajectory truth.

### Track C: independent MD transition benchmark

No MD intermediate frame may enter inference. Compare generated paths with held-
out MD transition ensembles using progress-aligned geometry and event-order
metrics.

### Track D: external physical validation

Evaluate generated frames with tools not used as the primary training target:

- OpenMM or Rosetta energy;
- peptide and bond-geometry violations;
- maximum clash severity;
- relaxation stability;
- estimated path barrier proxies, clearly labeled as proxies.

### Track E: downstream utility

The strongest practical claim is acceleration rather than replacement of MD:

```text
BINDRAE path proposal
  -> string / targeted-MD / weighted-ensemble initialization
  -> physical relaxation and sampling
  -> reduced convergence time or increased transition yield
```

Flexible docking or transient-pocket recovery can provide a second downstream
task if the protocol is defined independently of the path training metrics.

## Baselines

### Analytic and optimization baselines

- synchronous rigid/chi bridge;
- cubic or geodesic spline;
- test-time optimized spline with matched physical losses;
- linear and endpoint-rescaled morphs.

### Classical protein-path baselines

- ANM/ENM/NMA path methods;
- AdaptiveANM-style variants;
- eBDIMS/eBDIMS2;
- targeted or restrained MD on a curated subset.

### Learned internal baselines

- original free-flow model;
- warp-only bridge;
- residual-only bridge;
- unconstrained residual without tangent-normal projection;
- deterministic versus stochastic APNB.

No-ligand and no-ESM ablations are not headline experiments. They may appear as
secondary controls only if a reviewer-facing question requires them. The main
ablations must test the proposed path decomposition.

## Core Evaluation Metrics

### Endpoint and path

- endpoint frame/chi/atom14 error: lower is better;
- path MAE or geodesic error: lower is better;
- direction accuracy: higher is better;
- terminal correction magnitude: lower is better and should be zero by design.

### Event structure

- contact formation/release ordering correlation: higher is better;
- event-time rank correlation: higher is better;
- formed/released contact success: higher is better;
- transient-contact precision and recall: higher is better.

### Physical validity

- clash rate and severity: lower is better;
- peptide geometry violation: lower is better;
- external energy distribution: lower is better only under a matched protocol;
- relaxation survival: higher is better.

### Decomposition diagnostics

- phase deviation from identity: diagnostic, not monotonically better;
- phase monotonicity violations: lower is better and should be zero;
- residual norm: diagnostic and should remain controlled;
- projected parallel cosine: lower is better and should be near zero;
- residual concentration near high-curvature/clash/rotamer events: higher
  localization is better if path validity is preserved.

## Main Contributions For A Conference Paper

1. A general endpoint-exact path parameterization on product manifolds.
2. An identifiable decomposition into monotone asynchronous phase and
   normal-space spatial correction.
3. A ligand-conditioned protein instantiation on residue `SE(3)` frames and chi
   tori with atom14 decoding.
4. A stochastic extension that preserves exact endpoints for every sampled
   path and targets transition-ensemble coverage.
5. Evaluation across controlled manifold tasks, large apo/holo pairs, held-out
   MD transitions, and external physical/downstream tests.

The stochastic extension becomes a claimed contribution only after it is
implemented and validated. Until then it is an explicit next method stage, not
part of the reported model.

## Draft Abstract

Protein conformational transitions are often studied from paired endpoint
structures, yet standard morphing methods constrain all residues to synchronous
endpoint interpolation while unconstrained learned flows do not guarantee
arrival at the target state. We introduce an endpoint-exact asynchronous
phase-normal bridge for conformational path learning on product manifolds. The
method represents each residue using an `SE(3)` frame and side-chain torsions,
learns a monotone residue-wise phase that controls progress along an analytic
apo-to-holo bridge, and predicts a metric-orthogonal spatial residual for
off-bridge detours. An endpoint-zero envelope guarantees exact boundary states,
while the tangent-normal decomposition prevents the phase and residual branches
from duplicating the same first-order motion. We instantiate the framework with
ligand-conditioned geometric attention and atom14 forward kinematics for
protein-ligand induced-fit paths. [RESULT SENTENCE.] On held-out molecular-
dynamics transitions, [RESULT SENTENCE], while external physical evaluation
shows [RESULT SENTENCE]. These results position the method as a fast path
proposal mechanism for interpreting and accelerating endpoint-conditioned
conformational sampling rather than as a replacement for molecular dynamics.

## Falsification Criteria

The method claim should be weakened or abandoned if any of the following hold:

- full APNB does not beat both warp-only and residual-only;
- learned phase remains effectively identity across the validation set;
- the residual alone explains nearly all motion;
- tangent-normal projection improves diagnostics but harms path quality;
- gains exist only on losses also used for training;
- MD event ordering is not better than a synchronous bridge;
- stochastic samples are diverse but have poor ensemble precision or validity;
- results disappear under protein-family or ligand-scaffold splits.

## Implementation Status

| Component | Status on 2026-07-10 |
|---|---|
| Deterministic phase-normal parameterization | Implemented |
| Monotone normalized phase rates | Implemented |
| Product-metric tangent-normal projection | Implemented |
| Endpoint-zero envelope and exact endpoint insertion | Implemented |
| Dedicated zero-initialized residual heads | Implemented |
| Residual magnitude/temporal/neighbor/background controls | Implemented |
| Atom14 endpoint-consistent decoding | Implemented |
| Four-model launcher/evaluation support | Implemented in code paths; formal run pending |
| Canonical full OracleMotion train cache | Re-export pending |
| Controlled manifold benchmark | Not implemented |
| Independent MD benchmark | Planned |
| Global stochastic path latent | Proposed, not implemented |
| Multi-path ensemble objective | Proposed, not implemented |
| Stage-1 replacement for OracleMotion | Future paper track |

## Execution Order

1. Finish canonical OracleMotion cache export and merge.
2. Run deterministic four-model screening for 5-10 epochs.
3. Select stable phase/residual scales and integration resolution.
4. Run the controlled manifold benchmark.
5. Freeze a deterministic APNB checkpoint and evaluate on the MD benchmark.
6. Implement the global stochastic path latent only after deterministic
   component identification succeeds.
7. Fine-tune stochastic APNB on MD path ensembles and evaluate coverage versus
   precision.
8. Run full AHoJ-DB training, external physics, and downstream acceleration.
9. Write the method-first conference paper.

## Venue Posture

The primary target is a method-oriented AI conference. AAAI or IJCAI is the
realistic first lane. ICLR becomes plausible if the product-manifold method,
theoretical properties, controlled benchmarks, and stochastic extension are all
strong. Bioinformatics remains a domain-journal fallback rather than the design
template for the method.
