# BINDRAE Positioning And Benchmark Strategy

Date: 2026-06-29

This note records the current paper-positioning strategy after the Stage-2
OracleMotion, gated ESM, and REPA screening work. It incorporates the external
critique that the project must draw a sharp boundary against docking, cofolding,
generic protein-ensemble generation models, and static endpoint structure
prediction.

## Executive Position

BINDRAE should not be presented as a generic docking model, and the current
OracleMotion / holo-informed line should not be presented as a static endpoint
predictor. The clean current paper contract is:

> Given apo and holo endpoint states plus a ligand pose aligned in the apo frame,
> BINDRAE reconstructs a ligand-aware apo-to-holo induced-fit transition path over
> residue backbone frames and side-chain torsions.

This positioning is essential. Adjacent systems already cover broad versions of
flexible docking, protein-ligand complex prediction, conditional flow matching,
and protein conformational ensemble generation. The defensible BINDRAE claim is
not simply "ligand-conditioned apo-to-holo flow" and not "better static endpoint
prediction." The defensible current claim is endpoint-conditioned,
ligand-aware transition reconstruction, with path quality, contact switching,
pocket side-chain response, and intermediate structural validity treated as
first-class outputs.

Do not use a win-only endpoint strategy. The manuscript must not imply "compare
endpoint prediction if BINDRAE wins, hide it if BINDRAE loses." Instead, define
endpoint prediction as outside the main scope up front. Endpoint metrics are
reported as boundary closure, consistency, and validity diagnostics; the primary
benchmark is transition-path quality.

## Architecture Framing

The preferred one-paragraph architecture description is:

> BINDRAE is a two-stage, ligand-aware induced-fit transition model. In the
> current OracleMotion/holo-informed setting, Stage 1 or teacher features encode
> endpoint-conditioned local ligand-pocket interaction evidence. Stage 2 uses
> that local interaction field to drive a conditional bridge flow over residue
> `SE(3)` backbone frames and side-chain chi angles, reconstructing an apo-to-holo
> structural transition path between known endpoint states.

This wording is intentionally conservative. It does not claim that each primitive
is individually unprecedented. Instead, it claims a specific system contract and
evaluation target.

## Novelty Claims

Rank the novelty claims in this order:

| Rank | Claim | Manuscript posture |
| --- | --- | --- |
| 1 | Ligand-aware apo-to-holo transition path reconstruction between known endpoint states | Main contribution for the current paper |
| 2 | Contact switching, side-chain transition, and intermediate validity as first-class evaluation targets | Core scientific angle |
| 3 | Residue `SE(3)` frame plus side-chain chi transition modeling for pocket response | Core method detail, not unique alone |
| 4 | Soft Stage-1 local interaction prior guiding Stage-2 transition flow | Potential major biological contribution for the deployable line if ablations support it |
| 5 | Ligand-conditioned apo-to-holo bridge flow | Important but too broad as headline novelty |
| 6 | Gated multi-layer ESM residual fusion | Engineering improvement unless evaluator gains are large and reproducible |

The weakest claim is "we introduce a ligand-conditioned apo-to-holo flow over
protein frames and side chains." Reviewers can map that broad statement onto
DynamicBind, FlowDock, DynamicFlow-like methods, or other flexible-complex
generative models. A second weak claim is "we outperform static endpoint
predictors," because static complex prediction is crowded and not BINDRAE's
primary task. Always narrow the current claim to endpoint-conditioned transition
path reconstruction, contact-switch recovery, side-chain rearrangement, and
intermediate validity.

## Stage-1 Positioning

Do not lead with "teacher" or "posterior" in the abstract. Those terms are
useful internally but can imply a training trick or possible holo leakage.

Preferred language:

> Stage 1 is an amortized ligand-conditioned local interaction prior, trained by
> distillation from apo/holo structural evidence.

Method-section clarification:

> Stage 1 is trained as a teacher-distilled local posterior encoder over
> pocket/contact/side-chain variables, but at inference it functions as an
> apo-and-ligand-conditioned pocket prior for Stage 2.

Describe Stage-1 outputs as a local interaction field: pocket/contact logits,
residue switch likelihoods, side-chain rearrangement latents, and scalar or
latent conditioning signals. Avoid saying that Stage 1 predicts the holo
structure unless the experiment being discussed is explicitly an endpoint-anchor
ablation.

The two-stage design is justified only if ablations show that Stage 1 improves
where induced fit lives: pocket heavy-atom RMSD, chi recovery, formed-contact
recall, released-contact success, contact-LDDT-like scores, and clash reduction.

## Stage-2 And OracleMotion Boundary

Stage 2 is the deployable transition model architecture, but the current
OracleMotion-conditioned lane must be labeled carefully.

OracleMotion is an upper-bound, teacher, or diagnostic condition. It uses
information unavailable in a normal inference setting and must not be presented
as the paper's deployable main model. Every paper-facing table should separate:

- Oracle / teacher upper bound.
- Deployable BINDRAE.
- Ablated deployable baselines.

The current 24k OracleMotion matrix is still valuable because it tests whether
the Stage-2 architecture, ESM interface, and optional REPA alignment can exploit
residue-aligned motion information at scale. It does not by itself close the
deployable-method claim.

## Gated ESM And REPA Posture

The current 12k gated ESM result is useful but should not be oversold. The best
screened setting was ESM last-7 `gated_residual` fusion with `gate_bias=-2.0`
and `gate_context=none`, improving `val_total_no_repa` from the single-ESM
baseline around 5.1827 to 5.1625. That is a small loss gain and needs evaluator
metrics, seeds, and hard-subset support before it becomes a paper claim.

Use this wording if it holds at 24k:

> A conservative gated residual fusion over the last ESM layers modestly but
> consistently improves pocket/contact/path evaluator metrics, while preserving
> validity.

REPA should remain an auxiliary branch unless 24k evaluator metrics overturn the
current interpretation. Matched REPA beating shuffled REPA indicates a real
alignment signal, but noREPA currently remains the mainline if it wins on final
loss and evaluator quality.

Preferred REPA framing:

> A representation-alignment auxiliary loss showed non-random matched-vs-shuffled
> signal but did not yet improve final evaluator metrics, so it is excluded from
> the main model and reported as a controlled auxiliary analysis.

## Benchmark Tracks

BINDRAE needs predefined benchmark tracks so the paper does not drift into
cherry-picking. A single combined leaderboard would mix incompatible input
contracts, especially for static endpoint predictors.

### Track A: Main Endpoint-Conditioned Transition Benchmark

This is the current paper's home track. Inputs are apo endpoint, holo endpoint,
and fixed aligned ligand pose. The method is judged on whether it reconstructs a
ligand-aware structural transition path between the known endpoint states.

Primary baselines:

- apo-to-holo linear interpolation;
- `SE(3)` frame interpolation plus chi interpolation;
- cubic or geodesic interpolation on the same state variables;
- NMA / ENM / ANMPathway-style two-endpoint paths;
- targeted or steered MD-lite on selected systems;
- no-ligand BINDRAE;
- shuffled-ligand BINDRAE;
- no-Stage1 or zero-prior BINDRAE.

Primary transition metrics:

- path MAE to the reference structural transition;
- direction accuracy on active residues;
- geodesic or state-space path plausibility;
- path smoothness without over-smoothing contact switches;
- best-of-K path coverage where sampling is used.

### Track B: Contact-Switching And Side-Chain Transition Benchmark

This track should carry the strongest BINDRAE-specific biology. The goal is not
to ask whether the final endpoint is close in a generic RMSD sense. The goal is
to ask whether the path captures pocket response to the ligand.

Primary contact metrics:

- formed-contact recall;
- released-contact success;
- stable-contact preservation;
- ligand-contact trajectory lDDT or contact-LDDT-like score;
- pocket contact timing or ordering, if a reference transition is defined.

Primary side-chain metrics:

- chi-angle trajectory error on pocket and switch residues;
- side-chain heavy-atom RMSD along the path;
- rotamer-flip recovery;
- clash-free side-chain rearrangement;
- side-chain validity in ligand-facing residues.

### Track C: Intermediate Validity Benchmark

This track distinguishes BINDRAE from interpolation, NMA, and simple morphing
baselines, which may satisfy endpoint closure while producing impossible
intermediates.

Primary validity metrics:

- protein-ligand clash along the path;
- protein self-clash along the path;
- peptide geometry validity;
- bond and angle sanity where available;
- side-chain steric validity;
- ligand pocket penetration or impossible intermediate contacts;
- contact monotonicity violations when the path is expected to be binding-like.

Call these structural transition paths, not physical kinetic trajectories, unless
future experiments provide time-scale or free-energy validation.

### Track D: Auxiliary Endpoint Closure And Pipeline Context

Static endpoint prediction is intentionally outside the main scope. Do not put a
main-paper table that implies BINDRAE competes directly with AlphaFold 3, Boltz,
Chai, ESMFold, DynamicBind, FlowDock, or NeuralPLexer on static endpoint
prediction.

Endpoint metrics can be reported as predefined diagnostics:

> Since BINDRAE is endpoint-conditioned in this setting, endpoint metrics are
> reported only as boundary closure and validity checks, not as endpoint
> prediction benchmarks.

Useful endpoint diagnostics:

- final endpoint closure error;
- final-frame consistency;
- final chi consistency;
- endpoint pocket clash;
- peptide validity at the final frame;
- endpoint side-chain validity.

A future pipeline experiment can still involve static endpoint or docking models:

1. Run AF3, Boltz, Chai, DynamicBind, FlowDock, or another model to propose a
   holo-like endpoint or ligand pose.
2. Feed that endpoint or pose into BINDRAE.
3. Evaluate whether BINDRAE adds a plausible transition path and improves
   intermediate validity, contact switching, and side-chain rearrangement.

This positions BINDRAE as a path-completion and protein-response module
downstream of endpoint predictors, not as an endpoint predictor replacement.

## Main Metrics Versus Supplementary Metrics

Main paper tables should use path, contact, side-chain, and validity observables,
not training loss and not static endpoint prediction accuracy.

Main metrics:

- path MAE or direction accuracy, if the reference path is clearly defined;
- formed-contact recall and released-contact success;
- stable-contact preservation;
- ligand-contact trajectory lDDT or contact-LDDT-like score;
- pocket and switch-residue chi trajectory error;
- side-chain heavy-atom RMSD along the path;
- rotamer-flip recovery;
- path-level clash and peptide validity;
- intermediate steric validity;
- runtime;
- top-1 and best-of-K where sampling is used.

Endpoint closure diagnostics may appear in a small auxiliary table or supplement,
but they should be explicitly labeled as closure/validity diagnostics rather than
endpoint prediction benchmarks.

Supplementary metrics:

- `val_total_no_repa` and other training losses;
- FM rigid and chi components;
- frame translation and rotation error;
- final endpoint closure error;
- final-frame and final-chi consistency;
- chi MAE by residue type;
- stable-contact retention;
- path smoothness details;
- per-target and hard-subset breakdowns;
- gate statistics;
- REPA matched/shuffled controls;
- loss-weight and seed variance tables.

## Essential Ablations

The minimum near-term ablation set is:

- no ligand;
- shuffled ligand;
- no Stage-1 / zero-prior fallback;
- Stage-1 soft guidance;
- single ESM versus gated ESM `gate_bias=-2.0`;
- REPA matched versus REPA shuffled;
- path/contact/side-chain/validity evaluator comparison on the same target set;
- endpoint closure reported only as a diagnostic.

The stronger full ablation set adds:

- correct ligand chemistry with randomized pose;
- correct pose with masked ligand atom types;
- wrong ligand from related proteins;
- pose perturbation sensitivity at 0.5, 1.0, and 2.0 A;
- Stage-1 hard-anchor versus soft-posterior guidance;
- contact-only versus latent-only Stage-1 outputs;
- no ESM, last-layer ESM, soft last-K ESM, and gated residual ESM;
- REPA weight and target-layer sweeps;
- endpoint-only, path-only, no-contact, no-validity, no-chi, and no-frame loss
  ablations;
- subset analyses on pocket, switch, formed-contact, released-contact, stable
  contact, large-motion, side-chain-dominant, and backbone-dominant targets.

Leakage controls are mandatory: cluster/family split, ligand scaffold split where
possible, homolog overlap checks, and a clear separation of OracleMotion upper
bound from deployable inference.

## Reviewer Risks And Responses

| Reviewer risk | Response strategy |
| --- | --- |
| "This is just docking." | State known-pose assumption in title/abstract/Figure 1 and benchmark contracts. |
| "DynamicBind / FlowDock / DynamicFlow already do this." | Compare directly where feasible; emphasize fixed-pose transition modeling, path metrics, contact-switch recovery, and Stage-1 local interaction prior. |
| "AF3 / Boltz / Chai already solve the endpoint." | State that static endpoint prediction is outside the main scope; endpoint metrics are predefined closure diagnostics, not a win-only benchmark. |
| "The trajectory is not physical dynamics." | Use "structural transition path," avoid kinetic claims, compare to MD-lite only as a selected plausibility baseline. |
| "OracleMotion leaks the answer." | Label it as upper bound or teacher diagnostic; keep deployable model tables separate. |
| "Gated ESM gain is tiny." | Treat it as engineering unless evaluator metrics and seeds show hard-subset gains. |
| "Why two stages instead of one end-to-end model?" | Show no-Stage1 and Stage1-guided ablations, with gains localized to contacts, side chains, and switch residues. |
| "Known ligand pose is unrealistic." | Position use cases: lead optimization, crystallographic ligand transfer, template-guided modeling, docking refinement, and downstream refinement of external pose generators. |

## Compute Priorities

The immediate bottleneck is claim validity, not training scale.

1. **Freeze the path-first evaluator.** Define path, contact-switching,
   side-chain-transition, intermediate-validity, runtime, top-1, and best-of-K
   metrics before adding more model variants. Endpoint closure should be defined
   as a diagnostic, not a primary benchmark.
2. **Run the evaluator on all 12k and 24k matrix checkpoints.** Training loss is
   a screening proxy only.
3. **Separate endpoint-conditioned path reconstruction from deployable apo-only
   inference.** The current OracleMotion/holo-informed lane is a path paper or
   teacher/upper-bound lane. Deployable BINDRAE should be evaluated separately.
4. **Run the minimal path ablation set.** No ligand, shuffled ligand, no-Stage1,
   Stage1 soft guidance, single ESM, gated ESM, REPA matched, and REPA shuffled
   should be judged under the same path/contact/side-chain/validity evaluator.
5. **Add seeds only for the top candidates.** Prioritize single/noREPA and gated
   `gate_bias=-2.0` noREPA; add the best REPA variant only if evaluator metrics
   justify it.
6. **Implement cheap path baselines before expensive scale-up.** Start with
   apo-to-holo interpolation, `SE(3)` plus chi interpolation, NMA/elastic,
   ANMPathway-style methods where available, and selected MD-lite. Endpoint
   predictors should be treated as optional pipeline sources, not main baselines.
7. **Scale training only after evaluator movement is clear.** Larger training is
   useful only if it improves the scientific observables that will appear in the
   manuscript.

## Working Manuscript Story

One-sentence contribution:

> We introduce BINDRAE, a known-pose ligand-conditioned bridge-flow model that
> reconstructs ligand-aware apo-to-holo structural transition paths between known
> endpoint states by integrating local ligand-pocket interaction guidance with
> residue-frame and side-chain torsion dynamics.

Suggested title direction:

> Ligand-Aware Apo-to-Holo Transition Path Generation Between Known Endpoint
> States

Result structure:

1. Define endpoint-conditioned, ligand-aware induced-fit path reconstruction and
   show that it is distinct from blind docking, static endpoint prediction, and
   generic ensemble generation.
2. Demonstrate path quality against interpolation, `SE(3)` plus chi
   interpolation, NMA/elastic, ANMPathway-style, MD-lite, no-ligand, and
   shuffled-ligand baselines.
3. Demonstrate contact-switching recovery: formed-contact recall,
   released-contact success, stable-contact preservation, and contact timing.
4. Demonstrate side-chain transition quality on pocket and switch residues.
5. Demonstrate intermediate validity: protein-ligand clash, peptide geometry,
   side-chain sterics, and impossible-intermediate rates.
6. Report endpoint closure only as a diagnostic because the current setting is
   endpoint-conditioned.
7. Show that Stage-1 local interaction guidance improves contacts, side chains,
   and validity when deployable ablations are available.
8. Report gated ESM as a representation improvement only if evaluator metrics
   support it.
9. Report REPA as a controlled auxiliary analysis unless it improves final
   evaluator metrics.
10. Document failure modes: incorrect ligand poses, missing cofactors/metals,
    large allosteric rearrangements, multi-domain motion, ambiguous apo/holo
    states, and crystal-packing artifacts.

The bottom-line manuscript claim should be:

> Given a ligand pose from experiment, template transfer, or an external docking
> or cofolding model and endpoint structural information, BINDRAE reconstructs
> the protein's ligand-aware induced-fit transition path more accurately and with
> better contact-switch, side-chain, and intermediate-validity behavior than
> interpolation, morphing, NMA/elastic, MD-lite, no-ligand, and shuffled-ligand
> baselines.

The manuscript should also state the negative scope explicitly:

> BINDRAE does not aim to outperform static complex predictors on endpoint
> structure. It addresses a complementary problem: reconstructing ligand-aware
> apo-to-holo transition paths between known endpoint states, with residue-frame
> and side-chain resolution. Endpoint metrics are reported as closure and
> validity diagnostics rather than endpoint prediction benchmarks.
