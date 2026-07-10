# Stage-2 Comparison Metrics And Baselines

Date: 2026-07-01

> Status update (2026-07-10): the benchmark contract and metric definitions
> remain active. Replace the older generic OracleMotion/free-flow model label
> with the matched synchronous, warp-only, residual-only, and full APNB matrix.

This note consolidates the proposed Stage-2 comparison plan and metric
definitions for the current BINDRAE OracleMotion / holo-conditioned path line.

## Benchmark Contract

The current fair benchmark is:

```text
known apo endpoint + known holo endpoint + known ligand pose aligned in apo frame
-> ligand-aware apo-to-holo structural transition path
```

This is not docking, not blind holo endpoint prediction, and not a physical
kinetics claim. The main question is whether BINDRAE reconstructs a useful
ligand-aware induced-fit structural transition path, especially around pocket
contact switching, side-chain motion, and intermediate validity.

Manuscript shorthand:

```text
BINDRAE is an endpoint-conditioned, ligand-aware structural transition-path
model, not an endpoint predictor.
```

This sentence should be repeated in the method and benchmark sections to keep
the comparison away from docking and static complex prediction.

## Current 2048-Sample Snapshot

Checkpoint:

```text
checkpoints/stage2/stage2_stage2_24k_gm2_gated_m2_norepa_esm7sync_train24000_val2048_e10_bs4x4_20260630_054826/best_model.pt
```

Evaluator output:

```text
logs/stage2/transition_eval/stage2_stage2_24k_gm2_gated_m2_norepa_esm7sync_train24000_val2048_e10_bs4x4_20260630_054826_maxb2048.json
```

Important high-level read:

- Active motion direction is strong: `active/direction_acc = 0.803`.
- New contact formation is very strong: `formed_contact/direction_acc = 0.872`.
- Approach motions are strong: `approach/direction_acc = 0.847`.
- Release/contact-breaking motions are harder:
  `released_contact/path_mae_dist = 1.976 A`,
  `released_contact/endpoint_abs_dist = 3.706 A`.
- Stable noncontacts barely move toward holo:
  `stable_noncontact/improvement_to_holo = 0.071 A`, which is good because
  nonmoving background pocket residues should not be dragged artificially.

Important correction:

```text
improvement_to_holo = |d_apo - d_holo| - |d_final - d_holo|
```

It is a raw distance improvement in Angstroms, not a percentage. Do not report
`0.848` as `84.8%`. If a percentage is needed, add a normalized version:

```text
normalized_improvement = improvement_to_holo / (|d_apo - d_holo| + eps)
```

## Which Experiments To Run

### Track A: Internal Causal Controls

These should be run on the exact same 2048 validation subset first.

| Method | Purpose |
| --- | --- |
| BINDRAE matched OracleMotion gm2 | Current main model. |
| Zero / no-prior BINDRAE | Tests whether the Stage-2 architecture alone can guess the motion. |
| Residue-shuffled OracleMotion | Tests whether residue-aligned motion information is necessary. |
| Sample-shuffled OracleMotion | Tests whether sample identity or leakage-like signals explain the gain. |
| No-ligand BINDRAE | Tests whether ligand conditioning matters. |
| Shuffled-ligand BINDRAE | Tests whether the correct ligand pose/chemistry matters. |

Pass criterion:

```text
matched > zero and shuffled on active direction, contact-switch metrics,
path MAE, chi error, and validity metrics.
```

Ligand ablations should be split carefully:

| Ablation | Meaning |
| --- | --- |
| Stage2-only no-ligand | Keep Stage-1/OracleMotion posterior features, remove Stage-2 ligand tokens. Tests whether Stage-2 explicitly uses ligand tokens. |
| Stage1+Stage2 no-ligand | Remove ligand information from both stages where available. This is the cleanest ligand-aware claim control. |
| Correct endpoints + shuffled ligand pose/chemistry | Keep apo/holo endpoints correct, give a wrong ligand condition. Tests whether the path depends on correct ligand identity/pose rather than apo-holo geometry alone. |

Because OracleMotion is an upper-bound condition, ligand ablations must avoid
claiming deployable ligand causality unless ligand information is removed from
the relevant upstream prior as well as Stage-2.

### Track B: Cheap Endpoint/Path Baselines

These are mandatory reviewer-proof controls.

| Method | Purpose |
| --- | --- |
| Apo / no-motion | Lowest floor. |
| Linear Cartesian interpolation | Naive endpoint-aware baseline. |
| SE(3) frame + chi interpolation | Fair state-space interpolation baseline. |
| Cubic/geodesic SE(3) + chi interpolation | Stronger smooth interpolation baseline. |
| Cubic/geodesic + repack or relax, optional | Tests whether simple post-processing closes the gap. |

Expected story:

```text
Interpolation may look decent on endpoint/path distance, but should be much
worse on peptide geometry, steric clashes, and ligand-pocket side-chain validity.
```

### Track C: Main External Path Baselines

These are the most relevant external competitors for the current paper.

| Method | Priority | Why |
| --- | --- | --- |
| ANMPathway | Must compare | Classic two-endpoint ENM transition pathway baseline. |
| eBDIMS | Must compare | Established elastic-network/Brownian endpoint-conditioned path method. |
| eBDIMS2 | Strongly recommended | Modern optimized eBDIMS successor with code, Nat Commun 2026. |
| DeepPath | Must attempt | Closest deep-learning atomistic transition-path competitor, if reproducible. |

These are the real main battle opponents because their input contract is close:
known endpoint structures to transition intermediates/path.

### Track D: Small-Scale Physics References

Use these on a curated 20-50 case panel, not full scale.

| Method | Purpose |
| --- | --- |
| Targeted MD / Steered MD-lite | Endpoint-driven physical plausibility reference. |
| COMBAS | Physics-flavored two-state pathway construction baseline. |

### Track E: Optional / Supplementary Path Methods

Use if automation is practical.

| Method | Role |
| --- | --- |
| MinActionPath2 | Endpoint path server; good supplementary or subset comparison. |
| ICONGENI | Internal-coordinate NMA-guided morphing baseline. |
| SIDE / Langevin bridge | Recent endpoint-conditioned stochastic bridge/path method. |

### Methods Not For The Main Path Table

These are important context, but they should not be the primary path comparison.

| Method family | Examples | Why not main path baseline |
| --- | --- | --- |
| Flexible endpoint / docking models | DynamicBind, FlowDock, NeuralPLexer | Mostly output static/final complexes, not ordered apo-to-holo paths. |
| Static cofolding / complex predictors | AlphaFold 3, Boltz, Chai-1 | Strong endpoint predictors, but not transition-path methods. |
| Ligand-agnostic ensemble models | AlphaFlow, P2DFlow, EigenFold / Str2Str-like | Useful for best-of-K coverage, but not ligand-aware known-pose induced-fit paths. |

They can appear in related work, endpoint diagnostics, or future pipeline
experiments, but not as the core path-method battle.

## Proposed Paper Tables

Use two paper-facing tables rather than one overloaded leaderboard.

### Table 1: External Path-Method Comparison

This table answers:

```text
Does BINDRAE beat endpoint-conditioned transition-path baselines?
```

| Method | Active Dir ↑ | Active Path MAE ↓ | Formed Recall ↑ | Released Success ↑ | Stable Retention ↑ | Pocket Chi Err ↓ | Clash ↓ | Peptide ↓ | Runtime ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apo / no-motion | | | | | | | | | |
| Linear interpolation | | | | | | | | | |
| SE(3)+chi interpolation | | | | | | | | | |
| Cubic SE(3)+chi interpolation | | | | | | | | | |
| ANMPathway | | | | | | | | | |
| eBDIMS | | | | | | | | | |
| eBDIMS2 | | | | | | | | | |
| DeepPath, if reproducible | | | | | | | | | |
| BINDRAE matched | | | | | | | | | |

### Table 2: BINDRAE Causal Ablation

This table answers:

```text
Why does BINDRAE work, and is the signal residue-aligned, ligand-aware, and
not a leakage/shortcut artifact?
```

| Method | Active Dir ↑ | Active Path MAE ↓ | Formed Recall ↑ | Released Success ↑ | Stable Retention ↑ | Pocket Chi Err ↓ | Clash ↓ | Peptide ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-prior BINDRAE | | | | | | | | | |
| Residue-shuffled BINDRAE | | | | | | | | | |
| Sample-shuffled BINDRAE | | | | | | | | | |
| Stage2-only no-ligand BINDRAE | | | | | | | | |
| Stage1+Stage2 no-ligand BINDRAE | | | | | | | | |
| Shuffled-ligand BINDRAE | | | | | | | | |
| Single-ESM BINDRAE, optional | | | | | | | | |
| Gated-ESM BINDRAE | | | | | | | | |
| BINDRAE matched | | | | | | | | | |

Supplementary tables can include all subset-specific metrics, endpoint closure
details, smoothness, atom14 errors, seed variance, and hard-subset breakdowns.

## Residue Subsets

These subsets define where metrics are computed.

| Subset | Meaning | Why it matters |
| --- | --- | --- |
| `all_pocket` | All finite ligand-proximal pocket residues. | Overall pocket behavior. |
| `active` | Pocket residues with apo/holo ligand-distance change >= 0.75 A. | True induced-fit moving residues. |
| `approach` | Active residues whose holo state is closer to ligand. | Pocket closing / contact formation behavior. |
| `release` | Active residues whose holo state is farther from ligand. | Pocket opening / contact release behavior. |
| `formed_contact` | Apo noncontact, holo contact. | New ligand-contact formation. |
| `released_contact` | Apo contact, holo noncontact. | Ligand-contact release, usually harder. |
| `stable_contact` | Contact in both apo and holo. | Contacts that should be preserved. |
| `stable_noncontact` | Noncontact in both apo and holo. | Background residues that should not be over-moved. |

## Transition Path Evaluator Metrics

These are the metrics currently available from
`scripts/evaluate_stage2_transition_paths.py`.

| Metric | Direction | Meaning |
| --- | --- | --- |
| `direction_acc` | Higher is better | Whether the final predicted ligand-distance change has the same sign as the apo-to-holo target change. |
| `path_mae_dist` | Lower is better | Mean absolute deviation, along the path, from a linear apo-to-holo ligand-distance schedule. Unit: Angstrom. |
| `endpoint_abs_dist` | Lower is better | Final predicted side-chain-to-ligand distance error relative to holo. Unit: Angstrom. |
| `improvement_to_holo` | Higher is better | Raw reduction in holo-distance error relative to apo. Unit: Angstrom, not percent. |

Interpretation:

- `direction_acc` answers: did the residue move in the correct direction?
- `path_mae_dist` answers: did the path follow a reasonable apo-to-holo progress
  schedule?
- `endpoint_abs_dist` answers: how close is the final ligand-distance state to
  holo?
- `improvement_to_holo` answers: how many Angstroms of apo-to-holo distance
  error were removed?

Naming caution:

`path_mae_dist` should not be described as true "path MAE to reference
transition" unless a real MD/TMD/experimental intermediate reference path is
available. In paper text, prefer:

```text
linear_progress_mae_dist
```

or:

```text
distance_schedule_deviation
```

This metric measures monotonic endpoint progress in ligand-distance space, not
physical trajectory accuracy. It should be reported after contact-switching,
side-chain, and validity metrics, because interpolation baselines can be favored
by a linear schedule definition.

## Contact-Switching Metrics

These should be emphasized because they are the most BINDRAE-specific biology.

| Metric | Direction | Meaning |
| --- | --- | --- |
| `formed_contact/recall` | Higher is better | Fraction of holo-only ligand contacts recovered by the model. |
| `released_contact/release_success` | Higher is better | Fraction of apo-only contacts successfully released. |
| `stable_contact/retention` | Higher is better | Fraction of contacts present in both apo and holo that remain present. |
| Contact-LDDT / contact profile, optional | Higher is better | Whether the full ligand-contact pattern along the path resembles the reference schedule/profile. |

These metrics are crucial against interpolation, ANMPathway, eBDIMS, and
eBDIMS2 because they focus on ligand-pocket remodeling rather than only global
backbone movement.

## Side-Chain And Pocket Resolution Metrics

| Metric | Direction | Meaning |
| --- | --- | --- |
| `chi_err_rad` | Lower is better | Wrap-aware side-chain chi-angle error relative to holo. |
| Pocket/switch chi error | Lower is better | Same as chi error, restricted to ligand-facing or switch residues. |
| Rotamer-flip recovery | Higher is better | Whether large side-chain rotamer flips are recovered. |
| `atom14_endpoint_err_A` | Lower is better | Atom14 coordinate error relative to holo, especially useful for side-chain heavy atoms. |

This group is especially important against coarse-grained or backbone-centric
methods such as ANMPathway and eBDIMS-family baselines.

## Intermediate Validity Metrics

These are needed to show that the path is not just endpoint interpolation.

| Metric | Direction | Meaning |
| --- | --- | --- |
| `path/clash_penalty` | Lower is better | Steric clash penalty along generated intermediates. |
| `path/peptide_loss` | Lower is better | Peptide geometry violation along intermediates. |
| `smooth_frame_step_norm` | Lower/sane is better | Per-step backbone-frame change magnitude; catches jumps/explosions. |
| `smooth_chi_step_abs` | Lower/sane is better | Per-step chi-angle change magnitude; catches side-chain jumps. |

Expected important story:

```text
Cubic or geodesic interpolation can be endpoint-close by construction, but it
should be much worse on peptide geometry and steric validity. BINDRAE should win
by producing more valid ligand-pocket intermediates.
```

## Protocol Guardrails

### Path Resampling

Different methods may output different numbers of intermediate frames. Before
computing path-level metrics, resample every method to a common number of frames
or to a shared progress/arc-length convention. Otherwise smoothness, contact
timing, and schedule-deviation metrics are not comparable.

### Coarse-Grained Baseline Reconstruction

ANMPathway, eBDIMS, and eBDIMS2 can be backbone-centric or coarse-grained. For
fair all-atom pocket metrics:

```text
All non-side-chain baselines should be postprocessed with the same side-chain
reconstruction/repacking protocol before computing pocket chi, atom14, contact,
clash, and ligand-pocket validity metrics.
```

If reconstruction quality is uncertain, report two tables:

- backbone/path metrics on raw baseline output;
- all-atom pocket metrics after the unified rebuild/repack protocol.

### Contact Cutoff Sensitivity

Ligand-contact metrics depend on the distance cutoff. Main results can use
`4.5 A`, but supplement should include a cutoff sweep:

```text
4.0 A, 4.5 A, 5.0 A
```

### Counts And Statistical Tests

For every subset and method, report:

- number of targets;
- number of residues or contacts used by the metric;
- paired bootstrap confidence intervals;
- per-target paired differences;
- Wilcoxon or sign-test results where appropriate.

This is important because target-to-target variance is likely large and
`released_contact` counts may be sparse.

### Hard-Subset Breakdowns

Do not rely only on aggregate `active/direction_acc`. Report hard subsets such
as:

- active motion bins: `0.75-1.5 A`, `1.5-3.0 A`, `>3.0 A`;
- formed-contact residues;
- released-contact residues;
- switch residues with large chi flip;
- large pocket-motion cases;
- high-clash or difficult ligand environments.

Release/contact-breaking should be a visible hard subset rather than hidden in
the average.

### Failure Cases

Include representative failures:

- release failure;
- ligand clash failure;
- wrong side-chain rotamer;
- large domain motion failure;
- ambiguous apo/holo endpoint or ligand-pose mismatch.

This makes the scope credible and prevents overclaiming.

## Runtime Metric

Report wall-clock runtime per sample or per path, including number of path
frames/steps.

Runtime is important because BINDRAE should be much cheaper than TMD/SMD-like
physical simulations and potentially more scalable than expensive path-sampling
methods.

## Recommended Immediate Next Runs

1. Run trajectory reliability evaluator on the 2048 gm2 matched checkpoint.
2. Run the same evaluator on zero-prior and residue-shuffled checkpoints.
3. Ensure cubic `SE(3)+chi` interpolation metrics are reported from the same
   samples.
4. Add sample-shuffled OracleMotion if not already evaluated.
5. Add Stage2-only no-ligand, Stage1+Stage2 no-ligand, and shuffled-ligand
   ablations if code paths are ready.
6. Define a unified side-chain rebuild/repack protocol for ANMPathway/eBDIMS
   outputs before all-atom pocket comparisons.
7. Start adapter work for ANMPathway and eBDIMS outputs.
8. Audit eBDIMS2 and DeepPath reproducibility before promising full-scale
   comparison.

## Current Bottom Line

The current 2048 result is a strong engineering and upper-bound signal:

```text
BINDRAE-OracleMotion reliably recovers the direction of ligand-pocket motion,
especially for active and formed-contact residues, while avoiding large changes
in stable noncontact residues.
```

The result is not yet a complete paper claim by itself. It needs matched-vs-zero,
matched-vs-shuffled, interpolation, validity, and external path-method baselines
under the same metrics.
