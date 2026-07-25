# Path-4 Optimizer Gate-0 Contract

Date: 2026-07-20

## Decision

Continue Path-4 as a physical path-optimization research program, but stop
treating it as direct regression from endpoint-derived features to one MD
normal residual.

The next estimand is a reproducible canonical low-cost path under a declared
physical objective:

```text
frozen Path-3 guide
  -> paired global route seeds
  -> endpoint-exact projected-normal path optimization
  -> physical descent/validity acceptance
  -> best valid canonical candidate
```

This is a long-term method branch. The frozen protein Path-3 remains the current
validated model; implementing the optimizer does not retroactively validate a
protein normal-residual contribution.

## Why The Direct Force Residual Route Is Closed

`scripts/analyze_md_force_residual_alignment.py` tested a deliberately
optimistic upper bound: unbiased OpenMM forces were evaluated at the actual MD
replica states, aggregated into residue-local generalized forces, projected
normal to the endpoint bridge, and fitted with one train-scope amplitude.

On 30 disjoint validation systems with a 21-frame force window:

- rigid relative residual-MSE reduction: `-0.000731`, 95% CI
  `[-0.001715, +0.000173]`;
- rigid force/residual cosine: `-0.01910`, 95% CI
  `[-0.03972, +0.00018]`;
- rigid positive-alignment fraction: `0.4765`, 95% CI
  `[0.4481, 0.5039]`;
- translation relative residual-MSE reduction: `-0.002800`;
- decision: `force_signal_supported=false`.

Instantaneous force is therefore not a valid direct target for the silver MD
normal residual. Physical forces can still define and optimize a path-level
objective; that is a different use of force information.

## Existing Surrogate Optimizer Evidence

`src/stage2/modules/physical_path.py` already provided a single-start,
translation-only, differentiable normal refinement using atom14 peptide,
protein-clash, ligand-clash, contact-anchor, distance-anchor, residual, and
temporal-smoothness terms.

On the frozen validation lane, its conservative high-residual-weight setting:

- improved ligand-clash severity by `4.15%`, with 95% CI above zero;
- worsened path length by `2.29%`;
- changed MD Product RMSE from `1.305897` to `1.306687`, a `0.060%` worsening
  that passed the 1% non-inferiority margin;
- changed translation MAE from `0.636869 A` to `0.638286 A`, a `0.223%`
  worsening that also passed non-inferiority;
- improved clash severity on only `46.7%` of systems.

This is useful feasibility evidence, but it does not pass the final Gate 0. It
uses a local surrogate, one zero initialization, mean-dominated objectives, and
no independent OpenMM path optimization.

## Implemented Optimizer Scaffold V0

The active scaffold retains the old single-start Adam behavior by default and
adds opt-in capabilities for the Path-4 optimization program:

- deterministic paired route seeds `(z, -z)`;
- low-rank temporal sine basis shared across the whole path;
- chain-smoothed spatial route fields;
- route fields generated in global Cartesian coordinates and mapped into each
  bridge frame's body coordinates;
- product/block tangent-normal projection at every candidate evaluation;
- endpoint-zero path envelope and exact endpoint insertion;
- `mean`, `max`, or normalized softmax frame aggregation for physical tails;
- backtracking line search that accepts only declared-objective descent;
- strict fallback to the unmodified Path-3 guide when no route improves;
- objective-call accounting and per-start selection diagnostics;
- one reproducible Slurm screen entrypoint:
  `scripts/slurm/evaluate_stage2_path4_multistart_screen_1gpu.sh`.

The scaffold is not yet learned and does not yet optimize OpenMM energy. Its
current purpose is to make multi-route search, safety, and compute accounting
testable before adding a learned preconditioner.

## Validation Status

Local static checks pass. In the canonical remote BINDRAE environment:

- five focused physical-path tests pass;
- paired seeds are exact negatives;
- global route directions are invariant to a common rotation of bridge frames;
- a symmetric zero-gradient collision is escaped by multi-start search;
- endpoints remain exact;
- projected tangent leakage remains below the test tolerance;
- the selected objective never exceeds the Path-3 guide objective;
- 22 related evaluator/export tests pass.

The surrogate CUDA checkpoint smoke job `147108` was cancelled before
allocation (`00:00:00`, no node, and no Slurm log), so it provides no pipeline
or scientific evidence.

CPU engineering smoke job `147139` completed on `gpu38` in `00:02:36` with
exit code zero. On the single old training system `3zw1-E-FUC-3`, it exported
the frozen Path-3 candidate, obtained full residue/atom mapping, passed the
frozen reference-topology preflight (`5143.90 < 1e6 kJ/mol/nm` maximum protein
atomic force), and wrote all 21 OpenMM score frames to
`logs/stage2/path4_gate0/path4_gate0_dev_3zw1-E-FUC-3_147139/openmm/path3.json`.
This validates only the CPU Path-3 export-to-scorer plumbing. It does not test
the OpenMM optimizer, CUDA execution, a fresh cohort, or any Gate-0 scientific
criterion.

### Common all-atom frame-reference contract (2026-07-21)

The first one-system OpenMM optimizer smoke (`147166`) completed, but it exposed
that the scorer's hidden all-atom initialization had not been controlled. Two
candidate-specific scoring modes are rejected for method comparison:

- `previous_relaxed` keeps hydrogens adapted but makes later frames, including
  the apo endpoint, depend on the candidate's earlier frames;
- resetting every frame to the original prepared topology gives equal raw
  endpoints but creates hidden-atom nonbonded states with forces up to
  `3.72e9 kJ/mol/nm`, making minimization numerically unstable.

The active scorer contract is now `reference_cache`. A frozen Path-3 guide is
used once to precondition all atoms along the 21-frame path. The resulting
cache is bound to the sample, residue hash, time grid, topology atom count,
implicit-system contract, and cache SHA-256. Every cached frame must pass an
all-atom force preflight before either arm is scored, and the paired summary
hard-fails on scorer-contract or endpoint-energy disagreement.

The direct CUDA engineering run
`path4_gate0_reference_cache_3zw1-E-FUC-3_direct_gpu33_20260721_032435_v1`
completed this full cache -> Path-3 score -> optimized score -> paired summary
chain. The 21 cached frames had a maximum atomic force of
`7565.20 kJ/mol/nm`, below the frozen `1e6` threshold, and all four paired
raw/relaxed endpoint-energy differences were exactly zero. On this one old
training system, the optimized candidate improved raw excess p95 by `26.43%`,
raw maximum by `37.03%`, and every interior frame moved toward lower raw
energy. The initially reported relaxed p95/max improvements of
`81.37%`/`96.92%` are not valid headline results: Path-3 frame `t=0.25` retained
a `18588.31 kJ/mol/nm` maximum protein residue-net force under the configured
250-iteration cap and alone contributed `7993.51 kJ/mol` relaxed excess.

Before any fresh cohort, scorer schema v5 therefore freezes a separate relaxed
frame-validity criterion at maximum protein residue-net force
`<=500 kJ/mol/nm`, matching the existing MD-preparation readiness convention.
Paired-summary schema v2 recomputes relaxed energy metrics only on the
intersection of frames valid in both arms and reports invalid-frame incidence
separately. The direct CUDA v5/v2 rescore
`path4_gate0_relaxed_validity_3zw1-E-FUC-3_direct_gpu33_20260721_183700_v1`
completed with the same cache SHA and zero endpoint-energy differences.
`18/19` interior frames are jointly valid: relaxed p95 changes from `187.18`
to `195.16 kJ/mol` (`4.26%` worse), relaxed maximum changes from `257.91` to
`246.43 kJ/mol` (`4.45%` better), and invalid-frame incidence changes from
`1/19` to `0/19`. The broad raw p95/max improvements remain
`26.43%`/`37.03%`; on valid interior frames, residue-force p95 worsens by
`14.20%`, and both arms have zero severe-clash frames. This validates the new
engineering contract on one old system only, not Gate-0 efficacy.

### Training-scope development panel (2026-07-21)

A bounded gpu33 inventory found 150 systems in the exact intersection of the
241-system checkpoint training subset, the frozen phase cache, and completed
prepared-system reports. `scripts/run_path4_gate0_dev_panel.py` now selects a
12-system engineering panel from that intersection: one non-headline `3zw1`
sentinel plus 11 primary systems spread deterministically over residue count.
The primary selection hard-excludes the checkpoint validation list, the frozen
32/strict30 test lists, and the 12-system physical-calibration/post-selection
list. The sentinel is always excluded from the primary paired summary.

The runner freezes the optimizer settings from completed run `147166`, builds
and preflights the common all-atom frame-reference cache before optimization,
uses scorer schema v5 and paired-summary schema v2, and records rejection or
failure at each stage without terminating the remaining panel. This remains a
training-scope development check; neither its size nor its selection permits a
formal Gate-0 claim.

The first direct gpu33 run, tagged
`path4_gate0_dev_panel_train11_sentinel1_gpu33_20260721_v1`, completed. Four of
the 11 primary systems produced valid paired comparisons. Three were rejected
for residue-identity mismatch (`3km6`, `1l2s`, and `5amv`), and four were
rejected by the frozen `1e6 kJ/mol/nm` prepared-reference force threshold
(`3sbk`, `4zxa`, `5gm0`, and `5rgg`). These are preparation/identity exclusions,
not Path-4 losses. The non-headline `3zw1` sentinel completed separately.

Across the four valid primary systems, raw excess-energy p95 and maximum both
improved on all four. Paired-valid relaxed p95 improved on three of four, with a
mean relative improvement of `9.2%`, but its system-level interval crossed zero.
Relaxed maximum improved on two of four and had an approximately `-1.0%` mean
relative improvement. Invalid-frame incidence was unchanged, and paired-valid
residue-force p95 improved on only two of four. The optimizer therefore descends
its raw objective, but this small panel does not show stable transfer to the
independent relaxed/force outcomes. Gate 0 remains unpassed.

### Reserve preflight and Product-state contract (2026-07-22)

The development runner now has a `preflight` execution mode that exports the
frozen Path-3 candidate, builds the implicit system, verifies candidate/topology
identity and mapping, and evaluates the unmodified prepared topology once. It
does not build or minimize the 21-frame all-atom reference cache. Every accepted,
scientifically rejected, or engineering-failed system remains in structured
panel state, while accepted primary IDs are written to
`valid_primary_samples.txt`. A 24-system training-scope reserve launcher excludes
the entire first panel plus the frozen validation, test32, strict30, and
physical-calibration lists. This is candidate discovery only, not a Gate-0
cohort or efficacy result.

Newly exported path candidates use schema
`bindrae_stage2_path_candidate_v3`, which adds the exact per-frame rotation
matrices, frame translations in angstrom, and chi angles in radians. The OpenMM
optimizer applies its residue-global correction to both atom14 coordinates and
frame translations while leaving rotations and chi unchanged. The independent
candidate MD evaluator requires this v3 state, reconstructs targets only from
the audited inferred-phase `md_phase_normal_v1` replica cache, and reports the
predeclared 1% paired Product-RMSE non-inferiority decision. Legacy v2 candidates
remain scorer-readable but cannot be used to manufacture Product-RMSE evidence.

The reserve preflight run
`path4_gate0_preflight_reserve_train24_gpu33_20260722_011109_v1` finished with
seven accepted systems and 17 frozen-contract rejections. The accepted IDs are
`2e2o-A-BGC-400`, `2qje-D-Z8T-2`, `4wq2-B-3SU-301`, `6hfx-A-DMU-201`,
`6lr4-C-CLR-301`, `7mql-A-RIO-302`, and `8iy2-E-3AM-204`. There were no
unfinished systems. This preflight produced no optimizer, scorer, or Product
RMSE result. `scripts/run_path4_gate0_full_reserve7_gpu33.sh` consumes that
run's exact `valid_primary_samples.txt` artifact, includes no sentinel, and
runs the seven accepted systems through the frozen full comparison chain. The
result remains training-scope development evidence and cannot be counted as the
formal fresh 40--60-system Gate-0 cohort.

### Full reserve frame-reference audit (2026-07-23)

The full reserve run
`path4_gate0_full_reserve7_gpu33_20260723_011300_v3` finished with one completed
system (`8iy2-E-3AM-204`) and six `rejected_reference_preflight` records. The
six rejected systems do not enter the Path-3 versus Path-4 comparison. Gate 0
therefore remains unpassed.

The reserve/full artifact audit rules out a stale prepared-system cache as the
shared explanation. For every reserve/full pair, `implicit_system.xml`,
`small_molecule_system_cache.json`, and `contract.json` are byte-identical.
Although the topology PDB file hashes differ, OpenMM loads exactly identical
coordinates (maximum and RMS coordinate differences are both `0.0 A`). Direct
CPU and CUDA checks reproduce the reserve prepared-state force values, and the
full path's zero-strength restraint changes neither energy nor force. For
example, `6hfx-A-DMU-201` has a CUDA prepared-state maximum atomic force of
`33008.18335 kJ/mol/nm`, far below the frozen `1e6` threshold.

The earlier frame-cache report used the same generic prepared-topology error
text for both the initial state and the later 21-frame loop, and discarded the
failing frame context. Therefore the seven-system reserve preflight proves only
prepared-topology eligibility; it does not prove that the preconditioned
all-atom frame cache will pass. The frame-reference report is now schema v2 and
separates `prepared_topology_preflight`, `reference_preconditioning`, and
`frame_reference_preflight`. A rejected frame retains its index, time, energy,
maximum atomic force, mapped-heavy RMS, all previously accepted frame records,
and diagnostics for mapped path steps and carried hidden protein atoms. Panel
state propagates this exact failure stage instead of collapsing it to the
runner's generic `reference_preflight` stage.

The diagnostic-only rebuild
`path4_gate0_frame_reference_audit_reserve6_gpu33_20260723_110620_v2` completed
all six systems with no engineering failures. It kept the prepared-state
threshold frozen at `1e6 kJ/mol/nm`, used the original 25-step preconditioning,
and raised only the diagnostic frame threshold to `1e12 kJ/mol/nm` so all 126
frame records could be collected. Every system had at least one frame above the
frozen threshold: 21/126 frames exceeded `1e6`, 16 exceeded `1e8`, and 13
exceeded `1e9 kJ/mol/nm`. Four systems repeatedly reached approximately
`3.72e9 kJ/mol/nm`. `4wq2-B-3SU-301` already failed at holo endpoint frame 20,
the first holo-to-apo traversal frame with zero mapped path step, so traversal
accumulation is not a universal explanation. Correlations with hidden-protein
injection displacement varied in sign across systems and do not support a
single displacement-magnitude explanation.

Atom-level diagnostics now record the five highest-force atoms, their
mapped-protein/hidden-protein/environment scope, nearest atom not directly
bonded in the topology, and per-OpenMM-force-group contribution whenever a
frame exceeds `1e6`. Minimizer reporter callbacks are recorded without claiming
a convergence reason. The representative diagnostic run
`path4_gate0_atom25v250_rep3_gpu33_20260724_153351_v1` completed all six
system-iteration reports for `2qje`, `4wq2`, and `6lr4`. At the 25-iteration
cap, 9/63 frames exceeded the frozen `1e6` threshold; at the 250-iteration cap,
0/63 did. The respective per-system maximum forces changed from
`3.719e9`, `3.720e9`, and `1.617e7` to `1.495e4`, `1.355e4`, and
`1.245e4 kJ/mol/nm`, with all nine problem frames resolved and no new problem
frames.

Every 25-step problem frame was dominated by `NonbondedForce`. Representative
nearest non-directly-bonded contacts were a hidden `THR343:HA` against mapped
`TYR342:O` at `0.12--0.53 A` in `2qje`, mapped `GLN100:OE1` against
`GLU97:OE1` at `0.46 A` in `4wq2`, and mapped `GLU106:OE1` against the same
residue's backbone oxygen at `1.21 A` in `6lr4`. The two approximately
`3.72e9` maxima had Cartesian force components close to `+/-2^31`, consistent
with CUDA force-buffer saturation under catastrophic overlap rather than a
shared physical force scale. At 250 steps, the largest mapped-heavy relaxation
RMS across these systems was `0.083 A`.

This is strong engineering evidence that the original 25-step hidden-atom
preconditioning budget was insufficient on these three representatives. It is
not a Path-4 efficacy result or a formal convergence proof: 250 is an OpenMM
iteration cap, the reporter exposes no termination reason, and the run used a
diagnostic frame threshold of `1e12`. Its caches remain ineligible for scoring
or method comparison. The next check is a reference-only rebuild of all six
rejected reserve systems with the 250-step cap and the formal frame threshold
kept at `1e6`; only systems passing that contract may proceed to a rebuilt
reserve comparison. An earlier launcher attempt ending in `_110428_v1` failed
before OpenMM evaluation because a relative preparation-report path did not
match the existing cache contract's absolute path; that engineering failure has
no scientific interpretation and is not reused.

### Formal 250-step reserve reference eligibility (2026-07-24)

The reference-only run
`path4_gate0_frame_reference_formal250_reserve6_gpu33_20260724_212323_v1`
completed all six previously rejected reserve systems under the proposed
250-step contract with both the prepared and frame thresholds fixed at
`1e6 kJ/mol/nm`. All 126 frames passed, there were no engineering failures or
scientific rejections, and the largest frame force was
`17327.08 kJ/mol/nm`, about 58-fold below the frozen threshold. The largest
mapped-heavy relaxation RMS was `0.1088 A`; hidden protein atoms relaxed by up
to `0.5070 A`.

This clears the common-reference engineering blocker for the reserve panel. It
does not retroactively change the six rejections in the earlier 25-step run and
does not constitute a Path-4 method result. The active development-panel
reference contract is now re-frozen at 250 steps. A new full reserve7 run must
rebuild the common reference for all seven systems, including `8iy2`, and bind
both scoring arms to each resulting cache SHA-256. Old 25-step and new 250-step
caches must not be mixed, including during `--resume`.

### Rebuilt full reserve7 development result (2026-07-24)

The direct gpu33 run
`path4_gate0_full_reserve7_ref250_gpu33_20260724_222939_v2` completed all nine
stages for all seven primary systems, with no rejection or engineering failure.
All 147 common-reference frames passed the frozen `1e6 kJ/mol/nm` threshold;
the cohort maximum was `19979.24 kJ/mol/nm`. The maximum mapped-heavy reference
relaxation RMS was `0.1154 A`, all 28 paired endpoint-energy differences were
exactly zero, and each scoring pair used the same per-system cache SHA-256.

The optimizer selected an internally improved route for `7/7` systems, used
`2982` energy/force calls, and reported `63.48 s` of optimizer wall time. This
overstates the effective optimization depth: for six systems, every start
accepted only its initial zero or paired route seed and rejected the first
force-directed line-search update. Only `8iy2-E-3AM-204` completed the four
configured iterations. The current implementation therefore behaved mostly as
a three-route seed screen, not yet as a strong iterative physical optimizer.

On the frozen independent scorer, raw excess-energy p95 and maximum improved on
`6/7` systems, but the mean relative improvements were only `4.17%` and `4.27%`;
their medians were `3.56%` and `5.40%`, and only `1/7` systems reached a `10%`
improvement. Both paired mean-improvement intervals crossed zero. The absolute
cohort means worsened because `6hfx-A-DMU-201` reversed from an internal raw-p95
improvement (`374.16M -> 368.01M kJ/mol`) to an independent-score worsening
(`9.70B -> 10.84B kJ/mol`). This exposes sensitivity to the separately
preconditioned optimization and scorer reference realizations; it is not a
paired-arm cache or endpoint mismatch.

Only five systems had a valid paired relaxed-energy intersection. Relaxed p95
and maximum improved on `1/5`, with mean relative changes of `-6.53%` and
`-3.53%` (worse). Protein residue-force p95 improved on `2/7` and changed by
`-0.95%` on average. Invalid-or-severe-clash frames changed from `37/133` to
`30/133`, a pooled `18.9%` reduction; the mean of per-system relative changes
was `43.0%`, but its paired interval crossed zero. The severe-clash count alone
changed from one frame to zero, which is too sparse to establish a stable
effect.

The Product-state preservation result is positive but not an efficacy result:
system-macro Product RMSE changed from `1.237175` to `1.236770` (`0.0327%`
better), passed the frozen `1%` non-inferiority margin, and improved on `4/7`
systems. Rotation and chi metrics were exactly unchanged, as expected for this
translation-only optimizer.

This run passes the engineering chain but is a development promotion **no-go**
for the current optimizer/scorer configuration. It is not a formal Gate-0
failure because the seven systems are training-scope development data, not the
fresh 40--60-system cohort. Do not launch that cohort, a 1,000-system campaign,
or a learned/unrolled optimizer from this result. The next bounded development
step must first demonstrate real post-seed descent and remove or explicitly
stress-test optimization-to-scorer reference sensitivity on this same reserve
panel while keeping the frozen physical and Product-state thresholds unchanged.

### Exact-JVP direction audit status (2026-07-25)

The first bounded launch,
`path4_gate0_direction_exactjvp_4sys_gpu33_20260725_004018_v2`, rejected all four
systems before OpenMM evaluation because its relative preparation-report path
did not match the canonical absolute path stored in the existing
implicit-system cache contract. The cache was not rebuilt. The launcher now
canonicalizes the prepared-system directory, with a regression test, and the
replacement tag is
`path4_gate0_direction_exactjvp_4sys_gpu33_20260725_004428_v3`.

The audit covers `6hfx`, `2qje`, `6lr4`, and `8iy2`; it is an engineering
root-cause audit, not a new Gate-0 result. It reuses each frozen Path-3 cache,
canonicalizes the rebuilt carbonyl-O base before every variant, and compares
the current residue-force pullback with an exact all-atom coordinate JVP over a
ten-scale central finite-difference grid. A local direction conclusion is
eligible only when the direction is nonzero, atomic force stays below
`1e9 kJ/mol/nm`, three adjacent finite-difference scales form a stable plateau,
and both the total and softmax-tail derivatives agree with the all-atom
force-dot JVP. Cache-preflight coordinates and exact mapped-atom reinjection are
recorded as deliberately different states. Until the reports complete, no
optimizer fix or scientific conclusion is implied. The active log is
`logs/slurm/path4_gate0_direction_exactjvp_4sys_gpu33_20260725_004428_v3.out`.

## Gate 0: Necessity Before Learning

Before implementing an unrolled neural optimizer or generating 1,000 training
systems, run a strong non-learned multi-start physical optimizer on 40-60 fresh
systems that were not used for current validation, strict30 testing, or
post-processing selection.

Freeze before execution:

- endpoint preparation, protonation, ligand parameterization, and environment;
- Path-3 checkpoint and inference projection;
- route seed set and objective-call budget;
- primary physical objective and independent secondary scorer;
- raw-versus-relaxed reporting;
- failure handling and system-level paired bootstrap.
- one frozen all-atom Path-3 frame-reference cache shared by both arms.
- one frozen relaxed-frame validity threshold, with paired-valid relaxed energy
  summaries and invalid-frame incidence reported as separate outcomes.

Prepared systems must also pass a frozen reference-topology preflight before
either arm is evaluated. The current deliberately permissive engineering
threshold rejects a system when the unmodified ff14SB/OpenFF/GBn2 topology has
an all-atom maximum force above `1e6 kJ/mol/nm`. This separates corrupt or
severely clashing preparations from candidate-path failures; all rejected
systems and reasons remain in the cohort accounting.

Minimum positive evidence:

- maximum or p95 excess energy improves by at least `10%`;
- severe clash/invalid-frame incidence improves by at least `30%`;
- MD Product RMSE remains within the predeclared `1%` non-inferiority margin;
- at least `60%` of systems improve the primary physical objective;
- the paired system-level interval excludes zero;
- all failures and force/energy-call counts are reported.

These thresholds are engineering decision gates, not physical constants.

If Gate 0 fails, stop the learned Path-4 optimizer branch. If it passes, compare
at matched force-call budgets:

1. frozen Path-3;
2. single-step force-conditioned residual;
3. non-learned multi-start optimizer;
4. unrolled learned optimizer.

The learned optimizer must reduce force calls by at least `3x`, wall time by at
least `30%`, or produce a predeclared physical-tail improvement at equal budget.

## Claim Boundary

The optimizer may claim amortized proposal/refinement of a canonical low-cost
path under a declared physical contract. It must not call the optimized
potential-energy path a free-energy path, unique transition mechanism, or
kinetic trajectory.

The strongest eventual practical endpoint is lower string/NEB/restrained-MD
cost or higher convergence success on unseen systems. Internal surrogate loss
alone cannot validate Path-4.
