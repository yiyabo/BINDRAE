# Path-3 Final Experiment Contract

Date: 2026-07-25

## Decision

Path-3 is the deterministic conference candidate. The promoted model is the
endpoint-exact, chain-coupled, endpoint-fixed non-monotone residue phase field
over the Cartesian-backbone endpoint bridge. The protein normal-residual branch
remains a matched negative ablation and is not part of the promoted model.

A higher-budget anchor is warranted because the current canonical run stopped at
ten epochs and selected epoch nine, but the 241-system corpus is too small to
declare that anchor the final model. The three-seed final pass is deferred until
the data-scale gate below is resolved. More GPUs are an operational choice, not
a scientific contribution. The current launcher preserves the earlier global
batch size of 16:

```text
4 A100 x batch 4/GPU = global batch 16
```

The equivalent `2 A100 x batch 8/GPU` configuration is allowed when four
contiguous GPUs are unavailable. Changing the global batch requires a new
optimization study and must not be mixed into the matched final matrix.

## Data And Test Boundary

- Deterministic consensus corpus: 303 systems derived from the 326-system,
  1,340-replica silver corpus.
- Frozen development split: 241 train / 30 validation / 32 original test,
  disjoint by protein-family and ligand scaffold.
- Audited headline test: strict30 after excluding the two systems that fail the
  production coordinate contract.
- Training and checkpoint selection use only train241 and val30.
- MD intermediate frames are supervision and evaluation references only. They
  are never inference inputs.
- **Ligand chemistry addendum (2026-07-26).** The `ligand.sdf` files behind this
  corpus carried connectivity only, with no bond orders and no formal charges.
  209 of the 303 systems have had their ligand bond orders corrected against the
  PDB Chemical Component Dictionary; 198 of those are the silent class, whose MD
  ran to completion with a chemically wrong ligand. Endpoints, ligand
  coordinates, splits, thresholds and Stage-2 inputs are unchanged, and every
  pre-repair SDF is preserved as `ligand.legacy_connectivity_only.sdf`. Whether
  the silver paths must be regenerated is open. See
  `LIGAND_BOND_ORDER_DEFECT_20260726.md`.
- **Data-scale gate addendum (2026-07-26).** The gate's "3,000 novel pairs ->
  2,000 accepted consensus systems" implies a 67% conversion. The AHoJ smoke
  panel completed end to end for the first time and measured **21.9%**
  (7 of 32), and a paired budget probe showed the dominant loss is a physical
  saturation of the RMSD pull rather than an under-spent sampling budget, so it
  does not shrink with effort. At the measured rate 2,000 systems needs roughly
  9,100 pairs against a pool of 1,769 that a full rescan takes to at most 4,000.
  The 3,000 figure needs replacing, not raising. See
  `MD_CORPUS_YIELD_MEASUREMENT_20260726.md`.

The strict30 result has already been evaluated once. It remains the valid frozen
result for the existing checkpoint, but it cannot become a new blind-test result
for a post-test budget change. A larger-budget checkpoint may be reported on
strict30 as a clearly labeled post-freeze replication only. A new final headline
requires one of the following, in priority order:

1. a fresh family/scaffold-disjoint MD holdout acquired after this contract;
2. an audit of the 23 single-replica systems as an external reserve, retaining
   only coordinate-valid systems with no train/validation family or scaffold
   overlap and labeling the reference as single-replica silver evidence.

Neither lane may be used for hyperparameter or epoch selection.

## Data-Scale Gate

The current `241` training examples are independent endpoint systems with at
least two accepted silver MD replicas collapsed into one deterministic consensus
target. They are not the full AHoJ endpoint corpus. The larger endpoint corpus
does not contain intermediate paths or residue-phase labels and therefore cannot
be counted as direct Path-3 supervision.

The endpoint-only route has already been tested at larger scale. Leakage
filtering retained 59,216 of 64,724 endpoint records, canonical OracleMotion was
available for 59,086, and a three-A100, ten-epoch endpoint-flow pretraining run
completed 13,850 optimizer steps. Its `shared_trunk` transfer was worse than the
matched from-scratch full model by 3.41% Product RMSE and 7.35% translation MAE;
all five main paired geometry intervals were on the degradation side. Repeating
that objective on an arbitrary 2k or 10k endpoint subset is not an authorized
data-scaling experiment.

The 2026-07-25 capacity audit gives the actual acquisition boundary:

- 64,724 AHoJ records contain 10,897 unique apo/holo PDB pairs;
- the old equal-CA-count gate admitted 2,307 records but only 584 unique pairs;
- the historical `select1800` artifact therefore selected 584, not 1,800;
- `ca_count_mismatch` alone rejected 41,454 records spanning 7,170 unique pairs.

The old 584-pair lane is exhausted and cannot yield 2,000 path-labeled systems.
Expansion must use an explicit common-residue mapping rather than weaken motion,
force, or path-validity thresholds. The mapping-aware selector and RMSD-pull CV
freeze a minimum 0.95 exact common-residue fraction across apo, holo, and prepared
topology. Sequence or mapping failures remain recorded exclusions.

The mapping-aware scan completed on 2026-07-25. It retained 2,470 unique
endpoint pairs before holdout filtering, 2,262 after the frozen family/scaffold
leakage filter, and 1,769 after excluding historical attempted pairs. The
planning gate therefore fails by 1,231 systems (`1,769 < 3,000`). The 1,769-pair
artifact is an acquisition pool, not a path-labeled training corpus.

Before full MD scale-out:

1. close the planning-pool shortfall without weakening the mapping, leakage, or
   historical-attempt exclusions;
2. require at least 3,000 novel unique pairs before claiming that 2,000 accepted
   consensus systems are operationally plausible;
3. pass a small mismatched-residue context, pull, atomistic, mapping, and target-
   export smoke under the unchanged physical gates;
4. retain only systems with at least two accepted independent replicas;
5. freeze a fresh family/scaffold-disjoint validation and untouched test split.

Train nested unique-system curves at 241, 500, 1,000, and 2,000 only when those
post-gate counts exist. Replicas are repeated observations and never inflate the
system count. Report both epochs and optimizer updates; add a compute-matched
241-versus-largest-size sensitivity so a larger-data result is not presented as
purely a larger-compute result. A 10k path-labeled campaign is out of scope until
the 2k curve demonstrates a continuing validation benefit.

## External Endpoint-Index Audit

AHoJ-DB remains the primary acquisition database. External endpoint databases
are being audited only to replenish or annotate the acquisition pool; none of
them supplies the accepted multi-replica intermediate paths required for direct
Path-3 supervision.

The local `audit_external_endpoint_indexes.py` audit currently indexes:

| Source | Audited units | Endpoint role | Hard limit |
|---|---:|---|---|
| PSCDB | 839 PSCID records from 1,084 motion-component rows | Free/bound motion and segment annotation | Ligand label but no exact residue/site identity; no intermediate path |
| APObind | 12,267 directed apo/holo PDB pairs | Candidate apo/holo pairs plus binding-residue signatures | No ligand identity/site field in the CSV, no explicit repository data license, and no intermediate path |
| CoDNaS-Q | 3,649 conformer clusters | Maximum-tertiary endpoint-pair index | Query/target is not apo/holo; representative ligand fields cannot be reassigned to the maximum pair; data license unresolved |
| CryptoBench | 1,107 structures; net-new source count `0` | Cryptic-pocket benchmark annotation | Constructed from AHoJ-DB, so it cannot expand AHoJ provenance |

Across the first three indexed sources there are 16,755 source records and
16,586 unique undirected PDB pairs; 141 PDB-pair groups occur in more than one
source. PDB-pair-only overlap is:

| AHoJ reference | PSCDB records | APObind records | CoDNaS-Q records | Unique undirected PDB-pair matches |
|---|---:|---:|---:|---:|
| Full 64,724-row endpoint index | 53 | 246 | 149 | 436 |
| Mapping-aware 2,470 pool | 25 | 91 | 45 | 154 |
| Leakage-clean, unattempted 1,769 pool | 14 | 72 | 25 | 107 |

The per-source columns count source records and can overlap; the final column is
deduplicated only at the PDB-pair level. The full AHoJ index contains 64,723
valid endpoint rows, one row missing both endpoint IDs, and 10,897 unique
directed PDB pairs. Relative to that full pair universe, 16,150 external unique
undirected PDB pairs do not match. This remains a PDB-level acquisition upper
bound: the audit deliberately reports `exact_ligand_site_matches=null` and
`net_new_systems=null` because the sources do not expose a common exact
ligand/site key and no production quality gates have been applied.

APObind is the first expansion lane worth a structure-level smoke. Starting
from its 12,021 records that do not match a full-AHoJ PDB pair, a strict metadata
proxy retains 3,924 records and 3,918 unique undirected PDB pairs after requiring
single-chain endpoints, sequence identity and coverage in `[0.95, 1.0]`,
backbone RMSD in `[0.35, 5.0] A`, TM-score in `[0.5, 1.0]`, positive apo
resolution, and nonempty binding-site signatures. The audit excludes rather
than clips 246 sequence-identity values and 206 sequence-coverage values above
one. The 3,918 count is numerically sufficient to cover the 1,231-pair planning
shortfall, but it is not an acceptance estimate.

The deterministic APObind `smoke32_v1` structure preflight completed locally on
2026-07-25. Selection balanced backbone RMSD, apo resolution, and binding-site
size over empirical quartiles (eight systems per quartile for each axis), with
32 systems and 64 unique endpoint PDB IDs. RCSB mmCIF download succeeded for all
64 endpoints. All 32 systems passed declared-chain lookup, exact sequence and
residue mapping, APObind binding-signature resolution, mapped-site consistency,
and the frozen `[0.35, 5.0] A` global CA RMSD screen. Across the 32 systems,
structure-derived sequence identity was `0.9821-1.0000`, symmetric exact mapping
coverage was `0.9544-1.0000`, site mapping was `1.0000`, and global CA RMSD was
`0.3661-2.2670 A`.

After excluding water, crystallization components, metals, and mmCIF components
typed as polymer-linking, 25 of 32 systems had exactly one site-local organic
component with parseable CCD bond-order chemistry, at least 80% observed heavy
atoms, absolute formal charge at most two, and no metal. Four systems had no
plausible non-polymer site ligand and three had multiple plausible site ligands;
all seven remain explicit rejections. This `25/32` is a stratified engineering
smoke result, not an unbiased estimate for all 3,918 proxy pairs and not 25 new
Path-3 training systems. The selection and preflight report SHA256 values are
`84e3f4c57fb06e273f5007f55cb6e457c9145ffcd8b41fcbdbd6db9c6783cd7b`
and `cdda37c1fc9eb7c20432d4ea6325c8bd93d248650f8b228bc7ed9bc97f73bc0a`.

The aligned-triplet export then completed for all 25 chemistry-valid systems.
It writes the selected apo conformer and the holo protein/ligand transformed
into the apo frame, reconstructs the observed heavy-atom ligand with CCD bond
orders, and validates the SDF/NPY coordinate round trip. The canonical
transition manifest has 25 records, zero schema errors or warnings, and zero
records eligible for phase supervision or held-out benchmarking. The unchanged
MD-candidate screen retained 23 and rejected two proteins above the frozen
500-residue limit. Its export report SHA256 is
`3be09edd3c7752f785914162cc9ce83733d7cd54a40282bdc31f963197c8c8fd`.

The 23 screening-valid records were filtered on GPU33 against the frozen val30
and test32 exclusion sets using the existing 30% global sequence identity plus
80% coverage family rule, exact non-isomeric Bemis-Murcko scaffold overlap, and
exact endpoint-PDB overlap. Twenty records were retained. Three were excluded
for protein-family matches; there were no scaffold or endpoint-PDB exclusions.
The leakage report SHA256 is
`7056c5f5ee9c2f43e2c7cf31ba8fb3957112933ef07b62cdc1c5ba8ce049554b`.
These 20 remain endpoint candidates, not accepted Path-3 systems.

A deterministic eight-system preparation panel was selected from the retained
set with four `contact_switch` and four `moderate_motion` systems using max-min
coverage over protein size, ligand heavy atoms, global/pocket RMSD, maximum CA
displacement, and contact changes. The first direct GPU33 preparation completed
with six of eight systems ready. `apobind_1270_1L0E_1MY8_SM3` was rejected
because OpenFF 2.2.1 could not parameterize its `C-B` and `B-O` ligand bonds.
`apobind_9592_5I9U_5NKB_8ZT` minimized but exceeded the frozen maximum residue
net-force gate (`810.663 > 500 kJ/mol/nm`). The threshold was not changed.

One bounded rescue completed on 2026-07-25. The `9592` system received its only
longer full-system minimization attempt (2,500 iterations) and passed at
`249.312 kJ/mol/nm`. The force-field-unsupported slot was replaced by the
deterministic same-category max-min candidate
`apobind_5722_3RNN_3BKI_FQX`, which passed at `154.284 kJ/mol/nm` using the
original 500-iteration budget. The frozen prepared-ready panel therefore has
eight resolved slots: six original preparations, one longer-minimization
rescue, and one replacement. Its prepared-panel SHA256 is
`1169b778d9e3ce22868a97c77cc887c7ec13adf52be41e234c5e8045540ecc54`.

A short direct GPU33 NVT/NPT engineering smoke then completed on all eight
systems with zero process or gate failures. The maximum protein CA RMSD was
`0.722 A` after NVT and `0.921 A` after NPT; the maximum ligand heavy-atom RMSD
was `1.490 A` and `2.403 A`, respectively. Mean NPT densities were
`1.006-1.025 g/mL`, and the maximum production-volume coefficient of variation
was `0.0109`. These are endpoint-context stability results, not transition
paths.

The passed NPT states were registered as eight immutable context records, and
an independent-seed `8 x 2` silver-path pilot was launched directly on GPU33 as
PID `2297274`. It uses the unchanged fixed global CA-RMSD protocol: 500
pre-equilibration steps, 10,000 pulling steps, 2,000 endpoint-hold steps,
`200,000 kJ/mol/nm2`, final target `0.025 nm`, and minimum triple-mapping
fraction `0.95`. The replica-matrix SHA256 is
`3741cce1da979bfc0ffaee4961ec3db4c1159c603077c4cdfb11700b981089cc`.
Completion is defined only by
`processed_data/md_transition/apobind_replica_pilot8x2_20260725_v1/pilot_state.json`;
the launcher log is
`logs/stage2/apobind_replica_pilot8x2_20260725_v1/launcher.log`. The pilot
finished with nine replicas rejected by the unchanged pull gate. Seven replicas
(matrix indices `2,3,6,7,10,11,13`) passed pull, path, and atomistic audits, but
all seven failed target export before target construction because the external
APObind triplet root lacked the canonical `torsion_apo.npz` residue-axis cache.
This is one shared engineering-input failure, not seven physical-path failures;
the training-ready target count therefore remains zero at this point.

`build_apobind_torsion_cache.py` now extracts and hash-audits fresh apo/holo
torsion caches under the existing canonical residue-key schema, and
`run_apobind_target_recovery_gpu33.sh` is restricted to those seven frozen
indices. It snapshots their pre-recovery pipeline states, does not use `--force`,
and writes independent recovery finalization, consensus, and
`target_recovery_state.json` artifacts without replacing `pilot_state.json`.
The nine pull rejections remain excluded, every physical and `0.95` mapping gate
is unchanged, and no recovered Path-3 label is claimed until the recovery state
reports complete.

The bounded recovery then completed with `7/7` target exports, while preserving
the nine pull rejections. Three systems have two passed replicas and therefore
produced consensus targets: `apobind_11332_1N29_1DB4_8IN`,
`apobind_5406_1J1M_4MX1_1MX`, and `apobind_5722_3RNN_3BKI_FQX`.
`apobind_7268_1IA8_2BRO_DF2` has one passed replica and remains outside
consensus. The three paired phase offsets are reproducible (mean deterministic
explained energy `0.9825`; mean tau MAE `0.0428`), but the combined normal
residual is heterogeneous (explained energy `0.6976-0.9351`; pair cosine
`0.3915-0.8721`). Consensus interior residual-valid coverage is only `2.69%`,
`2.91%`, and `16.11%`, respectively. These are valid, immutable pilot silver
targets for engineering and quality control, not evidence for a scalable
training corpus: none is merged into the Path-3 expansion training manifest or
counted toward the 3,000-system acquisition target.

Before any external pair enters the 3,000-system planning pool, it must be
resolved against the full AHoJ provenance and exact ligand/site identity, pass
the same sequence/mapping and frozen holdout filters, and then pass the unchanged
context, atomistic, and multi-replica path gates. Endpoint annotations alone do
not close the current 1,231-system planning shortfall.

## Inference Inputs

The current Path-3 model consumes more than apo and holo coordinates:

1. apo and holo residue frames and chi angles, which define the exact analytic
   endpoint bridge;
2. frozen ESM last-seven residue embeddings and residue identity/masks;
3. one known ligand pose aligned into the apo coordinate frame, represented by
   ligand coordinates and atom/probe types;
4. apo-frame pocket weights and residue graph/geometric edge features;
5. privileged OracleMotion features derived from the known apo/holo endpoints
   and aligned ligand: local rigid displacement, chi displacement, motion
   magnitude/mask, apo/holo contacts, formed/released contacts, and signed
   ligand-distance change.

OracleMotion contains no MD intermediate coordinates, force, velocity, or
physical time. It does use exact holo-derived information and therefore makes
the current model an endpoint-informed upper-bound reconstruction system, not
an apo-only deployable predictor.

During training, audited MD paths are converted into system-level consensus
residue-phase targets. During inference, those MD paths and phase labels are not
available; the network predicts the phase field from the endpoint-conditioned
inputs above.

## Current 241-System Anchor Budget

Unless a validation-only budget probe falsifies this configuration before any
new holdout is inspected, use:

| Setting | Value |
|---|---:|
| Model initialization | from scratch |
| Train / validation systems | 241 / 30 (learning-curve anchor only) |
| Maximum epochs | 40 |
| Early-stop patience | 10 |
| Optimizer learning rate | `1e-4` |
| Global batch size | 16 |
| Preferred hardware | 4 x A100, batch 4/GPU |
| Equivalent fallback | 2 x A100, batch 8/GPU |
| Precision | bf16 AMP |
| Seeds | 7, 42, 137 |
| Checkpoint selector | validation phase/path objective only |
| Test access during training | forbidden |

The 40-epoch cap is four times the previous epoch budget while preserving its
global batch and learning rate. `submit_path3_final_training.sh` blocks a
train241 launch unless `ALLOW_TRAIN241_ANCHOR=1` explicitly labels it as the
learning-curve anchor. Report the selected epoch and optimizer-update count for
every run. Do not compare methods trained with different update budgets without
labeling that difference.

## Internal Experiment Matrix

### Main candidate

- chain-coupled endpoint-fixed non-monotone phase;
- chain residual scale `1.0`;
- one graph-smoothing step;
- phase-only path, with deterministic normal residual disabled;
- three training seeds.

### Reviewer-facing phase controls

Train with the same data, global batch, optimizer, budget, and seeds:

- global monotone phase;
- residue-wise monotone phase;
- independent residue-wise endpoint-fixed non-monotone phase;
- chain-coupled endpoint-fixed non-monotone phase.

These controls answer whether the result is merely a global speed change,
whether residue-specific timing matters, and whether chain coupling adds a
generalizable benefit.

### Path-decomposition controls

Use the same contract for:

- synchronous Cartesian bridge, analytic and untrained;
- phase-only candidate;
- identity-phase normal-residual-only model;
- full phase plus normal-residual model.

The residual-only and full rows test the decomposition. They do not reopen a
residual hyperparameter sweep. The observed/hidden-route synthetic benchmark is
retained as the controlled identifiability experiment.

### Secondary conditioning controls

No-ligand or no-ESM rows are not headline ablations. If the manuscript claims a
ligand-specific effect, use a matched correct-ligand versus wrong-pose/chemistry
counterfactual while keeping endpoints fixed. Because OracleMotion itself
contains holo-ligand contact information, any such result must distinguish the
explicit ligand-token effect from privileged endpoint-derived conditioning.

## External Comparison Set

### Required on every eligible test system

- linear Cartesian morph;
- smoothstep morph;
- synchronous Cartesian `SE(3) x chi` bridge;
- ANM20;
- AdaptiveANM50;
- eBDIMS2;
- Path-3 final candidate.

### Required reproducibility attempts

- TPS-Flow public-system case study, with a cross-system row only after matched
  retraining is possible;
- DeepPath code/weight audit and curated comparison if reproducible.

### Small physical/downstream panel

On a frozen, preflight-valid subset, compare Path-3 initialization against a
neutral interpolation initialization for string, restrained/targeted MD, or
weighted-ensemble sampling. The primary utility endpoints are force calls or
wall time to convergence, relaxation survival, and transition yield. CPU smoke
and failed reference-topology systems are engineering evidence only.

## Metrics And Statistics

### Primary common-axis metrics

- system-macro MD translation MAE in Angstrom, lower is better;
- residue pair-order accuracy, higher is better;
- phase tau MAE and contact-event timing MAE, lower is better;
- method coverage and failure rate.

### Full-state internal metrics

- product-state RMSE;
- rotation MAE and wrap-aware chi MAE;
- phase Spearman and phase monotonicity/backtracking violations;
- endpoint frame/chi/atom14 error as a contract check, not a selection win.

### Biological and path-validity metrics

- formed-contact recall, released-contact success, stable-contact retention;
- transient-contact precision and recall;
- pocket and large-rotamer-flip chi error;
- peptide bond/angle violation incidence;
- clash fraction and severity;
- frame/chi step smoothness and maximum path jump;
- runtime per generated path.

### Independent physical metrics

- prepared-reference preflight acceptance and rejection reasons;
- relaxed-frame survival;
- matched OpenMM force and energy-tail summaries, labeled as proxies rather
  than free energies or kinetics.

Every headline comparison uses system-level paired differences and 100,000-draw
paired bootstrap confidence intervals. Report the number of shared systems,
system-level win rate, failures, and excluded-system reasons. Replica-level rows
are secondary and must not be treated as independent system samples.

## Execution Order

1. Keep completed precheck job `148534` as cache/data evidence only; it performed
   no training and accessed no strict30 examples.
2. Preserve the completed mapping-aware scan, resolve its 1,231-pair planning
   shortfall, and pass the mismatched-residue pipeline smoke.
3. Build the accepted consensus corpus and fresh disjoint development/test split.
4. Run the nested 241/500/1k/2k data curve without test evaluation.
5. Freeze the promoted data size, optimization budget, and inference projection.
6. Run the three-seed candidate and matched phase controls at the promoted size.
7. Complete the residual-only/full negative controls without opening a sweep.
8. Freeze all checkpoints and evaluate the untouched expanded-corpus test once.
9. Run analytic/classical/learned baselines through the same evaluator.
10. Run the frozen physical/downstream utility panel.

The experiment is complete only when model, control, external baseline,
coverage, uncertainty, and failure-accounting rows are all present. A larger
checkpoint alone is not a final paper result.
