# BINDRAE Competitor Dataset And Metric Map

Date: 2026-06-24

## Executive Position

BINDRAE should not be benchmarked as a generic docking method. The clean task is:

```text
apo protein + aligned/known ligand pose
-> ligand-conditioned apo-to-holo protein motion/path
-> endpoint holo-like structure + physically usable intermediate trajectory
```

This creates an awkward but useful benchmark reality:

1. There are strong external baselines for **protein-ligand endpoint complex prediction**.
2. There are emerging baselines for **protein conformational ensembles / flows**.
3. There are very few direct baselines for **known-pose ligand-conditioned apo-to-holo trajectory generation**.

So the fair comparison should be layered, not one-table-fits-all:

| Layer | Question | Main comparison |
|---|---|---|
| A. Endpoint complex quality | Can the method recover a plausible holo complex? | AF3 / Boltz / Chai / FlowDock / NeuralPLexer / DynamicBind-style systems |
| B. Apo-holo induced-fit quality | Does ligand-conditioned motion improve apo->holo active residues? | BINDRAE zero / shuffled / oracle-motion / cubic interpolation / NMA-like controls |
| C. Trajectory reliability | Are intermediate frames usable, smooth, and low-clash? | BINDRAE learned path vs cubic SE(3)+chi interpolation; later short MD relaxation/stability |
| D. Downstream utility | Do generated intermediates help docking, pocket exposure, or lead optimization? | redocking/enrichment into generated ensembles; MD-lite validation |

The current strongest BINDRAE claim is not "we beat AF3 at endpoint complex prediction." It is:

```text
Matched ligand-conditioned motion guidance improves active-residue apo->holo
movement/contact recovery over zero and residue-shuffled controls, while the
learned path is far more geometrically reliable than naive endpoint interpolation.
```

## Competitor Classes

### 1. Closest Competitors: Flexible Protein-Ligand Complex Prediction

These methods are closest because they model protein-ligand complexes with protein flexibility. Most still optimize an endpoint complex, not a time-ordered apo->holo path.

| Method | Typical input | Output | Dataset/benchmark habits | Common metrics | Relation to BINDRAE |
|---|---|---|---|---|---|
| **DynamicBind** | apo/unbound or predicted protein, ligand | ligand-specific protein-ligand complex endpoint; often framed as flexible docking / induced fit | PDBBind-like, PoseBusters-style, DockGen-like external tests; exact dataset split must be rechecked for the target paper/version | ligand RMSD/top-k success, pocket/side-chain quality, clashes, binding-pose validity | Highly relevant endpoint competitor. If it does not produce reliable trajectories, BINDRAE can compare endpoint plus path reliability. |
| **FlowDock** | protein/ligand features; flexible complex prediction | endpoint complex, confidence/affinity-style outputs in some releases | PoseBench includes FlowDock alongside DynamicBind, NeuralPLexer, AF3, Chai, Boltz | ligand RMSD, centroid error, pocket RMSD, PoseBusters validity, ranking/confidence | Very relevant because it uses flow matching for docking-like endpoint prediction. BINDRAE differs by generating apo->holo paths on SE(3)xchi state. |
| **NeuralPLexer** | protein sequence/structure context + ligand | full protein-ligand complex endpoint | PDBBind/CrossDocked/PoseBusters-style benchmarks | ligand RMSD, protein-ligand contact recovery, pocket RMSD, clashes | Relevant endpoint baseline. It is not designed as explicit apo->holo trajectory generation. |
| **Apo2Mol / DynamicFlow-like pocket dynamics models** | apo pocket or conformational ensemble context | ligand/pocket-aware generated structures or molecules | often APObind / MD-derived / pocket-specific splits | pocket RMSD, coverage/diversity, docking/enrichment, geometry validity | Scientifically adjacent. DynamicFlow is especially worth tracking because it explicitly combines flow ideas with protein dynamics. |

Actionable reading note: use PoseBench as the first external benchmark index for endpoint methods, then inspect each method paper for exact split and metric definitions before putting hard numerical comparisons in a manuscript.

### 2. Strong Endpoint Co-Folding / Foundation Model Competitors

These methods are strong at static complex prediction. They are not direct trajectory baselines, but reviewers will ask about them.

| Method | Typical input | Output | Benchmark/dataset habits | Common metrics | Relation to BINDRAE |
|---|---|---|---|---|---|
| **AlphaFold 3** | sequences + ligands/nucleic acids/ions | static biomolecular complex structures | private/internal + public benchmarks; community benchmarks through PoseBench-style wrappers | DockQ-like interface quality, ligand RMSD, contact accuracy, confidence | Strong endpoint reference. Not an apo->holo path model. Use as endpoint teacher/reference when available, not as direct path competitor. |
| **Boltz-1 / Boltz-2** | biomolecular system specification | static complex; Boltz-2 also emphasizes affinity | open-source weights/code; PoseBench-style structure tests; affinity benchmarks for Boltz-2 | complex accuracy, ligand RMSD, PoseBusters validity, affinity correlation/RMSE where available | Useful open endpoint teacher/baseline. Good candidate for Stage-1 teacher labels later. |
| **Chai-1** | multimodal biomolecular input | static complex structures | open model/code; PoseBench-style external comparisons | complex/ligand RMSD, contact/interface quality, confidence | Useful endpoint teacher/baseline; not path generator. |

For BINDRAE, these should be used as endpoint reference teachers, external baselines for final holo-like complex quality, and not as direct competitors for intermediate trajectory reliability.

### 3. Protein Ensemble / Flow / Dynamics Generators

These methods are relevant to the trajectory/ensemble side but often lack explicit ligand conditioning.

| Method | Typical input | Output | Dataset/benchmark habits | Common metrics | Relation to BINDRAE |
|---|---|---|---|---|---|
| **AlphaFlow** | sequence / AF-style representation | protein conformational ensembles | MD ensembles / structural ensemble benchmarks | ensemble coverage, RMSF/contact-map consistency, diversity, lDDT/TM-style structure metrics | Relevant for ensemble generation and flow matching, but usually not ligand-conditioned apo+pose->holo path. |
| **P2DFlow** | protein structure/representation | SE(3) flow-based protein ensembles | ensemble/conformation benchmarks | coverage/diversity, RMSD/contact distribution metrics | Architecturally adjacent. Good related work, less direct as ligand-conditioned induced fit. |
| **EigenFold / Str2Str-like models** | single protein structure or sequence | alternative conformations / structure-to-structure samples | apo/holo or ensemble benchmarks depending on paper | RMSD to alternative states, diversity, distribution coverage | Useful conceptual baselines for protein conformational change but not ligand-specific endpoint/path with a known ligand pose. |
| **MD / enhanced sampling** | all-atom system with force field | physical trajectories | system-specific MD, MISATO/DD-style trajectory datasets, cryptic-pocket studies | RMSD/RMSF, contact lifetimes, energy, clashes, pocket volume, transition/path CVs | Gold-standard-ish for physical plausibility, but expensive and not a supervised ML baseline at BINDRAE scale. Best used for post-hoc validation on selected systems. |

### 4. Classical / Cheap Controls

These are not competitors in paper novelty, but they are essential controls.

| Control | What it tests | Metrics |
|---|---|---|
| zero guidance Stage-2 | whether motion feature capacity alone helps | same endpoint/path metrics as matched |
| residue-shuffled oracle motion | whether residue-aligned motion matters | same endpoint/path metrics; should collapse toward zero |
| sample-shuffled oracle motion | whether sample identity matters | same endpoint/path metrics; requires same-length or safe alignment control |
| cubic SE(3)+chi interpolation | endpoint-aware naive path baseline | endpoint, path MAE, clash, peptide geometry, smoothness |
| NMA / elastic-network interpolation | cheap physics-inspired directionality baseline | endpoint/path active-residue distance, smoothness, clashes |
| short restrained MD / minimization | physical sanity check | relaxation stability, clash removal, RMSD drift, contact retention |

## Dataset Map

### External Datasets / Benchmarks

| Dataset / benchmark | What it contains | Why it matters | Fit to BINDRAE | Notes / cautions |
|---|---|---|---|---|
| **PDBBind / CASF** | protein-ligand complexes with affinity labels; mostly holo complexes | standard docking/scoring benchmark source | good for endpoint/affinity, weak for apo->holo paths | CASF is not enough for BINDRAE Stage-2 because apo structures and trajectories are missing. |
| **PoseBusters Benchmark** | curated protein-ligand pose prediction benchmark plus validity checks | widely used to test whether predicted poses are physically valid | useful endpoint sanity benchmark | It targets docking/pose validity, not induced-fit paths. |
| **DockGen** | protein-ligand docking generalization benchmark | stresses novel binding-site / complex generalization | useful endpoint generalization test | Still endpoint-focused. |
| **PLINDER** | large protein-ligand interaction dataset/benchmark infrastructure | modern large-scale split and leakage control | useful for future endpoint teacher/baseline comparison | Need check whether apo/holo pairing exists for our task; likely not directly trajectory-ready. |
| **APObind / apo-holo paired sets** | apo/holo protein pairs, often ligand-associated | directly relevant to induced-fit endpoint changes | high fit if ligand pose and side-chain states are usable | Need strict filtering: aligned ligand pose, apo-holo sequence match, residue mapping, ligand quality. |
| **AHoJ-like apo/holo triplets / current BINDRAE triplets** | apo structure, aligned ligand pose, holo structure, ESM/torsions | current project-native training/eval source | highest fit for current experiments | Treat as our main benchmark lane; document split construction and leakage filters carefully. |
| **MISATO / MD protein-ligand trajectories** | molecular dynamics trajectories for protein-ligand complexes | possible path-level teacher or validation data | medium fit | Licensing/format/runtime must be checked. MD conditions may not match apo->holo transition labels. |
| **DD-style unbinding / dynamics datasets** | large-scale protein-ligand MD trajectories | useful for physical reliability and unbinding/contact dynamics | optional future validation | Mostly trajectory validation/teacher data, not direct apo->holo supervised endpoint labels. |

### Current BINDRAE Internal Split Scale

The project records currently support the following active lanes:

| Lane | Use | Scale |
|---|---|---:|
| train512 / val512 | smoke and quick upper-bound checks | 512 / 512 |
| train4096 / val512 | current robust oracle-motion scale-up | 4096 / 512 |
| train12000 / val1200 | main Stage-2 candidate scale from earlier LC-PGBF lane | 12000 / 1200 |
| full strict ligand-causal pool | future larger training | project records indicate tens of thousands; verify exact remote split before publication |

Publication-quality documentation should include exact split files, leakage control, apo-holo residue mapping success rate, ligand pose alignment convention, ligand quality filters, and whether protein families leak across train/val/test.

## Metric Map

### Endpoint Complex Metrics

These let BINDRAE talk to docking/cofolding baselines.

| Metric | Definition / implementation idea | Report on |
|---|---|---|
| ligand heavy-atom RMSD | predicted ligand pose vs holo ligand pose; only if ligand is predicted/refined | all complexes, pocket subset |
| protein pocket RMSD | pocket residue atom14 / side-chain RMSD vs holo | ligand-facing residues |
| backbone frame translation error | per-residue frame origin error in A | all, pocket, active |
| backbone frame rotation error | SO(3) log-norm error in rad | all, pocket, active |
| chi error | wrap-aware side-chain chi error | valid chi masks, pocket/active |
| atom14 endpoint error | full atom14 coordinate error vs holo | all, pocket, side-chain contact residues |
| formed-contact recall | fraction of apo-missing/holo-present contacts recovered | formed-contact residues |
| released-contact success | fraction of apo-present/holo-absent contacts released | released-contact residues |
| stable-contact retention | fraction of stable holo/apo contacts retained | stable-contact residues |
| clash / PoseBusters validity | steric and chemical validity checks | endpoint structures |

Important: for BINDRAE current known-pose setup, ligand RMSD is not the main metric unless the method is extended to move/dock the ligand. The protein response around the fixed/aligned ligand is the central endpoint question.

### Apo-Holo Motion Metrics

These are more central than generic endpoint RMSD.

| Metric | Meaning | Why it matters |
|---|---|---|
| active endpoint distance error | final side-chain-ligand min-distance vs holo on residues whose apo/holo distance changes by >= 0.75 A | directly tests induced-fit movement |
| active path MAE | deviation from a simple apo->holo distance schedule over integration frames | tests whether the path follows the intended transition |
| active direction accuracy | whether predicted contact distance moves in the correct sign | tests approach vs release direction |
| approach / release subset metrics | split active residues by holo moving closer/farther from ligand | prevents one class from hiding failure in the other |
| formed-contact recall | recovery of new holo contacts absent in apo | important for cryptic/induced pocket opening |
| released-contact success | removal of apo contacts that should disappear | detects path conservatism |

### Trajectory Reliability Metrics

These distinguish BINDRAE from endpoint-only competitors.

| Metric | Definition / implementation idea | Interpretation |
|---|---|---|
| path clash penalty | atom-level steric overlap along all generated frames | lower is better; compare to cubic interpolation and MD-relaxed structures |
| peptide geometry loss | C-N peptide bond/angle guard along path | lower is better; cubic per-residue interpolation can be very bad here |
| smooth frame step norm | SE(3) step magnitude between adjacent frames | detects jumps/explosions |
| smooth chi step abs | wrap-aware chi changes between adjacent frames | detects side-chain jumps |
| path contact profile | contact formation/release over t | should be plausible; not necessarily strictly monotonic |
| endpoint/path Pareto | endpoint improvement vs geometry degradation | avoids declaring a path good only because it reaches holo by construction |

Current BINDRAE reliability evaluator already reports the core set:

```text
model/active/endpoint_abs_dist_A
model/active/path_mae_dist_A
model/active/direction_acc
model/active/frame_trans_err_A
model/active/frame_rot_err_rad
model/active/chi_err_rad
model/active/atom14_endpoint_err_A
model/formed_contact/recall
model/released_contact/release_success
model/stable_contact/retention
model/path/clash_penalty
model/path/peptide_loss
model/active/smooth_frame_step_norm
model/active/smooth_chi_step_abs
cubic_ref/path/peptide_loss
cubic_ref/path/clash_penalty
```

### Ensemble / Downstream Metrics

These are important for the eventual useful-for-drug-design story.

| Metric | Use |
|---|---|
| pocket-open ensemble coverage | whether generated frames cover holo-like pocket opening states |
| diversity at fixed endpoint quality | whether multiple plausible paths/frames are produced |
| redocking success into generated frames | whether intermediate frames improve docking of holo ligand or analogs |
| enrichment / virtual-screening lift | whether generated ensembles recover active ligands better than apo-only docking |
| short-MD stability | whether selected frames survive minimization/short restrained MD without collapsing |
| contact lifetime under MD-lite | whether key ligand/protein contacts persist or form plausibly |

## Recommended Benchmark Design For BINDRAE

### Tier 0: Internal Causal Controls

This is mandatory before any external comparison.

| Experiment | Required comparison | Pass criterion |
|---|---|---|
| oracle-motion Stage-2 | matched vs zero vs residue-shuffled | matched improves active endpoint/path/contact; shuffled returns near zero |
| sample-shuffled oracle-motion | matched vs same-length sample-shuffled | matched wins; no silent length padding |
| steps robustness | steps 3/6/12 | ranking stable; no path geometry blow-up |
| scale robustness | train512 -> train4096 -> train12000 | gains do not disappear with more data |
| seed robustness | 2-3 seeds at fixed split | matched advantage exceeds seed noise |

### Tier 1: Cheap Baselines

| Baseline | Why |
|---|---|
| apo endpoint | measures how much movement is needed |
| holo endpoint oracle | numerical ceiling |
| cubic SE(3)+chi interpolation | endpoint-aware but geometry-weak path baseline |
| NMA / elastic interpolation | cheap physics-inspired direction baseline |

### Tier 2: External Endpoint Baselines

Run or cite carefully:

| Baseline | Use |
|---|---|
| DynamicBind / FlowDock | closest flexible endpoint competitors |
| NeuralPLexer | endpoint flexible complex predictor |
| AF3 / Boltz / Chai | strong static complex/teacher references |

For external baselines, compare only endpoint holo-like complex quality, pocket side-chain/contact recovery, protein clash/validity, and not time-ordered path reliability unless the method outputs a path.

### Tier 3: Physics Validation

Use on a smaller curated panel:

| Validation | Purpose |
|---|---|
| energy minimization | remove obvious impossible structures |
| short restrained MD | test stability of generated intermediates |
| pocket volume/contact tracking | validate mechanism plausibility |
| comparison with known MD/transition literature | system-level sanity check |

## What Would Count As A Strong Result

A paper-grade story should show all of the following:

1. **Causal guidance works**: matched oracle/student motion beats zero and shuffled controls.
2. **Path is not fake interpolation**: learned path has much lower peptide/clash penalties than cubic endpoint interpolation.
3. **Endpoint is useful**: final frames recover holo-like pocket contacts and side-chain geometry better than apo/zero controls.
4. **Scale holds**: train4096 results replicate on train12000 or larger.
5. **External context is fair**: endpoint numbers are compared against FlowDock/DynamicBind/AF3/Boltz/Chai-like baselines where possible, while path metrics are presented as BINDRAE-specific because competitors do not target explicit apo->holo trajectories.
6. **Physics sanity holds**: selected generated frames survive minimization/short MD and retain plausible contacts.

## Novelty Assessment

Current BINDRAE, using oracle motion, is not yet a complete deployable method because oracle motion uses apo/holo ground truth. But it is a valuable upper-bound system:

- It tests whether the Stage-2 architecture can exploit residue-aligned ligand-conditioned motion features.
- It creates a measurable target for a future Stage-1 motion-posterior student.
- It gives an immediate trajectory generation tool for cases where apo, holo, and ligand are known but the intermediate path is missing.

Novelty is strongest if framed as:

```text
posterior-guided bridge flow for known-pose ligand-conditioned induced-fit
trajectory generation
```

Novelty is weaker if framed as:

```text
another protein-ligand complex endpoint predictor
```

The RAEv2 / REPA / last-K ESM ideas should be treated as representation upgrades, not the core claim. They are useful if they improve the matched-vs-control gap or help a learned Stage-1 student approach oracle-motion guidance.

## Immediate Next Table To Fill With Numbers

After train4096/e10 reliability finishes, fill this table first:

| Model | Train size | Guidance | Steps | active endpoint ↓ | active path MAE ↓ | direction ↑ | chi err ↓ | formed recall ↑ | peptide ↓ | cubic peptide ↓ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage-2 | 4096 | zero | 12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Stage-2 | 4096 | oracle-motion matched | 12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Stage-2 | 4096 | oracle-motion residue-shuffled | 12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Then add the train12000/eN version:

| Model | Train size | Guidance | Steps | active endpoint ↓ | active path MAE ↓ | direction ↑ | chi err ↓ | formed recall ↑ | peptide ↓ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-2 | 12000 | zero | 12 | TBD | TBD | TBD | TBD | TBD | TBD |
| Stage-2 | 12000 | oracle-motion matched | 12 | TBD | TBD | TBD | TBD | TBD | TBD |
| Stage-2 | 12000 | oracle-motion residue-shuffled | 12 | TBD | TBD | TBD | TBD | TBD | TBD |

## Source Links To Recheck Before Publication

- AlphaFold 3: https://www.nature.com/articles/s41586-024-07487-w
- PoseBench: https://github.com/BioinfoMachineLearning/PoseBench
- FlowDock: https://github.com/BioinfoMachineLearning/FlowDock
- DynamicBind code/paper entry: https://github.com/luwei0917/DynamicBind
- NeuralPLexer: https://arxiv.org/abs/2209.15171
- Boltz: https://github.com/jwohlwend/boltz
- Chai-1: https://github.com/chaidiscovery/chai-lab
- AlphaFlow: https://arxiv.org/abs/2402.04845
- DynamicFlow: https://arxiv.org/abs/2503.03989
- P2DFlow: https://arxiv.org/abs/2411.17196
- PLINDER: https://www.plinder.sh/
- APObind: https://arxiv.org/abs/2108.09926

## Open Questions

1. Which exact external baselines can we run locally rather than cite?
2. Can we construct a PoseBench-like endpoint subset from our apo/holo triplets without leakage?
3. Does DynamicBind expose intermediate denoising states that can be fairly compared as a path, or are they only optimization artifacts?
4. Can APObind or PLINDER be filtered into a clean apo + aligned ligand + holo split for BINDRAE?
5. Should train12000 use only strict ligand-causal samples, or a larger mixed pool with stratified reporting?
