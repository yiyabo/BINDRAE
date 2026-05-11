# Consultation Brief: BINDRAE Stage-1 Ligand-Causality Failure and Next Directions

## How to use this brief

Give this document to another AI model or research advisor and ask them to propose concrete next research directions. The desired output is **not** a generic accuracy-improvement plan. We need a ligand-causality redesign that can pass negative controls.

Recommended prompt:

> We are developing BINDRAE, a known-pose protein-ligand induced-fit system. Stage-1 should provide ligand-conditioned local structural priors/posteriors for apo-to-holo modeling. Our current candidate-reranking objective learned ligand proximity but failed ligand identity/chemistry controls. Please critically review the evidence below and propose 3-5 implementable next directions. For each direction, specify the objective, required labels/data, code surfaces, positive/negative controls, expected failure modes, and a two-week screening plan.

## Project context

BINDRAE is a two-stage protein-ligand induced-fit framework for **known ligand poses**. It is not intended as a docking method. Ligand coordinates are assumed to be in the apo frame or consistently aligned into it.

Stage-1 currently consumes:

- apo protein structure;
- sequence embeddings;
- ligand atom/probe tokens;
- apo geometry;
- holo supervision during training.

Stage-1 outputs can include holo-like frames, side-chain chi predictions, contact signals, and candidate chi1 posterior information. Stage-2 should later consume Stage-1 information as a soft prior or guidance signal for apo-to-holo bridge flow.

The current scientific question is not “can we improve global chi1 accuracy?” Global chi1 accuracy can be dominated by a Dunbrack-like/base structural prior. The question is:

> Does correct ligand information improve local predictions over base-only, no-ligand, translated-away, scrambled-type, and batch-shuffled ligand controls on biologically relevant residues?

Relevant subsets include contact residues, ligand-facing residues, switch residues, apo-wrong residues, and pocket-switch residues.

## Current conclusion

The current Stage-1 candidate-reranking lane is **not paper-ready ligand-causal learning**.

It learned a strong signal when the decoy ligand was translated away from the pocket, but it failed stricter ligand identity/chemistry controls. Therefore, it appears to exploit ligand presence/proximity rather than the correct ligand’s chemical identity or interaction pattern.

A final internal decision memo records this conclusion:

- `docs/STAGE1_CANDIDATE_RERANKING_FINAL_DECISION_20260511.md`

## Key experimental evidence

### 1. Baseline posterior/candidate route was not ligand-causal

The previous Stage-1 posterior/candidate route completed training but failed ligand-causality diagnostics.

- Training job: `128821`, completed with exit `0:0`.
- Candidate-lift diagnostic job: `128847`, completed with exit `0:0`.
- Switch rotamer accuracy:
  - correct ligand: `0.3025324964199547`
  - no ligand: `0.3032306458369188`
  - translated-away ligand: `0.30208762911705206`
  - shuffled ligand: approximately `0.302601`

Interpretation: correct ligand did not improve over no-ligand or shuffled controls.

The dataset audit showed enough evaluable examples:

- audit job: `128849`, completed with exit `0:0`;
- samples: `2814`;
- chi1-valid residues: `842429`;
- switch residues: `308969`;
- CA-contact switch residues: `26670`;
- pocket-switch residues: `13182`;
- atom14 availability: `0`.

Important limitation: because `atom14_available=0`, we currently should not make atom-level contact-quality claims.

### 2. Explicit candidate-reranking against translated-away decoys produced a signal

We implemented an explicit candidate-reranking lane with a `candidate_decoy_rerank_loss`, plus config, CLI flags, trainer wiring, validation metrics, and a Slurm launcher.

The corrected translated-decoy rerank run completed successfully:

- job: `128852`, completed with exit `0:0`;
- run tag: `candidate_rerank_ligcausal_4gpu_20260510_090528`;
- selection metric: `candidate_decoy_lift_contact_switch_rotamer_acc`;
- best epoch: `27`;
- best selection metric: `0.06486689163854514`.

Best-epoch candidate lifts:

| subset | lift |
| --- | ---: |
| contact | `0.1512899835699167` |
| contact-switch | `0.06486689163854514` |
| pocket-switch | `0.0646335912608102` |
| switch | `0.07211403085746465` |

Interpretation: the machinery works and the model can use gross ligand proximity/presence. However, translated-away decoys are easy and do not test ligand identity or chemistry.

### 3. Full diagnostic showed the translated signal does not survive stricter controls

Diagnostic job `128894` evaluated the translated-rerank best checkpoint:

- checkpoint: `checkpoints/stage1/candidate_rerank_ligcausal_4gpu_20260510_090528/best_model.pt`;
- output JSON: `logs/stage1_diagnostics/candidate_rerank_ligcausal_best27_diag_20260510_212744/stage1_prior_diagnostics.json`;
- validation batches: `1424`;
- samples: `2814`;
- exit: `0:0`.

Correct ligand beat translated-away ligand strongly:

| subset | correct - translated-away |
| --- | ---: |
| all chi1 | `+0.15234755688609958` |
| pocket | `+0.1512055323655847` |
| ligand-facing apo CA | `+0.13248453983266645` |
| apo-wrong | `+0.07889820778765888` |
| switch | `+0.07147426426416675` |

But correct ligand did not meaningfully beat no-ligand, scrambled-type, or batch-shuffled controls:

| subset | correct - no ligand | correct - scrambled types | correct - batch-shuffled ligand |
| --- | ---: | ---: | ---: |
| all chi1 | `-0.0031242989023407275` | `-0.0000427335716126076` | `+0.00003561130967710824` |
| pocket | `+0.0009968226278736503` | `+0.0022428509127156993` | `+0.0030216185907420146` |
| ligand-facing apo CA | `+0.0009457984721716883` | `+0.002037104401600598` | `+0.0002182611858857486` |
| apo-wrong | `-0.0018170230983284297` | `-0.0001070267956176929` | `+0.0005399988324349958` |
| switch | `-0.002389944050630477` | `-0.00006819133840113567` | `+0.0010975558275992947` |

Interpretation: the model distinguishes a ligand moved away from the pocket, but it does not distinguish correct ligand identity/chemistry from close or structurally confounding controls.

### 4. Direct shuffled-decoy training failed

We then trained the same candidate-reranking objective directly against batch-shuffled ligand decoys.

- job: `128896`, completed with exit `0:0`;
- run tag: `candidate_rerank_shuffled_ligcausal_4gpu_20260510_215939`;
- decoy kind: `CANDIDATE_RERANK_DECOY_KIND=shuffled`;
- selection metric: `candidate_decoy_lift_contact_switch_rotamer_acc`;
- epochs: `23` through `29`;
- termination: early stopping at patience `6/6`;
- runtime failures: none observed (`Traceback=0`, `RuntimeError=0`, CUDA OOM `0`).

The candidate lift stayed exactly zero throughout:

| epoch | contact-switch lift | contact lift | pocket-switch lift | switch lift | patience |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | `0.0` | `0.0` | `0.0` | `0.0` | `0` |
| 24 | `0.0` | `0.0` | `0.0` | `0.0` | `1` |
| 25 | `0.0` | `0.0` | `0.0` | `0.0` | `2` |
| 26 | `0.0` | `0.0` | `0.0` | `0.0` | `3` |
| 27 | `0.0` | `0.0` | `0.0` | `0.0` | `4` |
| 28 | `0.0` | `0.0` | `0.0` | `0.0` | `5` |
| 29 | `0.0` | `0.0` | `0.0` | `0.0` | `6` |

Interpretation: direct correct-vs-shuffled candidate reranking does not learn a useful ligand-identity signal under the current representation, objective, and supervision.

## Current implementation surfaces

The most relevant code areas for redesign are:

- `src/stage1/models/ligand_condition.py`  
  Ligand atom/probe embedding, ligand-protein cross-attention, FiLM-style conditioning, RBF/probe features.

- `src/stage1/models/torsion_head.py`  
  Chi prediction heads and `GeometryCandidateScorer`, including base-prior plus ligand-residual decomposition.

- `src/stage1/modules/losses.py`  
  Current losses, including `candidate_decoy_rerank_loss`, contact loss, rotamer losses, and lift/no-harm objectives.

- `src/stage1/training/trainer.py`  
  Loss wiring, validation metrics, decoy construction, candidate metrics, checkpoint selection.

- `src/stage1/datasets/dataset_stage1.py`  
  Apo/holo/ligand triplet loading, pocket weights, masks, and data-derived labels.

- `scripts/diagnose_stage1_prior.py` and `scripts/slurm/diagnose_stage1_postcand4safe.sh`  
  Diagnostic variants and full validation evaluation against no-ligand, translated-away, scrambled, and batch-shuffled ligand controls.

## What we need help deciding

Please do **not** focus on global chi1 accuracy alone. We need a strategy that can pass ligand-causality controls.

### Question 1: What should Stage-1 learn?

Should Stage-1 continue predicting a holo-like endpoint/candidate posterior, or should it be reframed as one of the following?

- a calibrated local rotamer posterior;
- a residue-ligand contact compatibility model;
- a ligand-conditioned energy or score over candidate side-chain states;
- a soft prior for Stage-2 rather than a deterministic endpoint anchor;
- an uncertainty-aware posterior that tells Stage-2 where ligand evidence is actually informative.

Please recommend one primary target and justify why it is more likely to produce ligand-causal signal than the current candidate-reranking objective.

### Question 2: What objective should replace current candidate reranking?

The current objective works for translated-away decoys but fails for shuffled decoys.

Candidate alternatives include:

- multi-negative InfoNCE over chemically plausible in-pocket decoys;
- pairwise residue-ligand interaction energy scoring;
- masked ligand-token prediction tied to residue rotamer changes;
- contrastive learning over ligand probes/functional groups;
- rotamer-contact joint likelihood;
- calibration objective for ligand-sensitive subsets only;
- causal residual regularization that penalizes base-prior-only explanations.

For each proposed objective, please specify:

1. mathematical form;
2. positive examples;
3. negative examples;
4. required labels;
5. how it avoids trivial proximity shortcuts;
6. expected failure mode.

### Question 3: What decoy construction is scientifically appropriate?

Translated-away decoys are too easy. Batch-shuffled decoys may be too hard or too noisy.

What decoy families should we use?

Possibilities:

- same-pocket but chemically mismatched ligands;
- scaffold-preserving but functional-group-perturbed ligands;
- property-matched ligands from other proteins;
- pose-preserving atom-type scrambling;
- ligand probes with directionality removed or swapped;
- local pocket decoys that preserve distance distributions but alter chemistry.

We need decoys that prevent the model from solving the task by detecting ligand presence or distance alone.

### Question 4: Should atom14/contact labels be fixed first?

Current audit reported `atom14_available=0`, so atom-level contact supervision is not currently reliable. Should the next priority be repairing atom14/contact label generation before designing a new loss?

If yes, what is the minimal contact-label pipeline needed?

If no, what weak supervision can substitute for atom-level contact labels?

### Question 5: How should Stage-2 consume Stage-1 after this result?

Given that Stage-1 is not a reliable deterministic endpoint prior, should Stage-2 use Stage-1 only as:

- zero-prior fallback;
- soft local posterior;
- contact compatibility feature;
- uncertainty/confidence feature;
- auxiliary ablation-only guidance?

What interface minimizes the risk of propagating a non-causal Stage-1 prior into Stage-2 path generation?

## Requested output format from the consulted model/advisor

Please return a structured plan with the following sections.

### A. Diagnosis

- Why did translated-away reranking work?
- Why did shuffled/no-ligand/scrambled controls fail?
- Is the failure more likely caused by objective design, ligand representation, data labels, model capacity, or evaluation design?

### B. Top 3 next directions

For each direction:

1. hypothesis;
2. objective/loss;
3. data and labels required;
4. code surfaces to change;
5. controls that must pass;
6. go/no-go metrics;
7. expected runtime budget;
8. failure modes.

### C. Two-week screening plan

Design a minimal validation plan that can decide whether the direction is promising without full-scale training.

Required metrics should include:

- lift over no-ligand;
- lift over shuffled or chemically plausible decoys;
- contact/switch subset metrics;
- apo-wrong rescue and apo-correct harm;
- calibration or confidence quality if posterior outputs are used;
- explicit negative controls.

### D. Stage-2 interface recommendation

Specify whether Stage-1 should feed Stage-2 as a deterministic endpoint, soft posterior, contact feature, uncertainty feature, or be omitted until stronger controls pass.

## Non-negotiable constraints

- Do not reframe the task as docking; ligand pose is known.
- Do not optimize or select by global chi1 accuracy alone.
- Do not claim ligand causality unless correct ligand beats no-ligand and decoy-ligand controls on biologically relevant subsets.
- Do not rely on atom-level contact claims until atom14/contact labels are fixed.
- Preserve angle periodicity and SE(3)-consistent geometry.
- Full training must be launched through Slurm, not on the login node.

## One-sentence summary

Our current Stage-1 candidate-reranking objective can detect whether a ligand is near the pocket, but it cannot distinguish the correct ligand from no-ligand, scrambled, or shuffled controls; we need a new ligand-causal objective, decoy design, and Stage-2 interface that prioritize chemically meaningful ligand evidence over proximity shortcuts.
