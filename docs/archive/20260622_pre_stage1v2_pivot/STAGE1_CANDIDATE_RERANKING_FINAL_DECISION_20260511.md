# Stage-1 Candidate Reranking Final Decision (2026-05-11)

## Decision

The explicit Stage-1 candidate-reranking branch is **not a paper-ready ligand-causal solution**. It is a direction-determining partial/negative result:

- it learns a strong signal when the decoy ligand is translated far away, which shows sensitivity to local ligand proximity/pose availability;
- it does **not** learn reliable ligand identity or chemistry causality against no-ligand, scrambled-type, or batch-shuffled-ligand controls;
- therefore this branch should not be extended as-is for Stage-1 ligand-causal posterior claims.

The next scientific step should be a new objective or representation that directly separates correct ligand chemistry/identity from chemically plausible decoys, rather than another continuation of this same reranking lane.

## Evidence summary

### Pre-reranking baseline was not ligand-causal

The previous posterior/candidate route completed successfully but failed ligand-causality diagnostics.

- Training job: `128821`, completed with exit `0:0`.
- Best epoch: `22`, `selection_metric_value=0.508915291377671`.
- Candidate-lift diagnostic job: `128847`, completed with exit `0:0`.
- Switch rotamer accuracy:
  - correct ligand: `0.3025324964199547`
  - no ligand: `0.3032306458369188`
  - translated-away ligand: `0.30208762911705206`
  - shuffled ligand: approximately `0.302601`

This showed no meaningful correct-ligand lift.

The corrected dataset audit confirmed that the diagnostic was not empty:

- audit job: `128849`, completed with exit `0:0`
- samples: `2814`
- chi1-valid residues: `842429`
- switch residues: `308969`
- CA-contact switch residues: `26670`
- pocket-switch residues: `13182`
- atom14 availability: `0`

Because `atom14_available=0`, no atom-level contact-quality claim is made here.

### Translated-away reranking learned a proximity signal

The corrected translated-decoy reranking run completed successfully.

- Training job: `128852`, completed with exit `0:0`.
- Run tag: `candidate_rerank_ligcausal_4gpu_20260510_090528`.
- Selection metric: `candidate_decoy_lift_contact_switch_rotamer_acc`.
- Best epoch: `27`.
- Best selection metric: `0.06486689163854514`.
- Best epoch candidate lifts:
  - contact: `0.1512899835699167`
  - contact-switch: `0.06486689163854514`
  - pocket-switch: `0.0646335912608102`
  - switch: `0.07211403085746465`

This is a real finite signal, but the training decoy was a ligand translated far away from the pocket. That tests whether the model can distinguish ligand-present/local geometry from ligand-absent-by-distance, not whether it uses correct ligand identity or chemistry.

### Full diagnostic showed the translated signal does not survive stricter controls

Full diagnostic job `128894` evaluated the translated-rerank best checkpoint and completed successfully with exit `0:0`.

- Checkpoint: `checkpoints/stage1/candidate_rerank_ligcausal_4gpu_20260510_090528/best_model.pt`
- Diagnostic JSON: `logs/stage1_diagnostics/candidate_rerank_ligcausal_best27_diag_20260510_212744/stage1_prior_diagnostics.json`
- Validation batches: `1424`
- Samples: `2814`

Candidate rotamer lift from correct ligand over translated-away ligand was large:

| subset | correct - translated-away |
| --- | ---: |
| all chi1 | `+0.15234755688609958` |
| pocket | `+0.1512055323655847` |
| ligand-facing apo CA | `+0.13248453983266645` |
| apo-wrong | `+0.07889820778765888` |
| switch | `+0.07147426426416675` |

However, the same checkpoint did not beat stricter controls in a meaningful way:

| subset | correct - no ligand | correct - scrambled types | correct - batch-shuffled ligand |
| --- | ---: | ---: | ---: |
| all chi1 | `-0.0031242989023407275` | `-0.0000427335716126076` | `+0.00003561130967710824` |
| pocket | `+0.0009968226278736503` | `+0.0022428509127156993` | `+0.0030216185907420146` |
| ligand-facing apo CA | `+0.0009457984721716883` | `+0.002037104401600598` | `+0.0002182611858857486` |
| apo-wrong | `-0.0018170230983284297` | `-0.0001070267956176929` | `+0.0005399988324349958` |
| switch | `-0.002389944050630477` | `-0.00006819133840113567` | `+0.0010975558275992947` |

This pattern means the model distinguishes a ligand that has been moved away, but not the correct ligand from chemically scrambled or batch-shuffled ligand controls.

### Shuffled-decoy reranking failed completely

The stricter reranking lane trained directly against batch-shuffled ligands and completed successfully.

- Training job: `128896`, completed with exit `0:0`.
- Run tag: `candidate_rerank_shuffled_ligcausal_4gpu_20260510_215939`.
- Decoy kind: `CANDIDATE_RERANK_DECOY_KIND=shuffled`.
- Selection metric: `candidate_decoy_lift_contact_switch_rotamer_acc`.
- Epochs: `23` through `29`.
- Termination: early stopping at patience `6/6`.
- Failures: no `Traceback`, no `RuntimeError`, no CUDA OOM.

The selection metric stayed exactly zero across every validation epoch:

| epoch | contact-switch lift | contact lift | pocket-switch lift | switch lift | patience |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | `0.0` | `0.0` | `0.0` | `0.0` | `0` |
| 24 | `0.0` | `0.0` | `0.0` | `0.0` | `1` |
| 25 | `0.0` | `0.0` | `0.0` | `0.0` | `2` |
| 26 | `0.0` | `0.0` | `0.0` | `0.0` | `3` |
| 27 | `0.0` | `0.0` | `0.0` | `0.0` | `4` |
| 28 | `0.0` | `0.0` | `0.0` | `0.0` | `5` |
| 29 | `0.0` | `0.0` | `0.0` | `0.0` | `6` |

This is the decisive negative control. If the objective could learn ligand identity from the current representation and supervision, it should have produced positive correct-vs-shuffled lift. It did not.

## Interpretation

The translated-decoy branch is useful as an engineering check: it proves that the candidate-reranking machinery, metrics, checkpointing, and Slurm workflow work, and that the model can exploit gross ligand presence/proximity. It does not prove ligand-causal chemistry.

The shuffled-decoy branch is the relevant scientific test for ligand identity under this setup. Its all-zero lift through early stopping means the current reranking objective is insufficient.

## Consequence for Stage-2

Do not promote this Stage-1 candidate posterior as a deterministic holo endpoint or as a ligand-causal prior for Stage-2. If used at all, treat it as a weak proximity-aware local candidate signal with explicit zero-prior fallback and ablations.

## Recommended next direction

Stop extending this exact reranking lane. Future work should change the scientific object being learned, for example:

1. ligand-identity contrastive objectives over chemically plausible in-pocket decoys rather than translated-away decoys;
2. interaction-level supervision that distinguishes atom/probe chemistry, once atom14/ligand-contact labels are repaired;
3. pairwise residue-ligand energy or contact posterior targets with decoy separation at the ligand-token/probe level;
4. Stage-2-compatible soft posterior features only after these controls show lift over no-ligand and shuffled-ligand baselines.

Until such controls pass, the direction is determined as **not paper-ready**.
