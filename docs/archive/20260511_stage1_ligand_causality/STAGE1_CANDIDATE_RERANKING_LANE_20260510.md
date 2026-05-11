# Stage-1 explicit ligand-causal candidate-reranking lane (2026-05-10)

## Why this lane exists

The posterior/candidate diagnostic run `posterior_candidate_ligcausal_screen4safe_20260509_184220` did not show a meaningful correct-ligand advantage over no-ligand or decoy-ligand candidate scoring on switch residues. The previous checkpoint is therefore useful as a diagnostic artifact, but not as a ligand-causal Stage-2 prior.

This lane makes the next hypothesis actionable: train the candidate posterior itself to separate the correct ligand from decoys on residues where ligand information should matter.

## Implemented objective

`candidate_decoy_rerank_loss` acts directly on candidate χ1 logits. On apo→holo switch residues, optionally restricted to CA-ligand contact residues, it requires:

```text
log p_correct_ligand(holo_bin) > log p_decoy_ligand(holo_bin) + margin
```

An optional rank margin also requires the holo bin to outrank the other bins under the correct ligand. This is deliberately separate from the existing G-vector loss: the G loss measures lift over the base branch, while this objective trains the candidate posterior to make a direct correct-vs-decoy reranking decision.

## Entry points

- Loss: `src/stage1/modules/losses.py::candidate_decoy_rerank_loss`
- Config flags: `TrainingConfig.lambda_candidate_rerank`, `candidate_rerank_margin`, `candidate_rerank_rank_margin`, `candidate_rerank_contact_only`, `candidate_rerank_decoy_kind`
- CLI flags: `scripts/train_stage1.py --lambda_candidate_rerank ...`
- Launcher: `scripts/slurm/train_stage1_candidate_rerank_ligcausal_4gpu.sh`

## Validation posture

The launcher uses `selection_metric=candidate_decoy_lift_contact_switch_rotamer_acc`, because the scientific question is not global χ1 accuracy but correct-vs-decoy ligand separation in the candidate posterior on contact switch residues. A successful run still must be followed by candidate-lift diagnostics against no-ligand, translated-away, and shuffled-ligand ablations.

Atom-level contact claims remain disallowed until atom14 availability is repaired; this lane uses CA-ligand contact gating only.
