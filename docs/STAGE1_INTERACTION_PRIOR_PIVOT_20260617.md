# Stage-1 Interaction Prior Pivot

Date: 2026-06-17

## Context

Data audits showed that deterministic Stage-1 chi1 endpoint supervision does not expose a clean ligand-causal signal. The current Stage-1 role is therefore narrowed to a soft local residue-ligand interaction/proximity prior for known-pose induced fit. Stage-2 remains responsible for apo-to-holo path modeling.

This is not a docking reframing. Ligand coordinates are still treated as known and aligned to the apo frame.

## New Experiment

Implemented `scripts/train_interaction_prior.py`, a small explicit local pair-feature model that predicts holo sidechain-ligand contact labels from apo sidechain geometry and known-pose ligand tokens.

Primary validation criterion:

- correct ligand logits must separate from no-ligand, translated-ligand, and batch-shuffled-ligand controls.
- selection metric: `val_selection_interaction_lift_min`, the minimum logit lift over biologically relevant subsets and all controls.

Current launcher:

- `scripts/slurm/train_interaction_prior_4gpu.sh`
- 4 x A100 DDP
- default batch/GPU: 96
- train subset: 12,000
- val subset: 1,200

## First Run: Label-Enriched Local Crop

Job: `134519`

Run tag:

- `interaction_prior_bs96_local192_tc4.5_20260617_102740`

This first run used the initial local-crop behavior, which enriched the crop with holo/contact/switch labels. It is useful as a fast signal check but should not be treated as the final strict evidence.

Observed validation metrics:

| Epoch | val_loss | val_contact_ap | val_contact_auroc | val_selection_interaction_lift_min |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.478954 | 0.830136 | 0.979295 | 4.420846 |
| 1 | 0.343349 | 0.856099 | 0.980242 | 6.926480 |
| 2 | 0.339287 | 0.858692 | 0.980338 | 7.098671 |

Interpretation:

- Correct ligand separation is strong under this crop.
- This supports the existence of usable known-ligand local interaction signal.
- It does not rescue deterministic chi endpoint prediction, and it does not by itself prove induced-fit path modeling.

## Strict Follow-Up

The script was updated after the first run:

- default `--crop_mode pocket_only`, so local residue selection depends on apo pocket weights rather than holo/switch labels.
- added `apo_contact`, `contact_gain`, and `gain_switch` validation subsets.
- selection metric now includes `contact_gain` and `gain_switch`.

Submitted strict job:

- Job: `134520`
- Status at note time: pending
- Expected tag prefix: `interaction_prior_pocket_only_...`

Decision rule:

- If strict `pocket_only` keeps positive `val_selection_interaction_lift_min`, especially on `contact_gain` and `gain_switch`, Stage-1 interaction prior is worth integrating into Stage-2 as a soft contact/proximity feature or energy.
- If strict metrics collapse, narrow labels to high-confidence contact gain or clash-rescue subsets before Stage-2 integration.
