# Stage-1 Agent Guide

## Role
Stage-1 is BINDRAE's holo-prior and pocket-posterior subsystem. It consumes apo structure, frozen ESM-derived sequence features, ligand tokens/probes, and apo geometry, then predicts holo-like frames, chi torsions, contact signals, or candidate chi1 posterior information for Stage-2 and diagnostics.

Current work treats Stage-1 less as a deterministic endpoint oracle and more as a local ligand-conditioned posterior whose value must be proven against base-only and decoy-ligand baselines.

## Main files
- `models/stage1_model.py`: high-level wiring from ESM adapter through ligand conditioning, FlashIPA, torsion/contact heads, and FK outputs.
- `models/torsion_head.py`: chi heads and candidate chi1 scorers, including `GeometryCandidateScorer` with base-prior plus ligand-residual decomposition.
- `models/ligand_condition.py`: ligand atom/probe embedding, cross-attention, FiLM-style residue conditioning, and enhanced RBF options.
- `models/ipa.py`, `models/fk_openfold.py`, `models/forward_kinematics.py`: geometric trunk and FK reconstruction utilities.
- `training/trainer.py`, `training/config.py`: DDP loop, validation metrics, warmups, selection metric direction, and phase controls.
- `datasets/dataset_stage1.py`: apo/holo/ligand triplet loading, alignment assumptions, pocket weights.
- `modules/losses.py`: FAPE, torsion, clash, rotamer, lift/no-harm, and auxiliary objectives.

## Current modeling pattern
The geometry candidate path decomposes chi1 logits into a residue/backbone/apo-chi base prior plus a bounded ligand-distance residual. Use `freeze_base`, residual reset, gate override/freeze, beta scaling, and base-detach deliberately; these controls exist to test ligand-causal lift rather than inflate global rotamer accuracy.

`PocketRoutingAdapter` and ligand conditioning are meant to amplify pocket-relevant signals without destabilizing the full protein. If changing these paths, check whether non-pocket residues are being moved or harmed as a side effect.

## Metrics that matter
Do not judge ligand causality from `val_chi1_rotamer_acc` alone. Prefer contact/switch or pocket/switch subsets, full-vs-base lift, full-vs-decoy lift, apo-wrong rescue, apo-correct harm, and confidence-calibrated posterior quality. If `correct_ligand`, `no_ligand`, and `base_only` match, the result is base-prior performance, not ligand evidence.

When adding a validation metric, update both metric computation and metric direction/selection choices. A metric that can be selected for checkpointing must have an unambiguous maximize/minimize direction.

## Safety rules
Preserve angle periodicity with sin/cos or wrap-aware differences. Keep FK and FAPE globally SE(3)-consistent. Do not suppress type or tensor-shape errors; most failures here indicate mask, residue length, or ligand-token inconsistencies that should be fixed at the source.

Respect residue masks, chi masks, and ligand masks throughout reductions. Avoid trainer workarounds for corrupt ligands; malformed SDF inputs should be handled by preprocessing or dataset filtering.

## Common edit paths
For a new loss, wire `modules/losses.py`, config defaults, trainer accumulation, logging, and any Slurm CLI flags together. For a new candidate scorer behavior, update `models/torsion_head.py`, model output handling, validation ablations, and checkpoint loading compatibility. For a new selection metric, update `scripts/train_stage1.py` choices as well as trainer best-metric logic.

## Remote experiment pattern
Use Slurm scripts in `scripts/slurm/` for smoke, phase, and diagnostic runs. Branch-resume from a checkpoint should write to a new tag-specific log/checkpoint directory unless intentionally continuing an existing run.

## Validation
For lightweight checks, compile changed files with `python -m py_compile src/stage1/training/trainer.py scripts/train_stage1.py` and run targeted imports such as `python -c "from src.stage1.models.stage1_model import Stage1Model"`. For real training changes, launch a Stage-1 smoke or phase script through `scripts/slurm/`, never directly on the login node.
