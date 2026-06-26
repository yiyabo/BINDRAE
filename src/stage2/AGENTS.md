# Stage-2 Agent Guide

## Role
Stage-2 learns ligand-conditioned apo-to-holo bridge dynamics on product states of per-residue `SE(3)` frames and chi angles. It uses Conditional Flow Matching on reference bridges plus path-level geometry constraints, with optional Stage-1 prior features and NMA-informed mobility.

The deliverable is not only a final holo-like structure; the intermediate path must remain smooth, low-clash, chain-consistent, and biologically plausible around ligand contact formation.

## Main files
- `models/torsion_flow.py`: `TorsionFlowNet`, ligand-conditioned vector field, gates, Stage-1 relative features, twist and chi velocity heads.
- `training/trainer.py`: CFM losses, Heun integration, Stage-1 checkpoint loading, path losses, validation metrics.
- `training/config.py`: prior modes, flow/integration settings, NMA and loss weights.
- `datasets/dataset_stage2.py`: apo/holo/ligand triplets, pocket weights, ESM features, optional NMA features.
- `modules/se3.py`: exp/log/compose utilities for rigid transforms.
- `modules/geometry.py`: clash, peptide, contact, and effective-weight helpers.
- `modules/time_embed.py`: time embeddings for `t in [0,1]`.

## State and integration conventions
The backbone state is residue frames, not redundant phi/psi/omega torsions. Frame velocities are body-frame twists and updates should preserve the right-trivialized SE(3) convention used by the integration code. Chi velocities must remain wrap-aware on `S^1`. Heun integration is the default path-level validation mechanism.

Use existing SE(3) helpers for log/exp/compose and avoid mixing left- and right-update conventions. If changing integration, check endpoint consistency and equivariance-sensitive quantities, not just scalar loss values.

## Stage-1 prior use
Stage-1 guidance is safest as local soft information or chi/contact features. Be cautious with rigid endpoint priors: past diagnostics found rigid priors harmful and current deterministic Stage-1 chi guidance weak unless the signal is oracle-quality. If Stage-1 is absent or disabled, relative deltas should degrade to zero features, not broken tensors.

Any new Stage-1 interface should include a no-prior ablation and ideally a confidence-weighted or pocket-local variant. Do not assume a higher Stage-1 raw rotamer score means better Stage-2 guidance.

## Loss and metric expectations
CFM velocity losses alone are insufficient; path smoothness, clash, peptide geometry, contact monotonicity, prior alignment, background stability, and endpoint quality all define whether a path is usable. Watch for NaNs during integration, exploding twist norms, or apparent endpoint gains that worsen clash/contact behavior.

NMA features are optional mobility hints derived from apo structure. If enabled, ensure missing NMA data has a controlled fallback and does not silently change tensor shapes or residue alignment.

## Common edit paths
For model changes, update `models/torsion_flow.py` and the trainer call sites together. For new path losses, wire `modules/geometry.py`, config weights, trainer accumulation, validation logging, and Slurm flags. For dataset fields, update `datasets/dataset_stage2.py`, collate behavior, and any Stage-1 prior loading assumptions.

## Remote experiment pattern
Run small Stage-2 smoke jobs before long bridge-flow experiments. Inspect Slurm logs for Stage-1 checkpoint loading, DDP setup, and integration failures before interpreting metrics.

## Validation
Start with import or compile checks on changed Stage-2 files. For executable validation, use `scripts/slurm/train_stage2_smoke.sh` or a similarly small Slurm job. Inspect both `logs/slurm/` and `logs/stage2/`; a run that starts successfully but fails during path integration usually points to SE(3), mask, or chi wrapping bugs.
