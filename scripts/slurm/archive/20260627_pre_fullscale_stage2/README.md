# Pre-Fullscale Stage-2 Slurm Archive

Date: 2026-06-27

This folder preserves Slurm launchers from older exploratory lines:

- change-prediction RAE;
- chi-head-only probes;
- ligand discriminator/guidance probes;
- typed candidate-energy screens;
- LC-PGBF interaction-prior confirmations;
- one-off geometry and gradient diagnostics.

These launchers are not the supported starting point for new experiments. Use
top-level launchers under `scripts/slurm/` for active work, especially:

- `train_stage2_oracle_motion_ablation_4gpu.sh`
- `export_oracle_motion_features_1gpu.sh`
- `cache_esm_lastk_1gpu.sh`
- `train_stage1v2_posterior_4gpu.sh`
- Stage-2 evaluation and trajectory export wrappers

Archived launchers may contain stale cache paths, outdated defaults, or
historical selection metrics. Inspect and port intentionally before reuse.
