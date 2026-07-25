# Path-4 Pre-Gate-0 Archive

`train_stage2_physical_residual_2gpu.sh` amortizes a canonical physical-normal
Path-4 teacher into learned residual heads. It was removed from the active
Slurm surface on 2026-07-25 because Gate-0 has not established that the
non-learned OpenMM optimizer transfers to the independent relaxed/force
outcomes.

Do not launch this distillation route, alter its thresholds, or use it to
restart direct force-to-normal-residual regression. Retain it as a
reproducibility record only. Reopening requires a documented Gate-0 decision
that supersedes `docs/PATH4_OPTIMIZER_GATE0_20260720.md`.
