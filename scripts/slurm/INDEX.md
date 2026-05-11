# Slurm Launcher Index

Top-level launchers are the currently supported entrypoints. Superseded
experiment variants are archived under `scripts/slurm/archive/`.

## Stage-1 Diagnostics

- `audit_ligand_sensitive_dataset.sh` - ligand-sensitive dataset and contact-label audit.
- `diagnose_stage1_prior.sh` - Stage-1 prior diagnostics.
- `diagnose_stage1_postcand4safe.sh` - posterior/candidate diagnostic with ligand controls.

## Stage-1 Training

- `train_stage1_geom_baseprior.sh` - base geometry prior anchor.
- `train_stage1_posterior_candidate_screen4_safe.sh` - retained posterior/candidate screening baseline.
- `train_stage1_typed_candidate_energy_smoke.sh` - fast typed-energy smoke test.
- `train_stage1_typed_candidate_energy_screen8.sh` - active 8-GPU typed-energy screening run.

## Stage-2

- `train_stage2_smoke.sh` - Stage-2 smoke.
- `train_stage2_ddp.sh` - Stage-2 DDP training.

## Archive

- `archive/legacy_stage1/` - older Stage-1 launchers kept for reproducibility, not active use.
