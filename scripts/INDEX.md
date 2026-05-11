# Scripts Index

Top-level `scripts/` keeps active entrypoints and data-prep utilities. Older
helpers and one-off diagnostics are archived under `scripts/archive/`.

## Active Entry Points

- `train_stage1.py` - main Stage-1 training CLI.
- `train_stage2.py` - main Stage-2 training CLI.
- `validate_triplets_data.py` - validates Stage-1 triplet datasets.
- `audit_ligand_sensitive_dataset.py` - ligand-sensitive dataset audit.
- `diagnose_stage1_prior.py` - Stage-1 diagnostic entrypoint.
- `summarize_stage1_ligand_causality_validation.py` - validation report summarizer.

## Data Preparation

- `prepare_casf2016.py`
- `prepare_ligands.py`
- `extract_pockets.py`
- `extract_torsions.py`
- `cache_esm2.py`
- `split_dataset.py`
- `verify_casf2016.py`
- `verify_ligand_consistency.py`

## Alternate Dataset Line

- `download_ahojdb_pdbs.py`
- `extract_ahojdb_torsions.py`
- `prepare_ahojdb_triplets.py`
- `cache_ahojdb_esm2.py`

## Archive

- `archive/legacy_tools/` - older setup helpers and test wrappers.
- `archive/diagnostics/` - one-off audits and lineage/debug scripts.
- `slurm/` - current cluster launchers; see `scripts/slurm/INDEX.md`.
