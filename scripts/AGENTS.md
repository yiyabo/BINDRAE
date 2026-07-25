# Scripts Agent Guide

## Role
This directory contains Python entry points, diagnostics, preprocessing helpers, and Slurm launchers. Treat `scripts/*.py` as runnable local/remote program entry points and `scripts/slurm/*.sh` as cluster submission wrappers.

Scripts are part of the experiment API: changing a default, parser choice, or tag pattern can change reproducibility even when model code is untouched.

## Command families
- `train_stage1.py`, `train_stage2.py`: training CLIs; add new flags here only after wiring config/trainer support.
- `diagnose_stage1_prior.py` and related diagnostics: evaluate prior quality, rotamer/contact behavior, and ablations.
- `prepare_*.py`, `setup_server_data.sh`: data preparation and ligand/torsion/embedding preprocessing.
- `validate_*.py`, `test_*.py`, `test_*.sh`: focused validation utilities.
- `slurm/*.sh`: the only safe way to launch training or heavy diagnostics on the cluster.

Prefer adding experiment-specific launchers over mutating a known-good launcher in place when the run is scientifically distinct.

## Current active surface
For Stage-2 full-scale preparation, start from:

- `train_stage2.py`
- `export_oracle_motion_features.py`
- `slurm/train_stage2_oracle_motion_ablation_4gpu.sh`
- `slurm/export_oracle_motion_features_1gpu.sh`
- Stage-2 evaluation/export wrappers in `scripts/` and `scripts/slurm/`

For Stage-1-v2 posterior work, start from:

- `train_stage1v2_posterior.py`
- `build_teacher_posterior_labels.py`
- `export_stage1v2_posterior_cache.py`
- `audit_stage1v2_posterior.py`

Archived files under `scripts/archive/` and `scripts/slurm/archive/` are
reproducibility records. Do not use them as templates for new experiments unless
a current runbook explicitly revives that lane.

## Slurm rules
Before `sbatch`, ensure `PATH=/data/soft/slurm/24.11.4/bin:$PATH`. Launch from `/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE`, source conda with `source ~/miniconda3/etc/profile.d/conda.sh`, and activate `BINDRAE`. Write outputs to `logs/slurm/`, `logs/stage1/`, or `logs/stage2/` with unique tags.

Typical launchers use one node, one task per node, explicit A100 GPU requests, `PYTHONUNBUFFERED=1`, and `OMP_NUM_THREADS=4`. Keep these explicit so logs are self-describing.

Before a long 60k/full-scale Stage-2 run, submit a `PRECHECK_ONLY=1` job through
the same launcher and cache settings. For 4-6 GPU jobs, keep Slurm `--gres` and
`NPROC_PER_NODE` consistent; changing only one of them is an invalid setup.

## Agent remote launch hygiene
Full policy lives in the repo-root `AGENTS.md` section **Remote ops / anti false-positive cyber policy**. Scripts work must obey it.

- **Edit launchers locally**, validate with `bash -n`, then batch the changed files into a small number of remote sync/submit operations. Do not use the chat agent as a Slurm watchdog.
- Count `ssh`, `scp`, and `rsync` together; use at most 10 such calls per user
  turn. Continue batching work even when the full allowance is available.
- An occasional `squeue -u "$USER"` overview is allowed when the active JobID is unknown. Once known, prefer job-scoped `squeue`/`sacct` and matching `logs/slurm/...` reads.
- After submit: capture the JobID and use a bounded initial status/log check. Waiting jobs do not need repeated interactive monitoring.
- **Forbidden from agent loops**: `sleep`+`squeue` polling; `scontrol show nodes`; repeated `scancel`/`sbatch` churn; per-edit `rsync`/`scp`.
- Put retries, dependency chains, and multi-step pipelines **inside** `scripts/slurm/*.sh` (or a single remote bash script), not as many interactive SSH round-trips.
- Prefer host aliases from SSH config; do not default to `StrictHostKeyChecking=no` or raw user@IP:port in every command.
- If a session hits `cyber_policy` / gateway `403` session block: stop remote tools, do not fork the thread, continue in a new session with local-first work.

## Editing launchers
Keep launcher names and tags descriptive enough to recover the experiment intent from Slurm logs. When branching from checkpoints, set separate `save_dir` and `log_dir`; do not overwrite anchor runs. Match selection metrics to the scientific question, e.g. ligand-lift metrics for ligand-causality experiments rather than global rotamer accuracy.

If a launcher resumes from a checkpoint, record the source checkpoint path in the script. If it changes selection metric, loss weights, frozen modules, gate behavior, or decoy settings, encode that in the tag or comments.

## Validation
Run `bash -n scripts/slurm/<script>.sh` after shell edits. Run `python -m py_compile scripts/<entry>.py` after CLI edits. For new training flags, verify the parser choices, config fields, trainer usage, and checkpoint resume semantics all agree before submitting a cluster job.

After submission, a successful `sbatch` only proves scheduling. Prefer reading the job's `.out/.err` once it has clearly started (or when the user pastes status) over agent-side queue polling. Confirm node/GPU and metrics under the intended tag when logs exist—not via continuous `squeue` loops.

For REPA runs, use `val_total_no_repa` plus endpoint/contact/path metrics for
cross-run comparison. `val_total` includes auxiliary REPA loss, and `val_repa`
only measures target fitting.

## Login-node warning
Do not run long `python scripts/train_*.py` or `torchrun` commands on the login node. If a command will use GPUs, train for more than a tiny local syntax/import check, or touch the full dataset, submit it with `sbatch`.

Local runs are appropriate for parser help, `py_compile`, `bash -n`, and tiny import checks that do not load the full dataset or allocate GPUs.
