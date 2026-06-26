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

## Slurm rules
Before `sbatch`, ensure `PATH=/data/soft/slurm/24.11.4/bin:$PATH`. Launch from `/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE`, source conda with `source ~/miniconda3/etc/profile.d/conda.sh`, and activate `BINDRAE`. Write outputs to `logs/slurm/`, `logs/stage1/`, or `logs/stage2/` with unique tags.

Typical launchers use one node, one task per node, explicit A100 GPU requests, `PYTHONUNBUFFERED=1`, and `OMP_NUM_THREADS=4`. Keep these explicit so logs are self-describing.

## Editing launchers
Keep launcher names and tags descriptive enough to recover the experiment intent from Slurm logs. When branching from checkpoints, set separate `save_dir` and `log_dir`; do not overwrite anchor runs. Match selection metrics to the scientific question, e.g. ligand-lift metrics for ligand-causality experiments rather than global rotamer accuracy.

If a launcher resumes from a checkpoint, record the source checkpoint path in the script. If it changes selection metric, loss weights, frozen modules, gate behavior, or decoy settings, encode that in the tag or comments.

## Validation
Run `bash -n scripts/slurm/<script>.sh` after shell edits. Run `python -m py_compile scripts/<entry>.py` after CLI edits. For new training flags, verify the parser choices, config fields, trainer usage, and checkpoint resume semantics all agree before submitting a cluster job.

After submission, verify the job reached the expected node/GPU setup and wrote metrics under the intended tag. A successful `sbatch` only proves scheduling, not model startup.

## Login-node warning
Do not run long `python scripts/train_*.py` or `torchrun` commands on the login node. If a command will use GPUs, train for more than a tiny local syntax/import check, or touch the full dataset, submit it with `sbatch`.

Local runs are appropriate for parser help, `py_compile`, `bash -n`, and tiny import checks that do not load the full dataset or allocate GPUs.
