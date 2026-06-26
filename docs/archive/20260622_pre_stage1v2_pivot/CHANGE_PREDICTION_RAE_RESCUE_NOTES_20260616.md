# Change-Prediction RAE Rescue Notes

Date: 2026-06-16

## Why the previous final failure claim is not valid

The `change_fast_1gpu_20260616_164925` result should be treated as a base-only
baseline, not a ligand-conditioned RAE result.

Implementation issues fixed in this pass:

1. `train_change_prediction_fast.py` trained `DeltaZPredictor(z_apo, mask)` and
   never consumed ligand tokens.
2. `train_change_prediction.py` called `model(batch)` without advancing
   `current_step`, so the ligand conditioner warmup gate stayed at 0.
3. `diagnose_delta_z_predictor.py` also used `current_step=0` and mutated the
   same batch object for no-ligand/scrambled/translated controls.
4. `precompute_latents.py` appended multiple `.npy` arrays into one file, while
   normal `np.load()` reads only the first array.
5. Change-Prediction RAE latent precomputation used a newly initialized frozen
   encoder unless a Stage-1 checkpoint was manually injected.

Therefore the exact zero ligand sensitivity result is not decisive evidence
against the RAE idea.

## New code posture

- Full Change-Prediction RAE training now passes a real `current_step`.
- Validation forces the ligand conditioner open by default via `--eval_gate_step`.
- Diagnostics force `--gate_lambda 1.0` by default and report both:
  - `delta_z` sensitivity
  - ligand-conditioned feature sensitivity
- Fast training now requires `--allow_base_only` and should only be interpreted
  as a base-only latent-change predictor.
- Latent precomputation now writes shard/index files instead of appending `.npy`
  arrays.
- `DeltaZPredictor` lives in a lightweight module independent of `flash_ipa`.

## Recommended next experiment

Run the full RAE path with 2-4 GPUs and a real Stage-1 checkpoint:

```bash
STAGE1_ENCODER_CHECKPOINT=/path/to/stage1/best_model.pt \
sbatch scripts/slurm/train_change_prediction_4gpu.sh
```

Then diagnose:

```bash
CHECKPOINT=/path/to/change_rae/best_model.pt \
GATE_LAMBDA=1.0 \
STAGE1_ENCODER_CHECKPOINT=/path/to/stage1/best_model.pt \
sbatch scripts/slurm/diagnose_delta_z_predictor.sh
```

## Decision criteria

The experiment is promising only if:

1. conditioned-feature sensitivity is clearly nonzero;
2. `delta_z` sensitivity is nonzero against no-ligand, translated, scrambled,
   and batch-shuffled controls;
3. ligand sensitivity concentrates on contact/switch residues rather than all
   residues uniformly;
4. downstream rotamer or Stage-2 metrics show lift over base-only and decoy
   controls.

If feature sensitivity is nonzero but `delta_z` sensitivity remains zero, the
ligand conditioner is alive and the predictor/objective is collapsing.

If feature sensitivity is also zero with `gate_lambda=1.0`, the input
conditioning path or loaded checkpoint is the immediate bottleneck.
