# Stage-1 Posterior/Candidate Diagnostic Decision — 2026-05-10

## Executive summary

The 4-GPU posterior/candidate safe retry (`128821`) was an engineering success but not a sufficient ligand-causal result. The run completed cleanly through epoch 25, with epoch 22 retained as the best checkpoint. Follow-up diagnostics show that the validation split contains many apo→holo χ1 rotamer switches near ligand, so the absence of ligand-lift counts in the training metrics was not caused by data scarcity. Instead, the current posterior/candidate checkpoint has only weak ligand sensitivity and does not show a robust correct-ligand advantage over no-ligand or decoy-ligand baselines.

The next publishable direction should therefore shift from extending this training run to building an explicitly ligand-causal candidate-reranking experiment: train/select on contact/switch subsets and correct-vs-decoy separation, not global rotamer accuracy.

## Experiment lineage

- Training job: `128821` (`s1_postcand4safe`)
- Launcher: `scripts/slurm/train_stage1_posterior_candidate_screen4_safe.sh`
- Run tag: `posterior_candidate_ligcausal_screen4safe_20260509_184220`
- Best checkpoint: `checkpoints/stage1/posterior_candidate_ligcausal_screen4safe_20260509_184220/best_model.pt`
- Best epoch: 22
- Final epoch: 25
- Job outcome: `COMPLETED`, elapsed `07:51:57`, exit code `0:0`

## Training outcome

Epoch 25 did not meet the continuation threshold. It failed to beat the epoch-22 selection metric and did not raise contact-posterior F1 to the pre-declared `~0.17` threshold.

| epoch | selection / chi1 rotamer acc | χ1 acc | ligand-facing χ1 acc | contact posterior F1 | best |
|---:|---:|---:|---:|---:|:--|
| 22 | 0.508915 | 0.385953 | 0.391755 | 0.143613 | yes |
| 23 | 0.508365 | 0.384945 | 0.388876 | 0.153965 | no |
| 24 | 0.508909 | 0.385073 | 0.387395 | 0.159670 | no |
| 25 | 0.508309 | 0.384689 | 0.389240 | 0.162365 | no |

Interpretation: the run is stable and contact posterior improves gradually, but the global and ligand-facing χ1 signals do not justify more training of this exact setup.

## Diagnostics run after training

### Posterior/candidate diagnostic

- Job: `128847`
- Launcher: `scripts/slurm/diagnose_stage1_postcand4safe.sh`
- Output: `logs/stage1_diagnostics/candidate_lift_full_20260510_031658/stage1_prior_diagnostics.json`
- Patched script: `scripts/diagnose_stage1_prior.py` now includes `candidate_rotamer_metrics`, because the original `ligand_lift_*` training metrics only counted `geometry_chi1_*` outputs and missed this candidate-head experiment.

Key candidate/rotamer results:

| subset | n | correct ligand rotamer acc | no ligand | translated away | shuffled ligand |
|---|---:|---:|---:|---:|---:|
| all χ1 | 842,429 | 0.508915 | 0.509219 | 0.507853 | 0.508996 |
| pocket | 32,102 | 0.478381 | 0.475422 | 0.479970 | 0.478039 |
| ligand-facing apo Cα | 13,745 | 0.476173 | 0.473918 | 0.477846 | 0.476828 |
| apo-wrong | 411,112 | 0.391842 | 0.392467 | 0.391550 | 0.391932 |
| switch | 307,957 | 0.302532 | 0.303231 | 0.302088 | 0.302601 |

These differences are small and inconsistent. Correct ligand is not clearly better than no-ligand or decoy-ligand baselines. In some key subsets, no-ligand or translated-away ligand is equal or better.

### Dataset ligand-sensitive audit

- First audit job: `128844`, failed because many validation samples lack `atom14_holo`; this exposed a data-coverage issue rather than invalidating the audit.
- Retry job: `128845`, completed after making `scripts/audit_ligand_sensitive_dataset.py` tolerant of missing atom14 and reporting `atom14_available` coverage.
- Final corrected audit job: `128849`, completed after fixing the `apo_distance_pocket_fraction_of_chi1` denominator.
- Output: `logs/stage1_diagnostics/ligand_sensitive_dataset_audit_fixedfrac_20260510_034123/ligand_sensitive_dataset_audit.json`

Key validation split counts:

| quantity | count / fraction |
|---|---:|
| samples scanned | 2,814 |
| valid residues | 1,000,987 |
| χ1-valid residues | 842,429 |
| apo→holo rotamer switches | 308,969 (`36.7%` of χ1-valid) |
| apo wrong by 20° threshold | 411,112 |
| apo-distance pocket χ1 residues | 32,102 (`3.81%` of χ1-valid; `3.21%` of valid residues) |
| apo Cα ligand-contact χ1 residues | 67,559 (`8.0%` of χ1-valid) |
| apo Cα ligand-contact switches | 26,670 (`3.17%` of χ1-valid) |
| pocket switches | 13,182 (`1.56%` of χ1-valid) |
| atom14 contact switches | 0, because `atom14_available = 0` in this loader view |

Interpretation: the validation split does contain ligand-near switch cases. The zero `val_ligand_lift_*_n` values in the training log were primarily an evaluation wiring mismatch for this head family, not proof that ligand-sensitive cases are absent. However, atom14 coverage for this audit path is currently unusable and should be fixed before making atom-contact claims.

## Scientific interpretation

1. **Do not continue this checkpoint lineage as a claimed ligand-causal posterior.**  
   It is stable, but correct ligand does not outperform no-ligand/decoy consistently on switch, pocket, or ligand-facing subsets.

2. **The dataset has enough candidate cases for a ligand-causal experiment.**  
   The split includes tens of thousands of switch residues and more than 26k apo-Cα ligand-contact switch residues. This is enough to train and evaluate a targeted signal, provided selection and loss focus on these subsets.

3. **Current global selection metric is misaligned with the scientific goal.**  
   `chi1_rotamer_acc` is dominated by base-prior behavior. A publishable ligand-conditioned claim needs correct-vs-decoy separation, rescue/harm on apo-wrong and contact-switch residues, and calibrated local posterior quality.

4. **Contact-posterior behavior is promising but not yet sufficient.**  
   Contact posterior F1 improved from `0.1436` to `0.1624`, and diagnostic contact AP is high inside pocket-like subsets. However, the ligand variants show nearly identical contact AP/AUC, so this also needs stronger decoy-sensitive training or evaluation.

## Recommended next direction

The next experiment should be an explicitly ligand-causal candidate-reranking lane:

1. Add training/selection metrics for candidate-head ligand lift:
   - contact-switch rotamer accuracy;
   - full vs no-ligand / translated-away / batch-shuffled ligand deltas;
   - apo-wrong rescue minus apo-correct harm;
   - non-contact no-harm guardrail.

2. Train/select on ligand-sensitive subsets rather than global χ1:
   - primary selection: correct-vs-decoy lift on `contact_switch` or `ligand_facing_apo_ca ∩ switch`;
   - secondary guardrails: global rotamer not collapsing, non-contact changes bounded, contact posterior calibrated.

3. Repair or regenerate atom14 availability for validation if atom-level contact claims are needed. Until then, report CA-ligand/contact-proxy metrics honestly and avoid claiming atom-contact recovery from this audit path.

4. Keep epoch-22 best only as a diagnostic artifact, not as the Stage-2 prior checkpoint.

## Status

This branch is not ready as a paper result. It is useful because it clarified the failure mode: the problem is not lack of ligand-near switch examples, but insufficiently causal candidate/posterior training and misaligned metric wiring. The publishable route is a targeted ligand-causal reranking experiment with explicit decoy separation and contact-switch selection.
