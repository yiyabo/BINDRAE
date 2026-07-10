# Stage-1-v2 Posterior TODO

Date: 2026-06-22

## Goal

Implement the accepted **teacher-distilled student** plan:

```text
external teacher / holo_truth oracle
-> teacher posterior labels
-> compact Stage1PosteriorV2 student
-> Stage-2 scalar posterior and later latent guidance
```

The student is the default BINDRAE Stage-1-v2 model. External teachers are
offline label/latent sources and upper-bound diagnostics.

## Phase 0: Label Asset

- [x] Implement `scripts/build_teacher_posterior_labels.py`.
- [x] Add `scripts/slurm/build_teacher_posterior_labels_1gpu.sh`.
- [x] Run remote smoke on validation subset.
- [x] Export active validation labels with `teacher_source=holo_truth`
  (`stage2_lc_pgbf_val_512`).
- [x] Export active training labels with `teacher_source=holo_truth`
  (`stage2_lc_pgbf_train_4096`).
- [x] Summarize label statistics: valid, pocket, active, approach, release,
  formed contact, released contact, stable contact.

## Phase 1: Schema And Dataset

- [x] Create `src/stage1/posterior_v2/schema.py`.
- [x] Move schema version and required field list out of the exporter script.
- [x] Add strict `.npz` validator with hard failures for missing fields,
  length mismatch, stale schema, and non-finite distances.
- [x] Create `src/stage1/posterior_v2/dataset.py`.
- [x] Load apo/ligand/ESM inputs plus teacher posterior `.npz` labels.
- [x] Add collate function preserving residue masks and teacher masks.
- [x] Add a tiny import/schema smoke in the remote `BINDRAE` environment.

## Phase 2: Student Model

- [x] Create `src/stage1/posterior_v2/model.py`.
- [x] Reuse existing `ESMAdapter`.
- [x] Reuse existing `LigandConditioner` or a lightweight variant.
- [x] Add compact residue trunk.
- [x] Add heads for:
  - contact probability;
  - teacher distance or signed distance delta;
  - approach probability;
  - release probability;
  - switch/active probability;
  - confidence.
- [x] Keep optional `z_post` latent head behind a config flag.

## Phase 3: Training

- [x] Create `src/stage1/posterior_v2/losses.py`.
- [x] Create `src/stage1/posterior_v2/trainer.py`.
- [x] Create `scripts/train_stage1v2_posterior.py`.
- [x] Add first-pass metrics:
  - contact AUROC/AUPRC or balanced accuracy;
  - active/switch accuracy;
  - approach/release direction accuracy;
  - distance/delta MAE on pocket and active residues;
- [x] Add counterfactual metrics:
  - correct ligand vs zero/shuffled ligand separation.
- [x] Select checkpoints by an active-residue posterior metric, not global loss.
- [x] Add `scripts/slurm/train_stage1v2_posterior_4gpu.sh`.
- [x] Run remote smoke (`135568`) and train512/val512 counterfactual run
  (`135597`).
- [x] Run train4096/val512 class-balanced 20-epoch run (`135622`);
  training early-stopped at epoch 14, with best checkpoint at epoch 6 by
  `posterior_selection_score`.

## Phase 3.5: Checkpoint Audit And Posterior Cache

- [x] Add `src/stage1/posterior_v2/inference.py` for checkpoint loading and
  shared real/counterfactual inference.
- [x] Add `scripts/audit_stage1v2_posterior.py`.
- [x] Add `scripts/export_stage1v2_posterior_cache.py`.
- [x] Add Slurm wrappers:
  - `scripts/slurm/audit_stage1v2_posterior_1gpu.sh`;
  - `scripts/slurm/export_stage1v2_posterior_cache_1gpu.sh`.
- [x] Run remote smoke for best-checkpoint audit and posterior cache export
  (`135626`, `135627`).
- [x] Run full val512 best-checkpoint audit (`135628`):
  `logs/stage1v2_audits/stage1v2_best_val512_audit_20260622_231033.json`.
- [x] Export full val512 posterior cache (`135630`):
  `logs/stage1v2_student_posteriors/stage1v2_best_val512_cache_20260622_231033/`.
- [x] Export full train4096 posterior cache (`135633`; `135629`/`135631`
  were canceled before producing valid cache files because of subset-path
  handling):
  `logs/stage1v2_student_posteriors/stage1v2_best_train4096_cache_20260622_231423/`.

## Phase 4: Stage-2 Integration

- [x] Extend Stage-2 config from scalar `interaction_prior` to multi-channel
  `stage1v2_posterior_features`.
- [x] Add student posterior cache loading path.
- [x] Add inference modes:
  - `none`;
  - `zero`;
  - `student`;
  - `student_shuffled`;
  - `oracle_holo_truth`;
  - `external_teacher_cached`.
- [x] Run 1-GPU Stage-2 smoke for scalar posterior features:
  `student` (`135638`), `zero` (`135641`), `student_shuffled` (`135642`),
  and `oracle_holo_truth` (`135643`).
- [x] Add multi-GPU Stage-2 ablation launcher:
  `scripts/slurm/train_stage2_stage1v2_posterior_ablation_4gpu.sh`.
- [x] Train Stage-2 with scalar posterior features first:
  - head-only scalar input: `135645` none, `135646` zero, `135647`
    student, `135658` student_shuffled, `135659` oracle_holo_truth.
  - feature-scale stress test: `135661` student scale4, `135662`
    student_shuffled scale4, `135663` oracle_holo_truth scale4.
  - trunk-injected scalar input: `135666` zero, `135667` student,
    `135668` student_shuffled, `135669` oracle_holo_truth.
- [x] Add bias-free posterior/prior residue projection into the Stage-2
  residue trunk, while preserving zero-feature controls.
- [x] Add posterior-weighted Stage-2 objectives and `stage1v2_guidance` contact
  loss so scalar posterior guidance affects the training target, not only model
  inputs.
- [x] Run short objective-level oracle controls:
  - `135670` oracle smoke with `stage1v2_guidance` active;
  - `135671` zero, `135672` student, `135673` student_shuffled, `135674`
    oracle_holo_truth with `contact_active` loss reweighting and weak guidance;
  - `135675` zero and `135676` oracle_holo_truth with stronger contact
    guidance only.
- [ ] Reassess the Stage-1-v2 -> Stage-2 interface before full-scale training.
  Current scalar contact/posterior guidance separates shuffled from matched
  posteriors in the loss, but does not yet produce reliable path/contact gains.
- [ ] Evaluate transition paths with no-prior, zero, shuffled, oracle, and
  student controls.

## Phase 4.5: OracleMotion Upper-Bound

Motivation: scalar contact/posterior features are not enough to establish that
Stage-2 can use Stage-1 information as a movement prior. The next upper-bound
is privileged per-residue apo->holo motion conditioning.

- [x] Add `scripts/export_oracle_motion_features.py`.
- [x] Add `scripts/slurm/export_oracle_motion_features_1gpu.sh`.
- [x] Export schema `bindrae_oracle_motion_v1` with:
  - local frame translation `delta_trans_local`;
  - rotation log `delta_rot_log`;
  - Stage-2-compatible `delta_frame_log`;
  - raw `delta_chi` plus `delta_chi_masked`;
  - sin/cos chi delta features and chi masks;
  - motion/contact/distance masks and normalized feature matrix.
- [x] Add direct oracle-apply audit:
  `F_oracle = F_apo compose DeltaF`, `chi_oracle = chi_apo + delta_chi`.
- [x] Run 16-sample remote smoke (`135678`):
  `logs/stage2_oracle_motion/oracle_motion_val16_rawchi_direct_20260623_020005/`.
  Direct apply reached numerical ceiling:
  frame translation `3.7e-6 A`, frame rotation `2.4e-8 rad`,
  chi `5.5e-8 rad`, sidechain atom14 `3.9e-6 A`,
  ligand min-distance `2.7e-5 A`, contact F1 `1.0`.
- [x] Fix direct-apply chi handling: use raw `delta_chi` for the oracle
  baseline; retain `delta_chi_masked` and `chi_mask` for model features.
- [x] Finish val512 direct oracle-apply audit (`135679`) on
  `ablation_subsets/stage2_lc_pgbf_val_512_seed20260618.txt`:
  `logs/stage2_oracle_motion/oracle_motion_val512_direct_20260623_020114/`.
  Direct apply reached ceiling on 512 samples:
  frame translation `4.2e-6 A`, frame rotation `2.4e-8 rad`,
  chi `4.8e-8 rad`, sidechain atom14 `4.4e-6 A`,
  ligand min-distance `1.3e-5 A`, contact F1 `1.0`.
- [x] Add Stage-2 loading mode for oracle-motion feature caches:
  `zero`, `oracle_motion`, `oracle_motion_residue_shuffled`, and
  `oracle_motion_sample_shuffled`.
- [x] Add strict Stage-2 feature-cache validation before consuming
  Stage-1-v2/OracleMotion caches:
  - `sample_id` and `n_residues` must match;
  - `aatype` must match for matched/residue-shuffled modes;
  - cache masks cannot mark Stage-2-invalid residues as valid;
  - cross-sample shuffled controls require same-length donor samples instead of
    silent pad/truncate alignment.
- [x] Add Stage-2 audit metrics for this track:
  - unweighted endpoint rigid/chi metrics (`end_rigid_uw`, `end_chi_uw`);
  - sidechain-contact endpoint scores using the same 4.5 A / 0.75 soft-contact
    convention as posterior guidance;
  - formed-contact sidechain recall.
- [x] Add `scripts/slurm/train_stage2_oracle_motion_smoke_1gpu.sh`.
- [x] Add `scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh`.
- [x] Train Stage-2 matched oracle-motion vs zero/residue-shuffled controls:
  - zero `135684`:
    `logs/stage2/stage2_omotion_zero_train512_train512_val512_e3_bs2x4_20260623_025241/`;
  - matched `135686`:
    `logs/stage2/stage2_omotion_matched_train512_train512_val512_e3_bs2x4_20260623_025257/`;
  - residue-shuffled `135685`:
    `logs/stage2/stage2_omotion_res_shuffle_train512_train512_val512_e3_bs2x4_20260623_025257/`.
- [x] Evaluate generated endpoint/path metrics on moving, pocket, contact,
  formed-contact, and released-contact subsets.
- [x] Add oracle-motion-aware trajectory export and reliability benchmark
  entrypoints:
  - `scripts/generate_stage2_trajectories.py` now restores Stage-2
    checkpoints with multi-channel Stage-1-v2/OracleMotion features;
  - `scripts/evaluate_stage2_trajectory_reliability.py` compares the learned
    path against cubic SE(3)+chi interpolation on endpoint, contact,
    smoothness, clash, and peptide-geometry metrics.
- [x] Run an initial OracleMotion trajectory reliability smoke on val64:
  `model` vs `cubic_ref`.
  - Old-clamp matched OracleMotion improves over zero/residue-shuffled learned
    paths on active endpoint/path/contact movement metrics.
  - The endpoint-aware cubic reference reaches the holo endpoint/contact by
    construction but has much worse peptide-geometry loss.
  - Old learned integration clamps (`chi=1.0`, `rot=0.1`, `trans=0.2`) likely
    suppress backbone-frame motion; relaxed-clamp evaluation is now submitted
    as the next audit lane.
- [x] Complete relaxed-clamp OracleMotion trajectory reliability controls:
  matched, zero, and residue-shuffled on the same val64 subset.
  - Matched formed-contact recall improves from `0.043` to `0.085`.
  - Active endpoint distance barely moves (`1.703 A` -> `1.699 A`).
  - Zero and residue-shuffled are effectively unchanged.
  - Interpretation: old checkpoints likely learned a conservative vector field
    under old integration clamps; retraining is needed to test the relaxed
    path upper bound.
- [x] Retrain train512/val512 OracleMotion matched/zero/residue-shuffled
  controls with relaxed integration clamps.
  - Superseded the initial 4-GPU submissions (`135705`/`135706`/`135707`) due
    to fragmented cluster availability.
  - Resubmitted as 3-GPU jobs: `135709` zero, `135710` matched, `135711`
    residue-shuffled with `integration_chi_clip=5.0`,
    `integration_rot_clip=1.0`, `integration_trans_clip=5.0`.
  - Final epoch result strongly favors matched OracleMotion:
    `val_total` matched `12.3496`, zero `13.1535`, residue-shuffled
    `13.1576`; `val_end` matched `177.2053`, zero `186.5210`,
    residue-shuffled `185.7162`; `val_end_chi_uw` matched `1.8971`, zero
    `2.3176`, residue-shuffled `2.3172`; formed-contact sidechain recall
    matched `0.1719`, zero `0.0334`, residue-shuffled `0.0337`.
- [x] Run trajectory reliability benchmark for relaxed 3-GPU checkpoints on
  val64 with `n_integration_steps=3`.
  - This is inference/evaluation, not training: it loads the trained Stage-2
    checkpoints and uses OracleMotion caches as a privileged stand-in for the
    future Stage-1 motion-posterior student.
  - Slurm jobs (`136887`/`136888`/`136889`, then `136902`/`136903`/`136904`)
    stayed pending under fragmented GPU resources and were cancelled.
  - Direct `gpu33:GPU2` run completed:
    `logs/stage2/trajectory_reliability/omotion_relaxed_zero_3gpu_val64_reliability_steps3_direct_gpu33_gpu2_20260623.json`,
    `logs/stage2/trajectory_reliability/omotion_relaxed_matched_3gpu_val64_reliability_steps3_direct_gpu33_gpu2_20260623.json`,
    `logs/stage2/trajectory_reliability/omotion_relaxed_res_shuffle_3gpu_val64_reliability_steps3_direct_gpu33_gpu2_20260623.json`.
  - Matched OracleMotion beats zero/residue-shuffled on active endpoint
    distance, active path MAE, direction accuracy, chi error, and
    formed-contact recall, while learned paths keep peptide loss far below
    endpoint-aware cubic interpolation.
- [x] Run larger relaxed-checkpoint reliability benchmark on val512.
  - Direct `gpu33:GPU2` aggregate audit completed with
    `n_integration_steps=6`:
    `logs/manual/stage2_reliability_val512_steps6_direct_gpu33_gpu2_20260623_200344.out`.
  - Direct `gpu33:GPU7` aggregate audit completed with
    `n_integration_steps=12`:
    `logs/manual/stage2_reliability_val512_steps12_direct_gpu33_gpu7_20260624.out`.
  - `steps=12` outputs:
    `logs/stage2/trajectory_reliability/omotion_relaxed_zero_3gpu_val512_reliability_steps12_direct_gpu33_gpu7_20260624.json`,
    `logs/stage2/trajectory_reliability/omotion_relaxed_matched_3gpu_val512_reliability_steps12_direct_gpu33_gpu7_20260624.json`,
    `logs/stage2/trajectory_reliability/omotion_relaxed_res_shuffle_3gpu_val512_reliability_steps12_direct_gpu33_gpu7_20260624.json`.
  - `steps=12` confirms stable matched-over-shuffled lift: active endpoint
    distance matched `1.840 A` versus zero `2.045 A` and residue-shuffled
    `2.045 A`; active path MAE matched `0.970 A` versus `1.057 A`/`1.056 A`;
    direction accuracy matched `0.641` versus `0.511`/`0.510`; active chi
    error matched `1.324 rad` versus `1.536`/`1.536`; formed-contact recall
    matched `0.182` versus `0.0027`/`0.0027`.
  - Matched paths move more than zero/residue-shuffled and therefore have
    higher peptide loss (`0.017` versus about `0.0014`), but this remains far
    below endpoint-aware cubic interpolation (`17.44`); endpoint/contact
    metrics should be interpreted together with path geometry.
- [ ] Export a small set of OracleMotion-conditioned trajectory PDB/NPZ files
  for visual inspection when paper figures or qualitative debugging require it.
  This is paused by default after the 2026-06-25 baseline freeze.
- [x] Initial OracleMotion-UB decision: matched oracle motion beats zero and
  residue-shuffled controls after 3 epochs on train512/val512. Final epoch:
  - `val_total`: matched `12.6603`, zero `13.1469`,
    residue-shuffled `13.1525`;
  - `val_end`: matched `183.1509`, zero `186.5174`,
    residue-shuffled `186.3262`;
  - `val_end_chi_uw`: matched `2.0643`, zero `2.3224`,
    residue-shuffled `2.3224`;
  - sidechain holo-gap: matched `0.0620`, zero `0.0642`,
    residue-shuffled `0.0641`;
  - formed-contact sidechain recall: matched `0.1390`, zero `0.0204`,
    residue-shuffled `0.0247`.
- [ ] Build a Stage-1 motion-posterior student target from OracleMotion features
  after the Stage-2 representation-enhancement track is tested.
- [ ] Add a sample-shuffled OracleMotion control on an eligible same-length subset,
  or export a larger train cache to reduce singleton-length samples.
- [x] Scale OracleMotion Stage-2 training beyond train512.
  - Exported direct train4096 OracleMotion cache:
    `logs/stage2_oracle_motion/oracle_motion_train4096_direct_20260624_scaleup/`.
  - Completed train4096/e10 matched/zero/residue-shuffled controls and
    val512 reliability; matched strongly beat zero and residue-shuffled.
  - Exported train12000 OracleMotion cache:
    `logs/stage2_oracle_motion/oracle_motion_train12000_direct_20260624_scaleup/`.
  - Completed train12000/e10 matched/zero/residue-shuffled controls.
    Final epoch matched: `val_total=3.7005`, `val_end=99.8112`,
    formed-contact recall `0.6469`, direction accuracy `0.8926`.
    Zero/residue-shuffled remained near `val_total=13`.
  - Completed train12000/e10 best-checkpoint reliability on val512,
    `n_integration_steps=12`. Matched: active endpoint `0.7763 A`,
    path MAE `0.6287 A`, direction accuracy `0.9467`, chi error
    `0.2657 rad`, formed-contact recall `0.8859`.
- [x] Freeze the current OracleMotion upper-bound baseline:
  `docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md`.

Decision after the 2026-06-25 freeze: do not run more reliability step-count
sweeps, sample-level ranking, or visualization by default. They remain
available later for paper evidence, error analysis, and qualitative figures.

## Phase 5: RAEv2/REPA-Style Stage-2 Enhancement

Reference:

```text
docs/RAEV2_REPA_STAGE2_ENHANCEMENT_PLAN_20260625.md
```

- [x] Add backward-compatible ESM last-K fusion framework. Existing
  single-layer ESM caches and default Stage-2 checkpoints continue to use the
  old `ESMAdapter` path; `--esm_fusion_enabled` switches to
  `ESMLayerFusionAdapter` for `[B, N, K, D]` inputs. Remote smoke passed in the
  `BINDRAE` env on 2026-06-25 for 3D fallback, 4D fusion, layer-count mismatch,
  model instantiation, and CLI visibility. Last-K ESM cache export is still a
  later data-prep task.
- [x] Add Stage-2 dataset/cache support for real last-K ESM features.
  - `scripts/cache_esm2.py` and `scripts/cache_ahojdb_esm2.py` now accept
    `--last-k-layers`; when `K > 1`, they save `per_residue_layers`
    `[N, K, D]` plus `esm_layer_indices` while preserving legacy
    `per_residue` `[N, D]`.
  - `src/stage2/datasets/dataset_stage2.py` can load single-layer or last-K
    ESM caches. It returns `[B, N, D]` by default and `[B, N, K, D]` when
    fusion requests `K > 1`.
  - Missing or too-short `per_residue_layers` now hard-fails, preventing fake
    last-K experiments on single-layer caches.
  - Remote smoke passed on 2026-06-25 for synthetic `esm.pt` loading,
    too-short-layer failure, and 4D collate.
  - Existing real caches are still single-layer; before running an ESM fusion
    experiment, regenerate the selected train/val cache with
    `--last-k-layers 7`.
- [ ] Add optional REPA-style hidden-state alignment from Stage-2 `s_geo` to an
  OracleMotion/future motion-posterior target.
- [ ] Keep ESM fusion and REPA disabled by default behind config and CLI flags.
- [ ] Add a new 2-3 GPU enhancement Slurm wrapper instead of overloading the
  frozen baseline wrappers.
- [ ] Run a small controlled matrix only after smoke tests pass:
  zero baseline, oracle matched baseline, oracle matched + ESM fusion, oracle
  matched + REPA, oracle matched + both, and residue-shuffled + both.

## Stop Conditions

Pause the representation-enhancement line if matched enhanced Stage-2 does not
improve active path/contact metrics over the frozen OracleMotion baseline, or
if gains only appear in endpoint metrics while path geometry, clash, or
background stability worsens. Pause the later learned Stage-1 motion-posterior
line if student-guided Stage-2 cannot beat zero and shuffled-student controls.
