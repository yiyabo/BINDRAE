# BINDRAE Stage-1 Pipeline (Ligand-Conditioned Holo Decoder)

## 1. High-level Overview

Stage-1 in BINDRAE is designed as a **ligand-conditioned protein conformation decoder**:

- **Task**: given a *holo* protein backbone and the bound ligand pose, reconstruct the full-atom *holo* conformation.
- **Inputs**:
  - Frozen ESM-2 sequence embeddings.
  - Holo backbone coordinates (N, Cα, C).
  - Ligand represented as 3D points + 20D type/topology features.
- **Outputs**:
  - 7 torsion angles per residue (φ, ψ, ω, χ1–χ4) as (sin, cos) distributions.
  - A χ1 rotamer classification head (g− / t / g+).
  - Full-atom atom14 coordinates reconstructed via an FK module.

Importantly, Stage-1 **does not** attempt to model `apo → holo` transitions directly. It only learns
"**what a holo conformation should look like, given a ligand**". Continuous `apo → holo` pathways
are delegated to Stage-2.

### 1.1 Optimization history & results

- **Early Stage-1 configuration (baseline)**:
  - 12D ligand type features and a simpler single-shot ligand conditioning scheme.
  - FlashIPA with `z_rank=2` as the geometric backbone.
  - Achieved **χ1 ≈ 71.1%** on CASF-2016 validation, already above the initial 70% target.
- **Final Stage-1 configuration (current pipeline)**:
  - Introduces **multi-layer ligand conditioning**, a **χ1 rotamer auxiliary head**, and **20D ligand
    type/topology features** while keeping the FlashIPA configuration (`z_rank=2`) unchanged.
  - Reaches **χ1 ≈ 75–76%** on CASF-2016 validation.
  - At the same time maintains strong geometric quality with **FAPE ≈ 0.055 Å**, **pocket Cα
    iRMSD ≈ 0.01 Å**, and **clash% ≈ 9.4%**.

This optimization trajectory shows that most of the performance gain came from better ligand
conditioning and χ1-specific supervision, rather than changing the underlying FlashIPA backbone.

---

## 2. Data and Preprocessing

### 2.1 Dataset

- Source: CASF-2016 protein–ligand complexes.
- Only **holo complexes** are used:
  - Protein is in the ligand-bound (holo) state.
  - The ligand pose is the experimentally observed bound pose.
- No explicit apo structures are involved in Stage-1.

### 2.2 `IPABatch` structure

The dataset code (`dataset_ipa.py`) builds an `IPABatch` object with the following key fields:

- **Protein features**:
  - `esm`: per-residue ESM-2 embeddings (frozen encoder used offline / cached).
  - `N`, `Ca`, `C`: backbone atom coordinates.
  - `node_mask`: mask over valid residues.
- **Ligand features**:
  - `lig_points [B, M, 3]`: 3D coordinates of ligand atoms / pseudo-atoms.
  - `lig_types [B, M, 20]`: 20D scalar / topology feature vector per ligand point
    (extended from the original 12D type encoding).
  - `lig_mask`: mask over valid ligand points.
- **Torsion ground truth**:
  - `torsion_angles [B, N, 7]`: φ, ψ, ω, χ1–χ4 in radians.
  - `torsion_mask [B, N, 7]`: which torsions are defined.
- **Pocket weights**:
  - `w_res [B, N]`: per-residue pocket weight (≈ importance of residue in the binding pocket).
- **Metadata**:
  - `pdb_ids`, `sequences`, `n_residues`, etc.

This `IPABatch` is what the model consumes in both training and validation.

---

## 3. Model Architecture

The Stage-1 model (`Stage1Model`) can be decomposed into the following components.

### 3.1 ESM Adapter

- A frozen ESM-2 model (run offline / cached) provides per-residue sequence embeddings.
- An adapter MLP projects these embeddings from ESM dimension (e.g. 1280) to the internal
  single-residue channel dimension `c_s` (384).

### 3.2 EdgeEmbedder

- Uses the Cα coordinates to construct pairwise edge features between residues.
- Encodes geometric information (distance, orientation, RBF features, etc.) into `z` pair
  representations.
- These are fed into the IPA backbone as the pairwise features.

### 3.3 Ligand Featurization

- Each ligand point has:
  - 3D coordinate (x, y, z).
  - A 20D feature vector encoding element, aromaticity, ring membership, ring size buckets,
    degree buckets, heteroatom neighborhood, etc.
- A small MLP (`LigandTokenEmbedding`) maps these to a latent ligand token embedding.

### 3.4 LigandConditioner (Multi-layer ligand injection)

The LigandConditioner is responsible for injecting ligand information into the protein representation.
It has two main mechanisms:

1. **Protein–Ligand Cross-Attention**
   - Protein residues attend over the ligand tokens.
   - Produces a ligand-conditioned update for each residue.

2. **FiLM-style Modulation and Gating**
   - The ligand-conditioned signal is converted into FiLM parameters (scale/shift) applied to
     protein features.
   - A learnable gate (with warmup) controls how strongly the ligand influences the protein
     representation early vs late in training.

Injection schedule:

- After the ESM adapter, before geometric processing, to make the initial residue embeddings
  ligand-aware.
- After each IPA layer (multi-layer conditioning), so that geometric refinement is repeatedly
  influenced by the ligand.

### 3.5 FlashIPA Backbone

- A stack of Invariant Point Attention (IPA) layers (via FlashIPA) updates:
  - Single-residue features `s`.
  - Rigid-body frames for each residue.
- The initial frames come from the provided holo backbone; IPA then refines them under the influence
  of sequence, geometry, and ligand-conditioning.

### 3.6 TorsionHead and χ1 Rotamer Head

- **TorsionHead**:
  - Takes updated single-residue features (and optionally geometry context).
  - Predicts 7 torsion angles per residue as `(sin, cos)` pairs.
- **Chi1RotamerHead**:
  - Predicts a discrete χ1 rotamer class (g− / t / g+) per residue.
  - Acts as an auxiliary head to help the model learn the multi-modal structure of χ1.

### 3.7 OpenFold FK (Full-atom reconstruction)

- The predicted torsions, together with the updated rigid-body frames and residue types, are fed
  into an OpenFold-style FK module.
- FK expands to atom14 coordinates and masks.
- These coordinates are then used for FAPE, distance, and clash losses, and for evaluation metrics
  such as pocket iRMSD and clash percentage.

---

## 4. Training Pipeline

### 4.1 Configuration

`TrainingConfig` controls the Stage-1 run:

- **Data**:
  - `data_dir`, `batch_size`, `num_workers`.
- **Optimization**:
  - `lr`, `weight_decay`, `grad_clip`.
- **Scheduling**:
  - `warmup_steps`, `max_epochs`.
- **Loss weights**:
  - `w_fape`, `w_torsion`, `w_dist`, `w_clash`, `w_rotamer`.
- **Warmup behavior**:
  - `pocket_warmup_steps`: gradually ramp up pocket weights `w_res`.
  - `ligand_gate_warmup_steps`: gradually open ligand-conditioning gates.
- **Validation & checkpointing**:
  - `val_interval`, `early_stop_patience`, `save_top_k`.
- **Logging & device**:
  - `log_dir`, `save_dir`, `log_interval`, `device`, `mixed_precision`.

Stage1Trainer reads this configuration and sets up the model, data loaders, optimizer, and schedulers.

### 4.2 Forward Pass (per batch)

For a single `IPABatch`:

1. Move batch to the configured device.
2. Run the Stage1Model:
   - Apply ESM adapter.
   - Inject ligand information (LigandConditioner) into residue features.
   - Run multiple FlashIPA layers, with interleaved ligand conditioning.
   - Predict torsions and χ1 rotamer classes.
   - Run FK to obtain full-atom atom14 coordinates.
3. Collect outputs:
   - `pred_torsions [B, N, 7, 2]` (sin, cos).
   - `chi1_logits` (if enabled).
   - `atom14_positions`, `atom14_mask`, etc.

### 4.3 Loss Composition

The trainer computes a weighted sum of several loss terms:

- **Torsion loss (`w_torsion`)**
  - Angular loss on predicted vs. ground-truth torsions (all 7 angles, masked by `torsion_mask`).

- **χ1 rotamer cross-entropy (`w_rotamer`)**
  - Convert χ1 ground truth to rotamer bins (g− / t / g+) and train the classification head.
  - Encourages correct multi-modal behavior of χ1.

- **Distance loss (`w_dist`)**
  - L2 distance between predicted and true Cα positions, often with residue weighting.

- **FAPE loss (`w_fape`)**
  - Frame Aligned Point Error using N/Cα/C frame.
  - Encourages local rigid-body consistency of the predicted backbone.

- **Clash penalty (`w_clash`)**
  - Penalizes non-bonded atom pairs that are closer than a clash threshold.
  - Helps maintain physically plausible full-atom geometries.

Warmup strategies:

- **Pocket weighting warmup**:
  - Early in training, pocket weights `w_res` are smoothly ramped from a small baseline (e.g., 0.1)
    to their target values, stabilizing training before fully emphasizing pocket residues.

- **Ligand gate warmup**:
  - The ligand-conditioning gates are gradually increased from near 0 to 1 over a number of steps,
    so the model first learns a reasonable apo-like backbone prior before fully relying on ligand
    information.

### 4.4 Optimization Loop

Per training step:

1. Sample a mini-batch from the train loader.
2. Run the forward pass and compute all loss components.
3. Combine into a total loss using configured weights.
4. Backpropagate gradients.
5. Apply gradient clipping (if enabled).
6. Update model parameters with an Adam-style optimizer and the learning-rate schedule.
7. Log losses and selected metrics every `log_interval` steps.

### 4.5 Validation, Early Stopping, and Checkpointing

At each validation interval:

1. Switch model to eval mode.
2. Iterate over the validation loader without gradient computation.
3. For each batch, compute:
   - Loss components (torsion, rotamer, FAPE, distance, clash).
   - χ1 angle accuracy.
   - χ1 rotamer accuracy.
   - Pocket Cα iRMSD.
   - Clash percentage.
4. Aggregate metrics across the validation set.
5. Log validation metrics and, if improved, save model checkpoints (up to `save_top_k`).
6. An early stopping monitor observes validation metrics (e.g. total val loss) with
   `early_stop_patience` to terminate training when improvements plateau.

---

## 5. Evaluation Metrics (Stage-1)

Stage-1 focuses on the following validation metrics:

- **χ1 angle accuracy**
  - Fraction of residues where the predicted χ1 is within a given angular threshold of the ground
    truth (e.g. < 30° after wrapping).
- **χ1 rotamer accuracy**
  - Accuracy of the discrete χ1 rotamer head.
- **FAPE**
  - Frame Aligned Point Error over backbone atoms, measuring local geometric fidelity.
- **Pocket Cα iRMSD**
  - Interface RMSD over Cα for pocket residues (`w_res > threshold`), capturing how well
    ligand-proximal regions are reconstructed.
- **Clash percentage**
  - Fraction of structures (or atom pairs) with steric clashes beyond a predefined tolerance.

These metrics are printed in compact log lines per validation run to facilitate monitoring and model
selection.

---

## 6. Offline χ1 Error Analysis Pipeline

Beyond online validation, Stage-1 includes an **offline χ1 error analysis** pipeline implemented
as separate scripts. This pipeline is invoked after training finishes, using the best checkpoint.

### 6.1 Step 1: Per-residue χ1 error extraction

`src/stage1/training/chi1_error_analysis.py`:

- Loads the trained Stage-1 model checkpoint.
- Runs the model over the validation set.
- For each residue with a defined χ1:
  - Computes wrapped error `Δχ1` in degrees.
  - Records residue type, pocket weight `w_res`, pocket flag (`w_res > 0.5`), predicted and true
    χ1, and the absolute error.
- Aggregates statistics:
  - Global mean/median/p90/p95 of |Δχ1| and simple histograms.
  - Pocket vs. non-pocket error statistics.
  - Per–amino acid χ1 error (mean, p90, counts).
- Saves a structured file:
  - `logs/stage1/analysis/chi1_errors_val.npz` containing per-residue records.

### 6.2 Step 2: Post-hoc stratified analysis

`src/stage1/training/chi1_error_posthoc.py`:

- Reads `chi1_errors_val.npz`.
- Computes high-error fractions:
  - `frac(|Δχ1| > t1)` and `frac(|Δχ1| > t2)` with default thresholds t1=60°, t2=90°.
  - Pocket fraction among these high-error residues.
- Stratifies by amino acid type (global):
  - For each residue type, reports N, mean, p90, `frac(|Δχ1| > t1)`, `frac(|Δχ1| > t2)`, and the
    fraction of high-error cases that are in pocket.
- Stratifies by amino acid type within pocket only:
  - Same statistics restricted to residues with `is_pocket=True`.

This offline pipeline provides a detailed view of where χ1 errors concentrate (e.g. surface small
polar residues vs. pocket aromatic side chains), which in turn informs future adjustments to loss
weighting, sampling strategies, or Stage-2 supervision.

---

## 7. Role of Stage-1 in the Overall BINDRAE System

Within the broader BINDRAE framework:

- **Stage-1** provides a **high-fidelity ligand-conditioned holo decoder / prior**:
  - Given sequence, backbone, and ligand pose, it reconstructs realistic holo side chains and
    full-atom structures.
- **Stage-2** (planned) will operate on torsion / rigid / latent spaces to learn **continuous
  `apo → holo` pathways** under ligand conditioning.

Thus the Stage-1 pipeline is intentionally focused on *reconstruction quality and ligand-aware
geometry*, leaving long-range conformational dynamics and transport to Stage-2.
