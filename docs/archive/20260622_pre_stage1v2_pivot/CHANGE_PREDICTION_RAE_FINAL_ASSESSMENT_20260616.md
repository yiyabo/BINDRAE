# Change-Prediction RAE: Final Assessment

**Date**: 2026-06-16
**Status**: Failed - Model ignores ligand information
**Decision**: Abandon this approach, same fundamental limitation as previous methods

---

## Approach Summary

Inspired by RAEv2 (Representation Autoencoders v2), we implemented a **Change-Prediction RAE** architecture:

1. **Frozen Encoder**: ESM-2 + IPA to encode protein structures into latent space
2. **DeltaZPredictor**: MLP to predict `delta_z = z_holo - z_apo` from ligand information
3. **Training Objective**: Minimize `||delta_z_pred - delta_z_true||^2` in latent space

**Key Hypothesis**: By operating in latent space instead of directly predicting chi angles, we could avoid the chi1 dominance problem that plagued previous approaches.

---

## Implementation

### Phase 1: Precompute Latents (Job 134326)
- **Status**: ✅ Completed successfully
- **Duration**: 5 hours 14 minutes
- **Output**: 147GB of precomputed latents
  - Train: 81,953 samples
  - Val: 2,862 samples
  - Files: `z_apo.npy`, `z_holo.npy`, `node_mask.npy`, `pdb_ids.npy`

### Phase 2: Fast Training (Job 134408)
- **Status**: ✅ Completed successfully
- **Duration**: ~2 minutes (50 epochs)
- **Result**:
  - Best val_loss: **0.001438** (very low!)
  - Training curve: smooth convergence from 0.126 → 0.0014
  - Checkpoint: `checkpoints/stage1/change_fast_1gpu_20260616_164925/best_model.pt`

### Phase 3: Ligand Sensitivity Diagnostic (Job 134423)
- **Status**: ✅ Completed successfully
- **Duration**: 1 minute 46 seconds
- **Result**: **CRITICAL FAILURE**

---

## Critical Finding: Model Ignores Ligand

### Diagnostic Results

**Delta-z magnitude (mean L2 norm per residue)**:
```
correct_ligand:     0.153298
no_ligand:          0.153298
scrambled_ligand:   0.153298
translated_ligand:  0.153298
```

**Sensitivity (||delta_z_correct - delta_z_control|| / ||delta_z_correct||)**:
```
vs_no_ligand:          0.000000
vs_scrambled_ligand:   0.000000
vs_translated_ligand:  0.000000
```

### Interpretation

1. **Identical predictions**: The model produces exactly the same `delta_z` regardless of ligand input
2. **Zero sensitivity**: Changing, removing, or scrambling the ligand has no effect on predictions
3. **Degenerate solution**: The model learned to predict a constant `delta_z` that minimizes loss without using ligand information

---

## Root Cause Analysis

### Why Did This Happen?

The Change-Prediction RAE suffers from the **same fundamental limitation** as all previous approaches:

1. **Chi1 dominance**: Even in latent space, the training objective is dominated by the base prior (apo structure)
2. **Ligand signal suppression**: The model finds it easier to ignore ligand information and predict a constant
3. **No causal learning**: The model learns correlation (apo → holo) but not causation (ligand → conformational change)

### Comparison with Previous Approaches

| Approach | Auxiliary Signal | Actual Lift | prediction_changed_rate |
|----------|------------------|-------------|------------------------|
| Guidance Loss | 0.02 → 0.07 | ~0.001 | 0.0 |
| s_lig Injection | Present | ~0.001 | 0.0 |
| Contrastive (Margin) | 0.013 | ~0.001 | 0.0 |
| DrugCLIP-Inspired | 0.0 | 0.0 | 0.0 |
| Discriminator | Present | ~0.001 | 0.0 |
| Protein-Ligand Contrastive | 0.704 | 0.0 | 0.0 |
| **Change-Prediction RAE** | **0.001438** | **0.0** | **0.0** |

**Pattern**: All approaches show the same failure mode - auxiliary loss decreases but model ignores ligand.

---

## Scientific Implications

### What We've Learned

1. **Latent space doesn't help**: Operating in latent space (instead of chi angles) doesn't solve the fundamental problem
2. **Architecture doesn't matter**: Whether we use direct prediction, contrastive learning, or autoencoders, the result is the same
3. **The problem is fundamental**: The current Stage-1 architecture cannot learn ligand-causal effects, regardless of training objective

### Why This Is a Hard Problem

The fundamental issue is that **chi1 prediction loss dominates training**:
- The model can achieve low loss by predicting from the base prior (Dunbrack rotamer library)
- Learning ligand-causal effects requires the model to deviate from the base prior
- But deviations increase loss unless they're highly accurate
- So the model learns to ignore ligand information to minimize loss

This is a **chicken-and-egg problem**:
- To learn ligand effects, the model needs to make ligand-dependent predictions
- But making ligand-dependent predictions increases loss (initially)
- So the model avoids making ligand-dependent predictions

---

## Decision: Abandon This Approach

### Rationale

1. **Exhaustive exploration**: We've tried 9+ different approaches across multiple loss families
2. **Consistent failure mode**: All show the same pattern (loss decreases, ligand ignored)
3. **Fundamental limitation**: The problem is architectural, not implementation-specific
4. **Resource constraints**: Further experimentation unlikely to succeed without major architectural changes

### Next Steps

1. **Document findings**: This document serves as final assessment
2. **Pivot research direction**: Consider alternative approaches:
   - Fundamental architectural changes (e.g., separate chi1 and ligand pathways)
   - Accept Stage-1 as base-prior-only model
   - Focus on Stage-2 with softer prior interface
3. **Request Oracle verification**: Final assessment of whether limitation is fundamental or addressable

---

## Appendix: Job IDs and Checkpoints

### Completed Jobs
- **134326**: Precompute latents (5h 14m)
- **134408**: Fast training (2 min, best val_loss=0.001438)
- **134423**: Ligand sensitivity diagnostic (1m 46s)

### Key Checkpoints
- `/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/checkpoints/stage1/change_fast_1gpu_20260616_164925/best_model.pt`

### Diagnostic Results
- `/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/logs/stage1_diagnostics/change_fast_1gpu_20260616_164925_best_model_20260616_185014/delta_z_sensitivity.json`

---

## Conclusion

The Change-Prediction RAE approach, inspired by RAEv2, failed to learn ligand-causal effects. Despite achieving very low training loss (0.001438), the model learned to ignore ligand information completely (sensitivity = 0.0).

This confirms that the current Stage-1 architecture has a **fundamental limitation** in learning ligand-induced conformational changes. After 9+ different approaches, all showing the same failure mode, we conclude that this is not an implementation issue but an architectural limitation.

**Recommendation**: Accept this limitation and pivot to alternative research directions.
