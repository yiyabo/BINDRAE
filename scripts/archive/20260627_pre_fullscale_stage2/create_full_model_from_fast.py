#!/usr/bin/env python3
"""
Create a full Stage1Model checkpoint from fast training checkpoint.

This script loads:
1. Frozen encoder weights (ESM adapter + edge embedder + IPA + ligand conditioner)
2. Trained DeltaZPredictor weights from fast training
3. Chi head weights (random initialization)

And saves them as a unified Stage1Model checkpoint for diagnostics.
"""

import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.models.stage1_model import Stage1Model, Stage1ModelConfig
from src.stage1.models.delta_z_predictor import DeltaZPredictor


def main():
    fast_ckpt_path = "checkpoints/stage1/change_fast_1gpu_20260616_164925/best_model.pt"
    output_path = "checkpoints/stage1/change_fast_1gpu_20260616_164925/full_model_for_diag.pt"
    
    print(f"Loading fast training checkpoint: {fast_ckpt_path}")
    fast_ckpt = torch.load(fast_ckpt_path, map_location='cpu')
    
    print("Creating full Stage1Model...")
    config = Stage1ModelConfig.small()
    full_model = Stage1Model(config)
    
    print("Loading DeltaZPredictor weights...")
    predictor = DeltaZPredictor(c_s=384, hidden=256, n_layers=3)
    predictor.load_state_dict(fast_ckpt['model_state_dict'])
    
    print("Copying weights to full model...")
    full_model.delta_z_predictor = predictor
    
    print(f"Saving full model checkpoint: {output_path}")
    torch.save({
        'epoch': fast_ckpt['epoch'],
        'model_state_dict': full_model.state_dict(),
        'val_loss': fast_ckpt['val_loss'],
    }, output_path)
    
    print("✓ Done! Use this checkpoint for diagnostics:")
    print(f"  CHECKPOINT={output_path} sbatch scripts/slurm/diagnose_stage1_contrastive.sh")


if __name__ == '__main__':
    main()
