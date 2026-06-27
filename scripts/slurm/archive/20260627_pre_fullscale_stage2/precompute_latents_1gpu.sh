#!/bin/bash
#SBATCH --job-name=s1_precomp_lat
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/stage1_precomp_latents_%j.out
#SBATCH --error=logs/slurm/stage1_precomp_latents_%j.err

set -e
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
mkdir -p logs/slurm processed_data/latents
EXTRA_ARGS=()
if [[ -n "${STAGE1_ENCODER_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--stage1_encoder_checkpoint "$STAGE1_ENCODER_CHECKPOINT")
fi

echo "=============================================="
echo "Stage-1 Precompute Latents"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Stage1 encoder checkpoint: ${STAGE1_ENCODER_CHECKPOINT:-<random init>}"
echo "Start:      $(date)"
echo "=============================================="

python scripts/precompute_latents.py \
  --data_dir processed_data/triplets \
  --output_dir processed_data/latents \
  --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
	  --batch_size 8 \
	  --max_n_res 1600 \
	  --num_workers 2 \
	  --device cuda \
	  "${EXTRA_ARGS[@]}"

echo ""
echo "=============================================="
echo "Precomputation completed: $(date)"
echo "Latents saved to: processed_data/latents"
echo "=============================================="
