#!/bin/bash
# CPU launcher for the controlled product-manifold benchmark.

#SBATCH --job-name=s2_ctrl_man_cpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/stage2_controlled_manifold_cpu_%j.out
#SBATCH --error=logs/slurm/stage2_controlled_manifold_cpu_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE
export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage2/controlled_manifold
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

TAG="${TAG:-controlled_manifold_cpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-logs/stage2/controlled_manifold/${TAG}.json}"
MODEL_SEEDS="${MODEL_SEEDS:-7,42,137}"
TRAIN_ENDPOINT_PAIRS="${TRAIN_ENDPOINT_PAIRS:-256}"
VAL_ENDPOINT_PAIRS="${VAL_ENDPOINT_PAIRS:-64}"
TEST_ENDPOINT_PAIRS="${TEST_ENDPOINT_PAIRS:-128}"
N_RESIDUES="${N_RESIDUES:-8}"
N_STEPS="${N_STEPS:-40}"
EPOCHS="${EPOCHS:-300}"

echo "Output:       $OUTPUT"
echo "Model seeds:  $MODEL_SEEDS"
echo "Train pairs:  $TRAIN_ENDPOINT_PAIRS"
echo "Val pairs:    $VAL_ENDPOINT_PAIRS"
echo "Test pairs:   $TEST_ENDPOINT_PAIRS"
echo "Residues:     $N_RESIDUES"
echo "Path steps:   $N_STEPS"
echo "Epochs:       $EPOCHS"
echo "Device:       cpu"
echo "Start:        $(date)"

python scripts/run_controlled_manifold_benchmark.py \
  --output "$OUTPUT" \
  --model_seeds "$MODEL_SEEDS" \
  --train_endpoint_pairs "$TRAIN_ENDPOINT_PAIRS" \
  --val_endpoint_pairs "$VAL_ENDPOINT_PAIRS" \
  --test_endpoint_pairs "$TEST_ENDPOINT_PAIRS" \
  --n_residues "$N_RESIDUES" \
  --n_steps "$N_STEPS" \
  --epochs "$EPOCHS" \
  --device cpu

echo "Completed: $(date)"
echo "Output:    $OUTPUT"
