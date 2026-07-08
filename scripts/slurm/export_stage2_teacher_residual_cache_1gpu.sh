#!/bin/bash
# Export free-flow teacher residual cache for boundary-residual Stage-2 training.

#SBATCH --job-name=s2_teacher_cache
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage2_teacher_cache_%j.out
#SBATCH --error=logs/slurm/stage2_teacher_cache_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE}"

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage2_teacher_residual_cache

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:-}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
SPLIT="${SPLIT:-train}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/stage2_teacher_residual_cache/teacher_residual_${SLURM_JOB_ID:-manual}}"
MANIFEST="${MANIFEST:-$OUTPUT_DIR/manifest.json}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_BATCHES="${MAX_BATCHES:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TRUST_PRECHECKED_SAMPLES="${TRUST_PRECHECKED_SAMPLES:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PATH_PARAMETERIZATION="${PATH_PARAMETERIZATION:-flow}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-12}"
INTEGRATION_CHI_CLIP="${INTEGRATION_CHI_CLIP:-}"
INTEGRATION_ROT_CLIP="${INTEGRATION_ROT_CLIP:-}"
INTEGRATION_TRANS_CLIP="${INTEGRATION_TRANS_CLIP:-}"
T_MIN="${T_MIN:-0.08}"
T_MAX="${T_MAX:-0.92}"
MOTION_EPS="${MOTION_EPS:-1e-4}"
POCKET_THRESHOLD="${POCKET_THRESHOLD:-0.3}"
LIGAND_CLASH_DIST="${LIGAND_CLASH_DIST:-2.2}"
CLASH_FOCUS_DIST="${CLASH_FOCUS_DIST:-3.0}"
CLASH_RELIEF_GAIN_WEIGHT="${CLASH_RELIEF_GAIN_WEIGHT:-0.25}"
CLASH_RELIEF_WEIGHT_CAP="${CLASH_RELIEF_WEIGHT_CAP:-5.0}"
STAGE1V2_POSTERIOR_FEATURE_MODE="${STAGE1V2_POSTERIOR_FEATURE_MODE:-}"
STAGE1V2_POSTERIOR_CACHE_DIR="${STAGE1V2_POSTERIOR_CACHE_DIR:-}"
STAGE1V2_POSTERIOR_FEATURE_NAMES="${STAGE1V2_POSTERIOR_FEATURE_NAMES:-}"
STAGE1V2_POSTERIOR_FEATURE_SCALE="${STAGE1V2_POSTERIOR_FEATURE_SCALE:-}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: CHECKPOINT is required"
  exit 1
fi
case "$SPLIT" in
  train|val|test) ;;
  *)
    echo "ERROR: SPLIT must be train, val, or test"
    exit 1
    ;;
esac
case "$TRUST_PRECHECKED_SAMPLES" in
  0|1) ;;
  *)
    echo "ERROR: TRUST_PRECHECKED_SAMPLES must be 0 or 1"
    exit 1
    ;;
esac
case "$SKIP_EXISTING" in
  0|1) ;;
  *)
    echo "ERROR: SKIP_EXISTING must be 0 or 1"
    exit 1
    ;;
esac
case "$PATH_PARAMETERIZATION" in
  checkpoint|flow|projected_flow|boundary_residual_v1|boundary_residual) ;;
  *)
    echo "ERROR: unsupported PATH_PARAMETERIZATION=$PATH_PARAMETERIZATION"
    exit 1
    ;;
esac

ARGS=(
  --checkpoint "$CHECKPOINT"
  --data_dir "$DATA_DIR"
  --split "$SPLIT"
  --output_dir "$OUTPUT_DIR"
  --manifest "$MANIFEST"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --path_parameterization "$PATH_PARAMETERIZATION"
  --n_integration_steps "$N_INTEGRATION_STEPS"
  --t_min "$T_MIN"
  --t_max "$T_MAX"
  --motion_eps "$MOTION_EPS"
  --pocket_threshold "$POCKET_THRESHOLD"
  --ligand_clash_dist "$LIGAND_CLASH_DIST"
  --clash_focus_dist "$CLASH_FOCUS_DIST"
  --clash_relief_gain_weight "$CLASH_RELIEF_GAIN_WEIGHT"
  --clash_relief_weight_cap "$CLASH_RELIEF_WEIGHT_CAP"
  --device cuda
)
if [[ -n "$VALID_SAMPLES_FILE" ]]; then
  ARGS+=(--valid_samples_file "$VALID_SAMPLES_FILE")
fi
if [[ "$TRUST_PRECHECKED_SAMPLES" == "1" ]]; then
  ARGS+=(--trust_prechecked_samples)
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip_existing)
fi
if [[ -n "$MAX_BATCHES" ]]; then
  ARGS+=(--max_batches "$MAX_BATCHES")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [[ -n "$INTEGRATION_CHI_CLIP" ]]; then
  ARGS+=(--integration_chi_clip "$INTEGRATION_CHI_CLIP")
fi
if [[ -n "$INTEGRATION_ROT_CLIP" ]]; then
  ARGS+=(--integration_rot_clip "$INTEGRATION_ROT_CLIP")
fi
if [[ -n "$INTEGRATION_TRANS_CLIP" ]]; then
  ARGS+=(--integration_trans_clip "$INTEGRATION_TRANS_CLIP")
fi
if [[ -n "$STAGE1V2_POSTERIOR_FEATURE_MODE" ]]; then
  ARGS+=(--stage1v2_posterior_feature_mode "$STAGE1V2_POSTERIOR_FEATURE_MODE")
fi
if [[ -n "$STAGE1V2_POSTERIOR_CACHE_DIR" ]]; then
  ARGS+=(--stage1v2_posterior_cache_dir "$STAGE1V2_POSTERIOR_CACHE_DIR")
fi
if [[ -n "$STAGE1V2_POSTERIOR_FEATURE_NAMES" ]]; then
  ARGS+=(--stage1v2_posterior_feature_names "$STAGE1V2_POSTERIOR_FEATURE_NAMES")
fi
if [[ -n "$STAGE1V2_POSTERIOR_FEATURE_SCALE" ]]; then
  ARGS+=(--stage1v2_posterior_feature_scale "$STAGE1V2_POSTERIOR_FEATURE_SCALE")
fi

echo "=============================================="
echo "Stage-2 teacher residual cache export"
echo "=============================================="
echo "Job ID:       ${SLURM_JOB_ID:-NA}"
echo "Checkpoint:   $CHECKPOINT"
echo "Split:        $SPLIT"
echo "Samples file: ${VALID_SAMPLES_FILE:-OFF}"
echo "Output dir:   $OUTPUT_DIR"
echo "Manifest:     $MANIFEST"
echo "Batch size:   $BATCH_SIZE"
echo "Workers:      $NUM_WORKERS"
echo "Path mode:    $PATH_PARAMETERIZATION"
echo "Steps:        $N_INTEGRATION_STEPS"
echo "t range:      $T_MIN-$T_MAX"
echo "clash dist:   $LIGAND_CLASH_DIST"
echo "focus dist:   $CLASH_FOCUS_DIST"
echo "relief gain:  $CLASH_RELIEF_GAIN_WEIGHT"
echo "relief cap:   $CLASH_RELIEF_WEIGHT_CAP"
echo "Start:        $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
python scripts/export_stage2_teacher_residual_cache.py "${ARGS[@]}"

echo "Done: $(date)"
