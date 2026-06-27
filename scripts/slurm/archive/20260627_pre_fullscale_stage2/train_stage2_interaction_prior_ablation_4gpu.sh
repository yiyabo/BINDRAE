#!/bin/bash
# Stage-2 interaction-prior ablation on a fixed small ligand-causal subset.
#
# Submit paired runs:
#   W_INTERACTION_PRIOR=0.05 TAG_SUFFIX=prior_on  sbatch scripts/slurm/train_stage2_interaction_prior_ablation_4gpu.sh
#   W_INTERACTION_PRIOR=0    TAG_SUFFIX=prior_off sbatch scripts/slurm/train_stage2_interaction_prior_ablation_4gpu.sh

#SBATCH --job-name=s2_iprior_ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/stage2_iprior_ablation_%j.out
#SBATCH --error=logs/slurm/stage2_iprior_ablation_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

cd "$ROOT"
mkdir -p logs/slurm logs/stage2 checkpoints/stage2 processed_data/triplets/ablation_subsets

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
INTERACTION_PRIOR_CKPT="${INTERACTION_PRIOR_CKPT:-checkpoints/stage1/interaction_prior_pocket_only_bs96_local192_tc4.5_20260617_112147/best_model.pt}"
W_INTERACTION_PRIOR="${W_INTERACTION_PRIOR:-0.05}"
INTERACTION_PRIOR_MIN_SCORE="${INTERACTION_PRIOR_MIN_SCORE:-0.2}"
INTERACTION_PRIOR_T_MID="${INTERACTION_PRIOR_T_MID:-0.3}"
INTERACTION_PRIOR_FEATURE_MODE="${INTERACTION_PRIOR_FEATURE_MODE:-none}"
INTERACTION_PRIOR_FEATURE_SCALE="${INTERACTION_PRIOR_FEATURE_SCALE:-1.0}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-2e-5}"
VAL_T="${VAL_T:-0.5}"
SEED="${SEED:-42}"
TRAIN_N="${TRAIN_N:-2048}"
VAL_N="${VAL_N:-512}"
SUBSET_SEED="${SUBSET_SEED:-20260617}"
TAG_SUFFIX="${TAG_SUFFIX:-w${W_INTERACTION_PRIOR}}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
GEOM_EVERY="${GEOM_EVERY:-1}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-2}"
N_GEOM_STEPS="${N_GEOM_STEPS:-3}"

TRAIN_SOURCE="${TRAIN_SOURCE:-processed_data/triplets/train_valid_strict_ligcausal_20260512.txt}"
VAL_SOURCE="${VAL_SOURCE:-processed_data/triplets/val_valid_strict_ligcausal_20260512.txt}"
TRAIN_SUBSET_REL="ablation_subsets/stage2_iprior_train_${TRAIN_N}_seed${SUBSET_SEED}.txt"
VAL_SUBSET_REL="ablation_subsets/stage2_iprior_val_${VAL_N}_seed${SUBSET_SEED}.txt"
TRAIN_SUBSET="processed_data/triplets/${TRAIN_SUBSET_REL}"
VAL_SUBSET="processed_data/triplets/${VAL_SUBSET_REL}"

python - <<PY
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(".").resolve()))
from utils.ligand_utils import build_ligand_tokens_from_file

DATA_DIR = Path("processed_data/triplets")

def ligand_ok(sample_id):
    sample_dir = DATA_DIR / "samples" / sample_id
    coords = sample_dir / "ligand_coords.npy"
    sdf = sample_dir / "ligand.sdf"
    if not coords.exists():
        return False, f"missing_coords:{coords}"
    try:
        build_ligand_tokens_from_file(coords, sdf)
    except Exception as exc:
        return False, f"{type(exc).__name__}:{str(exc).splitlines()[0]}"
    return True, ""

def write_subset(src, dst, n, seed):
    src = Path(src)
    dst = Path(dst)
    ids = [line.strip() for line in src.read_text().splitlines() if line.strip()]
    if n > len(ids):
        raise SystemExit(f"requested {n} samples from {src}, only {len(ids)} available")
    rng = random.Random(seed)
    rng.shuffle(ids)
    subset = []
    rejected = []
    for sample_id in ids:
        ok, reason = ligand_ok(sample_id)
        if ok:
            subset.append(sample_id)
            if len(subset) >= n:
                break
        else:
            rejected.append((sample_id, reason))
    if len(subset) < n:
        raise SystemExit(
            f"only {len(subset)} ligand-valid samples available from {src}; "
            f"requested {n}; rejected={len(rejected)}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\\n".join(subset) + "\\n")
    rejected_path = dst.with_suffix(".rejected.txt")
    rejected_path.write_text(
        "\\n".join(f"{sample_id}\\t{reason}" for sample_id, reason in rejected) + ("\\n" if rejected else "")
    )
    print(f"Wrote {len(subset)} ligand-valid IDs to {dst} from {src}")
    print(f"Rejected {len(rejected)} IDs; details: {rejected_path}")

write_subset("$TRAIN_SOURCE", "$TRAIN_SUBSET", int("$TRAIN_N"), int("$SUBSET_SEED"))
write_subset("$VAL_SOURCE", "$VAL_SUBSET", int("$VAL_N"), int("$SUBSET_SEED") + 1)
PY

TAG="stage2_iprior_ablation_${TAG_SUFFIX}_train${TRAIN_N}_val${VAL_N}_e${MAX_EPOCHS}_bs${BATCH_SIZE}x${NPROC_PER_NODE}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage2/${TAG}"
LOG_DIR="logs/stage2/${TAG}"

echo "=============================================="
echo "BINDRAE Stage-2 interaction-prior ablation"
echo "=============================================="
echo "Job ID:              ${SLURM_JOB_ID:-NA}"
echo "Node:                ${SLURM_NODELIST:-NA}"
echo "GPUs:                ${NPROC_PER_NODE}"
echo "Tag:                 $TAG"
echo "Train source:        $TRAIN_SOURCE"
echo "Val source:          $VAL_SOURCE"
echo "Train subset:        $TRAIN_SUBSET"
echo "Val subset:          $VAL_SUBSET"
echo "Interaction prior:   $INTERACTION_PRIOR_CKPT"
echo "w_interaction_prior: $W_INTERACTION_PRIOR"
echo "min_score:           $INTERACTION_PRIOR_MIN_SCORE"
echo "t_mid:               $INTERACTION_PRIOR_T_MID"
echo "feature_mode:        $INTERACTION_PRIOR_FEATURE_MODE"
echo "feature_scale:       $INTERACTION_PRIOR_FEATURE_SCALE"
echo "max_epochs:          $MAX_EPOCHS"
echo "batch per GPU:       $BATCH_SIZE"
echo "lr:                  $LR"
echo "geom_every:          $GEOM_EVERY"
echo "amp_dtype:           $AMP_DTYPE"
echo "Save dir:            $SAVE_DIR"
echo "Log dir:             $LOG_DIR"
echo "Start:               $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

"$ENV_PREFIX/bin/torchrun" --standalone --nproc_per_node="$NPROC_PER_NODE" scripts/train_stage2.py \
  --data_dir processed_data/triplets \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_epochs "$MAX_EPOCHS" \
  --lr "$LR" \
  --grad_clip 0.3 \
  --accum_steps 1 \
  --warmup_steps 0 \
  --seed "$SEED" \
  --val_t "$VAL_T" \
  --no_stage1_prior \
  --w_prior 0.0 \
  --interaction_prior_ckpt "$INTERACTION_PRIOR_CKPT" \
  --w_interaction_prior "$W_INTERACTION_PRIOR" \
  --interaction_prior_min_score "$INTERACTION_PRIOR_MIN_SCORE" \
  --interaction_prior_t_mid "$INTERACTION_PRIOR_T_MID" \
  --interaction_prior_feature_mode "$INTERACTION_PRIOR_FEATURE_MODE" \
  --interaction_prior_feature_scale "$INTERACTION_PRIOR_FEATURE_SCALE" \
  --n_integration_steps "$N_INTEGRATION_STEPS" \
  --n_geom_steps "$N_GEOM_STEPS" \
  --geom_loss_every_n_steps "$GEOM_EVERY" \
  --valid_samples_file "$TRAIN_SUBSET_REL" \
  --val_samples_file "$VAL_SUBSET_REL" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --amp_dtype "$AMP_DTYPE" \
  --distributed

echo ""
echo "=============================================="
echo "Stage-2 interaction-prior ablation completed: $(date)"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
