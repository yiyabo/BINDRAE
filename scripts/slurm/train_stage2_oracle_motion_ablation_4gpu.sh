#!/bin/bash
# Stage-2 ablation with OracleMotion-UB feature caches.
#
# Required:
#   STAGE1V2_TRAIN_CACHE_DIR=logs/stage2_oracle_motion/<train_export>
#   STAGE1V2_VAL_CACHE_DIR=logs/stage2_oracle_motion/<val_export>
#
# Typical controls:
#   sbatch --export=ALL,STAGE1V2_MODE=zero,STAGE1V2_TRAIN_CACHE_DIR=...,STAGE1V2_VAL_CACHE_DIR=... scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=oracle_motion,STAGE1V2_TRAIN_CACHE_DIR=...,STAGE1V2_VAL_CACHE_DIR=... scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=oracle_motion_residue_shuffled,STAGE1V2_TRAIN_CACHE_DIR=...,STAGE1V2_VAL_CACHE_DIR=... scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh

#SBATCH --job-name=s2_omotion_ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm/stage2_oracle_motion_ablation_%j.out
#SBATCH --error=logs/slurm/stage2_oracle_motion_ablation_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage2 checkpoints/stage2 processed_data/triplets/ablation_subsets

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
STAGE1V2_MODE="${STAGE1V2_MODE:-oracle_motion}"
STAGE1V2_FEATURES="${STAGE1V2_FEATURES:-delta_trans_local_x_norm,delta_trans_local_y_norm,delta_trans_local_z_norm,delta_rot_log_x_norm,delta_rot_log_y_norm,delta_rot_log_z_norm,delta_chi1_sin,delta_chi2_sin,delta_chi3_sin,delta_chi4_sin,delta_chi1_cos,delta_chi2_cos,delta_chi3_cos,delta_chi4_cos,chi1_mask,chi2_mask,chi3_mask,chi4_mask,trans_mag_norm,rot_angle_norm,max_abs_delta_chi_norm,motion_active,contact_apo,contact_holo,formed_contact,released_contact,signed_delta_dist_norm,w_res}"
STAGE1V2_FEATURE_SCALE="${STAGE1V2_FEATURE_SCALE:-1.0}"
STAGE1V2_TRAIN_CACHE_DIR="${STAGE1V2_TRAIN_CACHE_DIR:-}"
STAGE1V2_VAL_CACHE_DIR="${STAGE1V2_VAL_CACHE_DIR:-logs/stage2_oracle_motion/oracle_motion_val512_direct_20260623_020114}"
TRAIN_N="${TRAIN_N:-512}"
VAL_N="${VAL_N:-512}"
SUBSET_SEED="${SUBSET_SEED:-20260623}"
SUBSET_TAG="${SUBSET_TAG:-}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-2e-5}"
VAL_T="${VAL_T:-0.5}"
SEED="${SEED:-42}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
ESM_FUSION_ENABLED="${ESM_FUSION_ENABLED:-0}"
ESM_NUM_LAYERS="${ESM_NUM_LAYERS:-1}"
ESM_FUSION_MODE="${ESM_FUSION_MODE:-sum}"
ESM_LAYER_DROPOUT="${ESM_LAYER_DROPOUT:-0.0}"
ESM_LAYER_ENTROPY_WEIGHT="${ESM_LAYER_ENTROPY_WEIGHT:-0.0}"
REPA_ENABLED="${REPA_ENABLED:-0}"
REPA_WEIGHT="${REPA_WEIGHT:-0.0}"
REPA_DIM="${REPA_DIM:-128}"
REPA_LOSS_TYPE="${REPA_LOSS_TYPE:-cosine}"
REPA_MASK_MODE="${REPA_MASK_MODE:-motion_active_or_pocket}"
REPA_TARGET_MODE="${REPA_TARGET_MODE:-full}"
REPA_TARGET_SHUFFLE_MODE="${REPA_TARGET_SHUFFLE_MODE:-none}"
RESUME_FROM="${RESUME_FROM:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"
GEOM_EVERY="${GEOM_EVERY:-1}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-3}"
INTEGRATION_CHI_CLIP="${INTEGRATION_CHI_CLIP:-1.0}"
INTEGRATION_ROT_CLIP="${INTEGRATION_ROT_CLIP:-0.1}"
INTEGRATION_TRANS_CLIP="${INTEGRATION_TRANS_CLIP:-0.2}"
N_GEOM_STEPS="${N_GEOM_STEPS:-4}"
W_CONTACT="${W_CONTACT:-0.1}"
TAG_SUFFIX="${TAG_SUFFIX:-omotion_${STAGE1V2_MODE}}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
STRICT_CACHE_NPZ_PRECHECK="${STRICT_CACHE_NPZ_PRECHECK:-0}"
AATYPE_PRECHECK="${AATYPE_PRECHECK:-1}"
NODE_MASK_PRECHECK="${NODE_MASK_PRECHECK:-1}"
USE_EXISTING_SUBSETS="${USE_EXISTING_SUBSETS:-0}"
PRECHECK_WORKERS="${PRECHECK_WORKERS:-8}"

case "$STAGE1V2_MODE" in
  zero|oracle_motion|oracle_motion_residue_shuffled|oracle_motion_sample_shuffled) ;;
  *)
    echo "ERROR: STAGE1V2_MODE must be one of zero, oracle_motion, oracle_motion_residue_shuffled, oracle_motion_sample_shuffled"
    exit 1
    ;;
esac

case "$ESM_FUSION_ENABLED" in
  0|1) ;;
  *)
    echo "ERROR: ESM_FUSION_ENABLED must be 0 or 1"
    exit 1
    ;;
esac
if [[ "$ESM_FUSION_ENABLED" == "1" && "$ESM_NUM_LAYERS" -lt 1 ]]; then
  echo "ERROR: ESM_NUM_LAYERS must be >= 1"
  exit 1
fi
case "$ESM_FUSION_MODE" in
  sum|mean|softmax_weighted|gated_residual) ;;
  *)
    echo "ERROR: ESM_FUSION_MODE must be one of sum, mean, softmax_weighted, gated_residual"
    exit 1
    ;;
esac
case "$REPA_ENABLED" in
  0|1) ;;
  *)
    echo "ERROR: REPA_ENABLED must be 0 or 1"
    exit 1
    ;;
esac
case "$REPA_LOSS_TYPE" in
  cosine|mse) ;;
  *)
    echo "ERROR: REPA_LOSS_TYPE must be one of cosine, mse"
    exit 1
    ;;
esac
case "$REPA_MASK_MODE" in
  node|pocket|motion_active|motion_active_or_pocket) ;;
  *)
    echo "ERROR: REPA_MASK_MODE must be one of node, pocket, motion_active, motion_active_or_pocket"
    exit 1
    ;;
esac
case "$REPA_TARGET_SHUFFLE_MODE" in
  none|residue) ;;
  *)
    echo "ERROR: REPA_TARGET_SHUFFLE_MODE must be one of none, residue"
    exit 1
    ;;
esac
case "$REPA_TARGET_MODE" in
  full|motion_continuous) ;;
  *)
    echo "ERROR: REPA_TARGET_MODE must be one of full, motion_continuous"
    exit 1
    ;;
esac
case "$PRECHECK_ONLY" in
  0|1) ;;
  *)
    echo "ERROR: PRECHECK_ONLY must be 0 or 1"
    exit 1
    ;;
esac
case "$STRICT_CACHE_NPZ_PRECHECK" in
  0|1) ;;
  *)
    echo "ERROR: STRICT_CACHE_NPZ_PRECHECK must be 0 or 1"
    exit 1
    ;;
esac
case "$AATYPE_PRECHECK" in
  0|1) ;;
  *)
    echo "ERROR: AATYPE_PRECHECK must be 0 or 1"
    exit 1
    ;;
esac
case "$NODE_MASK_PRECHECK" in
  0|1) ;;
  *)
    echo "ERROR: NODE_MASK_PRECHECK must be 0 or 1"
    exit 1
    ;;
esac
case "$USE_EXISTING_SUBSETS" in
  0|1) ;;
  *)
    echo "ERROR: USE_EXISTING_SUBSETS must be 0 or 1"
    exit 1
    ;;
esac
if [[ "$PRECHECK_WORKERS" -lt 1 ]]; then
  echo "ERROR: PRECHECK_WORKERS must be >= 1"
  exit 1
fi

if [[ -z "$STAGE1V2_TRAIN_CACHE_DIR" ]]; then
  echo "ERROR: STAGE1V2_TRAIN_CACHE_DIR is required"
  exit 1
fi
if [[ -z "$STAGE1V2_VAL_CACHE_DIR" ]]; then
  echo "ERROR: STAGE1V2_VAL_CACHE_DIR is required"
  exit 1
fi
FEATURE_DIM="$(python - <<PY
print(len("$STAGE1V2_FEATURES".split(",")))
PY
)"

if [[ -n "$SUBSET_TAG" ]]; then
  TRAIN_SUBSET_REL="ablation_subsets/stage2_oracle_motion_${SUBSET_TAG}_train_${TRAIN_N}_seed${SUBSET_SEED}.txt"
  VAL_SUBSET_REL="ablation_subsets/stage2_oracle_motion_${SUBSET_TAG}_val_${VAL_N}_seed${SUBSET_SEED}.txt"
else
  TRAIN_SUBSET_REL="ablation_subsets/stage2_oracle_motion_train_${TRAIN_N}_seed${SUBSET_SEED}.txt"
  VAL_SUBSET_REL="ablation_subsets/stage2_oracle_motion_val_${VAL_N}_seed${SUBSET_SEED}.txt"
fi
TRAIN_SUBSET="processed_data/triplets/${TRAIN_SUBSET_REL}"
VAL_SUBSET="processed_data/triplets/${VAL_SUBSET_REL}"

if [[ "$USE_EXISTING_SUBSETS" == "1" ]]; then
  if [[ ! -s "$TRAIN_SUBSET" || ! -s "$VAL_SUBSET" ]]; then
    echo "ERROR: USE_EXISTING_SUBSETS=1 but subset files are missing or empty"
    echo "  train: $TRAIN_SUBSET"
    echo "  val:   $VAL_SUBSET"
    exit 1
  fi
  TRAIN_SUBSET_COUNT="$(wc -l < "$TRAIN_SUBSET")"
  VAL_SUBSET_COUNT="$(wc -l < "$VAL_SUBSET")"
  if [[ "$TRAIN_SUBSET_COUNT" -ne "$TRAIN_N" || "$VAL_SUBSET_COUNT" -ne "$VAL_N" ]]; then
    echo "ERROR: USE_EXISTING_SUBSETS=1 but subset line counts do not match TRAIN_N/VAL_N"
    echo "  train: $TRAIN_SUBSET_COUNT != $TRAIN_N ($TRAIN_SUBSET)"
    echo "  val:   $VAL_SUBSET_COUNT != $VAL_N ($VAL_SUBSET)"
    exit 1
  fi
  echo "Using existing prechecked subsets:"
  echo "  train: $TRAIN_SUBSET ($TRAIN_SUBSET_COUNT IDs)"
  echo "  val:   $VAL_SUBSET ($VAL_SUBSET_COUNT IDs)"
else
python - <<PY
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from src.stage2.datasets.dataset_stage2 import (
    _align_array,
    _align_len,
    _coords_valid_mask,
    _esm_features_from_data,
    _load_backbone_npz,
    _sequence_to_aatype,
    align_by_residue_ids,
    extract_backbone_coords,
)

DATA_DIR = Path("processed_data/triplets")
STRICT_CACHE_NPZ_PRECHECK = "$STRICT_CACHE_NPZ_PRECHECK" == "1"
AATYPE_PRECHECK = "$AATYPE_PRECHECK" == "1"
NODE_MASK_PRECHECK = "$NODE_MASK_PRECHECK" == "1"
PRECHECK_WORKERS = int("$PRECHECK_WORKERS")
ESM_NUM_LAYERS = int("$ESM_NUM_LAYERS")
EXPECTED_META_CACHE = {}


def cache_path_for_record(record, manifest_path):
    raw_path = record.get("path")
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        return path
    sample_id = record["sample_id"]
    safe_id = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in sample_id)
    return manifest_path.parent / f"{safe_id}.npz"


def expected_stage2_meta(sample_id):
    cached = EXPECTED_META_CACHE.get(sample_id)
    if cached is not None:
        return cached
    sample_dir = DATA_DIR / "samples" / sample_id
    esm_path = sample_dir / "esm.pt"
    if not esm_path.exists():
        raise FileNotFoundError(esm_path)
    try:
        esm_data = torch.load(esm_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        esm_data = torch.load(esm_path, map_location="cpu", weights_only=False)
    esm_features = _esm_features_from_data(esm_data, esm_path, ESM_NUM_LAYERS)
    n_res = int(esm_features.shape[0])

    torsion_path = DATA_DIR / "samples" / sample_id / "torsion_apo.npz"
    if not torsion_path.exists():
        raise FileNotFoundError(torsion_path)
    with np.load(torsion_path, allow_pickle=False) as data:
        if "aatype" in data:
            aatype = _align_array(np.asarray(data["aatype"]).astype(np.int64), n_res)
        else:
            sequence_str = str(esm_data.get("sequence_str", ""))
            aatype = _sequence_to_aatype(sequence_str, n_res)

    apo_backbone = sample_dir / "apo_backbone.npz"
    holo_backbone = sample_dir / "holo_backbone.npz"
    apo_pdb = sample_dir / "apo.pdb"
    holo_pdb = sample_dir / "holo.pdb"
    if apo_backbone.exists():
        N_apo, Ca_apo, C_apo, apo_res_ids = _load_backbone_npz(apo_backbone)
    else:
        N_apo, Ca_apo, C_apo, _, apo_res_ids = extract_backbone_coords(apo_pdb)
    if holo_backbone.exists():
        N_holo, Ca_holo, C_holo, holo_res_ids = _load_backbone_npz(holo_backbone)
    else:
        N_holo, Ca_holo, C_holo, _, holo_res_ids = extract_backbone_coords(holo_pdb)
    if apo_res_ids is not None and holo_res_ids is not None:
        _, _, node_mask = align_by_residue_ids(
            (N_apo, Ca_apo, C_apo), apo_res_ids,
            (N_holo, Ca_holo, C_holo), holo_res_ids,
            n_res,
        )
    else:
        N_apo, Ca_apo, C_apo = _align_len(N_apo, Ca_apo, C_apo, n_res)
        N_holo, Ca_holo, C_holo = _align_len(N_holo, Ca_holo, C_holo, n_res)
        node_mask = _coords_valid_mask(N_apo, Ca_apo, C_apo) & _coords_valid_mask(N_holo, Ca_holo, C_holo)

    EXPECTED_META_CACHE[sample_id] = (n_res, aatype, node_mask)
    return n_res, aatype, node_mask


def record_matches_stage2(record, manifest_path):
    sample_id = record["sample_id"]
    cache_path = cache_path_for_record(record, manifest_path)
    if not cache_path.exists():
        return False, "missing_cache"
    try:
        n_res, expected_aatype, expected_node_mask = expected_stage2_meta(sample_id)
        if "n_residues" not in record:
            return False, "manifest_missing_n_residues"
        if int(record["n_residues"]) != n_res:
            return False, "n_residues_mismatch"
        if not STRICT_CACHE_NPZ_PRECHECK and not AATYPE_PRECHECK and not NODE_MASK_PRECHECK:
            return True, "ok"
        with np.load(cache_path, allow_pickle=False) as data:
            if "sample_id" in data and str(np.asarray(data["sample_id"]).item()) != sample_id:
                return False, "sample_id_mismatch"
            if "n_residues" not in data:
                return False, "missing_n_residues"
            cache_n_res = int(np.asarray(data["n_residues"]).item())
            if cache_n_res != n_res:
                return False, "n_residues_mismatch"
            if "oracle_motion_features" in data and int(data["oracle_motion_features"].shape[0]) != n_res:
                return False, "feature_length_mismatch"
            if AATYPE_PRECHECK:
                if "aatype" not in data:
                    return False, "missing_aatype"
                cache_aatype = np.asarray(data["aatype"]).astype(np.int64)
                if cache_aatype.shape[0] != n_res:
                    return False, "aatype_length_mismatch"
                if not np.array_equal(cache_aatype, expected_aatype):
                    return False, "aatype_mismatch"
            if NODE_MASK_PRECHECK:
                mask_key = "node_mask" if "node_mask" in data else ("valid_mask" if "valid_mask" in data else None)
                if mask_key is not None:
                    cache_mask = np.asarray(data[mask_key]).astype(np.bool_)
                    if cache_mask.shape[0] != n_res:
                        return False, "node_mask_length_mismatch"
                    invalid_claims = cache_mask & (~np.asarray(expected_node_mask).astype(np.bool_))
                    if invalid_claims.any():
                        return False, "node_mask_mismatch"
    except Exception as exc:
        return False, f"{type(exc).__name__}"
    return True, "ok"


def write_from_manifest(manifest_path, out_path, n, seed):
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    with manifest_path.open() as f:
        manifest = json.load(f)
    records = list(manifest["records"])
    if int(n) > len(records):
        raise SystemExit(f"requested {n} IDs from {manifest_path}, only {len(records)} available")
    rng = random.Random(int(seed))
    rng.shuffle(records)
    chosen = []
    skipped = {}
    checked = 0

    def check_record(record):
        ok, reason = record_matches_stage2(record, manifest_path)
        return record["sample_id"], ok, reason

    if PRECHECK_WORKERS > 1:
        executor = ThreadPoolExecutor(max_workers=PRECHECK_WORKERS)
        chunk_size = max(PRECHECK_WORKERS * 64, 256)
    else:
        executor = None
        chunk_size = 256
    try:
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            if executor is not None:
                results = executor.map(check_record, chunk)
            else:
                results = map(check_record, chunk)
            for sample_id, ok, reason in results:
                checked += 1
                if ok:
                    chosen.append(sample_id)
                    if len(chosen) >= int(n):
                        break
                else:
                    skipped[reason] = skipped.get(reason, 0) + 1
                if checked % 1000 == 0:
                    print(f"Checked {checked} records from {manifest_path}; chosen={len(chosen)} skipped={skipped}", flush=True)
            if len(chosen) >= int(n):
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if len(chosen) < int(n):
        raise SystemExit(
            f"requested {n} Stage-2/cache-consistent IDs from {manifest_path}, "
            f"only found {len(chosen)}; skipped={skipped}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\\n".join(chosen) + "\\n")
    print(f"Wrote {len(chosen)} Stage-2/cache-consistent IDs to {out_path} from {manifest_path}; skipped={skipped}")

write_from_manifest("$STAGE1V2_TRAIN_CACHE_DIR/manifest.json", "$TRAIN_SUBSET", "$TRAIN_N", "$SUBSET_SEED")
write_from_manifest("$STAGE1V2_VAL_CACHE_DIR/manifest.json", "$VAL_SUBSET", "$VAL_N", int("$SUBSET_SEED") + 1)
PY
fi

if [[ "$PRECHECK_ONLY" == "1" ]]; then
  echo "=============================================="
  echo "Stage-2 + OracleMotion-UB precheck passed"
  echo "=============================================="
  echo "Mode:         $STAGE1V2_MODE"
  echo "Train cache:  $STAGE1V2_TRAIN_CACHE_DIR"
  echo "Val cache:    $STAGE1V2_VAL_CACHE_DIR"
  echo "Train subset: $TRAIN_SUBSET"
  echo "Val subset:   $VAL_SUBSET"
  echo "TRAIN_N:      $TRAIN_N"
  echo "VAL_N:        $VAL_N"
  echo "No torchrun/training launched because PRECHECK_ONLY=1."
  exit 0
fi

TAG="${TAG:-stage2_${TAG_SUFFIX}_train${TRAIN_N}_val${VAL_N}_e${MAX_EPOCHS}_bs${BATCH_SIZE}x${NPROC_PER_NODE}_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-checkpoints/stage2/${TAG}}"
LOG_DIR="${LOG_DIR:-logs/stage2/${TAG}}"
ESM_ARGS=()
if [[ "$ESM_FUSION_ENABLED" == "1" ]]; then
  ESM_ARGS=(
    --esm_fusion_enabled
    --esm_num_layers "$ESM_NUM_LAYERS"
    --esm_fusion_mode "$ESM_FUSION_MODE"
    --esm_layer_dropout "$ESM_LAYER_DROPOUT"
    --esm_layer_entropy_weight "$ESM_LAYER_ENTROPY_WEIGHT"
  )
fi
REPA_ARGS=()
if [[ "$REPA_ENABLED" == "1" ]]; then
  REPA_ARGS=(
    --repa_enabled
    --repa_weight "$REPA_WEIGHT"
    --repa_dim "$REPA_DIM"
    --repa_loss_type "$REPA_LOSS_TYPE"
    --repa_mask_mode "$REPA_MASK_MODE"
    --repa_target_mode "$REPA_TARGET_MODE"
    --repa_target_shuffle_mode "$REPA_TARGET_SHUFFLE_MODE"
  )
fi
RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_ARGS+=(--resume_from "$RESUME_FROM")
fi
if [[ "$AUTO_RESUME" != "1" ]]; then
  RESUME_ARGS+=(--no_auto_resume)
fi

echo "=============================================="
echo "Stage-2 + OracleMotion-UB ablation"
echo "=============================================="
echo "Job ID:          ${SLURM_JOB_ID:-NA}"
echo "Node:            ${SLURM_NODELIST:-NA}"
echo "GPUs:            $NPROC_PER_NODE"
echo "Mode:            $STAGE1V2_MODE"
echo "Feature dim:     $FEATURE_DIM"
echo "Feature scale:   $STAGE1V2_FEATURE_SCALE"
echo "Train cache:     $STAGE1V2_TRAIN_CACHE_DIR"
echo "Val cache:       $STAGE1V2_VAL_CACHE_DIR"
echo "Train subset:    $TRAIN_SUBSET"
echo "Val subset:      $VAL_SUBSET"
echo "max_epochs:      $MAX_EPOCHS"
echo "batch/GPU:       $BATCH_SIZE"
echo "lr:              $LR"
echo "w_contact:       $W_CONTACT"
echo "ESM fusion:      $ESM_FUSION_ENABLED"
echo "ESM layers/mode: $ESM_NUM_LAYERS / $ESM_FUSION_MODE"
echo "ESM layer drop:  $ESM_LAYER_DROPOUT"
echo "ESM entropy wt:  $ESM_LAYER_ENTROPY_WEIGHT"
echo "REPA enabled:    $REPA_ENABLED"
echo "REPA weight:     $REPA_WEIGHT"
echo "REPA dim/loss:   $REPA_DIM / $REPA_LOSS_TYPE"
echo "REPA mask:       $REPA_MASK_MODE"
echo "REPA target:     $REPA_TARGET_MODE"
echo "REPA shuffle:    $REPA_TARGET_SHUFFLE_MODE"
echo "Resume from:     ${RESUME_FROM:-OFF}"
echo "Auto resume:     $AUTO_RESUME"
echo "geom_every:      $GEOM_EVERY"
echo "integration:     $N_INTEGRATION_STEPS"
echo "chi clip:        $INTEGRATION_CHI_CLIP"
echo "rot clip:        $INTEGRATION_ROT_CLIP"
echo "trans clip:      $INTEGRATION_TRANS_CLIP"
echo "geom steps:      $N_GEOM_STEPS"
echo "Save dir:        $SAVE_DIR"
echo "Log dir:         $LOG_DIR"
echo "Start:           $(date)"
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
  --interaction_prior_feature_mode none \
  --w_interaction_prior 0.0 \
  --stage1v2_posterior_feature_mode "$STAGE1V2_MODE" \
  --stage1v2_train_cache_dir "$STAGE1V2_TRAIN_CACHE_DIR" \
  --stage1v2_val_cache_dir "$STAGE1V2_VAL_CACHE_DIR" \
  --stage1v2_posterior_feature_names "$STAGE1V2_FEATURES" \
  --stage1v2_posterior_feature_scale "$STAGE1V2_FEATURE_SCALE" \
  --stage1v2_loss_weight_mode none \
  --stage1v2_loss_weight_alpha 0.0 \
  --w_stage1v2_guidance 0.0 \
  --contact_loss_mode holo_target \
  --w_contact "$W_CONTACT" \
  --n_integration_steps "$N_INTEGRATION_STEPS" \
  --integration_chi_clip "$INTEGRATION_CHI_CLIP" \
  --integration_rot_clip "$INTEGRATION_ROT_CLIP" \
  --integration_trans_clip "$INTEGRATION_TRANS_CLIP" \
  --n_geom_steps "$N_GEOM_STEPS" \
  --geom_loss_every_n_steps "$GEOM_EVERY" \
  --valid_samples_file "$TRAIN_SUBSET_REL" \
  --val_samples_file "$VAL_SUBSET_REL" \
  "${ESM_ARGS[@]}" \
  "${REPA_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --amp_dtype "$AMP_DTYPE" \
  --distributed

echo ""
echo "=============================================="
echo "Stage-2 + OracleMotion-UB ablation completed: $(date)"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
