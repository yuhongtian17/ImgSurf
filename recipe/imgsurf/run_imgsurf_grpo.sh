#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 1. Locate the SCAgent repository and Hydra configuration
# ==============================================================================
# Resolve paths from this script rather than from the caller's current working
# directory. Hydra interprets a relative --config-path from verl/trainer, so the
# ImgSurf primary config must be passed as an absolute directory.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)
CONFIG_DIR=${SCRIPT_DIR}/configs
CONFIG_NAME=imgsurf_multiturn_grpo
CLUSTER_CHECK_SCRIPT=${SCRIPT_DIR}/check_imgsurf_cluster.py
RUNTIME_CHECK_SCRIPT=${SCRIPT_DIR}/check_imgsurf_runtime.py

if [[ ! -f "${REPO_ROOT}/verl/trainer/main_ppo.py" ]]; then
  echo "Could not locate SCAgent-main from script path: ${SCRIPT_DIR}" >&2
  echo "Expected trainer entrypoint: ${REPO_ROOT}/verl/trainer/main_ppo.py" >&2
  exit 2
fi
if [[ ! -r "${CONFIG_DIR}/${CONFIG_NAME}.yaml" ]]; then
  echo "ImgSurf Hydra config is missing or unreadable: ${CONFIG_DIR}/${CONFIG_NAME}.yaml" >&2
  exit 2
fi

# Relative paths embedded in the YAML (custom dataset, reward, agent loop and
# tool config) are intentionally rooted at SCAgent-main.
cd "${REPO_ROOT}"

# ==============================================================================
# 2. Validate the pinned SGLang 0.5.6 training runtime
# ==============================================================================
# Run this before Ray creates workers so an incompatible import is reported once
# with package versions instead of being repeated in remote actor tracebacks.
if [[ "${IMGSURF_SKIP_RUNTIME_CHECK:-0}" == "1" ]]; then
  echo "WARNING: IMGSURF_SKIP_RUNTIME_CHECK=1; dependency/API validation is disabled." >&2
else
  python "${RUNTIME_CHECK_SCRIPT}"
fi

# ==============================================================================
# 3. User paths and dataset selection
# ==============================================================================
# MODEL_PATH may point to either Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B-Instruct.
MODEL_PATH=${MODEL_PATH:-/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct}
DATA_ROOT=${DATA_ROOT:-/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf}
FULL_DATASET=${FULL_DATASET:-1}

# ==============================================================================
# 4. Distributed GPU topology
# ==============================================================================
# TOTAL_GPUS is the total number used by the colocated ActorRollout resource pool.
# For multi-node jobs it must be divisible by NNODES. GPUS_PER_NODE may be given
# explicitly; otherwise it is derived from TOTAL_GPUS / NNODES.
NNODES=${NNODES:-1}
TOTAL_GPUS=${TOTAL_GPUS:-}
GPUS_PER_NODE=${GPUS_PER_NODE:-}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if ! is_positive_integer "${NNODES}"; then
  echo "NNODES must be a positive integer; received '${NNODES}'." >&2
  exit 2
fi

if [[ -z "${TOTAL_GPUS}" && -z "${GPUS_PER_NODE}" ]]; then
  GPUS_PER_NODE=8
  TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
elif [[ -n "${TOTAL_GPUS}" ]]; then
  if ! is_positive_integer "${TOTAL_GPUS}"; then
    echo "TOTAL_GPUS must be a positive integer; received '${TOTAL_GPUS}'." >&2
    exit 2
  fi
  if ((TOTAL_GPUS % NNODES != 0)); then
    echo "TOTAL_GPUS=${TOTAL_GPUS} must be divisible by NNODES=${NNODES}." >&2
    exit 2
  fi
  DERIVED_GPUS_PER_NODE=$((TOTAL_GPUS / NNODES))
  if [[ -n "${GPUS_PER_NODE}" && "${GPUS_PER_NODE}" != "${DERIVED_GPUS_PER_NODE}" ]]; then
    echo "GPUS_PER_NODE=${GPUS_PER_NODE} conflicts with TOTAL_GPUS/NNODES=${DERIVED_GPUS_PER_NODE}." >&2
    exit 2
  fi
  GPUS_PER_NODE=${DERIVED_GPUS_PER_NODE}
else
  if ! is_positive_integer "${GPUS_PER_NODE}"; then
    echo "GPUS_PER_NODE must be a positive integer; received '${GPUS_PER_NODE}'." >&2
    exit 2
  fi
  TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
fi

if ((TOTAL_GPUS < 2)); then
  echo "ImgSurf full-parameter Qwen-VL training requires at least two GPUs." >&2
  exit 2
fi
if ! is_positive_integer "${ROLLOUT_TP_SIZE}" || ((GPUS_PER_NODE % ROLLOUT_TP_SIZE != 0)); then
  echo "ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE} must divide GPUS_PER_NODE=${GPUS_PER_NODE}." >&2
  exit 2
fi

# ==============================================================================
# 5. Detect and validate the Qwen-VL model family
# ==============================================================================
# Reading model_type avoids guessing Qwen2.5-VL versus Qwen3-VL from a directory
# name and prevents applying the wrong M-RoPE implementation.
MODEL_CONFIG=${MODEL_PATH}/config.json
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Missing model config: ${MODEL_CONFIG}" >&2
  exit 2
fi

DETECTED_MODEL_TYPE=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("model_type", ""))' \
  "${MODEL_CONFIG}")
MODEL_FAMILY=${MODEL_FAMILY:-${DETECTED_MODEL_TYPE}}
case "${MODEL_FAMILY}" in
  qwen2_5_vl)
    MODEL_SLUG=qwen2.5-vl
    ;;
  qwen3_vl)
    MODEL_SLUG=qwen3-vl
    ;;
  *)
    echo "Unsupported model_type '${MODEL_FAMILY}'. Expected qwen2_5_vl or qwen3_vl." >&2
    exit 2
    ;;
esac
if [[ "${MODEL_FAMILY}" != "${DETECTED_MODEL_TYPE}" ]]; then
  echo "MODEL_FAMILY=${MODEL_FAMILY} does not match config.json model_type=${DETECTED_MODEL_TYPE}." >&2
  exit 2
fi

PROJECT_NAME=${PROJECT_NAME:-imgsurf-${MODEL_SLUG}}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo-${TOTAL_GPUS}gpu-${NNODES}node}

# ==============================================================================
# 6. Build the DeepEyes parquet list and output directories
# ==============================================================================
VSTAR_DATA=${DATA_ROOT}/data_0.1.2_visual_toolbox_v2.parquet
MATH_DATA=${DATA_ROOT}/data_thinklite_reasoning_acc.parquet
CHART_DATA=${DATA_ROOT}/data_v0.8_visual_toolbox_v2.parquet

for required_path in "${VSTAR_DATA}" "${MATH_DATA}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}" >&2
    exit 2
  fi
done

if [[ "${FULL_DATASET}" == "1" ]]; then
  if [[ ! -f "${CHART_DATA}" ]]; then
    echo "Missing chart subset: ${CHART_DATA}" >&2
    exit 2
  fi
  TRAIN_FILES="[\"${VSTAR_DATA}\",\"${MATH_DATA}\",\"${CHART_DATA}\"]"
else
  # Lower-I/O smoke mode excludes the high-resolution chart subset.
  TRAIN_FILES="[\"${VSTAR_DATA}\",\"${MATH_DATA}\"]"
fi

# ==============================================================================
# 7. Configure and validate the local CUDA runtime
# ==============================================================================
# If the caller did not choose devices, use the first GPUS_PER_NODE local GPUs.
# Ray will assign the corresponding local device to each distributed worker.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=""
  for ((gpu_index = 0; gpu_index < GPUS_PER_NODE; gpu_index++)); do
    if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
      CUDA_VISIBLE_DEVICES+=","
    fi
    CUDA_VISIBLE_DEVICES+="${gpu_index}"
  done
fi
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export IMGSURF_MAX_INPUT_PIXELS=${IMGSURF_MAX_INPUT_PIXELS:-4194304}

VISIBLE_GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPU_COUNT}" -ne "${GPUS_PER_NODE}" ]]; then
  echo "Expected ${GPUS_PER_NODE} visible GPUs on this node, but PyTorch reports ${VISIBLE_GPU_COUNT}." >&2
  exit 2
fi

# ==============================================================================
# 8. Validate a pre-existing multi-node Ray cluster
# ==============================================================================
# Single-node jobs let verl create local Ray automatically. Multi-node jobs must
# connect to a cluster started with ray start on the head and worker nodes.
if ((NNODES > 1)); then
  if [[ -z "${RAY_ADDRESS:-}" ]]; then
    echo "NNODES=${NNODES} requires a running Ray cluster and RAY_ADDRESS (normally 'auto' on the head node)." >&2
    exit 2
  fi
  python "${CLUSTER_CHECK_SCRIPT}" \
    --nnodes "${NNODES}" \
    --gpus-per-node "${GPUS_PER_NODE}" \
    --runtime-check-script "${RUNTIME_CHECK_SCRIPT}"
fi

# ==============================================================================
# 9. Scale and validate GRPO batch/concurrency settings
# ==============================================================================
# Keep the same number of prompts per GPU as the original 8-GPU recipe.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((TOTAL_GPUS * 4))}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
AGENT_WORKERS=${AGENT_WORKERS:-${TOTAL_GPUS}}

for integer_setting in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE AGENT_WORKERS; do
  integer_value=${!integer_setting}
  if ! is_positive_integer "${integer_value}"; then
    echo "${integer_setting} must be a positive integer; received '${integer_value}'." >&2
    exit 2
  fi
done
if [[ -n "${ROLLOUT_N:-}" ]] && ! is_positive_integer "${ROLLOUT_N}"; then
  echo "ROLLOUT_N must be a positive integer when set; received '${ROLLOUT_N}'." >&2
  exit 2
fi
if ((TRAIN_BATCH_SIZE % TOTAL_GPUS != 0)); then
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be divisible by TOTAL_GPUS=${TOTAL_GPUS}." >&2
  exit 2
fi
if ((PPO_MINI_BATCH_SIZE % TOTAL_GPUS != 0)); then
  echo "PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} must be divisible by TOTAL_GPUS=${TOTAL_GPUS}." >&2
  exit 2
fi
if ((TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0)); then
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be divisible by PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}." >&2
  exit 2
fi

echo "ImgSurf resources: ${TOTAL_GPUS} GPUs = ${NNODES} node(s) x ${GPUS_PER_NODE} GPU(s), rollout TP=${ROLLOUT_TP_SIZE}"
echo "ImgSurf batches: train=${TRAIN_BATCH_SIZE}, PPO mini=${PPO_MINI_BATCH_SIZE}, GRPO n=${ROLLOUT_N:-YAML default}"
echo "ImgSurf config: ${CONFIG_DIR}/${CONFIG_NAME}.yaml"

# ==============================================================================
# 10. Build optional backward-compatible environment overrides
# ==============================================================================
# Static defaults are defined only in imgsurf_multiturn_grpo.yaml. Keep the
# environment names accepted by earlier commands, but do not inject a duplicate
# default when the variable was not supplied. Positional arguments passed to
# this script are native Hydra overrides and take the highest precedence.
OPTIONAL_HYDRA_OVERRIDES=()

append_hydra_override_if_set() {
  local environment_name=$1
  local config_key=$2
  local environment_value=${!environment_name:-}
  if [[ -n "${environment_value}" ]]; then
    OPTIONAL_HYDRA_OVERRIDES+=("${config_key}=${environment_value}")
  fi
}

append_hydra_override_if_set LEARNING_RATE actor_rollout_ref.actor.optim.lr
append_hydra_override_if_set ROLLOUT_N actor_rollout_ref.rollout.n
append_hydra_override_if_set ROLLOUT_GPU_MEMORY_UTILIZATION actor_rollout_ref.rollout.gpu_memory_utilization
append_hydra_override_if_set SAVE_FREQ trainer.save_freq
append_hydra_override_if_set TOTAL_EPOCHS trainer.total_epochs
append_hydra_override_if_set RESUME_MODE trainer.resume_mode

# ==============================================================================
# 11. Launch verl GRPO training
# ==============================================================================
# The Hydra primary config uses an absolute directory. Paths inside that config
# remain repository-relative because this script changed to REPO_ROOT above.
#
# Configuration precedence (lowest to highest):
#   base verl config < ImgSurf YAML < resource-derived launcher values
#   < explicitly set legacy environment variables < positional Hydra overrides
python -m verl.trainer.main_ppo \
  --config-path="${CONFIG_DIR}" \
  --config-name="${CONFIG_NAME}" \
  "data.train_files=${TRAIN_FILES}" \
  "data.val_files=[\"${VSTAR_DATA}\"]" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS} \
  trainer.n_gpus_per_node=${GPUS_PER_NODE} \
  trainer.nnodes=${NNODES} \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_ROOT}/ckpts/${EXPERIMENT_NAME}" \
  "+trainer.tensorboard_dir=${OUTPUT_ROOT}/tensorboard/${EXPERIMENT_NAME}" \
  "${OPTIONAL_HYDRA_OVERRIDES[@]}" \
  "$@" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/${EXPERIMENT_NAME}.log"
