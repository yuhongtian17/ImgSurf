#!/usr/bin/env bash
set -euo pipefail

# ImgSurf v4-style GRPO launcher.
# Precedence: raw Hydra override > launcher flag > environment > defaults here > YAML fallback.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)
CONFIG_DIR=${SCRIPT_DIR}/configs
CONFIG_NAME=imgsurf_multiturn_grpo

usage() {
  cat <<'EOF'
Usage: bash recipe/imgsurf/run_imgsurf_grpo.sh [launcher flags] [Hydra overrides]

Paths and run identity:
  --model-path PATH                 Hugging Face model directory
  --data-root PATH                  DeepEyes dataset directory
  --output-root PATH                Checkpoint/log/tensorboard root
  --full-dataset on|off             Include the chart subset (default: on)
  --model-family FAMILY             qwen3_vl or qwen2_5_vl; normally auto-detected
  --project-name NAME               Logger project name
  --experiment-name NAME            Run/checkpoint name

Resources and distributed runtime:
  --total-gpus INT                  Total GPUs across all nodes (default: 8)
  --nnodes INT                      Number of nodes (default: 1)
  --gpus-per-node INT               GPUs per node (derived from total-gpus/nnodes)
  --rollout-tp-size INT             SGLang tensor parallel size (default: 1)
  --agent-workers INT               Async rollout workers (default: total-gpus)
  --cuda-visible-devices LIST       Visible local GPU indices (default: 0..N-1)
  --ray-address ADDRESS             Existing Ray cluster address for multi-node runs
  --nccl-debug LEVEL                NCCL log level (default: WARN)

Training and rollout:
  --train-batch-size INT            Global prompt batch (default: 4 per GPU)
  --ppo-mini-batch-size INT         Global PPO mini-batch (default: train batch)
  --rollout-n INT                   GRPO samples per prompt (default: 8)
  --rollout-gpu-memory-utilization FLOAT
                                      SGLang memory fraction (default: 0.50)
  --learning-rate FLOAT             Actor learning rate (default: 1.0e-6)
  --save-freq INT                   Checkpoint interval (default: 100)
  --total-epochs INT                Training epochs (default: 1)
  --resume-mode MODE                verl resume mode (default: auto)

ImgSurf trajectory:
  --reward-token all|outer          Reward outer + inner tokens (default: all)
  --semantic-reward rule|judge
                                      Deterministic rules (default) or VLM judge
  --consistency-reward on|off       Reward IoU convergence (default: on)
  --k FLOAT                         v4 expansion base (default: 0.4)
  --iou-thr FLOAT                   v4 convergence threshold (default: 0.5)
  --max-iter INT                    v4 maximum iterations (default: 4)
  --max-level INT                   v4 levels per iteration (default: 1)
  --expand-mode quarter|ctr|bbox    v4 expansion rule (default: quarter)

Token and image budgets:
  --max-prompt-length INT           Initial prompt limit (default: 8192)
  --max-response-length INT         Whole multi-turn response limit (default: 8192)
  --max-model-len INT               SGLang context (default: prompt + response = 16384)
  --max-batched-tokens INT          SGLang chunked-prefill budget (default: 8192)
  --max-turn-tokens INT             Per-policy-turn cap (default: 1024)
  --min-final-tokens INT            Tokens reserved for final turn (default: 512)
  --max-think-summary-tokens INT    Retained thought-summary budget (default: 512)
  --max-assistant-turns INT         Multi-turn assistant cap (derived from v4 limits)
  --max-user-turns INT              Multi-turn user cap (derived from v4 limits)
  --max-input-pixels INT            Initial image pixel budget (default: 4194304)
  --min-tool-pixels INT             Minimum tool image pixels (default: 4096)
  --max-tool-pixels INT             Maximum tool image pixels (default: 4194304)

Reward weights:
  --accuracy-weight FLOAT           Default: 1.0
  --format-weight FLOAT             Default: 0.05
  --tool-weight FLOAT               Default: 0.10
  --consistency-weight FLOAT        Default: 0.05
  --invalid-tool-weight FLOAT       Default: 0.10
  --excess-tool-weight FLOAT        Default: 0.02

VLM judge:
  --judge-base-url URL              Required when semantic-reward=judge
  --judge-model NAME                Judge model (default: first model from judge server)
  --judge-api-key KEY               OpenAI-compatible API key (default: EMPTY)
  --judge-timeout FLOAT             Request timeout in seconds (default: 60)
  --judge-max-retries INT           Retry count (default: 2)
  --judge-max-pixels INT            Judge image budget (default: 1048576)

Other:
  -h, --help                        Show this message

Both '--flag value' and '--flag=value' are accepted. Existing uppercase
environment variables remain backward compatible, but flags take precedence.
Arguments without a leading '--' are forwarded as raw Hydra overrides and have
the highest precedence.
EOF
}

# Environment values are backward-compatible secondary inputs. Launcher flags
# are parsed afterwards, so they always win over the corresponding environment.
MODEL_PATH=${MODEL_PATH:-/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct}
DATA_ROOT=${DATA_ROOT:-/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf}
FULL_DATASET=${FULL_DATASET:-1}
MODEL_FAMILY=${MODEL_FAMILY:-}
PROJECT_NAME=${PROJECT_NAME:-}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-}

NNODES=${NNODES:-1}
TOTAL_GPUS=${TOTAL_GPUS:-}
GPUS_PER_NODE=${GPUS_PER_NODE:-}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
AGENT_WORKERS=${AGENT_WORKERS:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
RAY_ADDRESS=${RAY_ADDRESS:-}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}
LEARNING_RATE=${LEARNING_RATE:-1.0e-6}
SAVE_FREQ=${SAVE_FREQ:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
RESUME_MODE=${RESUME_MODE:-auto}

IMGSURF_REWARD_TOKEN=${IMGSURF_REWARD_TOKEN:-all}
IMGSURF_SEMANTIC_REWARD=${IMGSURF_SEMANTIC_REWARD:-rule}
IMGSURF_CONSISTENCY_REWARD=${IMGSURF_CONSISTENCY_REWARD:-on}
IMGSURF_K=${IMGSURF_K:-0.4}
IMGSURF_IOU_THR=${IMGSURF_IOU_THR:-0.5}
IMGSURF_MAX_ITER=${IMGSURF_MAX_ITER:-4}
IMGSURF_MAX_LEVEL=${IMGSURF_MAX_LEVEL:-1}
IMGSURF_EXPAND_MODE=${IMGSURF_EXPAND_MODE:-quarter}

IMGSURF_MAX_PROMPT_LENGTH=${IMGSURF_MAX_PROMPT_LENGTH:-8192}
IMGSURF_MAX_RESPONSE_LENGTH=${IMGSURF_MAX_RESPONSE_LENGTH:-8192}
IMGSURF_MAX_MODEL_LEN=${IMGSURF_MAX_MODEL_LEN:-}
IMGSURF_MAX_BATCHED_TOKENS=${IMGSURF_MAX_BATCHED_TOKENS:-8192}
IMGSURF_MAX_TURN_TOKENS=${IMGSURF_MAX_TURN_TOKENS:-1024}
IMGSURF_MIN_FINAL_TOKENS=${IMGSURF_MIN_FINAL_TOKENS:-512}
IMGSURF_MAX_THINK_SUMMARY_TOKENS=${IMGSURF_MAX_THINK_SUMMARY_TOKENS:-512}
IMGSURF_MAX_ASSISTANT_TURNS=${IMGSURF_MAX_ASSISTANT_TURNS:-}
IMGSURF_MAX_USER_TURNS=${IMGSURF_MAX_USER_TURNS:-}
IMGSURF_MAX_INPUT_PIXELS=${IMGSURF_MAX_INPUT_PIXELS:-4194304}
IMGSURF_MIN_TOOL_PIXELS=${IMGSURF_MIN_TOOL_PIXELS:-4096}
IMGSURF_MAX_TOOL_PIXELS=${IMGSURF_MAX_TOOL_PIXELS:-4194304}

IMGSURF_ACCURACY_WEIGHT=${IMGSURF_ACCURACY_WEIGHT:-1.0}
IMGSURF_FORMAT_WEIGHT=${IMGSURF_FORMAT_WEIGHT:-0.05}
IMGSURF_TOOL_WEIGHT=${IMGSURF_TOOL_WEIGHT:-0.10}
IMGSURF_CONSISTENCY_WEIGHT=${IMGSURF_CONSISTENCY_WEIGHT:-0.05}
IMGSURF_INVALID_TOOL_WEIGHT=${IMGSURF_INVALID_TOOL_WEIGHT:-0.10}
IMGSURF_EXCESS_TOOL_WEIGHT=${IMGSURF_EXCESS_TOOL_WEIGHT:-0.02}

IMGSURF_JUDGE_BASE_URL=${IMGSURF_JUDGE_BASE_URL:-}
IMGSURF_JUDGE_MODEL=${IMGSURF_JUDGE_MODEL:-}
IMGSURF_JUDGE_API_KEY=${IMGSURF_JUDGE_API_KEY:-EMPTY}
IMGSURF_JUDGE_TIMEOUT=${IMGSURF_JUDGE_TIMEOUT:-60}
IMGSURF_JUDGE_MAX_RETRIES=${IMGSURF_JUDGE_MAX_RETRIES:-2}
IMGSURF_JUDGE_MAX_PIXELS=${IMGSURF_JUDGE_MAX_PIXELS:-1048576}

declare -A FLAG_TO_VARIABLE=(
  [--model-path]=MODEL_PATH
  [--data-root]=DATA_ROOT
  [--output-root]=OUTPUT_ROOT
  [--full-dataset]=FULL_DATASET
  [--model-family]=MODEL_FAMILY
  [--project-name]=PROJECT_NAME
  [--experiment-name]=EXPERIMENT_NAME
  [--total-gpus]=TOTAL_GPUS
  [--nnodes]=NNODES
  [--gpus-per-node]=GPUS_PER_NODE
  [--rollout-tp-size]=ROLLOUT_TP_SIZE
  [--agent-workers]=AGENT_WORKERS
  [--cuda-visible-devices]=CUDA_VISIBLE_DEVICES
  [--ray-address]=RAY_ADDRESS
  [--nccl-debug]=NCCL_DEBUG
  [--train-batch-size]=TRAIN_BATCH_SIZE
  [--ppo-mini-batch-size]=PPO_MINI_BATCH_SIZE
  [--rollout-n]=ROLLOUT_N
  [--rollout-gpu-memory-utilization]=ROLLOUT_GPU_MEMORY_UTILIZATION
  [--learning-rate]=LEARNING_RATE
  [--save-freq]=SAVE_FREQ
  [--total-epochs]=TOTAL_EPOCHS
  [--resume-mode]=RESUME_MODE
  [--reward-token]=IMGSURF_REWARD_TOKEN
  [--semantic-reward]=IMGSURF_SEMANTIC_REWARD
  [--consistency-reward]=IMGSURF_CONSISTENCY_REWARD
  [--k]=IMGSURF_K
  [--iou-thr]=IMGSURF_IOU_THR
  [--max-iter]=IMGSURF_MAX_ITER
  [--max-level]=IMGSURF_MAX_LEVEL
  [--expand-mode]=IMGSURF_EXPAND_MODE
  [--max-prompt-length]=IMGSURF_MAX_PROMPT_LENGTH
  [--max-response-length]=IMGSURF_MAX_RESPONSE_LENGTH
  [--max-model-len]=IMGSURF_MAX_MODEL_LEN
  [--max-batched-tokens]=IMGSURF_MAX_BATCHED_TOKENS
  [--max-turn-tokens]=IMGSURF_MAX_TURN_TOKENS
  [--min-final-tokens]=IMGSURF_MIN_FINAL_TOKENS
  [--max-think-summary-tokens]=IMGSURF_MAX_THINK_SUMMARY_TOKENS
  [--max-assistant-turns]=IMGSURF_MAX_ASSISTANT_TURNS
  [--max-user-turns]=IMGSURF_MAX_USER_TURNS
  [--max-input-pixels]=IMGSURF_MAX_INPUT_PIXELS
  [--min-tool-pixels]=IMGSURF_MIN_TOOL_PIXELS
  [--max-tool-pixels]=IMGSURF_MAX_TOOL_PIXELS
  [--accuracy-weight]=IMGSURF_ACCURACY_WEIGHT
  [--format-weight]=IMGSURF_FORMAT_WEIGHT
  [--tool-weight]=IMGSURF_TOOL_WEIGHT
  [--consistency-weight]=IMGSURF_CONSISTENCY_WEIGHT
  [--invalid-tool-weight]=IMGSURF_INVALID_TOOL_WEIGHT
  [--excess-tool-weight]=IMGSURF_EXCESS_TOOL_WEIGHT
  [--judge-base-url]=IMGSURF_JUDGE_BASE_URL
  [--judge-model]=IMGSURF_JUDGE_MODEL
  [--judge-api-key]=IMGSURF_JUDGE_API_KEY
  [--judge-timeout]=IMGSURF_JUDGE_TIMEOUT
  [--judge-max-retries]=IMGSURF_JUDGE_MAX_RETRIES
  [--judge-max-pixels]=IMGSURF_JUDGE_MAX_PIXELS
)

HYDRA_OVERRIDES=()
while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --*=*)
      flag=${1%%=*}
      value=${1#*=}
      shift
      ;;
    --*)
      flag=$1
      if (($# < 2)); then
        echo "Missing value for ${flag}" >&2
        exit 2
      fi
      value=$2
      shift 2
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      continue
      ;;
  esac

  if [[ -z "${FLAG_TO_VARIABLE[${flag}]+defined}" ]]; then
    echo "Unknown launcher flag: ${flag}" >&2
    echo "Use --help to list supported flags; raw Hydra overrides do not start with '--'." >&2
    exit 2
  fi
  printf -v "${FLAG_TO_VARIABLE[${flag}]}" '%s' "${value}"
done

is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nonnegative_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_integer() { [[ "$1" =~ ^-?[0-9]+$ ]]; }
is_number() { [[ "$1" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; }
check_float() {
  local value=$1
  local expression=$2
  local message=$3
  if ! is_number "${value}" || ! python -c "import sys; x=float(sys.argv[1]); assert ${expression}" "${value}"; then
    echo "${message}; received '${value}'" >&2
    exit 2
  fi
}

case "${FULL_DATASET,,}" in
  on|true|1) FULL_DATASET=1 ;;
  off|false|0) FULL_DATASET=0 ;;
  *) echo "--full-dataset must be on or off" >&2; exit 2 ;;
esac
case "${IMGSURF_REWARD_TOKEN}" in all|outer) ;; *) echo "--reward-token must be all or outer" >&2; exit 2 ;; esac
case "${IMGSURF_SEMANTIC_REWARD}" in rule|judge) ;; *) echo "--semantic-reward must be rule or judge" >&2; exit 2 ;; esac
case "${IMGSURF_CONSISTENCY_REWARD,,}" in
  on|true|1) IMGSURF_CONSISTENCY_REWARD=1 ;;
  off|false|0) IMGSURF_CONSISTENCY_REWARD=0 ;;
  *) echo "--consistency-reward must be on or off" >&2; exit 2 ;;
esac
case "${IMGSURF_EXPAND_MODE}" in quarter|ctr|bbox) ;; *) echo "--expand-mode must be quarter, ctr or bbox" >&2; exit 2 ;; esac

for setting in IMGSURF_MAX_PROMPT_LENGTH IMGSURF_MAX_RESPONSE_LENGTH; do
  if ! is_positive_integer "${!setting}"; then
    echo "${setting} must be a positive integer; received '${!setting}'" >&2
    exit 2
  fi
done
if [[ -z "${IMGSURF_MAX_MODEL_LEN}" ]]; then
  IMGSURF_MAX_MODEL_LEN=$((IMGSURF_MAX_PROMPT_LENGTH + IMGSURF_MAX_RESPONSE_LENGTH))
fi
for setting in IMGSURF_MAX_ITER IMGSURF_MAX_LEVEL IMGSURF_MAX_MODEL_LEN \
  IMGSURF_MAX_BATCHED_TOKENS \
  IMGSURF_MAX_TURN_TOKENS IMGSURF_MIN_FINAL_TOKENS IMGSURF_MAX_THINK_SUMMARY_TOKENS \
  IMGSURF_MAX_INPUT_PIXELS IMGSURF_MIN_TOOL_PIXELS IMGSURF_MAX_TOOL_PIXELS \
  IMGSURF_JUDGE_MAX_PIXELS; do
  if ! is_positive_integer "${!setting}"; then
    echo "${setting} must be a positive integer; received '${!setting}'" >&2
    exit 2
  fi
done
if ! is_nonnegative_integer "${IMGSURF_JUDGE_MAX_RETRIES}"; then
  echo "--judge-max-retries must be a non-negative integer" >&2
  exit 2
fi
if ((IMGSURF_MIN_TOOL_PIXELS > IMGSURF_MAX_TOOL_PIXELS)); then
  echo "--min-tool-pixels must be <= --max-tool-pixels" >&2
  exit 2
fi
if ((IMGSURF_MAX_MODEL_LEN < IMGSURF_MAX_PROMPT_LENGTH + IMGSURF_MAX_RESPONSE_LENGTH)); then
  echo "--max-model-len must be >= max-prompt-length + max-response-length" >&2
  exit 2
fi
if ((IMGSURF_MIN_FINAL_TOKENS >= IMGSURF_MAX_RESPONSE_LENGTH)); then
  echo "--min-final-tokens must be smaller than --max-response-length" >&2
  exit 2
fi

check_float "${IMGSURF_K}" '0 < x <= 1' '--k must be in (0,1]'
check_float "${IMGSURF_IOU_THR}" '0 <= x <= 1' '--iou-thr must be in [0,1]'
check_float "${ROLLOUT_GPU_MEMORY_UTILIZATION}" '0 < x <= 1' '--rollout-gpu-memory-utilization must be in (0,1]'
check_float "${LEARNING_RATE}" 'x > 0' '--learning-rate must be positive'
check_float "${IMGSURF_JUDGE_TIMEOUT}" 'x > 0' '--judge-timeout must be positive'
for setting in IMGSURF_ACCURACY_WEIGHT IMGSURF_FORMAT_WEIGHT IMGSURF_TOOL_WEIGHT \
  IMGSURF_CONSISTENCY_WEIGHT IMGSURF_INVALID_TOOL_WEIGHT IMGSURF_EXCESS_TOOL_WEIGHT; do
  check_float "${!setting}" 'x >= 0' "${setting} must be non-negative"
done
if [[ "${IMGSURF_SEMANTIC_REWARD}" == judge && -z "${IMGSURF_JUDGE_BASE_URL}" ]]; then
  echo "--semantic-reward=judge requires --judge-base-url" >&2
  exit 2
fi

if ! is_positive_integer "${NNODES}"; then echo "--nnodes must be positive" >&2; exit 2; fi
if [[ -z "${TOTAL_GPUS}" && -z "${GPUS_PER_NODE}" ]]; then
  GPUS_PER_NODE=8
  TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
elif [[ -n "${TOTAL_GPUS}" ]]; then
  if ! is_positive_integer "${TOTAL_GPUS}" || ((TOTAL_GPUS % NNODES != 0)); then
    echo "--total-gpus must be positive and divisible by --nnodes" >&2
    exit 2
  fi
  derived_gpus_per_node=$((TOTAL_GPUS / NNODES))
  if [[ -n "${GPUS_PER_NODE}" && "${GPUS_PER_NODE}" != "${derived_gpus_per_node}" ]]; then
    echo "--gpus-per-node conflicts with --total-gpus/--nnodes" >&2
    exit 2
  fi
  GPUS_PER_NODE=${derived_gpus_per_node}
else
  if ! is_positive_integer "${GPUS_PER_NODE}"; then echo "--gpus-per-node must be positive" >&2; exit 2; fi
  TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
fi
if ((TOTAL_GPUS < 2)); then echo "ImgSurf full-parameter training requires at least two GPUs" >&2; exit 2; fi
if ! is_positive_integer "${ROLLOUT_TP_SIZE}" || ((GPUS_PER_NODE % ROLLOUT_TP_SIZE != 0)); then
  echo "--rollout-tp-size must divide --gpus-per-node" >&2
  exit 2
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((TOTAL_GPUS * 4))}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
AGENT_WORKERS=${AGENT_WORKERS:-${TOTAL_GPUS}}
IMGSURF_MAX_ASSISTANT_TURNS=${IMGSURF_MAX_ASSISTANT_TURNS:-$((2 * IMGSURF_MAX_ITER * IMGSURF_MAX_LEVEL + 2))}
IMGSURF_MAX_USER_TURNS=${IMGSURF_MAX_USER_TURNS:-$((2 * IMGSURF_MAX_ITER * IMGSURF_MAX_LEVEL + 1))}
for setting in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE AGENT_WORKERS ROLLOUT_N TOTAL_EPOCHS \
  IMGSURF_MAX_ASSISTANT_TURNS IMGSURF_MAX_USER_TURNS; do
  if ! is_positive_integer "${!setting}"; then
    echo "${setting} must be a positive integer; received '${!setting}'" >&2
    exit 2
  fi
done
if ! is_integer "${SAVE_FREQ}"; then echo "--save-freq must be an integer" >&2; exit 2; fi
if ((TRAIN_BATCH_SIZE % TOTAL_GPUS != 0 || PPO_MINI_BATCH_SIZE % TOTAL_GPUS != 0 || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0)); then
  echo "--train-batch-size and --ppo-mini-batch-size must be compatible with --total-gpus" >&2
  exit 2
fi

if [[ ! -f "${REPO_ROOT}/verl/trainer/main_ppo.py" ]]; then
  echo "Could not locate SCAgent repository from ${SCRIPT_DIR}" >&2
  exit 2
fi
if [[ ! -r "${CONFIG_DIR}/${CONFIG_NAME}.yaml" ]]; then
  echo "Missing ImgSurf config: ${CONFIG_DIR}/${CONFIG_NAME}.yaml" >&2
  exit 2
fi
cd "${REPO_ROOT}"

MODEL_CONFIG=${MODEL_PATH}/config.json
if [[ ! -f "${MODEL_CONFIG}" ]]; then echo "Missing model config: ${MODEL_CONFIG}" >&2; exit 2; fi
DETECTED_MODEL_TYPE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("model_type", ""))' "${MODEL_CONFIG}")
MODEL_FAMILY=${MODEL_FAMILY:-${DETECTED_MODEL_TYPE}}
case "${MODEL_FAMILY}" in
  qwen2_5_vl) MODEL_SLUG=qwen2.5-vl ;;
  qwen3_vl) MODEL_SLUG=qwen3-vl ;;
  *) echo "Unsupported model_type '${MODEL_FAMILY}'" >&2; exit 2 ;;
esac
if [[ "${MODEL_FAMILY}" != "${DETECTED_MODEL_TYPE}" ]]; then
  echo "--model-family does not match config.json model_type '${DETECTED_MODEL_TYPE}'" >&2
  exit 2
fi

PROJECT_NAME=${PROJECT_NAME:-imgsurf-${MODEL_SLUG}}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo-${TOTAL_GPUS}gpu-${NNODES}node}
VSTAR_DATA=${DATA_ROOT}/data_0.1.2_visual_toolbox_v2.parquet
MATH_DATA=${DATA_ROOT}/data_thinklite_reasoning_acc.parquet
CHART_DATA=${DATA_ROOT}/data_v0.8_visual_toolbox_v2.parquet
for required_path in "${VSTAR_DATA}" "${MATH_DATA}"; do
  if [[ ! -f "${required_path}" ]]; then echo "Missing required file: ${required_path}" >&2; exit 2; fi
done
if [[ "${FULL_DATASET}" == 1 ]]; then
  if [[ ! -f "${CHART_DATA}" ]]; then echo "Missing chart subset: ${CHART_DATA}" >&2; exit 2; fi
  TRAIN_FILES="[\"${VSTAR_DATA}\",\"${MATH_DATA}\",\"${CHART_DATA}\"]"
else
  TRAIN_FILES="[\"${VSTAR_DATA}\",\"${MATH_DATA}\"]"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
fi
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG
if [[ -n "${RAY_ADDRESS}" ]]; then export RAY_ADDRESS; fi

export IMGSURF_REWARD_TOKEN IMGSURF_SEMANTIC_REWARD IMGSURF_CONSISTENCY_REWARD
export IMGSURF_K IMGSURF_IOU_THR IMGSURF_MAX_ITER IMGSURF_MAX_LEVEL IMGSURF_EXPAND_MODE
export IMGSURF_MAX_PROMPT_LENGTH IMGSURF_MAX_RESPONSE_LENGTH IMGSURF_MAX_MODEL_LEN
export IMGSURF_MAX_BATCHED_TOKENS IMGSURF_MAX_TURN_TOKENS IMGSURF_MIN_FINAL_TOKENS
export IMGSURF_MAX_THINK_SUMMARY_TOKENS IMGSURF_MAX_ASSISTANT_TURNS IMGSURF_MAX_USER_TURNS
export IMGSURF_MAX_INPUT_PIXELS IMGSURF_MIN_TOOL_PIXELS IMGSURF_MAX_TOOL_PIXELS
export IMGSURF_ACCURACY_WEIGHT IMGSURF_FORMAT_WEIGHT IMGSURF_TOOL_WEIGHT
export IMGSURF_CONSISTENCY_WEIGHT IMGSURF_INVALID_TOOL_WEIGHT IMGSURF_EXCESS_TOOL_WEIGHT
export IMGSURF_JUDGE_BASE_URL IMGSURF_JUDGE_MODEL IMGSURF_JUDGE_API_KEY
export IMGSURF_JUDGE_TIMEOUT IMGSURF_JUDGE_MAX_RETRIES IMGSURF_JUDGE_MAX_PIXELS

VISIBLE_GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPU_COUNT}" -ne "${GPUS_PER_NODE}" ]]; then
  echo "Expected ${GPUS_PER_NODE} visible GPUs, PyTorch reports ${VISIBLE_GPU_COUNT}" >&2
  exit 2
fi
if ((NNODES > 1)) && [[ -z "${RAY_ADDRESS}" ]]; then
  echo "--nnodes=${NNODES} requires a running Ray cluster and --ray-address" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/ckpts" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/tensorboard"
echo "ImgSurf: model=${MODEL_FAMILY}, reward_token=${IMGSURF_REWARD_TOKEN}, semantic_reward=${IMGSURF_SEMANTIC_REWARD}, consistency=${IMGSURF_CONSISTENCY_REWARD}"
echo "ImgSurf v4: k=${IMGSURF_K}, iou_thr=${IMGSURF_IOU_THR}, max_iter=${IMGSURF_MAX_ITER}, max_level=${IMGSURF_MAX_LEVEL}, expand=${IMGSURF_EXPAND_MODE}"
echo "ImgSurf context: prompt=${IMGSURF_MAX_PROMPT_LENGTH}, response=${IMGSURF_MAX_RESPONSE_LENGTH}, model=${IMGSURF_MAX_MODEL_LEN}, batched=${IMGSURF_MAX_BATCHED_TOKENS}"
echo "ImgSurf resources: total_gpus=${TOTAL_GPUS}, nnodes=${NNODES}, gpus_per_node=${GPUS_PER_NODE}, rollout_tp=${ROLLOUT_TP_SIZE}"

python -m verl.trainer.main_ppo \
  --config-path="${CONFIG_DIR}" \
  --config-name="${CONFIG_NAME}" \
  "data.train_files=${TRAIN_FILES}" \
  "data.val_files=[\"${VSTAR_DATA}\"]" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${IMGSURF_MAX_PROMPT_LENGTH} \
  data.max_response_length=${IMGSURF_MAX_RESPONSE_LENGTH} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
  actor_rollout_ref.rollout.max_model_len=${IMGSURF_MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${IMGSURF_MAX_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS} \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${IMGSURF_MAX_ASSISTANT_TURNS} \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=${IMGSURF_MAX_USER_TURNS} \
  trainer.n_gpus_per_node=${GPUS_PER_NODE} \
  trainer.nnodes=${NNODES} \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_ROOT}/ckpts/${EXPERIMENT_NAME}" \
  "+trainer.tensorboard_dir=${OUTPUT_ROOT}/tensorboard/${EXPERIMENT_NAME}" \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.resume_mode=${RESUME_MODE} \
  "${HYDRA_OVERRIDES[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/${EXPERIMENT_NAME}.log"
