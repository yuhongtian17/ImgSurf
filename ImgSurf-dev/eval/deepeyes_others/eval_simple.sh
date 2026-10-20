#!/usr/bin/env bash
set -euo pipefail

# Run from the sCoT-main checkout after copying ImgSurf-dev into it, or set
# REPO_ROOT explicitly.  DATA_ROOT contains the downloaded benchmark folders;
# the default matches the shared server layout: benchmarks is next to sCoT-main.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../benchmarks}"
WORK_DIRS="${WORK_DIRS:-$PWD/work_dirs/deepeyes_others}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-8b}"
MODEL_VER="${MODEL_VER:-v2}"
API_KEY="${API_KEY:-replace-with-a-long-random-key}"
API_URL="${API_URL:-http://127.0.0.1:18938/v1}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-${MODEL_NAME}}"
QWEN_VER="${QWEN_VER:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATASET="${DATASET:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="${SPLIT:-}"
USE_VLM_JUDGE="${USE_VLM_JUDGE:-0}"
split_args=()
if [ -n "${SPLIT}" ]; then
  split_args+=(--split "${SPLIT}")
fi
judge_args=()
if [ "${USE_VLM_JUDGE}" = "1" ]; then
  judge_args+=(--use_vlm_judge)
fi

DATASETS=(
  mathvista_mini mathverse_mini mathverse_vision
  mathvision_mini wemath_overall dynamath_overall logicvista_overall ocrbench
  refcoco refcoco_plus refcocog reasonseg pope
)
if [ "${DATASET}" != "all" ]; then
  DATASETS=("${DATASET}")
fi

for dataset in "${DATASETS[@]}"; do
  case "${dataset}" in
    mme_real_cn) dataset_path="${DATA_ROOT}/mme_real" ;;
    mathverse_vision|mathverse_mini) dataset_path="${DATA_ROOT}/mathverse_mini" ;;
    refcoco_plus|refcocog) dataset_path="${DATA_ROOT}/refcoco" ;;
    *) dataset_path="${DATA_ROOT}/${dataset}" ;;
  esac
  eval_script="${REPO_ROOT}/eval/deepeyes_others/${dataset}/eval_${dataset}_${MODEL_VER}.py"
  judge_script="${REPO_ROOT}/eval/deepeyes_others/${dataset}/judge_result_${dataset}.py"
  if [ ! -d "${dataset_path}" ]; then
    echo "[skip] missing ${dataset_path}; run download.sh or set DATA_ROOT" >&2
    continue
  fi
  python "${eval_script}" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${EVAL_MODEL_NAME}" \
    --api_key "${API_KEY}" \
    --api_url "${API_URL}" \
    --dataset_path "${dataset_path}" \
    --save_path "${WORK_DIRS}" \
    --num_workers "${NUM_WORKERS}" \
    --max_samples "${MAX_SAMPLES}" \
    --qwen_ver "${QWEN_VER}" \
    "${split_args[@]}"
  python "${judge_script}" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${EVAL_MODEL_NAME}" \
    --api_key "${API_KEY}" \
    --api_url "${API_URL}" \
    --save_path "${WORK_DIRS}" \
    --prediction_file "${WORK_DIRS}/${MODEL_NAME}/${dataset}/predictions_${MODEL_VER}.jsonl" \
    "${judge_args[@]}"
done
