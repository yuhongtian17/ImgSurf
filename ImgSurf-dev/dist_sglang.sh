#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct/}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-8b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18938}"
API_KEY="${API_KEY:-replace-with-a-long-random-key}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --tp-size "${TP_SIZE}" \
  --dtype bfloat16 \
  --context-length "${MAX_MODEL_LEN}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}"
