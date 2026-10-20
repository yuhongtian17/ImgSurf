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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" vllm serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --dtype bfloat16 \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-model-len "${MAX_MODEL_LEN}"
