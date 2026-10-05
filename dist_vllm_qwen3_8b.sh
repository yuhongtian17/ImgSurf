#!/usr/bin/env bash
MODEL_PATH="/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct/"
MODEL_NAME="qwen3-vl-8b"

vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port 18938 \
    --api-key "replace-with-a-long-random-key" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768
