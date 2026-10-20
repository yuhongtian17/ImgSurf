#!/usr/bin/env bash
MODEL_PATH="/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-7B/"
MODEL_NAME="deepeyes-7b"

vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port 18917 \
    --api-key "replace-with-a-long-random-key" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768
