#!/usr/bin/env bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --model-path=/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-4B-Instruct \
  --output-root=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_2 \
  --total-gpus=2 \
  --rollout-tp-size=2 \
  --rollout-gpu-memory-utilization=0.50 \
  --rollout-n=8 \
  --train-batch-size=8 \
  --ppo-mini-batch-size=8 \
  --agent-workers=1 \
  --max-input-pixels=4194304 \
  --full-dataset=on \
  --total-epochs=1 \
  "$@"
