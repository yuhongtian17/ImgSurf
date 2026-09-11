#!/usr/bin/env bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --model-path=/home/dataset-assist-0/workspace/deepeyes271/Qwen2.5-VL-7B-Instruct \
  --output-root=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen25_7b \
  --total-gpus=8 \
  --rollout-tp-size=4 \
  --rollout-gpu-memory-utilization=0.50 \
  --rollout-n=8 \
  --agent-workers=1 \
  --max-input-pixels=4194304 \
  --full-dataset=on \
  --total-epochs=1 \
  "$@"
