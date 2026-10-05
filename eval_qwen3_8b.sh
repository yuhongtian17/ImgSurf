#!/usr/bin/env bash
MODEL_NAME="qwen3-vl-8b"

VSTAR_BENCH_PATH="/home/dataset-assist-0/workspace/deepeyes271/vstar_bench/"
HRBENCH_PATH="/home/dataset-assist-0/workspace/deepeyes271/HR-Bench/"
WORK_DIRS="/home/dataset-assist-0/workspace/deepeyes271/work_dirs/"

MODEL_VER="deepeyes_v0"

# for vstar
python "./eval/deepeyes/eval_vstar_v0.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8
python "./eval/deepeyes/judge_result_vstar.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8

# for hrbench
python "./eval/deepeyes/eval_hrbench_v0.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8
python "./eval/deepeyes/judge_result_hrbench.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

MODEL_VER="scot"

# for vstar
python "./eval/scot/eval_vstar_scot.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8
python "./eval/deepeyes/judge_result_vstar.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8

# for hrbench
python "./eval/scot/eval_hrbench_scot.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8
python "./eval/deepeyes/judge_result_hrbench.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

MODEL_VER="imgsurf_v2"

# for vstar
python "./eval/deepeyes/eval_vstar_v2.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8 \
    --k 0.4 \
    --max_iter 4 \
    --max_level 1
python "./eval/deepeyes/judge_result_vstar.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8

# for hrbench
python "./eval/deepeyes/eval_hrbench_v2.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8 \
    --k 0.4 \
    --max_iter 4 \
    --max_level 1
python "./eval/deepeyes/judge_result_hrbench.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18938/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8
