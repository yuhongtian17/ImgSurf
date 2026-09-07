# ImgSurf

## Installation Step by Step for Evaluation

Yes indeed, it depends on [PyTorch](https://pytorch.org/), [DeepEyes](https://github.com/Visual-Agent/DeepEyes), [SCAgent](https://github.com/YWenxi/think-with-images-through-self-calling):

```shell
# ref: https://developer.nvidia.com/cuda-toolkit-archive
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run

# mv /usr/local/cuda-12.8/ /home/dataset-assist-0/workspace/envs/
# sudo ln -s /home/dataset-assist-0/workspace/envs/cuda-12.8/ /usr/local/

# Add CUDA path
echo "export PATH=/usr/local/cuda-12.8/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "" >> ~/.bashrc
cat ~/.bashrc
source ~/.bashrc
nvcc -V

# NO sudo when install anaconda
wget https://mirror.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
chmod +x ./Anaconda3-2024.10-1-Linux-x86_64.sh
./Anaconda3-2024.10-1-Linux-x86_64.sh

conda create -n qwen3ev python=3.12 -y

# mv /opt/conda/envs/qwen3ev/ /home/dataset-assist-0/workspace/envs/
# ln -s /home/dataset-assist-0/workspace/envs/qwen3ev/ /opt/conda/envs/

conda activate qwen3ev

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__)"

pip install vllm==0.16.0
pip install qwen-vl-utils==0.0.14 qwen-agent==0.0.34 transformers==4.57.1 huggingface_hub==0.36.2 openai pandas nvitop

# Download datasets
git clone https://huggingface.co/datasets/craigwu/vstar_bench
git clone https://huggingface.co/datasets/DreamMr/HR-Bench
git clone https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k

# Download models
hf download Qwen/Qwen2.5-VL-7B-Instruct
hf download Qwen/Qwen3-VL-4B-Instruct
hf download Qwen/Qwen3-VL-8B-Instruct
hf download ChenShawn/DeepEyes-7B
hf download ywenxi/SubagentVL-7B-Fine-Chart-80
```

## Evaluation for Qwen2.5-VL or DeepEyes

```
MODEL_PATH="/home/dataset-assist-0/workspace/deepeyes271/Qwen2.5-VL-7B-Instruct/"
MODEL_NAME="qwen25-vl-7b"

vllm serve "${MODEL_PATH}" \
    --port 18902 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --served-model-name "${MODEL_NAME}" \
    --trust-remote-code \
    --disable-log-requests

VSTAR_BENCH_PATH="/home/dataset-assist-0/workspace/deepeyes271/vstar_bench/"
HRBENCH_PATH="/home/dataset-assist-0/workspace/deepeyes271/HR-Bench/"
WORK_DIRS="/home/dataset-assist-0/workspace/deepeyes271/work_dirs/"
MODEL_VER="v0"

# for vstar
python "./eval/deepeyes/eval_vstar_v0.py" \
    --model_name "${MODEL_NAME}" \
    --api_url "http://127.0.0.1:18902/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8 \
    --qwen_ver 2
python "./eval/deepeyes/judge_result_vstar.py" \
    --model_name "${MODEL_NAME}" \
    --api_url "http://127.0.0.1:18902/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8

# for hrbench
python "./eval/deepeyes/eval_hrbench_v0.py" \
    --model_name "${MODEL_NAME}" \
    --api_url "http://127.0.0.1:18902/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8 \
    --qwen_ver 2
python ./eval/deepeyes/judge_result_hrbench.py \
    --model_name "${MODEL_NAME}" \
    --api_url "http://127.0.0.1:18902/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8
```

## Evaluation for Qwen3-VL or ImgSurf

```
MODEL_PATH="/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-4B-Instruct/"
MODEL_NAME="qwen3-vl-4b"

vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port 18934 \
    --api-key "replace-with-a-long-random-key" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768

VSTAR_BENCH_PATH="/home/dataset-assist-0/workspace/deepeyes271/vstar_bench/"
HRBENCH_PATH="/home/dataset-assist-0/workspace/deepeyes271/HR-Bench/"
WORK_DIRS="/home/dataset-assist-0/workspace/deepeyes271/work_dirs/"
MODEL_VER="v0"

# for vstar
python "./eval/deepeyes/eval_vstar_v0.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18934/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8
python "./eval/deepeyes/judge_result_vstar.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18934/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8

# for hrbench
python "./eval/deepeyes/eval_hrbench_v0.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18934/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8
python "./eval/deepeyes/judge_result_hrbench.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18934/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8
```

## Installation Step by Step for Reinforcement Learning

```
conda create -n qwen3tr python=3.12 -y

# mv /opt/conda/envs/qwen3tr/ /home/dataset-assist-0/workspace/envs/
# ln -s /home/dataset-assist-0/workspace/envs/qwen3tr/ /opt/conda/envs/

conda activate qwen3tr

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --resume-retries 100
python -c "import torch; print(torch.__version__)"

pip install qwen-vl-utils==0.0.14 qwen-agent==0.0.34 transformers==4.57.1 huggingface_hub==0.36.2 openai pandas nvitop
pip install scipy==1.17.1 numpy==1.26.4 ray==2.50.1 tensordict==0.9.1 torch-memory-saver==0.0.9
pip install cachetools codetiming hydra-core math-verify peft pybind11 pylatexenc ray[default] tensorboard torchdata wandb
pip install sglang==0.5.6.post2
pip install nvidia-cudnn-cu12==9.16.0.29 --no-deps
# pip install -r requirements_sglang.txt --index https://pypi.tuna.tsinghua.edu.cn/simple/
MAX_JOBS=8 pip install flash-attn==2.8.3.post1 --no-build-isolation

mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen25_7b/ckpts/"
mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen25_7b/logs/"

mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/ray/"
ln -s /home/dataset-assist-0/workspace/deepeyes271/work_dirs/ray/ /tmp/
ray stop --force
bash train_qwen25_7b.sh

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_8b/ckpts/grpo-8gpu-1node/global_step_1470/actor/" \
    --target_dir "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_8b/merged/"
```

