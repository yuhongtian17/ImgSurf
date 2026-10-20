# ImgSurf

## Installation Step by Step for Evaluation

Yes indeed, it depends on [PyTorch](https://pytorch.org/), [DeepEyes](https://github.com/Visual-Agent/DeepEyes), [sCoT](https://github.com/YWenxi/think-with-images-through-self-calling):

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

# Download models
hf download "Qwen/Qwen2.5-VL-7B-Instruct"
hf download "Qwen/Qwen3-VL-4B-Instruct"
hf download "Qwen/Qwen3-VL-8B-Instruct"
hf download "ChenShawn/DeepEyes-7B"
hf download "ywenxi/SubagentVL-7B-Fine-Chart-80"

# Download datasets
hf download "craigwu/vstar_bench" --repo-type dataset
hf download "DreamMr/HR-Bench" --repo-type dataset
hf download "ChenShawn/DeepEyes-Datasets-47k" --repo-type dataset

hf download "yifanzhang114/MME-RealWorld" --repo-type dataset --local-dir "mme_real"
hf download "yifanzhang114/MME-RealWorld-Lite" --repo-type dataset --local-dir "mme_real_lite"
hf download "AI4Math/MathVista" --repo-type dataset --local-dir "mathvista_mini"
hf download "AI4Math/MathVerse" --repo-type dataset --local-dir "mathverse_mini"
hf download "MathLLMs/MathVision" --repo-type dataset --local-dir "mathvision_mini"
hf download "We-Math/We-Math" --repo-type dataset --local-dir "wemath_overall"
hf download "DynaMath/DynaMath_Sample" --repo-type dataset --local-dir "dynamath_overall"

git clone https://github.com/Yijia-Xiao/LogicVista.git "logicvista_overall"

wget -c "https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip" -O "refcoco/refcoco.zip"
wget -c "https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip" -O "refcoco/refcoco+.zip"
wget -c "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip" -O "refcoco/refcocog.zip"
wget -c "http://images.cocodataset.org/zips/train2014.zip" -O "refcoco/train2014.zip"
wget -c "http://images.cocodataset.org/zips/val2014.zip" -O "refcoco/val2014.zip"
wget -c "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" -O "refcoco/annotations_trainval2014.zip"

# ref: https://github.com/JIA-Lab-research/LISA#dataset
git clone https://github.com/dvlab-research/LISA.git "reasonseg/LISA"

hf download "echo840/OCRBench" --repo-type dataset --local-dir "ocrbench"
hf download "lmms-lab-encoder/POPE" --repo-type dataset --local-dir "pope"
```

## Evaluation for Qwen2.5-VL or DeepEyes

```
MODEL_PATH="/home/dataset-assist-0/workspace/deepeyes271/Qwen2.5-VL-7B-Instruct/"
MODEL_NAME="qwen25-vl-7b"

vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port 18927 \
    --api-key "replace-with-a-long-random-key" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768

VSTAR_BENCH_PATH="/home/dataset-assist-0/workspace/deepeyes271/vstar_bench/"
HRBENCH_PATH="/home/dataset-assist-0/workspace/deepeyes271/HR-Bench/"
WORK_DIRS="/home/dataset-assist-0/workspace/deepeyes271/work_dirs/"
MODEL_NAME="qwen25-vl-7b"
MODEL_VER="v0"

# for vstar
python "./eval/deepeyes/eval_vstar_v0.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18927/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8 \
    --qwen_ver 2
python "./eval/deepeyes/judge_result_vstar.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18927/v1" \
    --vstar_bench_path "${VSTAR_BENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8

# for hrbench
python "./eval/deepeyes/eval_hrbench_v0.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18927/v1" \
    --hrbench_path "${HRBENCH_PATH}" \
    --save_path "${WORK_DIRS}${MODEL_VER}/" \
    --num_workers 8 \
    --qwen_ver 2
python "./eval/deepeyes/judge_result_hrbench.py" \
    --model_name "${MODEL_NAME}" \
    --eval_model_name "${MODEL_NAME}" \
    --api_key "replace-with-a-long-random-key" \
    --api_url "http://127.0.0.1:18927/v1" \
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
MODEL_NAME="qwen3-vl-4b"
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

cd /home/dataset-assist-0/workspace/deepeyes271/
git clone https://github.com/YWenxi/think-with-images-through-self-calling.git "sCoT-main"
cp -r ImgSurf-dev/* sCoT-main/
cd sCoT-main/

mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/ray/"
ln -s /home/dataset-assist-0/workspace/deepeyes271/work_dirs/ray/ /tmp/
ray stop --force

mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen25_7b/ckpts/"
mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen25_7b/logs/"
bash train8_qwen25_7b.sh

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen25_7b/ckpts/grpo-8gpu-1node/global_step_320/actor/" \
    --target_dir "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen25_7b/merged/"

mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_4b/ckpts/"
mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_4b/logs/"
bash train8_qwen3_4b.sh

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_4b/ckpts/grpo-8gpu-1node/global_step_320/actor/" \
    --target_dir "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_4b/merged/"

mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_8b/ckpts/"
mkdir -p "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_8b/logs/"
bash train8_qwen3_8b.sh

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_8b/ckpts/grpo-8gpu-1node/global_step_320/actor/" \
    --target_dir "/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_8b/merged/"
```

# Install Hugging Face with Windows Powershell

```
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

