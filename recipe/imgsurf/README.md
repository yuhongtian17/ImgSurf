# ImgSurf：Qwen2.5-VL-7B / Qwen3-VL-8B 的可扩展 A100 GRPO 配方

本目录是面向 `SCAgent-main` 的增量覆盖层。它把 `eval_vstar_v4.py` 和
`eval_hrbench_v4.py` 的“初始选区 → 递归放大并重定位 → 阅读最终裁剪 → 回答”逻辑映射到
SCAgent 的参数共享工具型 rollout。一次训练任务选择一个基座模型；同一套代码和命令入口支持
`Qwen2.5-VL-7B-Instruct` 与 `Qwen3-VL-8B-Instruct`。

## 1. 推理环境：PyTorch 2.9.1 对应版本

CUDA 12.8 下固定为：

| 包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| torchaudio | 2.9.1+cu128 |
| vLLM | 0.16.0（基于 cu128 PyTorch 源码编译） |
| qwen-vl-utils | 0.0.14 |
| qwen-agent | 0.0.34 |
| openai | 2.6.1 |
| huggingface-hub | 0.36.0 |

`vLLM 0.16.0` 是最后一个严格依赖 PyTorch 2.9.1 的发布系列；`vLLM 0.17.0` 已切换到
PyTorch 2.10.0。该版本实际没有发布可下载的 cu128 Release wheel，PyPI wheel 也不是针对当前
cu128 ABI 构建，因此安装脚本会检出 `vllm@v0.16.0`，按官方 existing-PyTorch 流程针对
`torch==2.9.1+cu128` 本地编译。创建独立推理环境：

```bash
cd /home/dataset-assist-0/workspace/SCAgent-main
CONDA_ENV_NAME=qwen-vl-eval bash recipe/imgsurf/install_inference_env.sh
conda activate qwen-vl-eval
```

该 PyTorch 约束对应的 vLLM 分支存在已公开修复于后续版本的网络服务安全问题。因此服务必须继续
绑定 `127.0.0.1`，使用随机 API key，并由防火墙禁止公网和不可信客户端访问；只加载已核验的本地
Qwen 权重。若必须提供公网服务，应放弃 PyTorch 2.9.1 约束并升级到已修复的 vLLM/PyTorch 组合。

## 2. 训练环境：从纯净 conda 环境创建

训练环境不再克隆、修改或依赖推理环境。先覆盖代码，再运行安装脚本：

```bash
cd /home/dataset-assist-0/workspace
cp -r ./ImgSurf-dev/* ./SCAgent-main/
cd ./SCAgent-main

CONDA_ENV_NAME=imgsurf-rl bash recipe/imgsurf/install_imgsurf_env.sh
conda activate imgsurf-rl
```

若希望安装结束时同时检查本机可见卡数，例如八卡节点，可执行：

```bash
IMGSURF_INSTALL_EXPECTED_GPUS=8 \
CONDA_ENV_NAME=imgsurf-rl bash recipe/imgsurf/install_imgsurf_env.sh
```

脚本从 `conda create -n imgsurf-rl python=3.11` 开始，核心版本为：

- `torch==2.9.1`、`torchvision==0.24.1`、`torchaudio==2.9.1`（cu128）；
- `sglang==0.5.6.post2`、`sgl-kernel==0.3.19`、`flashinfer-python==0.5.3`；
- `transformers==4.57.1`、`flash-attn==2.8.3`；
- `ray[default]==2.50.1`、`qwen-vl-utils==0.0.14`；
- 其余直接依赖全部固定在仓库根目录的 `requirements_imgsurf.txt`。

本覆盖层同时更新了两处 SGLang ABI：`setup.py` 的 `sglang` extra 不再声明已经移除的
`srt,openai` extra，并将 rollout 所需的 `ReleaseMemoryOccupationReqInput`、
`ResumeMemoryOccupationReqInput`、`UpdateWeightsFromTensorReqInput` 改为从 0.5.6 的正式位置
`sglang.srt.managers.io_struct` 导入。原 SCAgent 从 `tokenizer_manager` 导入的写法只适用于旧版，
会产生 `cannot import name 'ReleaseMemoryOccupationReqInput'`。多机地址检测也从已经删除的 `get_ip()`
切换为 `get_local_ip_auto()`；Ray 异步 actor 的初始化补丁则同步到 0.5.6 的环境变量和内核版本检查，
仅跳过 Python 不允许在非主线程中执行的 signal handler 注册。

安装末尾和每次训练启动前都会执行 `recipe/imgsurf/check_imgsurf_runtime.py`，检查上述精确版本、
CUDA 12.8 编译运行时以及 SCAgent/SGLang 的关键导入边界。例如环境中若仍是 `ray==2.57.0`，
自检会在创建 Ray actor 前直接报告它与本配方锁定的 `2.50.1` 不一致。已有的混合环境不建议原地
降级；请按上面的命令重新创建 `imgsurf-rl`。仅调试自检脚本本身时可临时设置
`IMGSURF_SKIP_RUNTIME_CHECK=1`，正式训练不应跳过。

`vllm` 与 `qwen-agent` 不参与 GRPO 训练，也不会安装到 `imgsurf-rl`。SCAgent 自带的 verl
代码以 `pip install -e . --no-deps` 安装，避免其旧 extra 把环境降级到 PyTorch 2.7.1。

若安装中途失败，可在问题修复后用
`IMGSURF_REUSE_ENV=1 CONDA_ENV_NAME=imgsurf-rl bash recipe/imgsurf/install_imgsurf_env.sh`
继续；正常情况下已存在的同名环境会被拒绝，以防无意继承旧包。

训练本身无需额外下载 GitHub 仓库，只要求已经存在：

- `SCAgent-main`（应用本覆盖层后）；
- `DeepEyes-Datasets-47k` 的三个 parquet；
- 两个基座模型中本次要训练的一个本地 checkpoint。

`DeepEyes-main`、VStar Bench 和 HR-Bench 仅用于训练后评测，不是 GRPO 训练依赖。
只有在创建严格 cu128 的独立推理环境时，`install_inference_env.sh` 会额外浅克隆
`https://github.com/vllm-project/vllm.git` 的 `v0.16.0` tag 用于本地编译；Qwen-Agent 与
qwen-vl-utils 直接从 PyPI 安装，不需要克隆 Qwen GitHub 仓库。

## 3. 双模型适配

训练脚本读取 checkpoint 的 `config.json:model_type`，只接受：

- `qwen2_5_vl`：使用 SCAgent/verl 原生的 Qwen2.5-VL M-RoPE 路径；
- `qwen3_vl`：使用覆盖层新增的 `verl/models/transformers/qwen3_vl.py` M-RoPE 兼容路径。

两者共用数据适配器、tool schema、递归裁剪工具和奖励函数。为避免 SCAgent 旧 fused forward 与
Qwen3-VL/Transformers 4.57.1 的 ABI 差异，两个模型默认均关闭 remove-padding 和 verl fused
forward，底层注意力仍使用 FlashAttention 2。脚本会拒绝模型目录与 `MODEL_FAMILY` 不一致的配置。

## 4. v4 工具轨迹

一次外层动作与评测 v4 对齐如下：

1. 当前策略读取原图，输出归一化到 `[0,1000]` 的 `bbox_2d` 和 `label`；
2. `ImgSurfZoomTool` 将框映射回原图，以原图宽高的 0.4 为窗口做 quarter-mode 扩展；
3. 同一套正在训练的权重读取扩展裁剪，并输出相对该裁剪的归一化新框；
4. 最多迭代 4 次，新旧框 IoU 大于 0.5 时提前停止，返回最后框的裁剪；
5. 外层策略读取裁剪并输出唯一的 `<answer>...</answer>`。

内层 refiner 与 actor 参数共享，但内层生成属于环境执行，其 token 不直接进入 PPO loss；外层的
“是否调用、初始选区、读取工具结果并回答”进入策略梯度。这与 SCAgent 的 self-calling 语义一致。

输入原图和工具裁剪默认各限制为 4,194,304 像素。A100 40GB 若峰值 OOM，可先设置
`IMGSURF_MAX_INPUT_PIXELS=1048576`，再把 `ROLLOUT_N=2`、
`ROLLOUT_GPU_MEMORY_UTILIZATION=0.50`；A100 80GB 可先使用默认值。

## 5. 奖励函数

总奖励为：

```text
R = clip(
      accuracy
      + 0.10 * format
      + 0.10 * I(correct visual answer and valid tool)
      - 0.05 * invalid_tool_count
      - 0.02 * max(tool_count - 2, 0),
      -0.2, 1.2)
```

- `accuracy` 为 0/1，始终是主奖励；
- HR-Bench/chart 多选题按选项字母精确匹配；
- ThinkLite 数学题优先用 `math-verify==0.9.0` 做符号等价验证；
- VStar 开放答案使用大小写/标点、Yes/No、方向词、数值、短答案包含和 token-F1 规则；
- 规则不能确定时，只有配置了 `IMGSURF_JUDGE_BASE_URL` 才调用 OpenAI-compatible judge，未配置
  时保守记 0；
- `format` 仅在恰好一个合法 `<answer>`、标签平衡且答案不过长时为 +1，否则为 -1；
- 工具 bonus 只在视觉问题答对且工具调用有效时给出，避免“无意义调用工具”超过准确率收益；
- 错误工具调用和超过两次的外层调用受惩罚。

若启用 judge，应放在另一台受信服务器上，不占用本节点八张训练卡：

```bash
export IMGSURF_JUDGE_BASE_URL=http://judge-host:18901/v1
export IMGSURF_JUDGE_API_KEY=replace-me
export IMGSURF_JUDGE_MODEL=qwen3-vl-8b
```

## 6. GPU 如何分配给 rollout、奖励和权重更新

本配方使用 verl 的 `hybrid_engine=true` 共置模式，不静态保留一张“奖励 GPU”：

1. **轨迹生成阶段**：`TOTAL_GPUS` 张卡全部运行参数共享的 SGLang rollout engine，执行外层回答、
   `ImgSurfZoomTool` 和内层 refiner 推理；TP=1 时每张卡对应一个 rollout data-parallel replica；
2. **奖励阶段**：生成完毕后，`compute_score` 在 Ray driver 的 CPU 上解析轨迹并计算规则奖励。它不加载
   reward model，所以不需要 GPU；可选 LLM judge 通过 `IMGSURF_JUDGE_BASE_URL` 调用集群外服务；
3. **更新阶段**：rollout engine 进入 sleep、释放 KV cache/权重占用，随后同一组 `TOTAL_GPUS` 以 FSDP
   计算 old log-prob、GRPO advantage 并更新 actor；下一步再同步最新权重进入 rollout。

因此每一步至少有 GPU 在执行轨迹推理，但它与更新 GPU 是**分时复用**，不是固定切成 1+N−1。这对
2/4 卡节点尤其重要。若改用本地神经 reward model 或本地 LLM judge，建议把它部署在训练 Ray 集群之外；
例如物理机有 9 张卡时，让 judge 使用第 9 张，训练命令仍设置 `TOTAL_GPUS=8`。

## 7. 可扩展 GPU 参数

统一入口为 `recipe/imgsurf/run_imgsurf_grpo.sh`：

- `TOTAL_GPUS`：训练使用的总 GPU 数，默认 8；
- `NNODES`：Ray 节点数，默认 1；
- `GPUS_PER_NODE`：可省略，由 `TOTAL_GPUS / NNODES` 推导；若显式传入则必须一致；
- `ROLLOUT_TP_SIZE`：单个 SGLang engine 的张量并行度，默认 1，必须整除每节点 GPU 数；
- 默认全局 batch 为 `TOTAL_GPUS * 4`，保持原八卡配置每卡 4 个 prompt 的比例；
- `AGENT_WORKERS` 默认等于 `TOTAL_GPUS`，它们是 CPU Ray actor，不是额外 GPU。

入口脚本通过自身位置定位 `SCAgent-main`，把 Hydra `--config-path` 转换为绝对路径，并在加载配置前
切换到仓库根目录。因此可以从任意当前目录调用脚本，不会把 `recipe/imgsurf/configs` 错误解析到
`verl/trainer/recipe/imgsurf/configs` 下。

脚本会检查 batch 整除关系、本机 CUDA 可见卡数以及多机 Ray 的逐节点 GPU 资源。多机模式还会把同一
版本/API 自检任务固定调度到每个选中的节点，防止 head 环境正确、worker 环境版本漂移后才在训练中途
失败。2/4 卡 A100 40GB 更容易因高分辨率图片或 KV cache OOM，建议从文末 smoke 配置开始。

## 8. 单机训练命令

单机 2 卡 Qwen3-VL-8B：

```bash
cd /home/dataset-assist-0/workspace/SCAgent-main
conda activate imgsurf-rl

TOTAL_GPUS=2 \
MODEL_PATH=/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct \
DATA_ROOT=/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k \
OUTPUT_ROOT=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf-qwen3-2gpu \
FULL_DATASET=1 \
bash recipe/imgsurf/run_imgsurf_grpo.sh
```

单机 4 卡只需改为：

```bash
TOTAL_GPUS=4 \
MODEL_PATH=/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct \
DATA_ROOT=/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k \
OUTPUT_ROOT=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf-qwen3-4gpu \
FULL_DATASET=1 \
bash recipe/imgsurf/run_imgsurf_grpo.sh
```

单机 8 卡 Qwen2.5-VL-7B：

```bash
cd /home/dataset-assist-0/workspace/SCAgent-main
conda activate imgsurf-rl

TOTAL_GPUS=8 \
MODEL_PATH=/home/dataset-assist-0/workspace/deepeyes271/Qwen2.5-VL-7B-Instruct \
DATA_ROOT=/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k \
OUTPUT_ROOT=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf-qwen2_5 \
FULL_DATASET=1 \
bash recipe/imgsurf/run_imgsurf_grpo.sh
```

## 9. 两台服务器各 8 卡

两个节点必须使用相同的 conda 环境和代码，并能通过同一个绝对路径访问模型、parquet 和输出目录
（通常使用 NFS/共享存储）。两台机器之间需开放 Ray、NCCL 和 worker 端口。

在 head 节点执行，其中 `10.0.0.10` 替换为 head 的内网 IP：

```bash
conda activate imgsurf-rl
ray start --head \
  --node-ip-address=10.0.0.10 \
  --port=6379 \
  --num-gpus=8
```

在第二台 worker 节点执行：

```bash
conda activate imgsurf-rl
ray start --address=10.0.0.10:6379 --num-gpus=8
```

回到 head 节点启动训练。`RAY_ADDRESS=auto` 会让 `main_ppo.py` 连接现有集群，而不是创建单机 Ray：

```bash
cd /home/dataset-assist-0/workspace/SCAgent-main
conda activate imgsurf-rl

RAY_ADDRESS=auto \
TOTAL_GPUS=16 NNODES=2 \
MODEL_PATH=/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct \
DATA_ROOT=/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k \
OUTPUT_ROOT=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf-qwen3-16gpu \
FULL_DATASET=1 \
bash recipe/imgsurf/run_imgsurf_grpo.sh
```

若这是训练任务独占的 Ray 集群，训练结束后可分别在两台节点运行 `ray stop --force`；不要在其他任务
共享的 Ray 集群上执行该命令。

## 10. Smoke test

首次运行建议先验证一个低 I/O step（排除高分辨率 chart）：

```bash
FULL_DATASET=0 \
TOTAL_GPUS=2 TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=2 \
SAVE_FREQ=1 TOTAL_EPOCHS=1 \
MODEL_PATH=/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct \
DATA_ROOT=/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k \
OUTPUT_ROOT=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf-smoke \
bash recipe/imgsurf/run_imgsurf_grpo.sh
```

训练时不要另行启动 vLLM；所配置的 GPU 已由 FSDP actor 和内嵌 SGLang rollout 共同使用。训练后的
FSDP actor checkpoint 需按 SCAgent/verl 的 checkpoint merge 流程导出为 Hugging Face 权重，再在
独立的 `qwen-vl-eval` 环境中运行 v4 评测。
