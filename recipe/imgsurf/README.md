# ImgSurf v4-style GRPO（Qwen2.5-VL / Qwen3-VL）

本目录是 `sCoT-main` 的差分覆盖层。请只在 `ImgSurf-dev` 中维护这些文件，训练前复制到同结构的
sCoT 仓库：

```bash
cd /home/dataset-assist-0/workspace
cp -r ./ImgSurf-dev/* ./sCoT-main/
cd ./sCoT-main
```

当前实现把 `eval/deepeyes/eval_hrbench_v4.py` 的主要轨迹变成可重放的 GRPO 多轮序列，训练时使用与
`eval_*_v4.py` 对齐的 system/tool XML、初始请求、`I{n}` zoom-out 观察、`R{n}` refinement 请求和
最终 `<tool_response>` 裁剪图请求：

1. 模型看初始图，决定是否输出 `image_zoom_in_tool`；
2. 工具保留原图，按 `k` 和 `expand_mode` 产生扩展区域；
3. 轨迹写入 v4 的合成 `image_zoom_out_tool` 消息和扩展裁剪；
4. 同一策略在裁剪上重新输出 `bbox_2d` 与 `label`；
5. 新旧区域 IoU 大于 `iou_thr` 时收敛，否则在 `max_iter * max_level` 次内继续；
6. 模型读取最终裁剪和最近两段思考，输出唯一的 `<answer>...</answer>`。

工具始终使用数据集原图做坐标映射，不会在已经缩小的初始输入图上重复裁剪。训练会在渲染初始
多模态 prompt 后检查 token 数；超过 `max_prompt_length` 会立即报错，避免把超长 prompt 静默送入 rollout。

## 训练环境

已验证的目标环境为 Python 3.12、CUDA 12.8、PyTorch 2.9.1、Transformers 4.57.1、
SGLang 0.5.6.post2、Ray 2.50.1、qwen-vl-utils 0.0.14。启动脚本不再执行固定版本检查器；依赖版本由
训练环境本身负责。建议先执行：

```bash
python -c "import torch, transformers, ray, sglang; print(torch.__version__, transformers.__version__, ray.__version__, sglang.__version__)"
```

训练入口会继续检查模型类型、数据文件、GPU 数量、batch 整除关系和上下文长度等与本次任务直接相关的
条件。

## 默认启动

八卡 Qwen3-VL-8B：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --model-path=/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct \
  --data-root=/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k \
  --output-root=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_8b \
  --total-gpus=8
```

所有启动配置现在都有命令行参数。`--name=value` 与 `--name value` 两种形式均可；用
`bash ./recipe/imgsurf/run_imgsurf_grpo.sh --help` 查看按路径、资源、训练、轨迹、预算、奖励和 judge
分类后的完整参数表。原有大写环境变量仍兼容，但仅作为次级配置来源。

## 两种 token 训练模式

默认训练初始定位、内层重定位和最终回答的所有策略生成 token：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh --reward-token all
```

只训练外层（初始定位与最终回答）时，内层重定位仍由当前策略生成并留在完整轨迹中，但其
`response_mask=0`，因此不进入 PPO loss：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh --reward-token outer
```

合成的 zoom-out 调用、用户消息、图像观察等环境 token 在两种模式下都不参与 loss。

## 两种语义奖励

默认 `rule` 使用保守的确定性规则：严格选项匹配、Yes/No、数值、方向关系（含主客体顺序）、短答案
以及高阈值 token/order 匹配。规则无法确认时记为错误，不会把高词汇重叠自动当成正确：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh --semantic-reward rule
```

`judge` 把问题、参考答案、候选答案以及压缩后的原图发送到 OpenAI-compatible VLM。judge 应部署在
训练 Ray 集群之外，以免占用八张训练卡：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --semantic-reward=judge \
  --judge-base-url=http://judge-host:18901/v1 \
  --judge-model=Qwen3-VL-8B-Instruct \
  --judge-api-key=EMPTY
```

可选参数：`--judge-timeout`（默认 60 秒）、`--judge-max-retries`（默认 2）、
`--judge-max-pixels`（默认 1,048,576）。judge 请求失败时该样本 accuracy 为 0，并记录警告，不会退回
较宽松的规则而造成奖励口径漂移。真实 API key 用命令行传入时会留在 shell 历史中；敏感环境下可继续
使用兼容的 `IMGSURF_JUDGE_API_KEY` 环境变量。

## 格式、工具与一致性奖励

每次策略 response 单独计算格式分数，合法为 `1`，非法为 `0`，最后对所有 response 取平均。初始
response 必须是 `<think>...</think>` 加一个工具调用或最终回答；每个 refinement response 必须是
`<think>...</think><tool_call>...</tool_call>`；最终 response 必须是
`<think>...</think><answer>...</answer>`。标签外文本、缺失标签、空思考/答案和超过该轮 token 上限都会使
该 response 的格式分数为 `0`。

每个 `image_zoom_in_tool` 调用单独计算工具分数，合法为 `1`，非法为 `0`，对所有调用取平均。合法调用
必须同时满足以下条件：

- `<tool_call>` 内是合法 JSON；
- 函数名是 `image_zoom_in_tool`；
- `bbox_2d` 是四个有限数，满足 `x1 < x2`、`y1 < y2` 且位于当前坐标范围；
- `label` 是非空字符串。

没有必须调用工具的直接回答不额外扣工具分；需要 refinement 但没有合法 `<tool_call>` 的 response
会贡献一个 `0`。没有任何 `image_zoom_in_tool` 调用时工具平均值定义为 `0`，只是不产生工具加分。
工具奖励不评价 bbox 与目标的重叠质量，bbox 内容由最终任务准确率和一致性信号间接约束。

默认开启一致性奖励：只要合法的迭代轨迹在次数上限内达到 `IoU > iou_thr`，就增加奖励。关闭方式：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh --consistency-reward off
```

一致性分数在合法 refinement 最终达到 `IoU > iou_thr` 时为 `1`，否则为 `0`。定位循环结束时，
根据展平后的 `k_list` 轮次编号 `iter_idx` 计算正向轮次奖励。令
`len_k_list = max_iter * max_level`，则 `iter_idx=0,1,...,len_k_list-1`：

```text
reward_iter = (len_k_list - 1 - iter_idx) / max(1, len_k_list - 1)
```

因此越早达到一致性、或越早结束循环，奖励越高；达到最后一轮时为 `0`。没有解析到初始
`image_zoom_in_tool` 区域、没有进入定位循环时，`reward_iter=0`。
当 `len_k_list=1` 时分母按 `1` 处理，因此唯一轮次的奖励为 `0`。

当前总奖励为：

```text
R = accuracy_weight * accuracy
  + format_weight * mean(response_format_scores)
  + tool_weight * mean(tool_validity_scores)
  + consistency_weight * consistency
  + iteration_weight * reward_iter
```

默认权重为 `accuracy=1.0`、`format=0.10`、`tool=0.10`、`consistency=0.10`、
`iteration=0.05`。准确率仍是主信号；格式、工具和轮次分数都属于 `[0,1]`，不再使用负向轮次
惩罚。权重可通过 `--accuracy-weight`、`--format-weight`、`--tool-weight`、
`--consistency-weight` 和 `--iteration-weight` 调整。

## v4 超参数

以下参数均可直接在启动命令中指定：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --k 0.4 \
  --iou-thr 0.5 \
  --max-iter 4 \
  --max-level 1 \
  --expand-mode quarter
```

它们分别对应 v4 的 `--k`、`--iou_thr`、`--max_iter`、`--max_level` 和 `--expand_mode`。

## 坐标系

- Qwen3-VL：初始图和每张扩展裁剪的 `zoom_in` 坐标均归一化到 `[0,1000]`；
- Qwen2.5-VL：`zoom_in` 使用当前实际展示图的像素范围，x 属于 `[0,w]`，y 属于 `[0,h]`；
- 两种模型最终都映射回原图坐标做扩展、IoU 和最终裁剪；
- 模型族由 checkpoint 的 `config.json:model_type` 检测，脚本会拒绝不一致的手工设置。

## token 与上下文预算

初始 prompt 超过 `max_prompt_length` 会直接报错，不做静默截断。整条多轮轨迹受
`max_response_length` 约束，每次生成还受 `max_turn_tokens` 约束，并预留 `min_final_tokens` 给最终答案。
当剩余预算无法容纳下一张 refinement 图时，轨迹会提前停止迭代，保留当前区域作为最终裁剪；最终图
必要时可继续降采样，绝不会超过设定的总上下文。

默认值：

```text
max_prompt_length   = 8192
max_response_length = 8192
max_model_len       = 16384
max_batched_tokens  = 8192
max_turn_tokens     = 1024
min_final_tokens    = 512
```

修改示例：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --max-prompt-length 8192 \
  --max-response-length 9216 \
  --max-model-len 17408 \
  --max-batched-tokens 8192 \
  --max-turn-tokens 768 \
  --min-final-tokens 512
```

脚本要求 `max_model_len >= max_prompt_length + max_response_length`。

## 参数优先级

从高到低为：

1. 命令末尾不带 `--` 的原生 Hydra override；
2. `run_imgsurf_grpo.sh` 的 `--kebab-case` 命令行参数；
3. 调用前显式设置的兼容环境变量；
4. `run_imgsurf_grpo.sh` 中的默认值；
5. `configs/*.yaml` 的兜底默认值；
6. verl 上游默认配置。

例如：

```bash
IMGSURF_MAX_ITER=3 bash ./recipe/imgsurf/run_imgsurf_grpo.sh --max-iter 5
```

实际使用 5，说明新参数会覆盖旧环境变量。对 verl 原生字段仍可在末尾传 Hydra override，例如：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --rollout-n=4 \
  actor_rollout_ref.rollout.n=8
```

实际使用 8。

## 资源与 smoke test

常用资源参数包括 `--total-gpus`（默认 8）、`--nnodes`（默认 1）、`--gpus-per-node`、
`--rollout-tp-size`（默认 1）、`--train-batch-size`（默认每卡 4 prompt）、
`--ppo-mini-batch-size`、`--rollout-n`、`--agent-workers`、`--full-dataset` 和
`--max-input-pixels`。多节点还可指定 `--ray-address`；本机卡号可用 `--cuda-visible-devices` 限制。

首次建议：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --full-dataset=off \
  --total-gpus=2 \
  --train-batch-size=8 \
  --ppo-mini-batch-size=8 \
  --rollout-n=2 \
  --save-freq=1 \
  --total-epochs=1 \
  --model-path=/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct \
  --data-root=/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k \
  --output-root=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf-smoke \
  --max-iter=1 \
  --max-level=1
```

四个 `train_*.sh` 也已经改成纯命令行参数包装；其末尾保留 `"$@"`，所以可临时覆盖包装脚本中的值，
例如 `bash train_qwen3_8b.sh --rollout-n=4 --max-iter=2`。若仍使用旧写法，环境变量名称与新参数的
对应关系遵循 `TOTAL_GPUS` → `--total-gpus`、`IMGSURF_MAX_INPUT_PIXELS` →
`--max-input-pixels` 这类规则，具体映射以 `--help` 为准。

训练保存的是 verl/FSDP actor checkpoint。使用 v4 评测前，仍需按 sCoT/verl 的 checkpoint merge
流程导出为 Hugging Face 权重。
