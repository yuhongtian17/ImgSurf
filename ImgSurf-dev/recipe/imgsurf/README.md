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
`bash ./recipe/imgsurf/run_imgsurf_grpo.sh --help` 查看按路径、资源、训练、轨迹、预算和奖励分类的完整
参数表。对应的大写环境变量可作为次级配置来源。

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

## accuracy judge 两种模式

指定 `--judge-base-url` 时，CPU reward manager 调用外部 OpenAI-compatible VLM。可同时指定
`--judge-model`、`--judge-api-key`、`--judge-timeout`、`--judge-max-retries` 和
`--judge-max-pixels`。外部 judge 使用独立对话，不携带策略轨迹；配置了 URL 时数据集才会编码并发送
原图。

不指定 `--judge-base-url` 时，每条 rollout 执行“策略 inference（包含 zoom loop）→ 独立 judge → 返回
奖励 metadata”。judge 使用当前 rollout 引擎和当前模型快照，只发送问题、标准答案和模型回答，不发送
原图或策略轨迹。judge 的 prompt/output token 是临时局部变量，不追加到策略 `response_ids`，也不写入
`response_mask`。完成本批 rollout 后训练器才执行 actor update，下一批 rollout 使用更新后的权重。

两种模式的 judge 都只返回 `CORRECT`/`INCORRECT`；请求失败、输出无法识别或最终 answer 不存在时，
`reward_accuracy=0`，不会退回规则匹配。

当 0 号卡专门部署外部 judge、其余 7 卡训练时，先在 0 号卡启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path=/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct \
  --served-model-name=Qwen3-VL-8B-Instruct \
  --host=127.0.0.1 \
  --tp-size=1 \
  --context-length=32768 \
  --port=18901 \
  --api-key=EMPTY \
  --trust-remote-code \
  --mem-fraction-static 0.85
```

另开终端确认服务可用：

```bash
curl http://127.0.0.1:18901/v1/models
```

然后只让 1~7 号卡参与训练。7 卡时应显式把总 GPU、每节点 GPU、rollout worker 和 batch 参数改为
7 卡对应值：

```bash
ray stop --force
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --model-path=/home/dataset-assist-0/workspace/deepeyes271/Qwen3-VL-8B-Instruct \
  --data-root=/home/dataset-assist-0/workspace/deepeyes271/DeepEyes-Datasets-47k \
  --output-root=/home/dataset-assist-0/workspace/deepeyes271/work_dirs/imgsurf_qwen3_8b_7train \
  --cuda-visible-devices=1,2,3,4,5,6,7 \
  --total-gpus=7 \
  --gpus-per-node=7 \
  --rollout-tp-size=1 \
  --agent-workers=7 \
  --train-batch-size=56 \
  --ppo-mini-batch-size=28 \
  --judge-base-url=http://127.0.0.1:18901/v1 \
  --judge-model=Qwen3-VL-8B-Instruct \
  --judge-api-key=EMPTY
```

这里 judge 服务独占物理 0 号卡，训练进程只看到物理 1~7 号卡；`56=7×8`、`28=7×4`，满足
launcher 对全局 batch 与 GPU 数量整除关系的检查。

## 格式、工具与一致性奖励

每次策略 response 单独计算格式分数，最后对所有 response 取平均。基本轨迹规则为：第一次 response
至少包含一个可识别的 `tool_call` 或 `answer`；最后一次 response 必须包含严格闭合的
`<answer>...</answer>`；中间 response 必须包含可被 `screen_tool_call_regions` 解析的
`tool_call`。格式奖励不再强制要求 `<think>...</think>`。

工具调用沿用 `screen_tool_call_regions` 的四种解析情况。严格闭合的
`<tool_call>...</tool_call>` 得到 `--reward_strict_tool`（默认 `1.0`），其余三种非严格闭合但仍能解析出合法区域的形式得到
`--reward_loose_tool`（默认 `0.9`），无法解析则为 `0`。bbox 只校验四元数、有限性、非负、坐标顺序
和当前坐标范围，不评价区域内容质量。第一次 response 如果工具解析失败但 answer 严格闭合，得到
`--reward_direct_answer`（默认 `0.8`）；中间 response 只接受工具；最后 response 只接受严格闭合的
answer。严格闭合的 `<think>...</think>` 会再乘以 `--reward_strict_think`（默认 `1.1`），没有严格
think 时乘数为 `1.0`。

对应环境变量为 `IMGSURF_REWARD_STRICT_TOOL`、`IMGSURF_REWARD_LOOSE_TOOL`、
`IMGSURF_REWARD_DIRECT_ANSWER` 和 `IMGSURF_REWARD_STRICT_THINK`。训练提示词仍显式要求 think，并继续使用与 `eval_hrbench_v4.py`
对齐的推理、zoom-out、refinement 和最终回答提示词。

`reward_exploration` 表示第一次 response 是否解析到了合法的初始 `image_zoom_in_tool` 区域。解析到
时基础值为 `1`，答案错误时乘以 `--inaccuracy_exploration`（默认 `0.5`），答案正确时乘以 `1`。

`reward_consistency` 表示 refinement 最终是否达到 `IoU > iou_thr`。达到时基础值为 `1`，答案错误时
乘以 `--inaccuracy_consistency`（默认 `0`），答案正确时乘以 `1`。

只要第一次 response 解析出合法初始区域，就进入 zoom loop。令
`len_k_list = max_iter * max_level`，根据跳出循环时的 `iter_idx=0,1,...,len_k_list-1` 计算基础值：

```text
reward_iteration = max(0, len_k_list - 1 - iter_idx) / max(1, len_k_list - 1)
```

如果没有合法初始区域，`reward_iteration` 直接为 `0`；否则基础值再按答案正确性乘以 `1` 或
`--inaccuracy_iteration`（默认 `0`）。当 `len_k_list=1` 时，iteration 基础值为 `0`。

当前总奖励为：

```text
R = accuracy_weight * reward_accuracy
  + format_weight * reward_format
  + consistency_weight * reward_consistency
  + exploration_weight * reward_exploration
  + iteration_weight * reward_iteration
```

默认权重为 `accuracy=1.0`、`format=1.0`、`consistency=1.0`、`exploration=1.0`、`iteration=1.0`。
权重可通过
`--accuracy-weight`、`--format-weight`、`--consistency-weight`、`--exploration-weight` 和
`--iteration-weight` 调整。

五项默认权重相同，分别鼓励答案正确、格式合法、定位收敛、使用有效初始工具以及尽早完成定位循环。
建议先使用默认组，观察 reward_accuracy、reward_format、reward_exploration、reward_consistency 和
reward_iteration 的独立日志后再调整。

初始 response 含有工具调用时，即使同时包含一个临时 answer，也仍会进入 v4 的 zoom loop，最后由最终
response 重新回答；初始只有 answer 时才跳过定位。训练提示词、坐标系和循环参数与 `eval_*_v4.py`
保持一致：`max_iter=4`、`max_level=1`、`k=0.4`、`iou_thr=0.5`。

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
- 图像缩放沿用 v4 的 `IMAGE_FACTOR`（Qwen3 为 32、Qwen2.5 为 28）和最小 2 patch；训练像素预算
  仍由 `--max-input-pixels`/`--max-tool-pixels` 控制，避免无意中突破上下文预算；
- 模型族由 checkpoint 的 `config.json:model_type` 检测，脚本会拒绝不一致的手工设置。

Qwen3-VL 的初始请求和每次 refinement 用户消息都会再次明确写出“使用 `[0,1000]` 归一化
坐标”；Qwen2.5-VL 保持 v4 的像素坐标提示。这样不会依赖模型是否把外部
`image_zoom_in_tool` 当作原生工具。

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
save_freq           = 80
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

1. 执行 `run_imgsurf_grpo.sh` 时传入的 `--[kwargs]` 命令行参数；
2. `run_imgsurf_grpo.sh` 内 `[KWARGS]` 变量的环境值或脚本默认值；
3. `configs/*.yaml` 的兜底默认值；
4. 相关函数参数列表的默认值。

末尾不带 `--` 的原生 Hydra override 仍可用于上游字段，但当它与本启动器参数冲突时，启动器的显式
`--[kwargs]` 参数优先。这样可以在不修改脚本的情况下临时覆盖上游字段，同时不会意外覆盖已选资源和奖励参数。

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

若同时指定启动器参数 `--rollout-n=4`，实际使用 4；只有未提供对应启动器参数时才使用该 Hydra 值。

## 资源与 smoke test

常用资源参数包括 `--total-gpus`（默认 8）、`--nnodes`（默认 1）、`--gpus-per-node`、
`--rollout-tp-size`（默认 1）、`--train-batch-size`（默认每卡 8 prompt）、
`--ppo-mini-batch-size`（默认全局 prompt batch 的一半）、`--rollout-n`、`--agent-workers`、`--dataset-mode` 和
`--max-input-pixels`。多节点还可指定 `--ray-address`；本机卡号可用 `--cuda-visible-devices` 限制。

`--dataset-mode` 用 1 到 7 的三位二进制掩码选择训练子集。最低位（`1`）是
ThinkLite-VL，中间位（`2`）是 ArxivQA，最高位（`4`）是 Vstar；多个子集可以按位或组合。
例如 `--dataset-mode=6`（二进制 `110`）使用 Vstar 和 ArxivQA，不使用 ThinkLite-VL，默认值
`7` 使用三个子集。Vstar 数据文件同时作为固定验证集，因此即使训练模式不包含 Vstar，仍需保留
`data_0.1.2_visual_toolbox_v2.parquet`。

八卡默认全局 prompt batch 为 `64`（每卡 `8`）。`train_qwen3_8b.sh`、`train_qwen3_4b.sh` 和
`train_qwen25_7b.sh` 默认传入 `--total-iterations 320`；指定 `--total-iterations 320` 时，
launcher 通过 `trainer.total_training_steps=320` 在 320 个 global step 后停止；它优先于
`--total-epochs`。未指定时仍按 `--total-epochs` 运行。当 `rollout-n=8` 时，每个 global step
会生成全局 batch 的 8 个 rollout 样本，实际生成量还会乘以 rollout-n。
三个八卡模型包装脚本将 `agent-workers` 设为 `8`，与 sCoT 的每卡并行 rollout 配置相符；
两卡 `train_2.sh` 仍保留较小的 worker 配置用于 smoke test。

首次建议：

```bash
bash ./recipe/imgsurf/run_imgsurf_grpo.sh \
  --dataset-mode=5 \
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
