# ImgSurf 修订记录

## 2026-10-13

- 统一 `reward_accuracy` 的两种 judge 路径：配置 `--judge-base-url` 时由外部 OpenAI-compatible VLM 判断；未配置时，在每条策略 rollout 完成后用当前 rollout 引擎执行独立的文本 judge。默认 judge 只接收问题、标准答案和模型答案，不发送原图、裁剪图或策略轨迹，judge token 不进入策略 `response_ids` 和 `response_mask`。
- 将每个策略 response 的格式分数按初始、内层和最终 response 分开计算并取平均，加入严格/非严格 tool-call、初始 direct-answer、严格 think 乘数，以及四元数、顺序、范围和面积检查。
- 将工具执行修复与格式奖励解耦：沿用 `screen_tool_call_regions` 的裁剪、逆序修正和越界修正继续 zoom loop，但原始几何不合法时格式奖励仍为 0。
- 明确 exploration、consistency 和 iteration 的答案错误衰减规则；默认 `k=0.4`、`max_iter=4`、`max_level=1`、`iou_thr=0.5` 及五项奖励权重均为 1.0。
- 保留 `--reward-token all|outer`：`all` 训练初始、内层和最终策略 token，`outer` 将内层生成 token 的 mask 设为 0。
- 保留 `--dataset-mode=1..7` 位掩码：bit_0 ThinkLite-VL、bit_1 ArxivQA、bit_2 Vstar；默认值 7。
- 修正启动参数优先级说明和传递顺序，使显式 `--[kwargs]` 高于脚本环境/默认值、YAML 默认值和函数默认值；`--total-iterations` 继续优先于 `--total-epochs`，并确保训练循环有足够 epoch 触达指定 step。
- 将 `--save-freq` 默认值改为 80，并把 ImgSurf 环境变量完整传入 Ray worker。
- 更新 0 号卡 Qwen3-VL-8B judge 服务及其余 7 卡训练示例，明确 7 卡的 GPU、TP、worker 与 batch 参数。
- 删除不再使用的 `eval/deepeyes_old` 覆盖目录，并清理本次检查生成的 Python 编译缓存。
