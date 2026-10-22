# M2 Transformer 窗口级增量执行

正式入口仍为 `projects/configs/OmniDrive/worldstream_m123_auto.py`，现继承
`worldstream_m2_window_auto.py`。M1 普通关联配置、模型/权重、检测/地图 Head、
Tokenizer 和 M3 Auto 均保持原配置；此前用户停止的长时评测不会恢复。

执行顺序：

1. 验证场景、时间、相机顺序、输入形状/步长、模型参数/缓冲区版本及数值策略。
2. 以完整像素比较建立真实变化掩码；M1 的失效区域只能增加重算范围。
3. 根据 Conv2d 的完整感受野计算 dirty patches，再传播窗口注意力依赖闭包。
4. 通过当前 block、实际输入布局和更新窗口数的成本/质量准入后，仅 gather dirty
   windows，执行原 EVA Attention/MLP，并将更新 scatter 到缓存窗口副本。
5. 拼回完整 BHWC 特征，再执行全局依赖部分和 Detection/Map，最后提交状态。

部署模型的 640×640 图像经 16×16 patch embedding 形成 40×40 tokens。
16×16 token 窗口需要补到 48×48，每相机有 9 个窗口；六相机为 54 个窗口/block。
只有 block 0、1 是全局注意力之前的独立窗口前缀，所以最多跳过 **108 个
block-window**。后续局部 block 不能越过 block 2 的全局依赖复用历史特征。
`actual_skipped_windows` 是 block-window 对，不是相机数，也不是跳过全部 EVA。

本次补齐的实现约束：

- padded positions 在计算后清零，缓存与原生 partition 的填充语义一致。
- 缓存独立持有输入和窗口输出；只在全部分配/校验成功后发布。
- 发布前检查真实内存大小，拒绝超预算；跟踪张量版本以检测外部原地修改。
- 零更新的单窗口输出不能别名历史缓存；训练/梯度模式不允许直接稀疏执行。
- 成本键包含内部归一化、FlashAttention softmax scale、fp16 设置、TF32、
  cuDNN、reduced precision 和 deterministic 策略。非法成本、错 block 以及字符串
  伪布尔质量标记不能取得准入。
- benchmark 使用只读 hooks，避免包装 EVA.forward 导致严格拓扑校验失效。
  窗口执行数与真实 Attention batch 数逐帧核对，不使用相机级计数代替窗口统计。

窗口级校准入口：

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m2.window_validate \
  --output outputs/m2_window_new_run --frames 4 --repeats 3
```

这个短程工具使用真实 checkpoint 和真实 nuScenes 预处理图像，分别保存：

- `natural`：连续真实帧，输入没有人为修改，独立统计实际复用数。
- `controlled_microbenchmarks`：完全相同输入、单窗口变化、填充边缘和多相机变化；
  明确属于受控视觉微基准，不作为真实流命中率或端到端加速比。
- `window_costs_provisional.json`：每 block/执行键/实际更新数的原始测量，
  `quality_certified=false`、`approved_counts=[]`，不会覆盖生产表或自动新增批准。

当前测量保存于 `outputs/m2_window_incremental_20261022_run01/`。
自然连续的四帧没有发生精确窗口复用。受控相同输入跳过 108 个 block-window，
特征位级一致；受控单窗口变化跳过 106 个，实际存在数值差异。
`mode='exact'` 表示输入和依赖闭包严格验证，不意味着任意 GPU packed 形状都位级一致。
因此正式 Auto 仍须独立的下游任务质量证据，不能凭窗口输入相同、M1 关联成功或
局部平均误差较小自动批准局部重算。当前生产窗口批准表保持为空。

正式入口保留普通 M1 的 `state_interface='legacy'`：unknown/失效区域仍强制重算，
不能期待仅切换执行粒度便出现复用。需要单独诊断已有的 source-linked 接口时，
可在临时加载的 `mmcv.Config` 上覆盖以下值，不修改正式 M1 配置：

```python
cfg.model.m1_config = dict(source_linked_state=True, temporal_visibility=True,
    local_support=True, logging=False, visualization=False)
cfg.model.m2_config.update(state_interface='source_linked', mode='exact',
    cache_capture='always')
```

这个接口逐帧验证真实 M1 快照和 M2 执行来源，对 unknown 区域仍需完整像素比较；
dirty hints、来源过期及上下文错误继续扩大重算或回退。上面的 Exact 覆盖仅用于诊断，
不能作为下游质量准入；`tests/worldstream_m2/test_live_m1_snapshot.py` 覆盖实际
precheck → M2 → commit 的连续调用。

只检查当前路线并另存验证结果、保留旧清理审计：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.verify_final_route \
  --output-dir outputs/m2_window_incremental_20261022_run01/route_verification \
  --tests-log outputs/m2_window_incremental_20261022_run01/tests.log
```

单独启用 Exact 仅用于受控实验时，可以临时覆盖
`model.m2_config.mode=exact` 与 `model.m2_config.cache_capture=always`。
这不会修改正式配置或产生质量批准；有 M1 失效掩码或 cheap gate 拒绝时仍可能回退。
