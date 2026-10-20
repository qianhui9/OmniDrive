# 大粒度跨帧计算边界审计

结论：**No-Go，未新增 Runtime 或启用 Production/Auto。** 保留 M1→M2→M3 研究约束、原始权重/LoRA/模型/Planning 及完整 Sampling 配置。旧 V-only、35 Token Prefix、EVA Window 不再优化或执行。

本轮服务人口为 8 个已曝光场景、96 个自然连续帧、两种原生服务共 192 个请求。`original_task=planning`，`task_classifier_label=global_safety`，`population=Exploratory`，`native_multi_cross_frame_requalification=Not Run`。详细 Profiler 为另一次回放的 2 个请求，来源/依赖 Witness 为 8 帧，不与主服务人口相加。

| 候选 | 独立 Profiler 平均 Kernel ms | GPU Compute 占比 P50 | 停止理由 |
|---|---:|---:|---|
| state_triggered_large_llm_prefill_decode | 2269.49 | 81.37% | planning_state_insufficient_and_multimodal_kv_changed |
| whole_visual_backbone | 478.06 | 17.14% | current_rgb_changed_global_dependency |
| detection_map_transformer_subgraphs | 39.02 | 1.40% | current_visual_queries_and_temporal_keys_changed |

Kernel 由原始 Trace 的真实 device activity 逐条计数，与独立 Profiler device event 总时长/数量核对；能建立唯一 CPU runtime correlation 的活动才分配阶段。归因时间覆盖率分别为 [99.99077, 99.99079]%，其余明确保留为 unattributed，不补齐、不按比例缩放。Torch 1.13 的 CPU FunctionEvent.kernel 关联存在重复/漏项，故未采用该错误总量。三级嵌套 Attention/FFN/Projection 仅作 inclusive 明细，不重复相加。CUDA Event Interval 含主机发射间隙，不等于 Kernel Sum。两次详细插桩不是延迟校准，开销及原始 Trace 单列。

Native Original 完整请求 E2E P50/P95 为 3033.81/3252.80 ms；Persistent Full 为 3652.52/4164.16 ms，均包含实际输入加载/预处理/Scatter、完整 VLM、原生回答写盘及同步。冷启动另存，以上不含每场景 ordinal 0。M1 内部 Wall P50/P95 为 526.81/1239.38 ms，可能包含既有 GPU 工作的等待，不能全部再次收费。88 个同帧服务配对增量 P50/P95 为 634.85/1167.60 ms，保留原始有符号值；这仍是单次、轮换顺序的描述性结果，尚非重复随机化因果结论。不同 P50 不能相减或相加当作配对增益。

本轮按实际热点筛选三种大边界。EVA 接收变化的当前 RGB，Global Attention 阻断局部历史扩张；Detection/Map 的当前视觉、查询和时序位置/Key 变化；LLM Prefill/Decode 消费变化的当前 Carrier、KV、生成历史与 RNG。M1 局部观测/对象历史既不能证明这些整个张量相同，也不能完整覆盖 Planning。原模型已有 Temporal Memory 不计作新节省。新人口未发现可合法跳过的大计算子图。

`optimistic_savings_bounds.json` 按每个实际请求计算整个候选免费、验证为零的宽松 Oracle 上界，保留输入流水线不可省成本。CPU+GPU envelope 刻意对候选有利，不能读作实测加速；同时单列真实 Kernel、GPU Event、CPU Wall 与 M1 维护成本。大 LLM Oracle 可以很大，但依赖/状态不合格；Oracle 不能授权跳算。旧局部候选另用旧隔离测量与新服务作描述性宽松上界，明确跨实验限制，不伪装成配对 E2E，也不按 Token 比例外推 GPU 时间。

| 候选 | GPU Event 整段最大 ms | 固定 Host、只消去 GPU 的条件估计 P50 | CPU+GPU 宽松上界最大值 |
|---|---:|---:|---:|
| state_triggered_large_llm_prefill_decode | 2387.060 | 2.672× | 32.596× |
| whole_visual_backbone | 481.188 | 1.150× | 1.211× |
| detection_map_transformer_subgraphs | 46.122 | 1.012× | 1.023× |

最后一列会重复计算 CPU/GPU 重叠，LLM 的宽松上界尤其松，仅用于不漏掉潜在大机会；它不是可信加速预测。条件估计也假设 GPU 区间可串行消除，包含发射间隙，不能当作实测。均逐请求计算，没有把不相关分位数相加。Native Original 目标还须抵消上述配对状态服务增量；Persistent Full→Active 只需另付新验证/缓存成本，其纯成本为 Not Run/null，因为三个候选均在依赖初筛失败。只读全输入核对与快照成本另存，不能冒充真正可复用 Cache 的成本。

旧 V-only 即使整个合格 V GEMM 免费，描述性 GPU 上界仅 0.366432 ms，服务倍率上界 1.000131×。35 Token Prefix 刻意放宽到整个 Prefill 免费，仍仅 186.882 ms、1.0716×；旧 Prefix 验证/发布 P50 为 7.970 ms。旧实测净成本负收益与 `insufficient_compute_scope` 停止结论保留，绝不重建缓存。

上一轮数值结论保持：Value 新 GEMM 行 MaxAbs=6.1988831e-06；Prefix 最后位置 logits 8 个样本中 3 个逐位相同，MaxAbs=0.046875。原生 Sampling 分布为 Not Certified。Stateful Active 数值、完整任务质量、四模式完整 E2E 均为 **Not Run**，不能写成执行后任务失败。

Native/Persistent 的输出、Token 与 Memory 守卫 96/96 帧逐位一致，额外 8 帧 Profiler/Witness 回放也全部一致，但不替代 Stateful Active 冻结质量资格。6 个合法历史转换上，各候选列出的必要变化输入均未出现整张量精确相同；这是完整输入相同这一闭包条件的反例，不是对未观测帧或其他数学不变量的穷尽证明。M1 见证状态的物理认证 Anchor 总数为 0，且并未捕获可证明整个子图等价的完整输入。实际减少 GPU Forward=0，实际跨帧复用=0，质量合格的服务净加速尚未建立。前置依赖失败，因此没有建立新 Cache、专用 Kernel 或多个执行器，也没有伪造四模式结果。

建议停止沿当前边界继续堆叠缓存。仅当出现自然帧中非零、来源合法、必要输入完整闭包、节省可覆盖状态维护、且可按原始 Sampling/任务质量验证的大子图，才重新进入最小实现。当前结果限制在该模型、实际 Planning 人口与所审计三类边界，不证明所有未来状态化模型都不可能。

完整回归 678 项，Failure=0，Error=0，Skip=0；旧 658 项历史记录保留。该回归不替代 Stateful Active 原始任务质量认证。

实际 HEAD/环境、完整模型与数据 SHA、全部生效生成默认项及覆盖项、源码前后哈希、原始服务帧/Kernel Trace 和失败日志保存在 outputs/stateful_cross_frame_boundary/gpu_run01。旧实验保持原样；复现见 commands.sh，变更见 implementation_changes.md。
