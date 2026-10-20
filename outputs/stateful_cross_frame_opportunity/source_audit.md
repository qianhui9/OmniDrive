# 源码边界核对

以 HEAD `c471d5d8bbc648995610a6ce5e360c9e5380c96c` 及实际文件为准。读取了 `docs/worldstream_m1_evidence.md`、`docs/worldstream_m2.md`、`docs/worldstream_m3.md`、`docs/worldstream_m3_refinement.md`、`docs/worldstream_system_optimization_v5.md` 和任务指定的四组输出。旧文档关于 Anchor 未保存 Source Frame 的描述已被后续实际实现修订，本轮直接记录现有字段，不沿用旧缺失假设。

| 代码边界 | 已核对的事实 | 复用阻断 |
|---|---|---|
| `worldstream/m1/state_manager.py` / `state_store.py` | Precheck/Commit 事务、Epoch、来源时间和签名；StateStore 与原生 Head Memory 分开 | 观测有效不意味着身份、几何、完整任务或 Transformer 等价；默认不捕获大型 Feature Tensor |
| `worldstream/m1/physical_certifier.py` | 限定 Claim、真实测量回执、来源/版本/时间及生产器独立资格 | 现有正式生产器资格缺失；不能由 Stable 或检测置信度升级 |
| `worldstream/m2/dependency_planner.py` / `source_alignment.py` | 真实卷积输入足迹、完整 Patch/Window、冲突和 Dirty/Unknown、Global 闭包 | 自然完整来源 Window=0 的既有证据保留；Block 2 后全局依赖，不恢复位移近似 |
| `worldstream/m2/conditional_prefix.py` / `runtime.py` | Approximate 与 Exact 缓存分开；Auto 没有资格则 Full | 位移后的 ABS/RoPE/Attention 历史上下文不能当作当前等价 |
| `projects/.../streampetr_head.py` | Memory 长 600，Top-k 300，Propagated 300；当前位姿和时间更新 Key Position | 同一个来源可从上一帧 Propagated Query 移入当前 Memory；只在真实 V 输入仍逐位相同的层考虑行级线性投影，不能跳过查询更新 |
| `projects/.../petr_head_map.py` | Memory 长 600，Top-k 300；当前 Pose/Time 改变 Key Position | FIFO 保留的 V 可以相同，Key/Attention/整 Head 仍有当前依赖 |
| `projects/.../petr_transformer.py` | Self-Attention 固定使用 PyTorch MHA；Cross-Attention 按配置使用 FlashMHA | 不根据配置标志臆测已执行的算子；观察真实 `_in_projection_packed` 输出 |
| `worldstream/m3/world_state.py` / `task_conditioner.py` | 使用 Precheck 后、当前 Commit 前历史；全景、Planning、数量/缺失命题保守回退 | 部分 Anchor 无法形成完整原生任务证据；插入文本也不是计算减少 |
| `worldstream/m3/visual_state.py` / `prefix_matcher.py` | 原生 Expanded Input、Token/Position/Mask、模型/LoRA/Backend、场景/相机隔离 | 只有值完全相同的因果前缀有数学闭包，不接受 Embedding 相似度 |
| `worldstream/m3/prefix_kv_cache.py` / `prefill_executor.py` | 真实历史源、Owned KV、Prefix Branch、Suffix Prefill；改变 Shape | 数学闭包不保证部署浮点实现的逐位 KV/Logits 或 Sampling 分布；完整命中才可能免整个 Prefill，短前缀仍要一次 Prefill 和全部 Decode |
| `worldstream/m3/runtime.py` / `scheduler.py` | `cross_frame` 只是扩大允许匹配的 Scope；Auto 还需要成本/任务资格 | 必须观察实际历史匹配与获准执行，不能把开关当作命中 |

新审计保留每个相机时间、真实来源与原始版本、Memory 来源 Frame 和投影计算 Frame。静态闭包、实际输入比较、GPU/CPU 微测和完整服务证据分开报告。数据中的 Annotation/GT 不进入任何审计候选决策；仅使用原模型执行值、历史来源和当前原生问题。

采用原配置的原生 Planning 负载，完整 Sampling 参数不变。原生 VQA 的其他任务覆盖引用旧报告并标为旧 Exploratory 证据，不把未重新执行的全部 VQA 宣称为本轮完整质量验证。
