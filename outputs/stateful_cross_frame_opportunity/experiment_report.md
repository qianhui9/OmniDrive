# OmniDrive 跨帧计算复用机会审计

本轮 P0 No-Go；按前置停止条件冻结 P1 新执行机制。原生模型、权重、LoRA、任务、Sampling 参数和冻结阈值不修改。新增工具不接入 Production/Auto。

实际人口：8 个已曝光场景、96 个自然连续帧、88 次跨帧转换、576 路六相机观测、96 个原生问题，全部为 Exploratory。原配置原生任务为 Planning；本轮不删减问题或更换输入。旧 EVA 96 帧的失败人口单独引用，绝不与新人口相加。

M1 观测有效 Anchor 累计 9，正式物理认证累计 0。每帧来源 Frame/Camera/Timestamp/Epoch/版本、Dirty/Unknown 和提交前历史快照保存在 GPU 原始记录。观测有效、物理身份、Transformer 等价和任务输入充分性分别判断。完整 Planning 证据没有由部分历史覆盖。

M1 Feature 数量是带版本的元数据记录；96 帧实际提交的 Tensor 字节 P50/P95 为 0/0，不能把未捕获的 Feature 当作可执行缓存。

Detection/Map 原有 Memory 已经是原模型基线的一部分。当前视觉、查询和位姿/时间相关 Key 变化，不能复用整个 Head 或 Attention。本轮发现真实保留的历史 Value 输入行累计 168000，只对 Wv 线性投影存在行级数学依赖闭包。测量完整投影与含来源核验、Gather、部分 GEMM、拼接的候选，按场景与分支汇总见 dependency_analysis.json；不能把原生 Memory 再记一次节省。

局部数值根因探针：全量诊断投影与原生投影逐位一致，继承的历史行也一致；首次差异位于缩小 GEMM Shape 后重新计算的新行。实际数值和首个差异索引保留在 temporal_cost_probes.json，不把它推演成已经测得的下游任务失败，也不为零净收益机制追加修复。

Expanded Input 的真实跨帧 LCP P50/P95 为 35.0/35.0 Token。历史源实际匹配 88 个问题；其中匹配视觉前缀 Token 总数 0。这是实际张量匹配而非仅允许跨 Frame ID。合法可执行 KV Token、实际执行命中和跳算均为 0；改变 Prefill Shape 的数值路径及原始 Sampling 分布未获认证。输入相同不能推出 FP16 输出逐位相同。8 次真实前缀探针中，仅 3 次最后位置 logits 逐位相同，最大绝对差 0.046875。

P0 成本探针保留 Native 全量执行。V-only 候选和前缀候选均有原始 GPU Event/CPU Wall 样本，验证与缓存成本单列。CUDA Event 区间包含主机发射空隙，不称为 Kernel 时间总和。没有将审计时延作为完整服务性能、将 Token 比例外推为 GPU 节省或把同帧 Prefix/Batch/Overlap 记为跨帧收益。

| 被测候选 | 相对完整计算的净增耗时 ms | 成本覆盖 |
|---|---:|---|
| detection V-only | 0.033–0.036 | 8 场景 |
| map V-only | 0.230–0.253 | 8 场景 |
| Prefix KV（含签名/匹配/发布） | 8.229–16.758 | 8 场景 |

上表是局部串行计划的实测成本范围，不能相加成为 E2E 增量。Detection 仅第 0 层满足历史 Value 闭包，Map 覆盖六层。没有少执行一次完整 Prefill 或任何 Decode。

本轮基准只覆盖逐层来源核验、Gather/Concat 的最小 V-only 计划和现有 PrefixKVCache 的匹配/签名/发布计划；No-Go 限这些实测计划。没有声称所有未来的缓存或专用 Kernel 都数学上不可能获益。新增专用 Kernel 不属于本轮任务范围。

本轮实际减少 GPU 操作：0；实际复用 KV Token：0；完整 VLM 净加速：未建立。P2 活跃机制质量与 P3 四模式配对/关闭历史消融均为 Not Run，原因见 formal_go_nogo.json。原生回答、Token ID、感知和 Memory 摘要已保存，但不冒充新策略通过冻结质量。

原 M1/M2/M3 加新增审计回归：658 项通过，失败 0，错误 0，跳过 0。这属于回归核验，不替代新执行机制的完整任务质量认证。

被否定的假设限本轮边界：Stable 观测可直接跳过 Transformer；已有 Temporal Memory 自动带来新增跳算；相同文本前缀自动获得有利净成本及严格 Sampling 等价。没有证明所有可能跨帧算法都不可行。建议停止继续投入当前 EVA Window、整 Head、原有 Prefix 边界的缓存复杂度；只有新增可观测且收益足够的依赖边界，才重新进入 P0。

运行证据：outputs/stateful_cross_frame_opportunity/gpu_audit02。环境、Commit、完整 Checkpoint SHA（含 LoRA）、Tokenizer 和数据文件 SHA、随机种子、Sampling 协议及源码前后哈希均已记录。审计用独立临时钩子，退出后恢复，原 M1/M2/M3 和模型文件未修改。已有未跟踪 SGVD 目录保留且没有读取或恢复。

复现见 commands.sh；交付代码边界见 implementation_changes.md。未运行阶段的 null 不是零时延或质量通过。
