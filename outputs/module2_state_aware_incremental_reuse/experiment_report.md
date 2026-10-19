# M2 状态感知增量视觉推理实验

本轮完成冻结 M1、来源对齐探针、默认关闭的条件性前缀复用原型、真实 GPU 不规则执行分析和完整 OmniDrive 对照。当前边界的生产结论为 **No-Go，保持 Disabled**。

自然探针覆盖 8 个已曝光场景、96 帧、576 路相机观测。实际支持 Token 129,110，严格完整 Patch 30,306，完整可复用 Window **0**。这些均为 `exploratory_reused_population`，不构成独立物理或任务认证。

实际 EVA 为 24 Block，0/1 是 16×16 Window，Block 2 首次进入每路相机的全局 Attention。单独执行 Block 0+1 的真实 Wall 为 39.505 ms。完整 EVA 减去缓存 Block 1 后的 Tail 得到偏宽 Oracle 上界 43.632 ms，其中连 PE/ABS 都免算；原型仍计算 PE，整层 PE 为 1.205 ms。自然完整 Window 为 0，因此可获准的实际节省上界为 0，验证/对齐/缓存仍需成本。

## 实现与边界

`source_alignment.py` 只使用 M1 真实来源点，统计实际像素支持、冲突、Dirty/Unknown、整 Patch 位移以及完整 Window 依赖。部分支持只进入诊断，不填充 Patch/Window。

`conditional_prefix.py` 使用独立缓存和现有 WindowExecutor；被复用 Window 的前缀 Attention/MLP 确实不执行，其余 Window 完整计算。Block 2 及以后全部执行。当前 PE 加当前 ABS，但历史上下文内的 ABS/RoPE 无法解析消除，因此位移复用明确标记 approximate，禁止伪造 Transformer/任务认证。生产配置关闭；研究配置显式开启，Auto 对所有条件复用保持回退。

GPU 优化只切换现有 original/spatial gather，保留索引顺序、Padding 和不可变历史来源；未另建调度器。成本记录含真实模型指纹/环境/Shape/Stride/Dtype、更新量和实测时间，所有条件记录未获任务资格，不安装进正式 Exact 成本表。

## 完整模型结果

| 输入 | 方法 | 计时请求 | E2E P50/P95 ms | 跳过 Block-Window | 冻结质量通过/比较 |
|---|---|---:|---:|---:|---:|
| natural | conditional | 16 | 3217.91/3996.96 | 0 | 32/32 |
| natural | conditional_optimized | 16 | 3332.15/3939.18 | 0 | 32/32 |
| natural | existing_exact | 16 | 3208.04/3910.02 | 0 | 32/32 |
| natural | full_maintain | 16 | 3174.02/3907.88 | 0 | 32/32 |
| natural | original | 16 | 2865.08/2896.38 | 0 | 基准 |
| natural | source_probe | 16 | 3225.67/3937.98 | 0 | 32/32 |
| repeat | conditional | 8 | 3222.86/3253.30 | 540 | 16/16 |
| repeat | conditional_optimized | 8 | 3217.04/3267.38 | 540 | 16/16 |
| repeat | existing_exact | 8 | 3243.86/3257.38 | 0 | 16/16 |
| repeat | full_maintain | 8 | 3228.03/3298.86 | 0 | 16/16 |
| repeat | original | 8 | 2868.28/2895.87 | 0 | 基准 |
| repeat | source_probe | 8 | 3280.28/3601.21 | 0 | 16/16 |
| local | conditional | 12 | 3217.65/3267.15 | 612 | 3/12 |
| local | conditional_optimized | 12 | 3224.97/3303.86 | 612 | 3/12 |
| local | existing_exact | 12 | 3250.82/3318.05 | 0 | 12/12 |
| local | full_maintain | 12 | 3217.90/3450.83 | 0 | 12/12 |
| local | original | 12 | 2879.81/2895.15 | 0 | 基准 |
| local | source_probe | 12 | 3265.56/3503.16 | 0 | 12/12 |
| local | unqualified_layout_intervention | 12 | 3204.08/3295.45 | 864 | 1/12 |

自然输入六方法在相同冻结 M1 持续维护契约下比较，Original 单独作为无 M1 的完整模型基准。B/C 是受控 RGB 输入，真实相机位姿、时间、Detection/Map/Memory/LLM 仍正常执行，不代表自然复用收益。

额外 `unqualified_layout_intervention` 使用已知受控图像置换的来源映射，刻意绕过 M1 准入，只用于暴露历史位置上下文的误差传播；不得作为可用 M1 候选或自然性能证据。局部普通研究路径的同坐标未变 Window 也可能因不同 packed batch 浮点数值产生任务误差。

局部普通研究路径没有获准的 M1 位移复用 Window，其跳算依赖同坐标完整输入检查。其最大 Detection 中心差异为 0.377 cm，最大 Map 坐标差异为 4.765 cm，匹配后 Memory 参考点最大差异为 2.155 cm。Map 超过冻结 1cm 门槛；这些结果不能当作 M1 来源对齐条件复用的合格人口。

刻意绕过准入的位移压力测试最大 Detection 中心差异为 78.944 m，只有 1/12 通过。视觉前缀刷新可将 EVA 误差恢复到零，但不能清除已经进入下游 Memory 的误差，详见逐帧刷新记录。

所有质量比较沿用冻结的 Detection 中心 1cm、Map 1cm、文本/Planning 一致性协议，保留 raw/matched Memory 和实际解析 ADE/FDE；无法解析/无参考不填零。第 6/11 帧只刷新视觉前缀，未清除下游 Memory，恢复情况见 `window_dependency_and_refresh.json`。

计时包含 M1 Precheck/Commit、来源验证、缓存及全部实际语言生成。输出/特征/Memory CPU 拷贝和质量计算在同步计时结束后；内部 Feature Probe 是方法工作量，计入计时。CUDA 事件与钩子具有观测开销，所有方法一致；分项间隔重叠不可直接相加。自然每场景一次预热和一次计时回放，局部一次无预热回放，因此时间为本轮描述性结果，不宣称独立重复性或泛化。

来源映射每路相机 CPU Wall P50/P95 为 1.390/141.445 ms；M1→M2 Adapter 为 1.985/3.225 ms。双倍帧间隔补充探针覆盖 48 帧、288 路观测，完整来源 Window 仍为零。逐场景转弯幅度、纹理和相机时间信息见 `population_census.json`。

自然两条研究路径都走 Full，因此其时延差不能归因于 packing 优化。GPU 优化对照来自实际执行的受控 Window microbenchmark；各头时延包含其内部 Memory 维护，不另行虚构独立 Memory 分项。

本轮真实 CUDA 回归：**623/623 通过**。M1/M3/原模型代码不修改，正式 Exact 的依赖规划、执行器与成本表保持原样。详细来源/版本审计、实际 Attention/MLP 调用 Shape 和 CUDA kernel traces 均已归档。

## 最终验收

| 项目 | 结论 |
|---|---|
| m1_interface_freeze | Go |
| exact_m2_preserved | Yes |
| conditional_reuse_feasible | No |
| conditional_reuse_quality_pass | No |
| conditional_reuse_profitable | No |
| production_conditional_reuse | Disabled |
| next_module_priority | 转向 M3 |

未具备完整来源支持的自然 Window；历史位置上下文无法安全视作当前位置等价表示；零自然跳算不能覆盖状态维护及验证开销。停止继续堆叠此边界的 Token/Window 复用算法，下一阶段转向完整 VLM 服务的任务条件化视觉计算/语义输入压缩边界。本轮不改动 M3。

复现命令见 `commands.sh`；所有命令排除故障 CPU 4/5。保留旧实验，不覆盖先前结论。若需要清理分析钩子，可删除独立实验工具；生产原模型没有植入 profiling 代码。
