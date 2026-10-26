# M2 执行与缓存管理优化

正式路径仍为 `M1 Precheck → WSGRR → DependencyPlanner → WindowExecutor → StateCache → Detection/Map → M1 Commit → M3`。本轮不修改刷新预算、必须事件、真实依赖闭包、模型权重、M1/M3 算法或 M3 质量预算，也没有成本门控和备用执行器。

## 执行器

`WindowExecutor.execute()` 仍同时处理全更新、部分更新和零更新。

- 缓存相机/窗口几何、padding 有效位置与网格重建索引，使用有界缓存。
- 部分更新直接按坐标 gather 当前 token，避免先完整打包整个输入再选择窗口。
- MLP 使用静态有效位置的 `index_select/index_copy_`，替代布尔高级索引引起的动态 CUDA `nonzero`。
- 用一次 gather 将 packed windows 写入原生 stride 的网格，避免 padded reshape/crop.contiguous/再一次 stride 对齐复制。非标准 stride 仍在同一个布局方法内完成拷贝，不构成另一条推理算法。
- Q、K、V 权重、bias、RoPE、FlashAttention、LayerNorm、残差和有效 token MLP 数值策略均保持原实现。未采用融合后改变 GEMM 数值的 QKV 权重副本。

规划器批量读取已闭合的窗口 mask，从 CPU 合法范围生成排序唯一 ID。私有 `_WindowIdProof` 绑定 Tensor 指针、版本、总窗口数和 CPU ID，执行器无需重复 GPU min/max/unique 扫描。ID 被修改或几何不符直接报错；独立 Tensor 调用仍完整验证范围与唯一性。

## 缓存所有权

公开 `StateCache.commit()` 默认继续 clone 所有外部 Tensor。

只有 Runtime 内部显式请求 `_transfer_outputs=True` 时，才接管 WindowExecutor 的 packed outputs。这些 buffer 来自新分配，且与输入、执行网格和最终返回特征没有 Storage Alias；零更新 buffer 来自已经发布的不可变缓存。Runtime 在提交后不再写入转移的 buffer。工具若截获私有 packed 输出，只能只读借用。

输入 images 和 block0_input 继续建立私有副本；ages、descriptors 和复杂区域/来源元数据也保留所有权拷贝。没有用 detach 替代外部输入的 clone。提交会拒绝 packed output 与输入/返回特征共享 storage 的情况。

有限值检查在原子发布前完成，执行器临时 buffer 不提前覆盖历史；失败保留上一发布版本。静态执行器元数据纳入提交容量检查。当前真实缓存载荷加静态索引远低于 512 MiB 预算。

## 状态与 CPU 工作

- 批量检查输入和 Block 0 有限值；最终视觉特征与缓存 Tensor 合并到发布前的检查。
- 缓存模型拓扑验证结果，用模块类型、子模块/参数/Buffer 身份、forward 方法和关键配置守卫检测结构变化；参数和 Buffer 的指针、Tensor 版本、shape/dtype/device 仍逐帧检查。
- 按实际卷积与窗口几何缓存 input footprints、诊断闭包映射和区域投影坐标。对外返回的 footprint 诊断独立复制，外部不能污染静态几何。
- 描述子有效 token 分母由相同整数几何预计算；区域矩形用等价的分离 OR 运算得到窗口覆盖，避免反复 padding 二维 token mask。
- 减少重复区域状态与来源行 deepcopy；发布边界继续拥有必要元数据。

## 测量边界

正式 `forward_from_m1()` 包括 M1 snapshot 导出。纯 M2 边际实验则预先导出同一个 snapshot，直接调用同一 `forward()` 核心，双方均排除 snapshot 导出；完整实时服务始终调用正式入口。不能将两个边界混用，也不能以 B0/B3 的 M1/M3 差异替代 M2 的因果测量。

Profiler 只存在于 `tools/worldstream_m2/profiling.py` 的上下文包装中，退出后恢复方法和 hooks；正式推理没有 record_function 开销。CUDA Event Span 包含主机提交间隙。实际 kernel active time 从 trace 的 kernel 事件单次计数；不能累计重叠的 operator CUDA 总时间。

提交 wall 时间包含此前累积 GPU 工作的等待。隔离实验在实际 commit 前 synchronize，再测原本提交参数，不将此前等待记为 clone 成本。当前 Trace 还可看到年龄数组标量 HtoD 之后的长等待：减少同步次数并不等于消除了全局后缀计算。

## 结果与复现

本轮输出：[experiment_report.md](../outputs/m2_execution_optimization_20261026_run01/experiment_report.md)。包含同 snapshot 的 40 帧优化前/后/原生配对、完整服务三轮交替顺序测量、Profiler 原始 trace、GT 子集质量、逐问题关键数字案例、测试和完整命令。

`tools/worldstream_m2/window_validate.py --reference-runtime <冻结旧包>` 只供 benchmark 使用；冻结包位于本轮输出目录。正式 M2 配置无法选择旧运行时或新算法。`--snapshot-boundary` 只改变实验计时的入口，不复制 GPU 执行实现。
