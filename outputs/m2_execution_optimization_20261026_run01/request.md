# OmniDrive M2：基于单一 WSGRR 主线的 GPU 执行与缓存管理性能优化

请直接在当前本地项目：

`/data/qh/phdCode/work4/OmniDrive`

基于最新代码对 M2 进行性能优化。

本次目标不是重新设计 WSGRR，也不是再次解决正式推理的缓存准入问题，而是**在保持当前正式窗口复用算法、M1/M2 依赖关系和基本任务精度的前提下，显著降低窗口增量执行的额外开销，使 M2 尽可能获得真实的 GPU 视觉计算和端到端净加速。**

## 一、当前基线与核心问题

上一轮已完成统一正式执行路径：

`M1 Precheck → WSGRR → DependencyPlanner → WindowExecutor → StateCache Commit → Detection/Map → M1 Commit → M3`

目前已经移除正式 M2 的 CostModel 成本批准与自适应缓存发布门控。

最新实验结果位于：

`outputs/m2_unified_wsgrr_20261026_run01/`

其中：

- 实时完整模型 M2 窗口复用率：8.1019%，350/4320。
- 历史 M1 状态重放连续帧复用率：12.2507%，516/4212。
- 完整视觉 CUDA Span：479.91 → 519.59 ms。
- 完整 M1+M2+M3 端到端平均时延：2980.10 → 3342.97 ms。
- 当前实际窗口复用发生在前两个 EVA Block，后续 22 个全局依赖 Block 仍需完整执行。
- 40 帧子集 Detection mAP、NDS 和 Lane DET_l 变化较小，但有 2 个问题出现关键数字代理退化。

**核心矛盾：窗口计算减少了，但窗口重组、GPU/CPU 同步、缓存提交和状态管理引入的开销超过了节省的计算量。**

本轮围绕这一矛盾直接优化正式代码。

## 二、严格要求：只保留一条主执行路线

整个项目正式 M2 仍然只有：

`WSGRR → DependencyPlanner → WindowExecutor → StateCache`

禁止：

1. 增加 GPU 成本准入和预先批准规则。
2. 增加独立的 full-compute fallback 执行器。
3. 增加根据耗时动态切换的第二套 M2 推理算法。
4. 引入多套不同复杂度的防御性退化路径。
5. 为了提高复用率关闭 M1 必须刷新事件。
6. 随意减少依赖闭包中的刷新窗口。
7. 强行复用后续 22 个具有全局依赖的 Transformer Block。
8. 修改 M1/M3 核心算法、模型权重或 M3 现有质量预算。

可以在同一个 WindowExecutor 中根据计划的窗口数量自然执行全部或部分更新，但不得维护两套独立的正式推理实现。

保留缓存来源、场景、版本、位置、Tensor 形状和非有限数值等基本正确性检查。不兼容或非法状态直接报错，不静默执行备用路线。

## 三、第一步：先定位真正的执行瓶颈

重点检查：

- `worldstream/m2/runtime.py`
- `worldstream/m2/window_executor.py`
- `worldstream/m2/packing.py`
- `worldstream/m2/state_cache.py`
- `worldstream/m2/region_refresh.py`
- `worldstream/m2/dependency_planner.py`
- `worldstream/m2/m1_adapter.py`

使用当前真实 checkpoint、正式 WSGRR 路线及自然连续帧，对 M2 的主要耗时进行分段测量。

重点区分：

- CPU 规划和元数据处理；
- CPU→GPU 和 GPU→CPU 数据传输；
- CPU/GPU 同步等待；
- 窗口 gather；
- QKV 投影和 Attention；
- 局部 MLP；
- 窗口 scatter/unpartition；
- Cache Tensor 复制；
- 缓存有限值检查；
- 后续 22 个全局 Block 的实际计算；
- 真实 CUDA kernel 执行时间。

优先使用 PyTorch Profiler 和必要的 CUDA Event，避免引入新的复杂性能分析框架。

特别注意：

目前报告中 `M2_cache_publication_ms≈430.98ms`，其中可能包含前面异步 GPU 计算在 `.item()` 处产生的同步等待。

必须将真实 Cache Clone/Commit 开销与之前累积的 GPU 等待时间区分开。

不要根据不准确的 CPU Wall Time 归因直接删除缓存操作。

本阶段性能测量只用于定位优化点，不引入新的正式算法分支。

## 四、第二步：优化 WindowExecutor 的窗口执行

重点修改：

`worldstream/m2/window_executor.py`

当前实现中存在重复窗口布局转换、创建有效 Token Mask、Tensor gather/scatter，以及部分 Tensor 复制。

### 4.1 减少重复窗口打包

检查 `partition()`、`unpartition()` 和 `_grid()` 中的内存布局转换。

要求：

- 避免同一输入在一个 Block 内重复执行不必要的 partition。
- 减少非必要的 `contiguous()`、`clone()` 和 `copy_()`。
- 保持模型期望的输出 shape、stride、dtype 和 Token 顺序。
- 保持 EVA 原有的位置编码、LayerNorm、残差连接和 Attention 语义。
- 不通过改变模型数学语义来制造加速。

可以根据现有真实几何布局预先建立必要的窗口索引和有效 Token Mask。

同样形状、相机数量和窗口大小的连续帧应重用这些静态元数据，而不是每帧重复构造。

### 4.2 减少 gather/scatter 与 GPU kernel 启动开销

重点检查：

- `index_select()`；
- `index_copy_()`；
- `cached_windows.clone()`；
- `torch.zeros_like()`；
- 按有效 Token 构造的中间张量；
- Q、K、V 独立投影产生的中间布局变化。

在不修改预训练权重和原生数值策略的前提下，优化计算组织。

目标是尽可能减少：

- 不必要的临时显存分配；
- 重复窗口打包；
- 小 Tensor CUDA Kernel 启动；
- GPU/CPU 往返；
- 不必要的整块数据拷贝。

如果尝试将 QKV 操作组织成更少的矩阵运算，必须保证原始权重、bias、RoPE 和 FlashAttention 输入顺序正确，并比较真实数值输出。

不要为了减少 Kernel 数量改变 Attention 依赖范围，也不要引入新的近似 Attention 算法。

### 4.3 保持完整更新与局部更新的单一接口

现有 `WindowExecutor.execute()` 同时接收当前特征、历史缓存和窗口 ID。

继续使用这一接口。

当计划要求更新全部窗口时，通过同一执行器进行正常完整窗口计算；当只需更新部分窗口时，使用同一执行器更新对应窗口。

不要新建 FullExecutor、FallbackExecutor 或其他备用执行类。

## 五、第三步：优化 StateCache 数据复制与提交

重点修改：

`worldstream/m2/state_cache.py`

当前 `StateCache.commit()` 会为缓存建立私有 Tensor 副本，而局部窗口执行本身也可能已经克隆过缓存。

这可能造成重复的大 Tensor 数据复制。

### 5.1 优化缓存所有权

检查：

- images；
- block0_input；
- Block 0/1 的 packed outputs；
- window ages；
- descriptors；
- provenance 和 region states。

重点分析：哪些 Tensor 已经由当前 WindowExecutor 新分配，并且可以通过明确的所有权转移直接成为新的不可变缓存。

在确保不会被后续请求或外部别名修改的条件下，避免再次复制已经具有独占所有权的 Tensor。

如果需要，可以围绕同一个 StateCache 实现受控的缓存缓冲区复用或双缓冲提交，但这只是同一路径的存储优化，不得新增第二套缓存系统。

必须保持：

- 原子提交；
- 旧缓存不会被提前修改；
- 非法数据不会污染已发布缓存；
- 未刷新的窗口保留正确的历史值；
- 更新窗口具有正确的新版本和来源；
- 场景切换时不会复用旧场景缓存；
- 显存不超过现有预算。

不允许简单使用 `detach()` 代替 `clone()` 而不解决 Storage Alias 问题。

### 5.2 优化元数据更新

减少每帧对大规模诊断和计算来源结构的重复 `deepcopy()`。

区分：

- 真实运行必需的来源元数据；
- 实验审计需要的详细信息。

保持正式 Runtime 使用轻量必要数据。

复杂的审计和原始窗口记录放在 Benchmark/Observer 中完成，不要让正式执行为了输出调试信息反复复制大型元数据。

不要删除对实际缓存一致性必要的字段。

## 六、第四步：减少 CPU/GPU 同步开销

重点检查：

`runtime.py`、`window_executor.py`、`region_refresh.py` 和 `state_cache.py`

当前存在需要重点分析的操作，包括：

- `torch.isfinite(...).all().item()`；
- `int(ids.min())`；
- `int(ids.max())`；
- `ids.unique()`；
- `ids.tolist()`；
- `torch.stack(...).cpu().tolist()`。

这些操作可能触发 GPU→CPU 同步或数据传输。

要求：

1. 找出实际热路径中的同步点，并通过 Profiler 确认其成本。
2. 合并可以合并的 Tensor 检查，减少重复同步。
3. 对由 DependencyPlanner 生成、已经有明确合法性保证的窗口 ID，避免在 WindowExecutor 内重复进行昂贵的范围扫描与唯一性检测；独立调用接口仍应保留正确的合法性验证。
4. 尽可能在 CPU 上处理纯元数据和来源信息。
5. GPU 上的窗口评分、特征变化计算和实际 Attention 保持 GPU 执行。
6. 对需要回传 CPU 的窗口统计进行批量读取，而不是对每个窗口多次同步。
7. 保证非有限值检查在缓存原子发布之前完成。

不要直接关闭非有限值检查或缓存来源检查来制造速度收益。

避免将异步 CUDA 执行误判为真正完成的 GPU 计算。测试和性能统计必须采用正确的同步边界。

## 七、第五步：优化 WSGRR 的窗口选择开销

重点检查：

`worldstream/m2/region_refresh.py`

继续使用现有方法：

- 必须刷新；
- 普通刷新；
- 漂移评分；
- 缓存年龄；
- M1 状态事件；
- 每相机普通预算 3；
- 最大复用年龄 8。

不改变上述核心算法及参数。

但可以优化实现，例如：

- 复用静态窗口几何和窗口有效性 Mask；
- 避免每帧对相同相机布局重复构造 Patch 位置索引；
- 缓存仅依赖模型配置的投影几何；
- 将相同区域的 Window Mask 计算合并；
- 减少 Python 循环中的重复 Tensor 创建；
- 减少对完整 GPU 评分 Tensor 的多次 CPU 读取。

优化后必须保证在相同输入、M1 快照、缓存和参数下，刷新窗口集合与修改前完全一致。

如果出现窗口集合变化，必须确认是原有代码缺陷修复，不能将其当作正常的性能优化。

## 八、第六步：优化运行时中不必要的重复工作

检查：

`worldstream/m2/runtime.py`

重点检查：

- 每帧模型拓扑检查；
- `signature()` 中对全部模型参数和 Buffer 的遍历；
- `input_footprints()`；
- 静态窗口几何；
- 重复的 M1 来源元数据拷贝；
- 每帧重复创建的 Camera/Block 映射；
- 不参与真实决策的重复诊断。

对于与输入帧无关的静态模型信息，允许在模型初始化时生成并复用。

但需要保留明确的模型版本与 Tensor 版本管理，使模型权重或结构改变后不会错误使用旧缓存。

本次只允许减少重复计算，不允许简化真实 Transformer 依赖闭包或者跳过 M1 明确要求刷新区域。

## 九、测试：保持 WSGRR 算法行为不变

运行现有：

- `tests/worldstream_m1/`
- `tests/worldstream_m2/`
- `tests/worldstream_m3/`
- `tests/worldstream_m123/`

并增加必要的针对性测试。

重点验证：

1. 修改前后在同一真实帧序列上的窗口规划结果一致。
2. 实际执行窗口 ID 完全一致。
3. 首帧缓存初始化正确。
4. 连续帧缓存版本和窗口年龄正确。
5. 缓存所有权、不可变性和原子提交正确。
6. 新场景不复用旧场景数据。
7. 非有限值、异常来源、版本错误等非法状态能够明确报错。
8. 实际复用窗口数量没有因为减少必须刷新窗口而虚假增加。
9. WindowExecutor 输出保持原有数值策略和精度容差。
10. M1/M2/M3 完整推理流程能够运行。

特别测试优化后的 Cache Buffer 是否发生 Storage Alias、提前覆盖或错误复用。

不要只通过 Mock 测试证明 GPU 优化完成。

## 十、真实性能测试与质量验证

使用现有权重：

`ckpts/iter_10548.pth`

优先使用：

`outputs/m2_unified_wsgrr_20261026_run01/`

对应的 40 帧自然连续序列。

### 10.1 测试 M2 自身的边际性能

构建一个严格控制变量的配对实验，仅比较：

- 相同 M1 快照下的原生完整 EVA 视觉计算；
- 相同 M1 快照下的优化后 M2 WSGRR 增量计算。

两者必须使用相同的模型、图像、预处理、M1 状态和前置条件。

记录：

- WindowExecutor 分段 GPU 耗时；
- Cache Copy/Commit 真实耗时；
- WSGRR 评分耗时；
- CPU/GPU 同步耗时；
- 完整 EVA 视觉计算 Wall 和 CUDA Span；
- 实际窗口复用率；
- 峰值显存。

不要把包含 M1/M3 差异的 B0/B3 结果直接当作 M2 自身的因果加速。

### 10.2 测试完整实时服务

继续运行实际 M1+M2+M3 闭环。

与原生完整计算路径进行配对比较，固定 checkpoint、问题、生成设置和随机种子。

尽量执行 3 轮交替顺序的性能测量，避免只选择最好的一轮。

报告：

- M2 正式实际复用窗口数和复用率；
- M1/M2/M3 各自的时延；
- 完整视觉 CUDA Span；
- Prefill 和 Decode 时延；
- 端到端平均/P50/P95；
- 峰值 GPU 显存；
- 整体吞吐量。

注意 CUDA Event Span 包含可能的 Host 提交间隙，不应直接称为纯 GPU Kernel Active Time。

### 10.3 检查模型精度

保持原来的 nuScenes/OpenLane 评估和 VLM 配对质量流程。

重点检查：

- Detection mAP、NDS；
- Lane DET_l；
- VLM GT Token F1；
- 数值/计数；
- 空间关系；
- 运动及驾驶相关问题。

由于本轮只修改 M2 的执行效率，不改变 WSGRR 窗口选择规则，因此合理预期修改前后的 WSGRR 模型输出应当保持一致或在已验证的数值容差内。

如果明显超出原数值容差，应修复实现错误，而不是直接提高容差掩盖错误。

本研究允许少量相对于原生完整计算的任务精度损失，但不允许本轮性能优化引入无法解释的新增误差。

上一轮存在的 2 个关键数字退化案例需要保留并检查，不得删除这些问题来改善最终结果。

如果有时间可以补充少量新场景质量验证，但不要为了本次性能优化运行大量无关基线。

## 十一、验收标准

本轮核心验收有三部分。

### A. 工程正确性

- 正式 M2 仍然只有一条 WSGRR 主线。
- 不重新引入成本准入和自动退化执行器。
- 缓存发布和窗口复用正常。
- 依赖闭包与窗口执行集合正确。
- GPU 优化没有破坏输出语义或缓存状态。

### B. 性能改进

优先要求：

- WindowExecutor 和 Cache 管理的可测开销下降。
- 完整 EVA 视觉时延相对于当前 WSGRR 版本下降。
- 与完整原生 EVA 相比的性能差距明显缩小。
- 尽可能取得真实视觉净加速与端到端净加速。

本轮不能为了满足指标而人为提高复用率或减少必须刷新窗口。

如果经过合理优化后仍然没有净加速，请明确给出剩余理论收益上限和已经测量到的固定开销，说明是否需要在下一阶段改变 M2 算法的可复用计算范围。

不允许通过重新添加全量计算回退或运行时成本选择器来让性能指标表面变好。

### C. 任务质量

- 优化后的 WSGRR 不产生新的不可解释数值偏差。
- Detection/Map 及 VLM 的整体任务表现不明显恶化。
- 数值、空间关系等关键问题保留逐样本诊断。
- 不将 40 帧子集指标冒充完整验证集质量。

## 十二、最终交付

请直接完成实际代码修改、必要测试和真实 GPU 实验，并提交：

1. 修改的文件及核心优化说明。
2. 优化前后各执行阶段的性能比较。
3. WindowExecutor gather/scatter 和缓存复制的优化证据。
4. CPU/GPU 同步点减少情况。
5. 正式 M2 窗口复用率及与旧版本的逐帧一致性检查。
6. 实际完整 EVA 视觉净时延变化。
7. M1+M2+M3 完整服务质量与时延结果。
8. GPU 回归测试结果。
9. 可复现实验命令与完整输出。
10. 尚未解决的计算瓶颈。

新实验输出保存在独立目录，例如：

`outputs/m2_execution_optimization_20261026_run01/`

不得覆盖之前的实验结果。

最后请明确回答：

- 当前正式 M2 是否仍然只有一条主执行路径？
- GPU 窗口执行器减少了哪些实际开销？
- Cache Clone/Commit 和同步是否真正降低？
- 窗口复用率是否仍保持算法原有行为？
- EVA 完整视觉计算是否比优化前更快？
- 是否实现相对于原生计算的净加速？
- 端到端推理是否加速？
- 精度是否出现新增退化？
- 如果仍然无法加速，究竟是固定开销过高，还是只有前两个 EVA Block 可以复用导致计算收益上限太低？

**本轮最重要的目标是：保持单一 WSGRR 正式执行路径和原有精度基础，通过优化 GPU 窗口执行、缓存管理和 CPU/GPU 协同，将已经实现的窗口复用转化为实际计算收益。不要新增防御性分支，也不要重新回到成本表准入机制。**