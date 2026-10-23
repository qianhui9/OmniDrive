# WSGRR 归因、同预算实验与执行器优化

本轮唯一主线仍是 `worldstream/m1` + `worldstream/m2`。生产配置和正式批准表保持原样。结果、失败日志和重试写入 `outputs/wsgrrr_20261022_followup/`；原始 `outputs/wsgrrr_20261022/` 不修改。

## 比较指标与正式精度的边界

`tools/worldstream_m2/task_diagnostics.py` 保留原来的置信度 ≥0.3、正反点序取最小距离、无距离门限的匈牙利匹配。新增类别兼容、弧长均匀采样（101 点）的双向点到线段平均距离及带 dummy 的匈牙利匹配，优先最大化有效匹配数量。3 m 门限在运行前声明，依据是仓库 OpenLane-V2 最大 Frechet 门限；新几何定义与官方 Frechet AP 不同，不能互相替代。被拒绝的旧配对、原始索引、坐标、置信度及未匹配原因仍归档。

Detection 同时保留原始类别内匈牙利结果，另列 2 m 的三维中心距离诊断及未匹配索引。该三维距离也不等于 nuScenes 的平面中心距离正式指标。

`task_quality.py` 用真实 GT 调用原 nuScenes `DetectionEval.evaluate` 和 OpenLane-V2 `evaluate`。40 帧的官方函数结果明确标记为子集结果；只有覆盖全部 6019 个验证 token 才标记 FULL。无效的官方值保留 null 及其路径，不补零。地图模型只有 lane-centerline 输出，没有交通元素或拓扑预测头；其空交通元素和零邻接矩阵产生的分数不能视作这些能力得到验证。

`official_detection_audit.py` 额外执行未修改的 SDK `accumulate`（官方 2 m XY 距离、全保留分数、严格小于门限、按置信度匹配）。只读 callback 记录其实际访问和返回的 TP 决策，逐项验证一致；没有另写匹配算法替代正式评分。导出匹配对象、过滤前预测索引、官方过滤后 GT 索引、未匹配真实对象、逐帧预测状态及类别计数。这里的未插值 precision/recall 是全部保留分数下的审计操作点，不是 mAP/NDS，也不是 3D 预测一致性。没有 GT 的类别按 SDK 的 no_predictions 语义标为未评分，不能当作零漏检。

`map_eval.py` 用 CPU 子进程预计算原始官方几何函数，并按完整坐标字节、形状、dtype、掩码和类别绑定缓存。AP、拓扑、F-score 的聚合和顺序仍来自原 evaluator；缓存输入不符直接报错。单元测试比较串行与并行官方输出的完全一致性。它只加速离线评估，不计入模型推理加速。

`official_map_audit.py` 对声明的 40 帧子集或完整 6019 帧真实输出直接调用官方 `pairwise`、Chamfer/Frechet 和 `_tpfp`，保留官方 1/2/3 m 门限、ego relaxation、预筛选、全部分数与匹配顺序，导出逐线 GT/预测索引、匹配距离、拒配原因、未匹配 GT 及图节点度分层。原始 Frechet 与官方距离分别归档；预筛选占位值 1024 不计作实际误差。节点度大于 1 仅为图分支分层，不是人工交叉口标签；零预测邻接表明模型缺少拓扑头，不能将 GT 节点覆盖解释为预测连接正确。操作点计数不是 AP，实际范围以各目录状态和 `--limit` 为准；全量审计完成前不能使用子集冒充全量。全部全量 CPU 审计完成后才启动独立性能对照。

## 离线对照

- Native：相同 checkpoint、预处理、帧顺序、任务头和问题的原生 EVA。
- D3：现有 WSGRR，每相机普通预算 3，必须刷新优先。
- NoM1Region：仅在离线 API 中禁用世界区域/对象事件的额外优先级，保留普通预算 3、漂移、年龄、基础不确定性、来源验证、像素失效、TTL、数值检查和全局依赖屏障。
- NoM1RegionMatched：附加的离线诊断控制。只读取 D3 每帧每相机实际执行窗口数量，不读取其位置，按 NoM1Region 自身评分选取相同数量。安全刷新可以超出目标，不能被目标截断。此组用于分离“额外区域窗口增加计算量”的混杂；边缘窗口有效 token 数不同，窗口数相同仍不代表所有 FLOPs 完全相同。它不是普通预算为 3 的生产策略。

`event_missed_windows=0` 只表示已识别必须事件未被选择器漏掉，不是现实世界变化的漏检率。跨组事件覆盖使用 D3 轨迹的窗口集合做共同审计参考，也不是真实 GT 事件召回率。

## 最小执行优化及数值政策

此前部分执行只缩小 Attention；QKV、输出投影和 MLP 仍消耗完整或填充网格计算。现在同一执行器对选中窗口计算 QKV 和输出投影，仅对真实有效 token 计算 MLP。全刷新直接调用原生 Block，虚拟 padding 在 norm1 后清零，输出保持原生 grid stride。无 dtype 改动，未扩大可缓存前缀，Block 2 及其后仍完整执行。

部分 GEMM 的形状变化会改变 FP32 舍入，因此数值政策版本改为 `selected_window_qkv_projection_valid_token_mlp_v2`；旧成本签名失效，不能复用旧准入证据。测试用原有 3e-6 的执行器舍入容差验证，真实复用特征误差及真实任务损失另行实测，不能用执行器测试代替质量验收。

实际点式工作读取每阶段 `pointwise_tokens`、`qkv_tokens`、`pointwise_window_work`。兼容字段 `skipped_pointwise_window_work` 仍只统计整块零执行的窗口数量，不能用它估算 v2 部分窗口的总节省。

事件投影使用相同 Conv2d 采样格点在 CPU 批量映射，避免逐对象 CUDA 标量同步。缓存年龄更新直接复用本次执行的 GPU 索引，避免后缀网络提交后又上传 CPU 索引。事务提交仍保留必要的张量独立副本、来源元数据、有限值检查及原子发布。

M1 的 `export_m2_inputs(computation_only=True)` 仍校验待提交帧及来源摘要，但只复制 M2 所需的计算状态、来源和 lease，不复制无用的稠密观测审计数组。默认导出接口不变。来源摘要仍绑定完整元数据，包括 ndarray 内容、dtype、shape；改为直接哈希字节，避免先深拷贝再转列表/JSON，对 object 数组递归绑定值而不是内存地址。

## 计时与复现

所有运行通过 `taskset -c 0-3,6-31` 使用 OmniDrive 环境，避开故障 CPU。GPU 空闲才启动任务。正式预测进程结束后可启动 CPU-only 官方评估子进程；预测完整保留，评估状态看各运行的 `official_eval/status.json`，不能把预测完成误报为评估完成。

完整验证主要用于质量。其 CPU-only 评估可能与下一组预测重叠，因此不使用这些单次全量运行的请求延迟证明跨组速度收益；性能结论使用另行执行的同输入、无评估并发的重复视觉/完整任务配对实验。

历史 `outputs/wsgrrr_20261022/continuous40` 使用初版 WSGRR，该轮报告已注明后续增加过期区域处理。因此它只作历史参考。本轮严格优化前基线为新目录中的 `baseline_source_snapshot/`，逐文件取自任务开始时本仓库 commit `4756dfac5c673b1a02abab65d0c485e37f3b5558`，不复制或改写模型/数据/生产配置。`run_baseline_control.py` 复用最终计时函数，仅枚举 D3 并移除旧接口不支持的 `computation_only` 关键字（执行原始完整导出），改写后的离线测量函数和 hash 一并归档。`baseline_final40_control/` 同样使用 40 帧、预热 2、重复 5；在全量任务、CPU 评估、回归全部结束且 GPU 空闲后运行。

`run_baseline_task.py` 用同一当前任务测量器在该 Git 基线核心上运行两轮 Native/D3 完整语言请求，40 帧、相同问题、种子及生成参数。原始 M3 帮助函数仅供离线 observer 定位原生 LM，M3 执行始终关闭；共享文件 hash 另存 `shared_observer_manifest.json`。旧 D3 同一 40 帧另执行真实官方 GT 子集评估，与最终代码比较。全部为隔离实验，不增加生产配置或批准。

`profiling.py` 的 `torch.profiler` / cProfile 包装、模块 hooks 和 `record_function` 只在离线剖析启用；不替换模型 forward。profile 运行排除在普通计时样本之外。CPU 范围包含其同步等待；CUDA event 跨度包含提交空隙，不是活跃 kernel 之和。父子范围和同一时段的 CPU/GPU 时间均不能叠加。

完整请求从数据读取/预处理前开始，到输出 materialization 并同步后结束；包括 M1 precheck/commit、M2、任务头和真实语言生成。M2 独立视觉基准与完整请求分别报告。语言生成长度可能改变，因此同一问题的完整请求与视觉速度同时呈现，不把较短回答直接归因为视觉加速。

可复现命令见新结果目录各 `invocation.json`、`source_hashes.json`、`benchmark_source_hashes.json`、`m1_m2_source_hashes.json`、`run_manifest.json`。派生汇总：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.followup_summary --root outputs/wsgrrr_20261022_followup
```

剖析插桩清单：

| 文件 | 类型 | 内容 | 定位 |
|---|---|---|---|
| `tools/worldstream_m2/profiling.py` | 新增 | scoped record_function、CPU/GPU operator 表、trace、cProfile；退出时恢复 hooks/包装 | 第 14、56 行，`instrumentation`, `profile_call` |
| `tools/worldstream_m2/window_validate.py` | 修改 | `--followup-profile` 离线入口；预热、重复计时和原生样本；M1 CPU 组件记录 | 第 140 行，`run_wsgrr` |
| `tools/worldstream_m123/benchmark/observe.py` | 修改 | 观测实际调用的 `precheck_with_observation`，避免漏记 M1；不双重包裹别名 | 第 73 行，`Observer.__init__` |
| `tools/worldstream_m123/benchmark/worker.py` | 修改 | 输出已有 M1 CPU 计时；记录源文件哈希；计时后写审计、独立官方评估 | 第 140 行，`run` |

这些离线插桩可通过不使用 `--followup-profile` 关闭；归档 trace/CPU 表保留用于复核。后续需要清理时，可以按此清单移除离线插桩并保留已有证据。

最终真实权重回归另使用相同测量器与精确 Git 基线执行器做新鲜缓存对照：旧执行器十种 case 的局部和最终差异均为 0；v2 部分执行局部最大约 8.20e-4，最终视觉平均最大约 2.80e-4、极值约 0.8903。它未保持原有位级等价，不能用小型单元测试容差代替真实整网数值准入。v2 为未认证研究实现，不自动批准；同一 40 帧的 Detection/Token F1 小幅回归与 Map 变化在报告并列，不能宣称无损。
