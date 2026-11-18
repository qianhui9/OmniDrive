# WorldStream Core Component Latency Breakdown

独立实验入口。当前默认采用 **forward_only_v2 纯前向口径**：输入准备完成并已在 GPU 后，只计一次 WorldStream `model(...)` 调用，外部数据加载、预处理、H2D、问题读取/编码及结果解码/导出均不计时。模型内部的 OGSM、DAVP、感知头、五个语义请求、自回归生成和缓存更新仍属于真实方法推理。OGSM 必需的输入/输出适配及状态绑定仍在方法前向中，排除的是外部通用数据 pipeline。

所有新增代码位于本目录；不修改 `worldstream/`、`projects/`、主实验脚本或历史结果。默认完整实验为 **5 个独立进程 × 同一正式负载**，`--trials 10` 切换为 10 轮。完整负载必须与 `outputs/worldstream_ablation_full_run01/manifest.json` 的 18 Scene、723 帧、每帧 3 个事实问题 + 2 个原 QA、种子和权重相同，准备阶段逐项核验。

## 运行

只做资产、正式负载、四帧独立预热序列和输入哈希检查，不使用 GPU：

```bash
taskset -c 0-3,6-31 bash tools/component_latency_eval/run.sh --prepare-only
```

小规模真实 GPU Smoke Test（一个 Scene 的连续三帧；两对独立进程，插桩/无插桩交替运行；每个进程各预热四帧）：

```bash
taskset -c 0-3,6-31 bash tools/component_latency_eval/run.sh --smoke
```

完整实验命令（本次交付不自动执行）：

```bash
taskset -c 0-3,6-31 bash tools/component_latency_eval/run.sh --trials 5
# 可选 10 轮：
taskset -c 0-3,6-31 bash tools/component_latency_eval/run.sh --trials 10
```

单元测试和真实小型 CPU 契约检查：

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m pytest tools/component_latency_eval/tests -q
```

输出默认为 `tools/component_latency_eval/outputs/<时间戳>/`，也可传 `--output`。目标目录已存在时拒绝覆盖。准备阶段得到的目录仅用于审核，不会自动启动实验。`--report-only --output <已完成推理目录>` 可为尚未生成汇总的推理结果生成报告；汇总文件也使用独占创建，不能覆盖已有报告。

使用 Python `/home/qh/miniconda3/envs/OmniDrive/bin/python`，进程继承 CPU `0-3,6-31` 亲和性。GPU 固定为物理 GPU 0；先等项目已有文件锁，再等 GPU 空闲，不停止其他进程。启动器可复用已经校验的 `outputs/worldstream_driver_compat/580.173.02` 进程内驱动库，不安装或更新系统驱动。

## 真实函数及计时边界

所有 GPU 项同时记录 Host elapsed 与 Device stream span。下表中 Host 指 `perf_counter_ns()` 入口/出口差值，并非操作系统 CPU busy time；它可以包含原代码自身的隐式等待。对混合组件不把 Host 和 Device 相加。

| 组件 | 真实函数/边界 | 时钟及范围 |
|---|---|---|
| OGSM.Precheck | `worldstream/m1/state_manager.py:WorldStateManager.precheck_with_observation` | Host；完整事务预检查，含 `_precheck` 的投影、光流、外观验证、失效/刷新需求及事务绑定；不含外部输入适配器 |
| OGSM.Commit | `WorldStateManager.commit` | Host；候选状态、对象/地图关联、版本/上下文检查及最终原子提交；不含外部检测结果适配器 |
| OGSM.Total | 上述两个非连续但互不重叠区间之和 | 逐帧 Host sum；不会把中间视觉/感知工作计入 |
| DAVP.Patch Embedding | EVA `PatchEmbed.forward` | Host + CUDA Events；当前帧完整 patch embedding 一次；位置编码仍在 DAVP 总时延中 |
| DAVP.Visual Planning | `IncrementalVisionRuntime._forward_frame` 的两个区间 | Host + CUDA Events；入口至 `block0_input` 前，以及 `input_finite` 至 `_execute` 前；包括元数据/来源校验、observation-to-dependency mapping、特征漂移、优先级、闭包、执行计划及世界风险信号；两个区间逐帧汇总，排除 embedding/位置编码 |
| DAVP.Incremental EVA | `IncrementalVisionRuntime._execute` 入口至 `global_context` 构建前 | Host + CUDA Events；执行 ID 生成、整个窗口前缀和对应来源元数据；整个前缀只计一次，不逐层/逐窗口插桩；保存实际执行/复用窗口数 |
| DAVP.Global Refresh / Residual Reuse | `IncrementalVisionRuntime._update_global` → `global_residual.update_global` | Host + CUDA Events；每帧只执行并记录一条路径；包括共享风险决策和实际 suffix 刷新或残差加法/锚点构造。决策与分支未人为分开；实际返回的 diagnostics 决定标签 |
| DAVP.Total | `IncrementalVisionRuntime.forward_from_m1` | Host + CUDA Events；包括 M1 snapshot 读取、规划、embedding、前缀/全局路径、校验与缓存发布。总量不等于仅列出四个子项的和 |
| Detection Head | `Petr3D.simple_test_pts` 中 `if self.with_pts_bbox` 的完整分支 | Host + CUDA Events；真实 `pts_bbox_head(...)`、carrier clone、`get_bboxes` 及结果构建 |
| Mapping Head | 同函数 `if self.with_map_head` 完整分支 | Host + CUDA Events；真实 `map_head(...)`、carrier clone 和 `get_bboxes` |
| Original Perception | 上述两个非重叠 head 分支之和 | Host sum、Device span sum 分开；不含共享 ROI/position preparation，该部分留在 E2E residual |
| QCSE.Semantic Routing | `answer_validity.propose_fact_answer` | Host；当前问题解析、世界状态读取、支持事实与准入选择；原函数同时构建部分事实文本，未虚构分离该步骤 |
| QCSE.Direct Answer | `M3Runtime._decode_current_frame` 中已准入直答分支 | Host + CUDA Events；tokenizer encode、EOS/BOS 处理、结果 Tensor 和直答 diagnostics 构造；仅实际触发时记录 |
| QCSE.Candidate Retrieval / Construction | `AnswerMemory.retrieve` / `construct` | Host；候选检索、有效性/世界差异和候选 token 构造，独立记录真实调用 |
| QCSE.Metadata Binding | `M3Runtime._decode_binding` | Host；每批共享的模型/tokenizer/生成配置绑定，不摊成伪查询调用 |
| QCSE.Verification and Continuation | `PrefillExecutor.generate`，且实际 `verification_forward_calls > 0` | Host + CUDA Events；完整当前 prefill、候选选择/验证/接受及后续生成的合并区间，条件统计；不代表纯验证 kernel 时间 |
| QCSE.Native Generation | 同函数，实际没有执行候选验证 | Host + CUDA Events；正常 prefill/decode，包括无候选和候选过滤后生成；不把无候选误称为验证失败 |
| QCSE.Total | `M3Runtime.generate` | Host + CUDA Events；一帧全部五个请求及共享绑定/准备的完整批次开销，`sample_unit=frame` |
| QCSE.Query Total / Query E2E | `_decode_current_frame` 的实际逐查询 body 入口/出口 | Host、CUDA Events、同步 Wall；两者为同一计时边界的别名，`sample_unit=query`；包含该请求路由、直答或生成、记忆更新；排除批次共享绑定、共享视觉 carrier 和稍后的文本导出 |
| Frame Forward（当前主指标） | 已准备的 GPU 输入 → 起始 GPU fence → `model(return_loss=False, rescale=True, **data)` → 结束 GPU fence | 同步 Wall；只计完整方法前向，排除数据读取/预处理、collate/scatter/H2D、问题读取/编码、模型加载/预热及稍后结果解码/JSON 写盘。 |
| Frame E2E（历史 v1） | 仅旧实验保留 | 包含加载/预处理/问题编码/导出，不能改称纯前向或与新口径混合统计 |

### 为什么合并候选验证和 fallback

实际 `GenerationFacade._history_decode` 在同一 while 循环中进行 block 验证、接受处理和逐 token 正常续写。独立测量其中的 fallback/验证需要进入 Token 循环重复计时，违反本实验的低频插桩要求，因此**保留真实完整 executor 区间并合并报告**。不生成纯 Candidate Verification 或纯 rejection fallback 的虚假时延。`fallback_after_rejection` 调用比例来自实际 verification log 和最终输出长度；拒绝后立即 EOS/结束不算 fallback。该路径的孤立时延标记为不可分，包含它的时间位于 Verification and Continuation。候选检索/构造已单列；不能再与包含这些区间的父模块总量相加。

## 计时行为及非干扰

- 通过实验进程内 AST scope 替换函数 `__code__`，保留函数对象身份，不触发模型/执行缓存基于 forward identity 的失效。异常或结束时恢复。磁盘源文件保持原样。纯前向模式将 `simple_test_pts` 中 M3 完成后的原有文本解码/写盘尾段封装为回调，在前向结束同步及计时后执行，填充同一个返回列表并生成相同答案文件。该延后处理在无插桩对照中也启用；只支持完整 QCSE，禁止把 Native LLM 生成本身误移到计时外。
- 去除新增 scope、展开延后的原导出回调后，AST 必须与原函数逐语句相等，否则拒绝启动。每轮保存 `instrumentation_audit.json`，列出函数、源代码 SHA256、变换种类与保持原语句的核验结果。
- 没有逐 token、逐 Patch、逐 Transformer 层计时；真实 EVA 前缀循环外只有一个 scope。被包装的 `PrefillExecutor.generate` 每个请求只进入一次。
- 未发现实际 WorldStream/Petr3D 执行代码显式建立第二条 `torch.cuda.Stream`。Events 在当前流 record，起止使用同一流；不改流调度。Device 数值是 stream span，会包含 host 调度间隙，不能解释为纯 kernel busy time。
- 组件出口只 record Event，不 synchronize、不读取 elapsed。每 `event_batch_frames=32` 帧，复用已有 Frame Forward 完成 fence 统一解析事件；最后再解析尾批。预热后清空预热记录。
- 同步仅用于 Frame Forward / Query 边界；不新增组件级 fence。Query 同步本身可能带来开销，因此 Smoke Test 的无插桩对照不装 scope，也不加 query fence，实测总体观察效应。
- 两组均关闭已有 `worldstream_profile`：旧开关会逐 forward 记 Event 并在每个请求末尾 synchronize，不适合本实验的计时控制。算法、缓存、生成参数、精度、batch size、流和调用顺序保持正式配置。
- 每轮独立启动进程、严格加载同一 checkpoint。复用已有补充实验的四帧独立 Scene 预热策略；这来自 frozen main 的四帧 pilot 规模，正式 ablation runner 自身没有额外 warmup 阶段。预热帧与全部测试 Scene 分离。仅在 Scene 边界重置感知记忆及 M1/M2/M3，帧间不清缓存。
- 无插桩与插桩各运行两次，顺序 baseline→instrumented、instrumented→baseline。比较输入 digest、逐问题答案及 Frame Forward。偏移绝对值超过 5% 写 `REVIEW_REQUIRED`，不能直接当作无开销证明；小样本还受启动/温度波动影响。

## 统计与输出

每个 `trial_XX/` 含原始 `raw_component_latency.csv`、真实推理 `receipts.jsonl`、`frame_component_latency.csv`、`trial_component_summary.csv`、权重和运行配置、源码插桩核验、阶段/进度和完成标记。

`raw_component_latency.csv` 包含任务要求的全部字段，并补充窗口数、父组件和聚合类型。CPU-only 的 GPU 字段留空；混合组件没有同步墙钟的 `component_wall_ms` 留空。失败调用标为 FAILED，报告拒绝把它们作为成功样本。

- 最终组件表以 frame/query 聚合样本统计，`statistical_unit` 和 `sample_unit` 明示。原始记录仍保留真实调用；同帧 Visual Planning 的两个不重叠片段先相加，`invocation_count=2`，最终 `sample_count=1`。Host/Device 永不相加。
- 各轮分别计算 Mean 和线性插值 P95（NumPy `percentile(method='linear')`），轮内调用波动 `std_ms` 单独保存；最终 Mean 为各轮 Mean 的算术均值，Mean Std 为各轮 Mean 的样本标准差 `ddof=1`。P95 同理。不同轮样本绝不混池求最终 SD。
- 未触发的路径 `sample_count=0`、条件 Mean/P95/SD 留空、invocation_rate=0；全局平均开销为 0 是基于已观察到的零调用，不是虚构的零耗时样本。只触发一轮时跨轮 SD 留空，`trials_with_samples` 和 `p95_caution` 保留缺样本信息。
- `invocation_rate` 是触发 frame/query 数除以全部 frame/query 数；另报真实调用总数。条件均值与 `global_average_ms`（组件耗时总和/全部机会数）分开。
- 辅助 `e2e_share_percent` 分别按对应粒度的 Host 或 Device span 总和除以对应总区间 wall 总和（新帧级分母为 Frame Forward，查询级为 Query E2E；历史实验仍用 Frame E2E）计算。这些是非可加比例，不能把父子、Host/Device 或 frame/query 占比相加当作 critical-path partition。
- 仅 OGSM 和两个 heads 计算有依据的非重叠模块之和；DAVP/QCSE 采用真实包含范围的总区间。E2E 单独实测，绝不相加组件 P95。
- 汇总核验各轮 frame 顺序、查询、输入 digest 和逐问题答案完全一致，少轮/少帧或失败就拒绝完成报告。

根目录输出：`component_latency_summary.csv`、`module_latency_summary.csv`、`execution_path_summary.csv`、`e2e_latency_summary.csv`（兼容文件名）、纯前向模式的 `forward_latency_summary.csv`、`trial_component_summary.csv`、`experiment_config.json`、Smoke 的 `instrumentation_overhead.json`。

图像在 `figures/`：Mean/P95 Component Latency、Module Latency Breakdown，各为 PDF 和 320 dpi PNG。图 1/2 顺序一致，误差线为跨轮样本标准差；混合项采用 GPU span、CPU-only 项采用 Host，轴标签明确区分。图 3 比较同一 Host 时钟下的模块 inclusive totals，使用并排柱，避免嵌套/互斥路径错误堆叠。缺少调用明确标 N/A；Smoke 图含 SMOKE ONLY 标记。没有真实 GPU 数据时不生成论文数据或图。

## 插桩变更清单

所有文件均为新增，原模型与主实验源文件没有变更。

| 文件 | 类型 | 用途 |
|---|---|---|
| `__init__.py` | 新增 | 独立工具包 |
| `config.json` | 新增 | 可直接运行的正式 5 轮配置 |
| `run.sh`, `driver.py` | 新增 | 亲和性、环境、已有进程内驱动兼容 |
| `run.py` | 新增 | 固定负载预检、独立进程、预热、真实推理、对照 |
| `timing.py` | 新增 | Host 时钟、低频 scope、延迟 CUDA Event 解析 |
| `forward.py` | 新增 | 准备好输入后的纯前向计时与计时后的结果导出队列 |
| `instrumentation.py` | 新增 | 可恢复的进程内边界插桩、原语句相等核验 |
| `report.py` | 新增 | 逐轮/跨轮统计、路径比例、论文图表 |
| `tests/*.py` | 新增 | 时钟、统计、边界、真实 OGSM/EVA 小型 CPU 契约 |
| `README.md`, `VALIDATION.md` | 新增 | 运行方式、口径和实际验证状态 |
| `outputs/*` | 新增 | 本任务的预检/测试/Smoke 证据；不写入原实验目录 |

## 口径版本与历史数据

`config.json` 固定 `latency_scope=forward_only_v2`；新 manifest 保存 `timing.frame_component=Frame Forward`。收据使用 `forward_ms`，原始记录使用 `Frame Forward`；`forward_latency_summary.csv` 是新的整帧主指标，`share_denominator` 明示占比分母。旧目录的 CSV/收据及旧 `Frame E2E` 数字保持原样。报告按 manifest 选择分母，禁止将新旧口径跨轮混合。

已经完成的 `outputs/20261117_140846_299699` 没有独立记录完整模型调用时延，无法准确事后扣除输入准备和导出，也不能以子组件之和代替前向。其组件时延仍可参考，旧 E2E 与基于旧 E2E 的占比仅作历史记录。要得到新口径的正式五轮数字，需要使用上述完整命令重新测量并保存至新目录。

### forward_only_v2 修订清单

| 文件 | 本次变更 | 用途 |
|---|---|---|
| `forward.py` | 新增 | 单次真实模型调用的同步墙钟边界；计时后执行导出回调 |
| `run.py`, `config.json` | 修改 | 输入准备移至计时前；新 scope/manifest/forward_ms 字段；相同口径对照 |
| `instrumentation.py` | 修改 | 只将 M3 后的原导出尾段延后；推理计算、状态/缓存及源码文件不变 |
| `report.py` | 修改 | 独立 Frame Forward 指标、forward_latency_summary.csv 和明确占比分母；兼容旧结果口径 |
| `tests/test_forward.py` | 新增 | 前向边界、计时后导出、失败处理及分母隔离验证 |
| `README.md` | 修改 | 新指标定义、历史数据限制及运行方式 |
| `outputs/20261117_140846_299699/结果.md` | 修改说明 | 标明旧全流程指标不能当作纯前向；原数值/CSV/收据不变 |

## 最新正式纯前向结果

`outputs/20261117_191525_562820` 已完成实际配置的三轮独立正式测量（非 Smoke）。帧级纯前向 Mean 为 2704.0925 ± 9.3333 ms，P95 为 3608.6217 ± 15.4242 ms，详见 [结果报告](outputs/20261117_191525_562820/结果.md)。旧五轮全流程结果保留作历史口径，不与新三轮混合。
