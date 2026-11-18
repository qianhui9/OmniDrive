# 连续运行与状态老化：五问、三阶段（Protocol v3）

2026-11-18 更新。主负载严格为每帧三个原固定事实问题 + 两个原始 QA。阶段为 Early / Middle / Late；不在阶段边界重置状态，不将短日志循环播放当作长视频。


## 推荐：18 条真实日志扩样重跑

新增独立配置 `expanded_config.yaml` 和入口 `run_expanded.sh`；原 `config.yaml` 仍保留单日志模式。扩样协议保持每帧五问、每条日志内 Early/Middle/Late 三等分、跨阶段状态连续，真实日志边界重置。

```bash
# 先核验全体 18 条日志的真实输入、问题和 GT；不加载 GPU 模型。
bash tools/experiments/long_run_state_aging/run_expanded.sh --check-only

# 可选：两条不同日志各首 9 帧的短 Smoke，核验真实日志边界重置。
bash tools/experiments/long_run_state_aging/run_expanded.sh --smoke

# 正式扩样：18 条完整日志 × 两方法 × 3 次独立重复。
bash tools/experiments/long_run_state_aging/run_expanded.sh --run-full
```

正式配置严格要求 18 条完整日志，不能静默退化到较小子集；同时核验官方 scene.log_token，要求来自 18 个不同的 driving log，映射保存在 manifest 中。共有 723 个不同帧、每方法每次 3615 个原查询；正式总执行 4338 帧次、21690 个查询。输出独立保存到 `outputs/experiments/long_run_state_aging_expanded/run_<time>/`，不覆盖已有结果。

**扩大的是场景覆盖，不是单日志连续时长。** 同一日志的三次重复不会被当成三个独立场景，也不会人为消除真实的 0 或 N/A。扩样场景仍来自既有主实验，不能称为新增独立测试集。

预检新增 `sample_coverage.csv/json`：逐日志、逐阶段列出不同帧数、known/UNKNOWN GT 和唯一的采样问题事实变化查询数，后者不乘重复次数，也不等同于独立物理事件。正式结果同时报告实际可评分分母，不能只给百分比。

`stage_summary.csv` 新增场景数、唯一可评分查询、唯一 Stale-eligible 查询转变和重复观测数量。`overall_summary.csv`、`scene_summary.csv`、`repeat_summary.csv` 分别给出总体、逐日志和逐重复结果。两原始 QA 的有限事实评分继续单列。

统计采用 **2000 次配对场景聚类 bootstrap**：每次重采样完整日志，保留该日志的全部时间帧、五问和重复运行，两个方法用相同的场景抽样。报告 Accuracy、Stale、Mean E2E、P95 E2E 的经验 95% 区间；`paired_comparison.csv` 报告配对差值和时延相对降低率的区间。

有以下限制时不伪造区间：有效场景不足两个、有效分母的 bootstrap 抽样比例不足 80%、零/全错误率导致非参数 bootstrap 无法反映未观测错误。点估计仍如实保留 0%/100%，CI 的 N/A 原因和有效抽样数单列。经验区间退化为点时也标出 `ci_empirically_degenerate`，不能据此宣称真实不确定性为零。

每次重复的阶段均值及 P95 另计算 mean ± sample SD（ddof=1）。这是重复运行间波动，区别于场景 bootstrap CI；每次 P95 的平均也区别于主表的 pooled P95。Peak GPU/Cache 仍是实际观测最大值，不包装成长期内存上界。

自动生成 `结果.md` 与 `report.md`（相同内容），并单独生成 `表1.md` 和 `paper_table.csv`，可直接取出所需七列表格。中间方法完成后会更新阶段性报告；只计两个方法均完整完成的同一 repeat，不把中断前缀混入主表。完整模型没有新算法开关、额外缓存清理或标注输入。

## 运行

从项目根目录执行：

```bash
# 默认只准备真实数据、GT、输入和权重哈希，不启动 GPU 推理。
bash tools/experiments/long_run_state_aging/run.sh --check-only

# 短验证：首 9 个相邻真实帧，三阶段各 3 帧，两方法独立进程。
bash tools/experiments/long_run_state_aging/run.sh --smoke

# 完整日志推理，默认 3 次独立初始化的重复；重复结果不拼成时间轴。
bash tools/experiments/long_run_state_aging/run.sh --run-full
```

脚本自动调用 `/home/qh/miniconda3/envs/OmniDrive/bin/python` 和 `taskset -c 0-3,6-31`。每次运行自动创建时间戳输出目录 `outputs/experiments/long_run_state_aging/run_<time>/`。GPU 被占用时只等待，使用项目 GPU lease，不干预其他任务。每个方法和每个 repeat 均使用新进程、独立 CUDA 上下文。

可用 `--config <yaml>`、`--repeats N`、`--output <新目录>` 配置。`other_questions` 必须为 2；缺少任一原始 QA、问题重复或固定三问/顺序不符，预检及收据校验均报错，不默默减少问题。原始 QA 从原 conv/vqa JSON 按主实验顺序读取，不新造问题和答案。

## 连续日志与阶段

默认从主实验的完整 18 条日志中按视频真实时长选择最长日志 `scene-0962` / `2086743226764f268fe8d4b0b7c19590`：41 帧、19.949757 秒。三阶段分别为 **14 / 14 / 13 帧**。选择不依赖模型结果；全体候选保存在 manifest。

阶段公式为 `min(2, floor(3 * frame_index / sequence_length))`，按真实时间序帧索引分为三份。Smoke 只对短前缀重新分阶段，用于工程验证；正式运行使用全部帧。

核验相邻 next/prev 双向链接、严格递增时间戳、官方 scene 首末 sample 和帧数、原始六相机文件、标定和位姿。`sequence_pool: full-val` 可搜索所有验证集日志；`sequence_count` 和 `scene_ids` 可分别评估多条完整日志。不同日志在实际边界重置，从不拼接历史状态。

WorldStream 在日志内部持续保留原生状态。逐帧记录前后 OGSM epoch/commit、DAVP publication version/源帧、QCSE frame index/源帧/条目数，强制核验状态连续，包括两个阶段边界。OGSM 原子提交可能替换 StateStore，因此不以 Python 对象地址推断重置。Native 按无跨帧状态的定义逐帧清空感知 temporal memory。

**时长限制：nuScenes 单日志约 20 秒。3 次独立重复不能变成 60 秒连续视频，也不能证明小时级稳定性。** 本地当前入口没有可用的真实长日志适配；不得通过循环帧或拼接不同场景扩大“连续时长”。正式报告保留实际视频时长。

## 统计口径

每帧五问在同一次完整模型调用内执行，计时不为问题子集拆分或估算。CUDA 同步的 E2E 包括读图、预处理、H2D、问题编码、完整推理和最终答案导出；冷首帧计入，不预热测量前缀。模型加载、离线评分、只读采集和审计输出不在 E2E 中。FPS = 帧数 / 累计 E2E 秒，沿用 main_protocol_v1；辅助处理墙钟吞吐另存。

|主指标|计算方式|
|---|---|
|Factual QA Accuracy|保持主实验 grade：五问中具有可靠 known GT 的事实得分平均；UNKNOWN 排除并计数，自然 QA 的有限部分分保留；完全正确率另列|
|Stale Answer Rate|同日志同问题的前次可比 known GT 已变化，回答对旧 GT 满分而对新 GT 未满分；分母为可评分的变化查询，没有分母为 N/A|
|Average / P95 E2E Latency|完整五问帧 E2E 样本的均值与 P95，不只测模型内部时间|
|Peak GPU Memory|每帧开始前 reset_peak_memory_stats（只清统计计数，不清缓存），结束同步后读取 max_memory_allocated / max_memory_reserved；阶段取最大值，单位 MiB，包含驻留模型和临时计算内存|
|Peak Cache Size|OGSM/DAVP/QCSE 各自逐帧结束时实际保留载荷的阶段最大值；平均值、各自峰值与同帧合计峰值单列，不把不同时间的模块峰值简单相加|

Host RSS 在帧开始和结束采样，阶段 `max_observed_host_rss_mb` 是最大观测值，**不是精确主机内存峰值**。`host_process_lifetime_peak_rss_mb` 是 Linux ru_maxrss 的进程生命周期高水位，可能包含加载阶段，不伪称阶段峰值。GPU 指标是当前进程 PyTorch allocator 范围，不等于整个设备显存。没有 CUDA empty_cache 或新增缓存淘汰。

缓存大小使用 MiB（1024² 字节），遍历明确状态/缓存根，计 Python 元数据与去重的 ndarray/Tensor storage，是逻辑保留载荷估计，不是 RSS 或显存分配器占用。DAVP 另存原生 cache.bytes 精确张量载荷。各模块可能持有共享状态，合计也不称为互斥进程内存。采集在 E2E 外只读执行，并记录耗时。

## 三模块诊断

|模块|真实字段与用途|不能推断的内容|
|---|---|---|
|OGSM|StateStore 的 anchors/features/provenance、对象/地图锚点、track_birth_timestamp 存储年龄、source_timestamp 观测年龄、last_verified_timestamp 验证年龄；Precheck invalid IDs；Commit 实际 association_pairs/selected indices、几何匹配率和 unmatched 数|几何关联率不等于真实身份关联准确率；缺少预测锚点与 GT instance 的可靠绑定，identity error rate 和完整 ambiguity 总数 N/A|
|DAVP|StateCache 的 outputs/sources/window_ages/global_anchor；复用窗口年龄、原生上限、复用比例、全局锚点年龄/刷新、global_feature_drift 与视觉时延|高复用比例不代表质量更好，需同时检查 QA/Stale；不改刷新策略|
|QCSE|AnswerMemory.records/acceptance/performance 的大小、条目、生成 Token/Prompt Token；真实驻留与被选候选年龄；实际提出/接受 Token、验证拒绝率、直接作答比例、语言时延|运行时 cache_bytes=0 指没有持久 KV，不能当作候选内存为零；不假设超龄条目立刻删除；完整 entry-validation-failure 接口缺失时 N/A|

比例以实际分子分母合并，不平均逐帧百分比；无候选或复用窗口为 N/A。年龄原始列表、分布、窗口复用上限违反数、实际候选来源年龄分桶接受率均保存。OGSM 只读记录关联对供审计，不根据几何成功率编造“没有错误关联”的结论。状态更新频率是视频时间域的提交频率，非部署吞吐。

QA GT 来自原 QA 引用与权威 nuScenes 标注。约 2 Hz 关键帧支持采样事实变化上的 Stale，但无法定位精确物理变化时间；Change Response Delay / Stale Duration 为 N/A，不生成 change_response.csv。不把无查询帧视为正确响应，不用回答语义相似度伪造 GT。

## 两个原始 QA 的补充评价

`query_group_summary.csv` 独立列出 `fixed_facts`（q0–q2）与 `original_qa_limited_facts`（q3–q4）的事实分、可评分数量、UNKNOWN、完全正确率和 Stale 分母。全部推理仍执行五问，不另跑三问推理，也不为子集估算时延。

当前现有评价器仅可靠检查有限事实，不能覆盖回答流畅性、完整性和全部语义。因此完整生成质量标为 N/A，不将有限事实分包装成生成质量分；可靠事实部分可作为补充结果。

## 输出、恢复及历史协议

- manifest.json：协议 v3、完整配置、源码/GT/输入/权重 SHA256、Git Commit 或不可读取原因、真实日志及阶段数、指标定义。
- summary.json、stage_summary.csv、per_frame_metrics.csv、ogsm_state_aging.csv、davp_cache_aging.csv、qcse_cache_aging.csv、query_group_summary.csv、scored_queries.csv、report.md。
- runs/repeat_NNN/{native,worldstream}/attempt_NNN：有效模型配置、身份、完整实际问答、逐帧收据及状态连续性、运行日志、完成标记。
- figures/figure{1,2,3}_*.{pdf,png}：完整配对记录生成趋势、缓存峰值、缓存年龄与候选接受率三张图，图一按日志内原帧序号对齐。

prepared 模式只生成 N/A 表格和空明细，不生成虚构图。部分尝试保留真实记录，但不进入完整配对汇总。原始收据按帧 flush/fsync；末尾半行忽略；阶段仍按预定完整序列划分，不能因中断缩短而重新分阶段。

```bash
# 重生成派生报告，无推理。
bash tools/experiments/long_run_state_aging/run.sh --report-only --output <v3输出目录>
# 完整尝试保留；失败尝试使用新进程从首帧重跑，原部分记录保留。
bash tools/experiments/long_run_state_aging/run.sh --resume --output <v3输出目录>
```

恢复不意味着恢复完整系统内存，不从半段接续，也不将不完整尝试合并成长序列。源码/输入变动拒绝恢复。旧 v1/v2 收据和报告保持原文件，v3 不重新解释或覆盖它们，必须新建输出目录。此前验证见 VALIDATION.md 和 VALIDATION_v2.md；扩样验证见 VALIDATION_v3.md。

## 代码复用与验证

完整入口为 projects/configs/OmniDrive/worldstream_m123_auto.py 和 configs/worldstream_r0.json；通过 benchmark.worker 的 effective_config/load_model 调用 Petr3D 真正推理。复用 r0_task_evaluation 的事实问题与连续日志、environment_stability_eval.data_adapter 的数据/GT、continuous_stream_report.grade 的事实评分。OGSM/DAVP/QCSE 算法文件未修改。

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m pytest -q \
  tools/experiments/long_run_state_aging/test_protocol.py
```

CPU 测试夹具不写入实验结果。GPU Smoke 用真实连续六相机帧和原五问，正式实验需实际完成才有结果。
