# WorldStream 环境语义稳定性实验

2026-11-18 主协议已更新为**仅 FQ-3 的 Stable/Changed 变化响应评估**，见 [FQ3.md](FQ3.md)；使用 `fq3.yaml`（`other_questions: 0`）与 `python -m tools.environment_stability_eval.fq3`，不再将本页以下旧五问命令用于新论文主表。旧运行与原始结果保留。新版重新执行真实三问负载，重点报告 Change Response Delay、Stable/Changed Factual QA Accuracy 及 Stale Answer Rate。

**最新统计口径：** [正式结果](outputs/run_20261117_113951_740642/结果.md)已核验与主实验使用同一评分器和 mean-score 公式，主 FPS/QPS 已改用累计 E2E 分母；原持续吞吐保留为辅助值。跨实验负载差异和离线重统计说明见 [MAIN_PROTOCOL.md](../worldstream_ablation/MAIN_PROTOCOL.md)。下文原始协议和推理代码保留不变。

独立实验，仅执行 Native VLM（OGSM/DAVP/QCSE 全关）与当前冻结完整 R0（全开）。不修改模型、三个模块的算法、主实验脚本或历史输出。所有新增代码、测试及结果位于本目录。没有训练、参数搜索、简化 WorldStream 或额外视觉标注模型。

## 使用

从项目根目录执行，使用已有 `OmniDrive` 环境。输出目录必须尚不存在；省略 `--output` 会创建时间戳目录。默认行为仅做 CPU 数据/GT 预检查；必须明确加 `--run-full` 才会启动完整 GPU 实验。

```bash
# 单元测试、真实配置检查、临时目录内的报告/绘图契约测试
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/environment-stability-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m unittest discover -s tools/environment_stability_eval/tests -v

# 真实 GT 覆盖率、固定窗口、阈值和完整数据依赖检查；不构建模型
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.run --check-only

# 短时真实 GPU Smoke：每种方法 4 帧独立预热 + 3 帧正式推理
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/environment-stability-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.run --smoke

# 完整实验：18 场景、723 帧/方法、每帧相同的 5 条查询
# 交付阶段不会自动执行此命令。
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/environment-stability-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.run --run-full

# 已完整推理但尚未生成报告时，可离线报告；不能覆盖已有报告
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/environment-stability-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.run --report-only --output <该次实验目录>
```

启动前核验 CPU 亲和性。GPU 阶段使用全新解释器，先初始化 CUDA，再导入模型插件/数据，沿用服务器已验证的启动顺序。两种方法分别加载模型，独立缓存；整个配对运行继承同一项目 GPU 文件锁。若 GPU 有其他任务，只检查并等待。进程 PID 和日志在各方法目录；不会停止其他任务。

GPU 模式在 GT 预处理前检查 NVML。遇到已确认的 `Driver/library version mismatch`（退出码 18）时，自动复用 `tools.worldstream_ablation.driver_compat` 已准备的内核版本匹配库，以新的进程启动；原命令无需增加环境变量。选择的库清单及摘要写入实验清单与两方法的生效配置。此修复仅作用于本实验进程，不安装系统包、不重载内核模块。匹配库缺失或验证失败会报告具体原因，不跳过 GPU 检查，也不无限重新启动。

## 复用模块与真实资产

| 用途 | 已有实现/资产 |
|---|---|
| 完整最佳配置 | `configs/worldstream_r0.json`、`projects/configs/OmniDrive/worldstream_m123_auto.py`、`worldstream.r0.model_settings` |
| Native 配置 | `projects/configs/OmniDrive/mask_eva_lane_det_vlm.py` |
| 相同模型构建、严格权重加载、序列状态重置 | `tools.worldstream_m123.benchmark.worker.effective_config/load_model/reset` |
| 原模型推理及数据预处理 | 已有 `Petr3D`、`build_dataset`、`collate/scatter`，没有新增推理算法 |
| 完整场景选择和既有 QA 负载 | `r0_task_evaluation.selected_scenes/facts_questions` |
| 原始对话 prompt 编码 | `selective_answer_reuse.encode_questions` |
| 请求 RNG 隔离 | `frozen_main_benchmark.frame_rng`、`worldstream.m3.request_rng` |
| 当前帧真实 GT | `answer_validity_quality.load_current_annotations`，从 `sample_annotation.sample_token` 反向连接，保留曝光时相机姿态和可见性规则；用 `iter_json_array` 补充官方属性 |
| GT 目标和问答评分 | `fact_bypass_identity.question_target`、`continuous_stream_report.grade`、原来的有限 QA facts parser |
| 实际执行计算量 | `tools.worldstream_ablation.instrumentation.Meter`，只用计数器，不调用任何消融/替代算法 |
| 模型权重 | `ckpts/iter_10548.pth`，冻结 SHA256 `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d` |
| Info / RGB / 标定 | `data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl` 及其六相机路径、ego pose、相机曝光姿态和内参 |
| 查询与参考 | `data/nuscenes/{conv,vqa}/val/<sample_token>.json`；每帧三个固定事实问题 + 按既有顺序最多两个原始问题 |
| 评价序列 | `outputs/m123_r0_only_reproduction_run01/scene_selection.json` 中的 18 个完整场景 |

`config.yaml` 中的数据/权重路径均来自本机现存资产。父进程在推理前保存配置、代码、GT、RGB、查询文件的摘要和冻结分组。原始输入摘要、prompt token IDs、请求 seeds、相机顺序及时间戳会在两种方法间逐项核验。GT 只用于离线统计/评分，不进入模型输入或缓存。

## 分组协议和已声明的选择

原项目没有语义稳定性窗口协议。本实验新增 **5 秒、不重叠、以场景首帧为原点**的固定时间区间。数据是约 2 Hz 的 nuScenes keyframe 序列，因此每窗约 10 帧；不是高频视频。窗口完整性按末次真实采样时间加该场景的中位采样周期判定。所有帧均执行；不完整尾窗为 `Unclassified`，不补齐、不丢帧。

每个连续有效帧对构建原子事实转换：

1. 对官方标注支持的类别，以 `instance_token` 作为跨帧对象标识。两个样本的对象 ID 并集提供标注集合存在性转换；进入/离开集合分别记 appearance/disappearance。必须有两帧的权威 sample 行，不能把缺失帧当空集合。
2. 持续存在的对象提供类别转换；非空官方属性集合在两帧都有标签时才比较。缺失属性为 UNKNOWN。
3. 两帧均满足原有可见性规则的对象，比较真实 ego 坐标的方向区域；直接复用 `fact_directions` 的 ±1 米区域定义。区域内部的坐标移动不算变化。
4. 两帧均可见的持续对象对，按固定 instance ID 顺序，比较相对向量的方向区域。空间关系轴以当前 ego 朝向为准；不是世界坐标中永久不变的方位标签。

`R_change = changed atomic facts / valid atomic transitions`，各原子事实等权。appearance 和 disappearance 共用存在性分母，每次转换只计一次。对象对转换按一对一个事实计数；**对象对数量较多的场景会在这个定义中占较大比重**，所以同时保存 `valid_by_type`、每类事件数和逐窗分母。该定义只描述已声明的标注语义范围。

窗口第一帧可以使用同一场景前一窗口末帧的 GT；转换按后一个样本所在窗口归属。跨场景不关联，标注/链接缺失或时间间隔超过 1 秒的帧对不参与分组分母。未使用 RGB 差、光流、模型答案、时延或准确率划组。

用主实验 18 个场景之外的 **132 个 nuScenes val 场景/5,296 帧**计算完整窗口变化率的 25%、50%、75% 分位数；推理前固定到主实验场景。同值总在较低等级，阈值重合允许空等级。当前真实覆盖率预检查得到：

| 组 | 变化率区间 | 完整窗口数 | 帧数 | 窗口变化率均值 |
|---|---|---:|---:|---:|
| S1 Highly Stable | `[0, 0.025806451612903226]` | 24 | 247 | 0.015497 |
| S2 Moderately Stable | `(0.025806451612903226, 0.03827227993439038]` | 11 | 111 | 0.032965 |
| S3 Dynamic | `(0.03827227993439038, 0.05045871559633028]` | 9 | 90 | 0.045395 |
| S4 Highly Dynamic | `(0.05045871559633028, 1]` | 21 | 213 | 0.065459 |
| Unclassified | 尾部不足 5 秒 | 7 | 62 | 不纳入四级结论 |

实际阈值始终以该次 `stability_thresholds.json` 为准。上述 18/132 场景划分同属已有、曾暴露的验证数据，**不是独立 TEST 泛化认证**。分位数校准不使用任何模型性能。

## 指标

| 指标 | 具体定义 |
|---|---|
| Average / P95 E2E | CUDA 同步的每帧及全部查询实际 host wall；从读取图像/预处理、prompt 编码和 H2D 起，到模型感知、OGSM precheck/commit、DAVP、QCSE 路由、必要 Prefill/Decode、答案导出全部结束。模型加载、预热、GT 离线评分、审计摘要和收据序列化不计入 E2E。沿用当前 R0 事实任务边界，包含场景冷首帧。|
| Overall Accuracy | 100 × 既有评分器在有效 GT 问题上的平均 fact score；额外给完全正确答案比例。原始 QA 的有限事实得分可为小数，不冒充通用开放问答准确率。UNKNOWN 排除并计数。|
| Stable / Changed Accuracy | 在相同场景、完全相同查询的上一有效可比较 GT 和当前 GT 间，比较规范化的**回答所需事实**。类别问题只比较类别；位置问题比较方向；仅目标 ID 改变不算回答事实变化。不存在历史或事实 schema 不可比时不分类，仍可参与 Overall。缺失当前 GT 不覆盖先前有效历史。|
| Stale Answer Rate | 当前 GT 已变化，当前预测对当前 GT 的得分 `<1`，且同一预测对上一有效 GT 的得分 `=1`。分母为两端都可按同一评分器评分的变化查询。Native 与 WorldStream 完全相同规则；无变化机会为 N/A。另存 change type 分组。|
| Sustained FPS | 实际完成帧数 / 窗口处理区间的真实 `perf_counter` wall seconds；独立记录，包含区间内收据/审计等测量工具开销；不是平均 E2E 的倒数。|
| Visual Compute Reduction | `100 × (1 − 当前实际 visual MACs / 相同帧 Native visual MACs)`。形状推导的实际执行 Linear/Conv2d、QKV 及 attention MACs，包含完整 patch embed 和全局 suffix；不称为硬件 FLOPs。窗口刷新数和复用比例单独给，不代替计算量。|
| LLM Compute Reduction | 相同的 paired MACs 比值，包含实际 Prefill 和 Decode 计算；输出长度差异保留。另存 direct-answer rate、实际 candidate/accepted token 数及比值、prefill/decode MACs、prefill 输入 token 和 decode forward 输入 token 数。|
| Response Delay（可选） | 离线视频只能识别数据集采样时间域的首次正确响应间隔，保存为 `sampled_video_response_delay_ms`。同时记录 5 秒观察范围内未响应、右删失、被后续变化替代的事件。缺少 live GT 事件与答案完成的同步墙钟，因此 `live_wall_response_delay_ms` 明确为不可评估；不把视频时间与离线推理耗时混加。|

所有零分母为 JSON null / CSV 空格，不填零。S1–S4、Unclassified 和 ALL 分开报告。均值、P95、FPS 及四类 QA 指标提供以完整序列为 cluster 的 2,000 次 bootstrap 95% 区间；少于两个独立序列时区间不可评估。重复运行次数默认 1，与冻结主实验的 repeat=0 一致；多个 repeat 保持配对，bootstrap 不把重复当新独立样本。

每种方法先在 4 个独立校准帧上预热（对应原主实验 4 帧 pilot 的规模），预热输出在单独目录。正式场景开始时重置原运行时和任务头，窗口边界不重置。Smoke 仅取一个场景前 3 帧，使用完整场景的 GT 窗口等级；不声称测完该窗口，散点图只接受实际完整测量的窗口。

## Change Response Delay 补充评估

`change_response.py` 为已完成的配对实验增加独立补充报告，直接使用已有真实回答和官方 GT，无需再次加载模型。原报告的简版 `response_delays` 不会覆盖；新增版本在查询缺失、UNKNOWN 和观测间隔处停止追踪，结果可能与旧版跨未知帧的统计不同。

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/environment-stability-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.change_response --source tools/environment_stability_eval/outputs/run_20261117_113951_740642
```

默认在来源目录下新建带时间戳的 `change_response_*` 子目录；可用 `--output <尚不存在的目录>` 指定。执行前校验原实验源码/GT/输入冻结摘要、推理收据和区间摘要及两方法输入/运行配置配对，然后重新离线评分。补充报告保存来源文件和本次实现的 SHA256，保留原始结果可复查。默认观察上限和最大连续间隔分别沿用来源配置的 5 秒和 1 秒，也可显式用 `--horizon-seconds`、`--max-gap-seconds` 生成另一个敏感性分析。

定义 `T_response = t_first_correct - t_change`。`t_change` 是当前查询所需 GT 回答状态首次被观测到变化的帧时间，真实变化时刻仅能落在上一有效 GT 与当前 GT 之间。`t_first_correct` 是同一场景、同一查询、同一 repeat 中，对这个变化后的状态第一次完全正确的输出时间；当前变化帧就正确时为 0。时间统一来自视频采样时间戳，导出单位为 ms，不把离线推理耗时与视频时间混加。

后续状态再次变化记 `SUPERSEDED`；GT 未知、该帧不再提问同一查询、帧间隔超过阈值、场景结束分别记录为删失原因；到达观察上限单独计数。未观察到正确回答时 delay 为 null，不能填 0。报告成功样本的 mean/median/P95 时，同时给总事件数、已响应数、响应率、删失和替代数，并另算两方法均响应的相同事件，避免直接比较不同成功子集。

“历史错误事实”在每个后续帧重新计算：预测对当前 GT 得分 `<1`，且同一预测对**本次变化之前冻结的 GT**得分 `=1`。即使该后续帧的转换标签是 Stable，也仍需检测旧事实；普通错误但不吻合旧 GT 的回答不计历史错误。输出包含：

- 首次正确响应前连续错误回答次数、起始连续历史错误次数、最长连续历史错误次数、历史错误回答总数。
- 每段历史错误的起止采样帧、已观测跨度和后继非历史错误输出的时间边界。已观测跨度是持续时间下界；单次错误跨度为 0，不代表真实错误持续时间为 0。删失段不推断精确结束时刻。
- 每个观测回答的当前/变化前 GT 得分、原始文本及 `llm_skipped` 诊断；仅用作路由描述，不作为模块因果归因。
- S1–S4/Unclassified/ALL、变化类型分组；ALL 统计的关键指标附序列 cluster bootstrap 区间。

固定报告三个集合：`ALL_OBSERVED_GT_CHANGES` 保留原有全部变化事件；`CONTIGUOUS_ONSET` 要求上一已知 GT 来自紧邻前帧且间隔不超过阈值；`PERSISTENT_TARGET_CONTIGUOUS_ONSET` 进一步要求 GT 能确定同一持久目标 instance ID。原始 QA 与“最近对象”选择变化不自动解释为同一车辆的移动。

输出 `change_response_events.jsonl/csv`、`change_response_summary.csv`、`paired_response_summary.csv`、`change_response_report.json`、`结果.md` 和 PDF/PNG 图；全部成功后写 `completion.json`。完整 WorldStream 与 Native 的结果支持系统级比较；证明 OGSM 与 QCSE 的协同价值仍需相同事件、相同协议下的模块消融。真实在线墙钟响应时延仍不可评估。

## GT 覆盖范围

当前完整 723 帧有 16,874 个支持类别的对象标注，其中 10,222 个满足现有可见性规则；13,155 个对象有非空官方属性。3,615 个问题中有 **1,969 个可评分 GT（54.47%）**，包括类别 308、存在性 591、最近行人位置 315、原始有限事实 QA 755。1,123 次查询转换稳定，72 次变化，其余没有可比较历史或 GT。

官方标注集合的出现/离开不等于物理对象凭空生成/消失，也不等于可见对象不存在的证明。主 QA 评分器采用正例可见性政策；**不存在可见正例时不新增负例标签**。属性/对象对的真实标注可用于环境分组，但查询集合没有覆盖所有关系/属性问题，因此不宣称这些类型具有独立完整问答准确率。

缺失或模糊的 Ground Truth、无法解析的原始 QA 参考、最近目标歧义均保留 UNKNOWN。没有用模型预测或未经验证视觉模型补齐。实时 Change Response Delay 和独立 TEST 仍没有数据支持；运行已配置的主实验补充评估不需要下载或人工指定新的数据资产。

## 输出

预检查生成 `manifest.json`、`experiment_summary.json`（明确 PREPARED_NO_INFERENCE）、`data_coverage.json`、`stability_thresholds.json`、校准/评价逐窗 JSONL、`query_ground_truth.jsonl`、`evaluation_objects.jsonl` 和 `stability_distribution.csv`。没有真实推理时不生成性能数字或假图。

配对推理完成后生成：

- `raw_frame_results.csv`、`raw_qa_results.csv`；详细原始收据在各方法 `receipts.jsonl`。
- `latency_summary.csv`、`accuracy_summary.csv`、`compute_summary.csv`、`response_delay_summary.csv`。
- `accuracy_by_query_group.csv`、`stale_by_change_type.csv`、`paired_window_summary.csv`、`scored_queries.jsonl`。
- 两种方法的生效配置、严格权重检查、独立预热、实际吞吐区间和推理完成状态。
- `figures/` 中 E2E（平均及 P95）、Overall Accuracy、Stable/Changed Accuracy、Stale Answer Rate、语义变化率对加速比共五组图，各有 PDF 和 PNG。缺失组不补值；Smoke 图有明确标记，不作为论文结果。
- `experiment_summary.json` 完整配置和统计定义；`completion.json` 只有全部配对、评分、导出成功才写出。

本目录 tests 中的构造输入只用于单元测试；报告/绘图测试写入名称包含 `SYNTHETIC_TEST` 的临时目录并自动清理，不进入 `outputs`。真实交付验证状态和输出位置见 `VALIDATION.md`。
