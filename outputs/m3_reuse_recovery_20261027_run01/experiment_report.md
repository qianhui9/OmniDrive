# M3 WSDR-KV 冻结历史策略诊断与验收（最终部署见 FINAL_REPORT.md）

完成真实 `ckpts/iter_10548.pth` 实验。当前正式代码在历史工作负载重现 `5607` Tokens 复用；旧输入隔离回放为 `5444` Tokens、`72/72` 回答与相同视觉的 Native 一致。最新单问题流的零复用主要由工作负载与原有质量/成本证据解释，未发现 PrefillExecutor 丢弃获准前缀的回归。

## 真实基线与代码差异

`baseline_evidence.json` 对历史 4522d3b6 与任务开始 main 09e258d3 逐文件比较，`historical_source.diff` 保存实际差异。Runtime、Planner、Scheduler、Matcher、VisualState、TaskConditioner、PrefixKVCache 和策略字节一致；PrefillExecutor 仅改变 Decode position_ids 的准备方式：预计算 prompt unmasked count，加上生成步偏移。没有替换 Prefill、KV 分支、因果 Mask 或最后 Token logits。

历史连续 24 帧×3 个原标注问题、顺序、生成设置、checkpoint 和策略保持，当前整个 M1/M2/M3 管线复用分解为 `{'cross_frame_bounded_approx': {'requests': 7, 'tokens': 1141}, 'same_frame_exact': {'requests': 8, 'tokens': 4466}}`。当前整个管线与旧管线答案相同 `57/72`，不能把不同 M1/M2 视觉输出的答案差异归因于 M3。为隔离这一点，另使用历史 native_reference 的实际 Carriers、IDs、逐问题 RNG、M1 Snapshot、M2 Receipt、原生布局和生成设置，只运行当前 M3；全部 Native 原始 Token 已逐项复现。

旧输入隔离回放：平均 GT F1 下降 `0.000000`，新增关键代理错误 `0`。隔离回放不含当前图像读取、Detection/Map 或 M1/M2，绝不作为完整系统 E2E 时延。

## 为什么原始 48 问的 42 个热缓存请求仍复用零 Token

| 原始任务 Profile | 问题数 |
| --- | --- |
| object_attributes | 6 |
| full_visual | 42 |

| 逐请求首个 Exact 候选拒绝条件 | 请求数 |
| --- | --- |
| cold_cache | 6 |
| critical_regressions | 37 |
| quality_loss_exceeds_budget | 5 |

| 请求至少一个候选遇到的条件（允许重叠） | 请求数 |
| --- | --- |
| critical_regressions | 37 |
| quality_evidence_missing | 37 |
| quality_loss_exceeds_budget | 5 |
| input_tokens_out_of_range | 3 |
| reuse_tokens_below_range | 2 |
| risk_limit | 1 |

6 次冷启动之外，42 个请求确有至少 35 Tokens 的精确因果前缀。37 个 Full Visual 的 Exact 首先因原策略 `critical_regressions=1` 被拒，其近似规则没有质量校准记录；5 个 Object Attributes 的 Exact 因 `quality_loss_upper=0.025974025974025983 > 0.02` 被拒。Object Attributes 跨帧近似候选还会遇到风险、573–576 输入长度范围或最少 558 复用 Tokens 等限制。普通新帧最多复用前 128 个视觉 Tokens（加 35 个文本 Tokens，共 163），不能满足 558 的测量下限；已有规则主要覆盖同帧或继承近似的长前缀。Full Visual 的 35 Token 文本前缀仅代入尚未获准的旧成本式，35×.261811−16.048986−实际开销也为负；有 Exact 前缀本身不等于有净收益。

逐候选首次拒绝数值、完整策略规则、未获准的成本预测、每视觉 Block RMS/最大变化、实际 Prefix/Forward 长度与源帧见 `original48_rejection_details.jsonl`。这是原 Risk0 已记录实际统计的 CPU 重放，不执行 KV，不构造虚假命中。`single_before_rejection_details.jsonl` 是当前正式 AR2 下相同问题的真实重跑；`single_after_diagnostic_rejection_details.jsonl` 另记录真实 KV Tensor _version、存储 revision SHA、来源帧/帧索引/epoch。所有被拒请求仍执行完整 Prefill，没有 Executor 违背规划结果的证据。

## 本次正式代码修改

Scheduler 的同一套证据谓词现在同时支持选择与离线首次拒绝解释。正式 choose 不分配候选日志列表。Planner 只在跨帧、M1 连续性有效且近似规则有合格质量证据时计算逐视觉 Block RMS；同帧、Exact-only 或缺失/不合格近似规则只计算原 Exact/Legal Prefix。Prefix 的 Embedding、Token、Position、Mask、Finite 与缓存生命周期验证保留。阈值、成本长度区间、风险上限、深度和年龄约束均未放宽。

TaskConditioner 仅保守收紧：整体区域描述、通行空间、停车态势和驾驶动作响应不再仅因出现 vehicle 等词就归为 Object Attributes。没有把安全、数量、运动或复杂场景问题提升为 Map。原单问题流本身缺少可批准 Map Relations；零 Map 不是通过错误忽略问题制造的。策略未重新拟合，未新增放行规则，因而不使用失败的验证场景调阈值。

真实相同输入/缓存/M1/M2/任务 Profile 控制：`{'requests': 72, 'all_actual_fixed_histories_equal': True, 'initial_cache_storage_immutable': True, 'planner_equivalence_all_checks_pass': True, 'headline_performance': False, 'reused_requests': 5}`。`planner_equivalence.json` 使用实际观测调度开销固定成本项，只隔离统计删除，比较冻结旧 Planner 与当前决策、风险、来源和年龄。统计微基准见下表。

| 真实相邻输入 Prefix 统计 | Wall mean/P50/P95 ms | CUDA span mean/P50/P95 ms |
| --- | --- | --- |
| before | 1.194 / 1.192 / 1.203 | 1.187 / 1.185 / 1.196 |
| after | 0.207 / 0.208 / 0.215 | 0.201 / 0.201 / 0.208 |

Profiler 实际操作计数：`{'before': {'aten::sum': 87, 'aten::sqrt': 34, 'aten::max': 17, 'aten::stack': 1, 'cuda_kernel_events': 347, 'sampled_active_kernel_total_ms': 1.049}, 'after': {'aten::stack': 1, 'cuda_kernel_events': 41, 'sampled_active_kernel_total_ms': 0.166}}`。active_kernel_total 为单次 Profiler 采样的实际 CUDA Kernel 总时间；微基准 CUDA span 还含 host 提交空隙，两者分别解释。额外 GPU 标签、旧 Planner 检查和日志探针只在离线诊断执行，整段 `same_input_control` 不参与 headline 性能；不将统计的微秒级收益夸大为系统加速。

## 自然单问题与多问题流的机会和质量

单问题保留上一轮 6 场景×8 帧、48 个原问题及顺序；多问题使用未参与历史 M3 拟合与此前测试的 4 个新场景×6 帧，每帧 3 个原生标注问题。场景按输入自车位移分位冻结；多问题选择沿用预先冻结的任务分布平衡规则，问题名称显式写入 Manifest，不人工重复问题，也不改变原生 object→ego→map 布局。多问题收益不能外推到单问题流。

| 工作负载 | 实际 Profiles | KV 分类：请求数/Token | 总 Prefill Token | 实际 Prefill Token | 跳过比例 |
| --- | --- | --- | --- | --- | --- |
| multi | {'object_attributes': 12, 'full_visual': 43, 'map_relations': 13, 'object_ego_motion': 4} | {'same_frame_exact': {'requests': 2, 'tokens': 1117}, 'cross_frame_bounded_approx': {'requests': 3, 'tokens': 489}} | 41939 | 40333 | 3.829% |
| single | {'full_visual': 48} | {} | 28169 | 28169 | 0.000% |

| M3 相对相同 M1/M2 零复用 | 自然答案相同 | 平均 F1 绝对下降 | 最大单请求下降 | 新增关键代理错误 |
| --- | --- | --- | --- | --- |
| multi | 71/72 | -0.000635 | 0.000000 | 0 |
| single | 48/48 | 0.000000 | 0.000000 | 0 |

| 仅实际复用请求 | 请求数 | 平均 F1 绝对下降 | 质量上界 | 最大请求下降 | 新增关键代理错误 |
| --- | --- | --- | --- | --- | --- |
| multi | 5 | -0.009142 | 0.008776 | 0.000000 | 0 |
| single | 0 | — | — | — | 0 |

数字/空间/运动/安全检查使用原有词法代理，不能替代人工语义认证；每个实际复用请求的原问题、GT、两个自然答案及逐项差异在 `*_m3_natural_quality_pairs.json`。同样输入下前三个增量步及 Prefill 末 Token 的未强制选择 logits 误差保存在 `same_input_controls.jsonl`，包括精确/近似复用、真实来源与实际 Forward 步数；强制答案绝不参与质量评分。

| 完整系统相对 Native | 指标 | Native | M123 | 绝对下降 | 相对下降 |
| --- | --- | --- | --- | --- | --- |
| multi | mAP | 0.342245 | 0.338691 | 0.003554 | 0.010384 |
| multi | NDS | 0.368276 | 0.355298 | 0.012978 | 0.035239 |
| multi | DET_l | 0.189853 | 0.189905 | -0.000052 | -0.000274 |
| multi | Token_F1 | 0.465060 | 0.464334 | 0.000727 | 0.001562 |
| single | mAP | 0.256546 | 0.245408 | 0.011138 | 0.043417 |
| single | NDS | 0.371669 | 0.373210 | -0.001540 | -0.004145 |
| single | DET_l | 0.118298 | 0.120381 | -0.002083 | -0.017610 |
| single | Token_F1 | 0.471328 | 0.470793 | 0.000536 | 0.001137 |

| 工作负载/场景 | mAP loss | NDS loss | Lane DET_l loss | Token F1 loss |
| --- | --- | --- | --- | --- |
| multi/6741d407b1b44511853e5ec7aaee2992 | 0.008494 | 0.006078 | -0.003481 | 0.010052 |
| multi/b94fbf78579f4ff5ab5dbd897d5e3199 | 0.000760 | 0.000324 | -0.013620 | 0.000000 |
| multi/d3b86ca0a17840109e9e049b3dd40037 | -0.007156 | -0.000339 | -0.009136 | -0.002003 |
| multi/fcc020250f884397965ba00c1d9ad9e6 | 0.024161 | 0.043100 | 0.003601 | -0.005143 |
| single/3a2d9bf6115f40898005d1c1df2b7282 | 0.011327 | 0.008199 | -0.004519 | 0.000000 |
| single/3dd2be428534403ba150a0b60abc6a0a | -0.001769 | -0.000956 | -0.007028 | 0.000000 |
| single/91f797db8fb34ae5b32ba85eecae47c9 | -0.005580 | -0.002086 | 0.005990 | 0.000459 |
| single/952cb0bcd89b4ca4b904cdcbbf595523 | 0.024425 | 0.007654 | 0.002935 | -0.002013 |
| single/bd338b912ce9434995b29b6dac9fbf1d | 0.002613 | 0.005305 | -0.035581 | 0.004768 |
| single/ed242d80ccb34b139aaf9ab89859332e | 0.009111 | 0.002513 | 0.009477 | 0.000000 |

M3 内部 .02 和完整系统每项 .10 预算分开判断。Detection/Map 质量来自真实预测与官方 GT 评价；零复用与有复用的感知逐帧摘要完全一致。完整系统与 Native 的差异包含固定 M1/M2，不能把它们全部归给 KV。全部数字仅对声明的小子集成立。

| 自然生成（单轮观测） | 请求 mean/P50/P95 ms/帧 | M3 Wall mean/P50/P95 ms/帧 | 实际生成 Tokens | 实际 Prefill Tokens |
| --- | --- | --- | --- | --- |
| multi/native | 4449.869 / 3685.819 / 7108.516 | 3569.391 / 2789.124 / 6184.850 | 2897 | 41939 |
| multi/reuse | 4673.230 / 3782.643 / 7662.332 | 3581.327 / 2747.089 / 6230.795 | 2898 | 40333 |
| single/native | 2392.329 / 1953.440 / 4528.933 | 1505.299 / 1053.716 / 3645.380 | 2507 | 28169 |
| single/reuse | 2296.268 / 1846.810 / 4200.012 | 1500.451 / 1110.765 / 3605.253 | 2486 | 28169 |

自然生成表仅为一次未截短的真实服务观测；输出长度可能不同，不替代三轮固定 Token 计算收益验收。带输入采集的 free_zero 时延不作为性能对照。

人工式图像核查结果：`{'frame_id': '3513b64493ee42b3a231d5fffc0f0d81', 'scene_id': 'd3b86ca0a17840109e9e049b3dd40037', 'question': 'What kind of road markings can you see, and what might they indicate about traffic flow?', 'reference': 'There are dashed white lines on the road indicating traffic lanes, with arrows suggesting straight movement or a turn to the right, implying options for traffic flow at the intersection ahead.', 'zero_answer': 'The road markings include a pedestrian crossing and lane dividers. These indicate that the road has designated areas for pedestrians to cross and separate lanes for vehicles, suggesting a well-organized traffic flow.', 'reuse_answer': 'The road markings include a double yellow line, which typically indicates a no-passing zone, and white dashed lines marking the lanes. These markings suggest a two-way traffic flow with one lane for each direction.', 'reused_tokens': 163, 'automatic_critical_regressions': 0, 'manual_semantic_rejections': 1, 'reviewer': 'Codex direct inspection of all six original camera images; conservative engineering review, not new official GT annotation', 'newly_unsupported_claims': ['double yellow line', 'no-passing zone', 'two-way traffic with one lane in each direction'], 'visible_support': 'White pedestrian crossing/lane markings and physical road separation are visible; the new double-yellow/no-passing assertion is not supported by the current views or reference.', 'judgement': 'Reject this approximate rule for deployment: the higher lexical F1 did not establish semantic correctness. Preserve automatic scores and add separate failure evidence; do not relabel automatic proxy output.', 'images': {'CAM_FRONT': {'path': './data/nuscenes/samples/CAM_FRONT/n008-2018-09-18-15-12-01-0400__CAM_FRONT__1537297966262404.jpg', 'sha256': '3ad7dff5233482cc0b3268c99ade49977d7a62ead8e403a973651901e63c3320'}, 'CAM_FRONT_RIGHT': {'path': './data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-09-18-15-12-01-0400__CAM_FRONT_RIGHT__1537297966270482.jpg', 'sha256': 'df7a68fc7f432299a29cdb5b4d625423352224e90d47ce4118f2a3efc516b1d7'}, 'CAM_FRONT_LEFT': {'path': './data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-09-18-15-12-01-0400__CAM_FRONT_LEFT__1537297966254799.jpg', 'sha256': 'e30cfc3e1ce22f540c7bafd19625e084a15f3e088116610d622e3e419b33e820'}, 'CAM_BACK': {'path': './data/nuscenes/samples/CAM_BACK/n008-2018-09-18-15-12-01-0400__CAM_BACK__1537297966287565.jpg', 'sha256': 'dcd382a19bd3373bc302b31659ce5ed8a9ad9390ad7a5dab20c23446cf7e5889'}, 'CAM_BACK_LEFT': {'path': './data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-09-18-15-12-01-0400__CAM_BACK_LEFT__1537297966297405.jpg', 'sha256': '2500f00c952ab2be1dd2ab52f8c505788af667b0428d5b567868f13759bc68de'}, 'CAM_BACK_RIGHT': {'path': './data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-09-18-15-12-01-0400__CAM_BACK_RIGHT__1537297966278113.jpg', 'sha256': '97f146b8bd3d02c9117631b27622ac1a1665c6b624c22e733381fc9657614cf9'}}}`。自动关键词代理零错误不等于语义通过；出现新增双黄线/禁超车/车道数断言，本阶段冻结 Map Approx 候选按语义风险拒绝。原 F1 和自动代理分数完整保留，后续限制性策略在新场景验证，未利用这些已观察场景重调阈值。

## 相同 Token 的 Prefill、完整 M3 和 E2E

三个必要条件为 Native 完整计算、相同 M1/M2 下正式 Planner/Cache/Executor 固定复用长度零、有真实 WSDR-KV 的完整管线。零对照仅为 Benchmark 的决策覆盖，不是第二个 Runtime。固定序列来自每个原问题的 Native 自然输出，所有条件仍执行真实 HF Forward、Attention、Position 和 KV 更新，逐请求验证相同输出 ID、增量输入历史与实际 Forward 次数。选择开销计入请求，未扣除估算 Token 时间。另有相同视觉、原始 IDs、RNG 和初始真实 KV 存储的三条件隔离回放；为逐请求探针拆开批次时，保持原三问题真实批次测得的调度开销，并校验获准长度与原批次相同，不将拆分探针时延混入主表。

正式自然生成仍使用原 max_new_tokens=320、temperature=.1、top_p=.75。固定 Token 只在 Harness 中应用统一选择处理器并禁用提前 EOS，正常答案和 M3 generation binding 保持。自由生成 zero 条件包含输入采集，只用于质量/诊断，不能与未采集条件作时延结论。所有 headline 时延来自无 trace/无采集的三轮固定 Token 完整请求，轮次顺序 Native/Zero/Reuse、Reuse/Zero/Native、Zero/Native/Reuse。

| 负载/条件（三轮） | Prefill CUDA mean/P50/P95 ms/帧 | 完整 M3 Wall mean/P50/P95 ms/帧 | 模型 Wall mean/P50/P95 | 请求 Wall mean/P50/P95 | 实际生成 Tokens | 实际 Decode Forward |
| --- | --- | --- | --- | --- | --- | --- |
| multi/native | 555.231 / 545.489 / 592.066 | 3579.424 / 2801.355 / 6241.455 | 4111.188 / 3331.770 / 6773.372 | 4326.408 / 3546.788 / 7099.621 | 8691 | 8475 |
| multi/zero | 549.350 / 540.689 / 587.080 | 3592.024 / 2810.509 / 6252.538 | 4286.949 / 3305.040 / 6871.773 | 4561.625 / 3671.732 / 7290.698 | 8691 | 8475 |
| multi/reuse | 535.111 / 539.760 / 582.867 | 3581.269 / 2750.598 / 6263.386 | 4277.626 / 3312.069 / 6882.470 | 4470.623 / 3518.977 / 7117.060 | 8691 | 8475 |
| single/native | 188.979 / 183.486 / 225.126 | 1503.661 / 1051.904 / 3973.914 | 2031.284 / 1576.873 / 4499.432 | 2273.489 / 1843.347 / 4650.585 | 7521 | 7377 |
| single/zero | 185.385 / 179.597 / 221.047 | 1512.625 / 1059.581 / 3986.579 | 2146.715 / 1691.270 / 4266.604 | 2311.851 / 1857.166 / 4424.519 | 7521 | 7377 |
| single/reuse | 185.221 / 179.616 / 220.941 | 1511.661 / 1058.546 / 3980.115 | 2136.629 / 1697.111 / 4244.686 | 2311.029 / 1868.682 / 4404.335 | 7521 | 7377 |

| 负载/对照 | 阶段 | 加速比 | 配对节省 mean ms | 场景 CI95 ms | 稳定净收益 |
| --- | --- | --- | --- | --- | --- |
| multi/reuse_vs_zero | prefill_ms | 1.026610 | 14.239374 | [1.249, 36.403] | False |
| multi/reuse_vs_zero | m3_ms | 1.003003 | 10.755426 | [-4.162, 32.769] | False |
| multi/reuse_vs_zero | model_ms | 1.002179 | 9.322864 | [-4.843, 31.517] | False |
| multi/reuse_vs_zero | request_ms | 1.020356 | 91.002434 | [70.810, 118.090] | True |
| multi/reuse_vs_zero | preprocessing_ms | 1.433235 | 81.637005 | [72.343, 88.067] | False |
| multi/reuse_vs_zero | non_model_ms | 1.423216 | 81.679570 | [72.413, 88.119] | False |
| multi/reuse_vs_native | prefill_ms | 1.037599 | 20.119622 | [6.794, 43.404] | True |
| multi/reuse_vs_native | m3_ms | 0.999485 | -1.844732 | [-16.254, 23.261] | False |
| multi/reuse_vs_native | model_ms | 0.961091 | -166.437487 | [-210.434, -88.176] | False |
| multi/reuse_vs_native | request_ms | 0.967742 | -144.214734 | [-188.689, -59.249] | False |
| multi/reuse_vs_native | preprocessing_ms | 1.117662 | 22.171715 | [14.369, 28.978] | False |
| multi/reuse_vs_native | non_model_ms | 1.115145 | 22.222753 | [14.395, 29.092] | False |
| single/reuse_vs_zero | prefill_ms | 1.000887 | 0.164310 | [-0.008, 0.340] | False |
| single/reuse_vs_zero | m3_ms | 1.000638 | 0.964516 | [0.351, 1.601] | False |
| single/reuse_vs_zero | model_ms | 1.004720 | 10.085823 | [3.968, 14.489] | False |
| single/reuse_vs_zero | request_ms | 1.000355 | 0.821440 | [-7.815, 9.527] | False |
| single/reuse_vs_zero | preprocessing_ms | 0.946494 | -9.083704 | [-16.062, -2.768] | False |
| single/reuse_vs_zero | non_model_ms | 0.946879 | -9.264383 | [-16.343, -2.883] | False |
| single/reuse_vs_native | prefill_ms | 1.020293 | 3.758643 | [3.506, 3.934] | True |
| single/reuse_vs_native | m3_ms | 0.994708 | -7.999687 | [-8.750, -7.339] | False |
| single/reuse_vs_native | model_ms | 0.950696 | -105.344475 | [-188.130, -17.347] | False |
| single/reuse_vs_native | request_ms | 0.983756 | -37.540680 | [-119.906, 48.418] | False |
| single/reuse_vs_native | preprocessing_ms | 1.398514 | 67.656185 | [61.751, 73.804] | False |
| single/reuse_vs_native | non_model_ms | 1.388782 | 67.803794 | [61.762, 74.057] | False |

Bootstrap 固定 seed=20261027、10,000 次，先对场景内全部帧和三轮重复的配对节省求均值，再重采样场景。独立场景只有单问题 6 个、多问题 4 个，不能把重复帧当独立场景。稳定判断要求场景区间下界>0、三轮平均与每场景平均节省均为正；否则只报告观察值与不确定性。

| 负载/对照/阶段 | 第0/1/2轮加速比 | 场景平均节省 ms |
| --- | --- | --- |
| multi/reuse_vs_zero/prefill_ms | 1.0250 / 1.0265 / 1.0283 | {'6741d407b1b44511853e5ec7aaee2992': 3.299, 'b94fbf78579f4ff5ab5dbd897d5e3199': -0.613, 'd3b86ca0a17840109e9e049b3dd40037': 6.834, 'fcc020250f884397965ba00c1d9ad9e6': 47.438} |
| multi/reuse_vs_zero/m3_ms | 1.0006 / 1.0047 / 1.0038 | {'6741d407b1b44511853e5ec7aaee2992': 2.831, 'b94fbf78579f4ff5ab5dbd897d5e3199': -7.046, 'd3b86ca0a17840109e9e049b3dd40037': 4.489, 'fcc020250f884397965ba00c1d9ad9e6': 42.748} |
| multi/reuse_vs_zero/model_ms | 1.0033 / 1.0026 / 1.0007 | {'6741d407b1b44511853e5ec7aaee2992': -1.703, 'b94fbf78579f4ff5ab5dbd897d5e3199': -7.888, 'd3b86ca0a17840109e9e049b3dd40037': 4.291, 'fcc020250f884397965ba00c1d9ad9e6': 42.591} |
| multi/reuse_vs_zero/request_ms | 1.0018 / 1.0276 / 1.0314 | {'6741d407b1b44511853e5ec7aaee2992': 66.259, 'b94fbf78579f4ff5ab5dbd897d5e3199': 75.361, 'd3b86ca0a17840109e9e049b3dd40037': 90.058, 'fcc020250f884397965ba00c1d9ad9e6': 132.333} |
| multi/reuse_vs_zero/preprocessing_ms | 0.9588 / 1.5025 / 1.6923 | {'6741d407b1b44511853e5ec7aaee2992': 67.872, 'b94fbf78579f4ff5ab5dbd897d5e3199': 83.242, 'd3b86ca0a17840109e9e049b3dd40037': 85.759, 'fcc020250f884397965ba00c1d9ad9e6': 89.675} |
| multi/reuse_vs_zero/non_model_ms | 0.9594 / 1.4928 / 1.6775 | {'6741d407b1b44511853e5ec7aaee2992': 67.961, 'b94fbf78579f4ff5ab5dbd897d5e3199': 83.249, 'd3b86ca0a17840109e9e049b3dd40037': 85.766, 'fcc020250f884397965ba00c1d9ad9e6': 89.742} |
| multi/reuse_vs_native/prefill_ms | 1.0355 / 1.0401 / 1.0372 | {'6741d407b1b44511853e5ec7aaee2992': 8.05, 'b94fbf78579f4ff5ab5dbd897d5e3199': 5.537, 'd3b86ca0a17840109e9e049b3dd40037': 11.702, 'fcc020250f884397965ba00c1d9ad9e6': 55.189} |
| multi/reuse_vs_native/m3_ms | 0.9980 / 1.0011 / 0.9994 | {'6741d407b1b44511853e5ec7aaee2992': -13.761, 'b94fbf78579f4ff5ab5dbd897d5e3199': -17.898, 'd3b86ca0a17840109e9e049b3dd40037': -11.322, 'fcc020250f884397965ba00c1d9ad9e6': 35.602} |
| multi/reuse_vs_native/model_ms | 0.9608 / 0.9627 / 0.9597 | {'6741d407b1b44511853e5ec7aaee2992': -196.048, 'b94fbf78579f4ff5ab5dbd897d5e3199': -214.666, 'd3b86ca0a17840109e9e049b3dd40037': -206.202, 'fcc020250f884397965ba00c1d9ad9e6': -48.834} |
| multi/reuse_vs_native/request_ms | 1.0111 / 0.9444 / 0.9484 | {'6741d407b1b44511853e5ec7aaee2992': -185.035, 'b94fbf78579f4ff5ab5dbd897d5e3199': -190.125, 'd3b86ca0a17840109e9e049b3dd40037': -184.379, 'fcc020250f884397965ba00c1d9ad9e6': -17.32} |
| multi/reuse_vs_native/preprocessing_ms | 2.5277 / 0.5982 / 0.7028 | {'6741d407b1b44511853e5ec7aaee2992': 11.001, 'b94fbf78579f4ff5ab5dbd897d5e3199': 24.47, 'd3b86ca0a17840109e9e049b3dd40037': 21.866, 'fcc020250f884397965ba00c1d9ad9e6': 31.348} |
| multi/reuse_vs_native/non_model_ms | 2.4851 / 0.6044 / 0.7094 | {'6741d407b1b44511853e5ec7aaee2992': 11.013, 'b94fbf78579f4ff5ab5dbd897d5e3199': 24.541, 'd3b86ca0a17840109e9e049b3dd40037': 21.823, 'fcc020250f884397965ba00c1d9ad9e6': 31.514} |
| single/reuse_vs_zero/prefill_ms | 0.9988 / 1.0015 / 1.0023 | {'3a2d9bf6115f40898005d1c1df2b7282': 0.147, '3dd2be428534403ba150a0b60abc6a0a': 0.191, '91f797db8fb34ae5b32ba85eecae47c9': -0.131, '952cb0bcd89b4ca4b904cdcbbf595523': 0.519, 'bd338b912ce9434995b29b6dac9fbf1d': -0.064, 'ed242d80ccb34b139aaf9ab89859332e': 0.323} |
| single/reuse_vs_zero/m3_ms | 0.9994 / 1.0012 / 1.0013 | {'3a2d9bf6115f40898005d1c1df2b7282': 0.653, '3dd2be428534403ba150a0b60abc6a0a': -0.083, '91f797db8fb34ae5b32ba85eecae47c9': 0.328, '952cb0bcd89b4ca4b904cdcbbf595523': 1.91, 'bd338b912ce9434995b29b6dac9fbf1d': 0.849, 'ed242d80ccb34b139aaf9ab89859332e': 2.129} |
| single/reuse_vs_zero/model_ms | 1.0041 / 1.0022 / 1.0079 | {'3a2d9bf6115f40898005d1c1df2b7282': 14.199, '3dd2be428534403ba150a0b60abc6a0a': -3.599, '91f797db8fb34ae5b32ba85eecae47c9': 13.616, '952cb0bcd89b4ca4b904cdcbbf595523': 13.715, 'bd338b912ce9434995b29b6dac9fbf1d': 6.789, 'ed242d80ccb34b139aaf9ab89859332e': 15.794} |
| single/reuse_vs_zero/request_ms | 1.0026 / 0.9825 / 1.0167 | {'3a2d9bf6115f40898005d1c1df2b7282': -1.718, '3dd2be428534403ba150a0b60abc6a0a': -12.881, '91f797db8fb34ae5b32ba85eecae47c9': 12.982, '952cb0bcd89b4ca4b904cdcbbf595523': -10.36, 'bd338b912ce9434995b29b6dac9fbf1d': 1.31, 'ed242d80ccb34b139aaf9ab89859332e': 15.596} |
| single/reuse_vs_zero/preprocessing_ms | 0.9806 / 0.8053 / 1.1558 | {'3a2d9bf6115f40898005d1c1df2b7282': -15.691, '3dd2be428534403ba150a0b60abc6a0a': -9.086, '91f797db8fb34ae5b32ba85eecae47c9': -0.485, '952cb0bcd89b4ca4b904cdcbbf595523': -23.704, 'bd338b912ce9434995b29b6dac9fbf1d': -5.295, 'ed242d80ccb34b139aaf9ab89859332e': -0.241} |
| single/reuse_vs_zero/non_model_ms | 0.9813 / 0.8068 / 1.1514 | {'3a2d9bf6115f40898005d1c1df2b7282': -15.917, '3dd2be428534403ba150a0b60abc6a0a': -9.283, '91f797db8fb34ae5b32ba85eecae47c9': -0.634, '952cb0bcd89b4ca4b904cdcbbf595523': -24.075, 'bd338b912ce9434995b29b6dac9fbf1d': -5.479, 'ed242d80ccb34b139aaf9ab89859332e': -0.198} |
| single/reuse_vs_native/prefill_ms | 1.0178 / 1.0210 / 1.0221 | {'3a2d9bf6115f40898005d1c1df2b7282': 3.985, '3dd2be428534403ba150a0b60abc6a0a': 3.77, '91f797db8fb34ae5b32ba85eecae47c9': 3.847, '952cb0bcd89b4ca4b904cdcbbf595523': 3.796, 'bd338b912ce9434995b29b6dac9fbf1d': 3.997, 'ed242d80ccb34b139aaf9ab89859332e': 3.157} |
| single/reuse_vs_native/m3_ms | 0.9934 / 0.9954 / 0.9953 | {'3a2d9bf6115f40898005d1c1df2b7282': -7.418, '3dd2be428534403ba150a0b60abc6a0a': -9.477, '91f797db8fb34ae5b32ba85eecae47c9': -8.946, '952cb0bcd89b4ca4b904cdcbbf595523': -7.707, 'bd338b912ce9434995b29b6dac9fbf1d': -7.412, 'ed242d80ccb34b139aaf9ab89859332e': -7.038} |
| single/reuse_vs_native/model_ms | 0.9492 / 0.9510 / 0.9519 | {'3a2d9bf6115f40898005d1c1df2b7282': -229.161, '3dd2be428534403ba150a0b60abc6a0a': -216.984, '91f797db8fb34ae5b32ba85eecae47c9': -213.783, '952cb0bcd89b4ca4b904cdcbbf595523': -22.705, 'bd338b912ce9434995b29b6dac9fbf1d': 41.999, 'ed242d80ccb34b139aaf9ab89859332e': 8.568} |
| single/reuse_vs_native/request_ms | 1.0488 / 0.9469 / 0.9568 | {'3a2d9bf6115f40898005d1c1df2b7282': -165.409, '3dd2be428534403ba150a0b60abc6a0a': -145.375, '91f797db8fb34ae5b32ba85eecae47c9': -135.514, '952cb0bcd89b4ca4b904cdcbbf595523': 37.646, 'bd338b912ce9434995b29b6dac9fbf1d': 99.559, 'ed242d80ccb34b139aaf9ab89859332e': 83.849} |
| single/reuse_vs_native/preprocessing_ms | 2.5646 / 0.9100 / 1.0320 | {'3a2d9bf6115f40898005d1c1df2b7282': 63.599, '3dd2be428534403ba150a0b60abc6a0a': 71.328, '91f797db8fb34ae5b32ba85eecae47c9': 78.104, '952cb0bcd89b4ca4b904cdcbbf595523': 60.438, 'bd338b912ce9434995b29b6dac9fbf1d': 57.427, 'ed242d80ccb34b139aaf9ab89859332e': 75.042} |
| single/reuse_vs_native/non_model_ms | 2.5214 / 0.9104 / 1.0316 | {'3a2d9bf6115f40898005d1c1df2b7282': 63.753, '3dd2be428534403ba150a0b60abc6a0a': 71.609, '91f797db8fb34ae5b32ba85eecae47c9': 78.268, '952cb0bcd89b4ca4b904cdcbbf595523': 60.351, 'bd338b912ce9434995b29b6dac9fbf1d': 57.56, 'ed242d80ccb34b139aaf9ab89859332e': 75.281} |

| 负载/条件 | 视觉 CUDA mean/P50/P95 ms | Decode CUDA mean/P50/P95 ms | 吞吐帧/s | 峰值 allocated/reserved GiB | 固定选择 CUDA mean ms |
| --- | --- | --- | --- | --- | --- |
| multi/native | 482.977 / 483.896 / 485.732 | 2971.868 / 2218.351 / 5562.503 | 0.231139 | 15.258/15.742 | 0.572623 |
| multi/zero | 446.162 / 498.557 / 509.022 | 2971.137 / 2215.342 / 5558.825 | 0.219220 | 16.099/16.840 | 0.560856 |
| multi/reuse | 446.768 / 499.639 / 510.409 | 2973.425 / 2218.569 / 5566.662 | 0.223682 | 16.098/16.834 | 0.563286 |
| single/native | 480.953 / 481.829 / 483.495 | 1291.983 / 854.227 / 3729.726 | 0.439853 | 15.226/15.740 | 0.247326 |
| single/zero | 398.999 / 496.823 / 510.330 | 1293.441 / 854.915 / 3732.919 | 0.432554 | 16.098/16.832 | 0.244436 |
| single/reuse | 397.840 / 495.770 / 509.422 | 1292.998 / 854.622 / 3730.650 | 0.432708 | 16.098/16.832 | 0.244990 |

| 相对零 KV 的收益归因 | 请求观测稳定 | 请求节省 ms | 模型节省 ms | 模型之外节省 ms | 其中预处理节省 ms | M3/模型 CI 均为正 | E2E 计算净收益已证明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi | True | 91.002434 | 9.322864 | 81.679570 | 81.637005 | False | False |
| single | False | 0.821440 | 10.085823 | -9.264383 | -9.083704 | True | False |

各阶段 CUDA span 包含 host 提交间隙，并且存在嵌套，不能直接相加重建请求 Wall。请求均值改善若主要来自预处理等模型之外时间，而 M3/模型区间跨零，不能把整段请求改善归功于 KV。完整保留所有正负样本，不扣除估算时间，也不把局部 Prefill 的显著收益替代整个模型的净收益。

## 唯一正式配置、回归与复现

本阶段实验策略与开始 main 的文件逐字节一致，内部 budget=.02；没有新增放行规则、删除 critical_regressions 或修改成本区间。最终正式策略如经收紧，以 qualification/deployment_decision.json 为准。M2 保持 AR2/.7721027880907059、ordinary_budget=3、max_reuse_frames=8。完整系统预算通过：`True`；M3 自然质量通过：`False`。相对零 KV 的跨场景稳定 E2E 判断：`{'multi': False, 'single': False}`。当前仍只有 `M1→M2→Detection/Map→M1 Commit→M3 ReusePlanner→PrefillExecutor→PrefixKVCache→Decode` 正式路线。

GPU 回归：`554 passed, 23 warnings in 9.91s`，日志 `tests_gpu_delivery.log`。所有命令使用 OmniDrive、RTX3090 和 `taskset -c 0-3,6-31`，未停止其他任务。历史输出不覆盖。复现入口见 REPRODUCE.md，完整 argv 与退出状态在 commands.json/control_commands.json。

### 插桩变更记录

| 文件 | 类型 | 用途 | 行数 |
| --- | --- | --- | --- |
| worldstream/m3/scheduler.py | 修改 | 统一证据谓词与离线首次拒绝解释 | 113 |
| worldstream/m3/reuse_planner.py | 修改 | 删除无资格近似候选的 Block 归约 | 155 |
| worldstream/m3/prefix_matcher.py | 修改 | 保留精确/合法前缀，按需统计视觉 Block | 82 |
| worldstream/m3/task_conditioner.py | 修改 | 保守收紧整体场景与驾驶动作分类 | 28 |
| tools/worldstream_m123/benchmark/worker.py | 修改 | 显式零复用控制与逐问题 RNG 采集 | 794 |
| tools/worldstream_m123/benchmark/observe.py | 修改 | 保存实际每个 Prefill 请求开始时 RNG | 252 |
| tools/worldstream_m123/benchmark/decision_replay.py | 修改 | 缓存实际版本与诊断 Block/候选解释 | 127 |
| tools/worldstream_m123/benchmark/m3_recovery.py | 新增 | 仅 Harness 固定零长度决策 | 18 |
| tools/worldstream_m123/m3_recovery_experiment.py | 新增 | 冻结对照调度 | 70 |
| tools/worldstream_m123/m3_recovery_control.py | 新增 | 原历史输入、真实同初始状态、logits 和归约控制 | 306 |
| tools/worldstream_m123/m3_recovery_analysis.py | 新增 | 质量、计数与场景级配对统计 | 251 |
| tools/worldstream_m123/m3_recovery_report.py | 新增 | 原始数据生成本报告与图 | 84 |
| tools/worldstream_m123/m3_recovery_reproduce.py | 新增 | 在新目录复现冻结的完整协议 | 43 |
| tools/worldstream_m123/m3_recovery_qualification.py | 新增 | 只收紧失败规则并在新场景验证 | 140 |
| tools/worldstream_m123/m3_recovery_final_report.py | 新增 | 区分失败候选与最终部署结果 | 69 |

正式运行不挂诊断回放或额外标签执行器。新增采集、强制选择与零长度控制均由 Benchmark flags 显式启用；可按上述工具文件单独移除，不影响正式单一路径。

单问题流仍可能天然没有获准长前缀；本轮没有为非零 Token 修改问题或安全边界。多问题场景证明适用任务仍可发生真实跨帧复用；是否构成稳定 M3/E2E 净收益，以配对表和场景区间为准，不能用命中数代替速度结论。
