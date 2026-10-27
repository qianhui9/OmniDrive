# M3 WSDR-KV：最终代码、质量与真实计算验收

**完成实现和真实 checkpoint 验证。旧 M3 的 KV 执行能力没有消失，最新单问题零复用主要来自任务分布和已冻结的质量/成本约束。新增语义失败已保留并处理：最终唯一策略禁用 Map Relations Approx，保留其余原规则及合法 Exact 复用。**

## 历史复现和零复用原因

任务开始 main 在历史24帧×3问题上精确复现5607 Tokens（8次Exact、7次跨帧Bounded Approx）。保守修正任务分类后，使用原始实际Carriers、IDs、RNG、M1 Snapshot、M2 Receipt隔离回放当前M3，Native原始Token全部复现，复用 `5444` Tokens，`72/72` 自然回答与相同视觉的Native一致。少一次163 Token近似与将“道路坡度下如何管理车速”从Map收紧为Full Visual有关，不能称作Executor失效。

原始48问的具体首次拒绝：`{'cold_cache': 6, 'critical_regressions': 37, 'quality_loss_exceeds_budget': 5}`。37次Full Visual Exact先被已有critical=1拒绝，5次Object Exact先被质量上界.025974>.02拒绝，另6次冷启动。42次热缓存都存在35 Token的真实精确前缀，但不是已证明可获净收益的前缀；Object Approx普通跨帧最大163 Token，达不到其成本证据至少558 Token的范围。输入长度、风险和缺失校准的详细数值在原始候选日志中。

## 直接修改的正式代码

只改现有TaskConditioner、Scheduler、ReusePlanner和PrefixMatcher：保守修正驾驶动作/通行空间/整体区域问题；共享证据谓词支持精确首次拒绝解释；同帧或近似证据不合格时不再计算无用的Block RMS。Exact/Legal Prefix、Finite、场景/相机/模型/来源/版本、位置和Mask检查保留。PrefillExecutor及PrefixKVCache没有被旧文件覆盖。

| 真实Prefix统计 | Wall mean/P50/P95 ms | CUDA span mean/P50/P95 ms |
| --- | --- | --- |
| before | 1.194 / 1.192 / 1.203 | 1.187 / 1.185 / 1.196 |
| after | 0.207 / 0.208 / 0.215 | 0.201 / 0.201 / 0.208 |

单次Profiler采样：`{'before': {'aten::sum': 87, 'aten::sqrt': 34, 'aten::max': 17, 'aten::stack': 1, 'cuda_kernel_events': 347, 'sampled_active_kernel_total_ms': 1.049}, 'after': {'aten::stack': 1, 'cuda_kernel_events': 41, 'sampled_active_kernel_total_ms': 0.166}}`。零需求归约移除34次sqrt、87次sum、17次max；必要的前缀stack保留。微基准前同步、交替测量6组真实相邻输入，不作为完整E2E时延。相同输入/缓存/任务Profile和固定观测调度开销的旧/新Planner决策一致，实际Forward长度与跳过Token数逐项核对。

## 为什么没有把原策略的F1提高当作质量通过

第一批新多问题流曾复用1606 Tokens（2次同帧Exact、3次跨帧Bounded Approx），71/72答案一致，另一答案F1提高且原自动关键词代理未报警。但六相机核查显示：新增“双黄线、禁止超车、每方向单车道”断言缺少图像和标注支持；零复用回答所述斑马线有图像依据。这是独立的新增语义风险，不能被F1上升抵消。

完整失败证据在 semantic_failure_review.json。原自动分数不修改，仅给已有Map Approx规则追加critical拒绝证据，不调风险/长度/成本阈值，不新增规则。该批失败场景已被观察，不再用于新规则拟合或新的独立通过声明。

## 最终收紧策略：另一批新独立场景

另4场景×6连续帧×3个真实标注问题，选择后保持原标注顺序。自然答案 `72/72` 与相同M1/M2条件下零历史KV对照一致，平均F1下降 `0.0`，新增答案差异 `0`。仅此小子集支持相对零KV无新增回答退化，不是全验证集保证。

| 工作负载/策略 | 实际KV分类 | 输入Prefill Tokens | 有效Prefill Tokens | 跳过Tokens | 下降比例 |
| --- | --- | --- | --- | --- | --- |
| 此前单问题48问 | {} | 28169 | 28169 | 0 | 0.000% |
| 最终新多问题72问 | {'same_frame_exact': {'requests': 7, 'tokens': 3908}} | 41841 | 37933 | 3908 | 9.340% |

两批多问题场景不同，1606与最终3908等计数不能用于宣称参数或算法更优。跨帧Map Approx因新增语义风险不作为最终获准收益；历史功能复现和最终部署结论分别保留。单问题流不能因多问题Exact命中而被宣称也能加速。

| 最终系统指标 | Native | M123 | 绝对下降 | 相对下降 |
| --- | --- | --- | --- | --- |
| mAP | 0.215788 | 0.214526 | 0.001263 | 0.005851 |
| NDS | 0.264042 | 0.263404 | 0.000638 | 0.002415 |
| DET_l | 0.167848 | 0.169862 | -0.002014 | -0.011997 |
| Token_F1 | 0.456529 | 0.454117 | 0.002413 | 0.005285 |

四项绝对下降≤.10：`True`。内部M3 budget=.02不变。所有任务质量仅使用正常自由生成及真实GT评价；固定Token答案不计质量。

## 固定相同Token的真实计算性能

同一新Manifest、相同六相机输入、相同问题及相同参考Token内容/数量，三个条件各做三轮交替完整请求。零复用只通过Benchmark将同一个Planner决策长度固定为零，仍使用原Executor/Cache；不建立第二Runtime。每次真实HF Forward/Attention/位置/KV更新均执行，实际增量输入历史和Forward次数校验通过。

| 最终策略对照 | Prefill CUDA加速比 | 完整M3 Wall加速比 | 完整模型加速比 | 完整请求加速比 |
| --- | --- | --- | --- | --- |
| 相同M1/M2零KV | 1.086758 | 1.013673 | 1.011271 | 0.997218 |
| Native完整计算 | 1.095676 | 1.009258 | 0.966792 | 0.967736 |

| 最终策略相对零KV | 平均节省 ms/帧 | 场景CI95 ms |
| --- | --- | --- |
| prefill_ms | 43.729139 | [12.353, 67.899] |
| m3_ms | 43.007556 | [11.407, 66.902] |
| model_ms | 43.180700 | [22.359, 60.795] |
| request_ms | -11.321042 | [-43.545, 14.425] |
| preprocessing_ms | -54.424977 | [-64.325, -46.360] |
| non_model_ms | -54.501741 | [-64.254, -46.481] |

由M3/模型计算共同支持的稳定E2E净收益：`False`。请求节省为 `-11.321` ms，其中模型之外为 `-54.502` ms。若M3或模型区间跨零，不能把整段请求改善归因于KV。原冻结策略曾出现请求改善2.04%但约82/91 ms来自模型之外，本轮没有隐藏该结果或据此宣称计算加速。

Bootstrap先在场景内合并三轮重复后按4个场景重采样10,000次，seed=20261027；重复帧不是新增独立场景。完整每轮、P50/P95、显存、吞吐、自然生成计数、来源、拒绝数值和未强制选择logits误差见两个阶段的详细报告/JSON。

## 唯一正式状态与复现

正式文件 `outputs/final_route/m3_reuse_policy.json` 的SHA为 `326ecb70d4065ab788659b55ed540226ace4dd0e6b5751bece69edaae4140c27`。只上调已失败Map Approx规则的critical字段并附原始证据，旧Full Visual critical=1与Object Exact .025974拒绝记录保留，其他所有规则和成本参数逐项未改。M2仍是AR2/.7721027880907059；ordinary_budget=3、max_reuse_frames=8及M3内部.02不变。仍只有一条M1+M2+M3正式路线，无自动算法切换。

最终完整回归：`555 passed, 23 warnings in 10.01s`，见 `tests_gpu_qualified.log`。全部真实实验使用OmniDrive与RTX3090，taskset排除CPU4/5；没有停止其他用户任务。此前失败/中间日志和历史输出保留。

复现命令见 REPRODUCE.md；冻结候选详细报告为 experiment_report.md，最终策略详细报告为 qualification/experiment_report.md，最终部署判定为 deployment_decision.json。原始大Tensor/trace在本机按SHA索引，不能把诊断额外Forward计入headline时延。
