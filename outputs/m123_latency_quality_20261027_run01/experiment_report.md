# M1+M2+M3：0.10 任务预算与端到端时延实验

四项独立质量预算：**未通过**。完整请求加速比 **1.0321×**；**未证明稳定净加速**，不得把视觉收益或预算放宽解释为完整服务加速成功。

开发集固定选择候选 AR3 / drift=0.912088304758。**独立质量失败，候选不发布；保留原 AR2 / 0.7721027880907059 正式参数，不用独立集再次调参。以下 Formal 计时均指冻结的失败候选，不是已发布正式默认值的性能。**

## 质量标准及数据

四项默认预算由 0.05 改为 0.10，逐项检查 native−candidate≤0.10，包含等号；提升不能抵消另一项超标。这是绝对 10 个百分点，非相对 10%。M3 内部 WSDR-KV quality_budget=0.02 与策略文件字节保持不变。质量预算只用于离线选择和验收。

真实 checkpoint：`ckpts/iter_10548.pth`；六相机预处理、权重、必须刷新事件、ordinary_budget=3、max_reuse_frames=8、前两块窗口选择、依赖闭包及 HF 生成参数保持原值。

开发：2 个新场景×8 连续帧；独立：4 个新场景×6 连续帧。来自未参与 run01/run02/run03 的场景，按场景 ID 排序取固定数量连续帧，没有按模型输出筛选。每帧一个有真实标签的问题。这里是小规模子集任务评分，不能代替完整 nuScenes 验证集。场景/帧/时间/问题清单见 development_sequence.json、validation_sequence.json 及各运行的 per_frame_metrics.jsonl、per_question_metrics.jsonl。

开发场景：`080a52cb8f59489b9cddc7b721808088`, `0ac05652a4c44374998be876ba5cd6fd`。

独立场景：`0dae482684ce4cd69a7258f55bc98d73`, `0e7ede02718341558414865d5c604745`, `112ca771e318478a88cfa692f61ffcac`, `16be583c31a2403caa6c158bb55ae616`。

选择规则在完成开发输出前固定：All four .10 task budgets must pass; shortlist within 1% of the lowest complete service mean, then minimize worst per-scene candidate/native latency ratio, then mean and age; no heldout retuning。只比较同一算法的 2/3/4 三个间隔、已有漂移阈值 0.9120883047580719；没有使用额外漂移阈值。独立结果不参与选择。

## 历史结果只读重评

historical_quality_reassessment/ 内重评 run03 已有预测，quality_budget=0.10，new_inference_performed=false。原始证据 SHA256 和旧报告保留；这不是新的独立验证。旧完整服务仍为 0.9851×，性能结论没有改变。

| 指标 | 旧 Native | 旧 AR4 | 绝对下降 | 相对下降 | 新预算通过 |
|---|---:|---:|---:|---:|---|
| mAP | 0.122287 | 0.051313 | 0.070974 | 0.580386 | True |
| NDS | 0.169732 | 0.120477 | 0.049255 | 0.290193 | True |
| DET_l | 0.077293 | 0.063222 | 0.014071 | 0.182042 | True |
| Token_F1 | 0.439463 | 0.458659 | -0.019196 | -0.043680 | True |

## 开发曲线

| 间隔 | 完整请求 mean ms | 视觉 span mean ms | 后缀复用帧 | mAP/NDS/DET_l/F1 绝对下降 | 四项通过 |
|---:|---:|---:|---:|---|---|
| 2 | 1828.781 | 352.271 | 5 | 0.028861 / 0.014277 / -0.000554 / 0.005168 | True |
| 3 | 1730.794 | 325.458 | 6 | 0.038176 / 0.020663 / 0.000231 / 0.005168 | True |
| 4 | 1776.983 | 299.739 | 7 | 0.059191 / 0.028249 / 0.004680 / 0.004275 | True |

[开发质量—时延曲线](quality_latency_curve.svg)。calibration_curve.json 保留每个开发场景的质量和完整服务时延。开发阶段预处理存在较大波动（Native 370.4 ms，选中候选 137.3 ms）；因此开发加速比不能全归因于模型，正式结论使用下面的三轮交替实测，不用独立数据反向重选。

## 新独立精度

| 指标 | Native | 正式候选 | 绝对下降 | 相对下降 | 通过 |
|---|---:|---:|---:|---:|---|
| mAP | 0.311478 | 0.206472 | 0.105006 | 0.337121 | False |
| NDS | 0.349219 | 0.288990 | 0.060229 | 0.172468 | True |
| DET_l | 0.074603 | 0.057118 | 0.017484 | 0.234366 | True |
| Token_F1 | 0.445313 | 0.446552 | -0.001239 | -0.002782 | True |

相对下降是 (Native−候选)/Native；负值表示改善。以下逐场景指标只作透明诊断，不新增质量硬门槛。

| 独立场景 | mAP native→formal | NDS native→formal | DET_l native→formal | F1 native→formal | 各项绝对下降 |
|---|---|---|---|---|---|
| 0dae482684ce4cd69a7258f55bc98d73 | 0.1929→0.0474 | 0.2003→0.1213 | 0.0287→0.0458 | 0.3674→0.3652 | 0.145534 / 0.078961 / -0.017184 / 0.002218 |
| 0e7ede02718341558414865d5c604745 | 0.3111→0.3074 | 0.3129→0.3113 | 0.0827→0.0968 | 0.3944→0.3944 | 0.003661 / 0.001605 / -0.014033 / 0.000000 |
| 112ca771e318478a88cfa692f61ffcac | 0.1604→0.1180 | 0.1666→0.1405 | 0.0908→0.0514 | 0.6124→0.6236 | 0.042477 / 0.026106 / 0.039414 / -0.011171 |
| 16be583c31a2403caa6c158bb55ae616 | 0.0774→0.0126 | 0.1336→0.0796 | 0.0270→0.0202 | 0.4071→0.4031 | 0.064794 / 0.054012 / 0.006750 / 0.003998 |

数字/空间/运动/安全词法代理出现退化的问题数：{'safety': 1}；退化事件数 1。逐问原生答案、正式答案、GT、数字值和各类匹配保留在 r0_formal_quality_pairs.json；代理指标不等于人工语义或安全判断。

## 三轮交替完整服务

顺序：第 0/2 轮 Native→Formal，第 1 轮 Formal→Native。每轮相同 24 帧、4 个独立场景，warmup 1 后计时。三轮是重复测量，不是 12 个独立场景。所有正常负收益请求保留。

| 条件 | 请求 mean/P50/P95 ms | 吞吐 fps | 生成 token 总数 | 复用 KV token | 峰值 allocated/reserved GiB |
|---|---|---:|---:|---:|---|
| Native | 2083.450 / 1737.478 / 3619.352 | 0.47997 | 3243 | 0 | 15.226 / 15.742 |
| Formal | 2018.595 / 1655.766 / 3792.146 | 0.49539 | 3198 | 489 | 16.070 / 16.791 |

| 条件 | 预处理 mean/P50/P95 ms | transfer/collate mean/P50/P95 ms | 模型 mean/P50/P95 ms |
|---|---|---|---|
| Native | 227.793 / 142.634 / 470.819 | 6.836 / 6.766 / 6.993 | 1848.780 / 1500.580 / 3328.868 |
| Formal | 176.799 / 181.892 / 216.502 | 6.941 / 6.817 / 7.109 | 1834.812 / 1468.406 / 3604.453 |

三轮重复输出一致性（任务摘要/回答）：{'native': (True, True), 'formal': (True, True)}。独立精度对第 0 轮评分，未把重复帧当作新增独立精度样本。

| 同一独立请求分项 CUDA span | Native mean/P50/P95 ms | 候选 mean/P50/P95 ms |
|---|---|---|
| EVA/M2 wrapper | 477.453 / 477.880 / 482.223 | 291.614 / 477.107 / 505.937 |
| Detection | 14.359 / 14.350 / 14.547 | 14.042 / 14.204 / 14.417 |
| Map | 30.374 / 30.538 / 30.982 | 29.702 / 29.818 / 30.525 |
| Prefill | 192.179 / 182.612 / 224.323 | 187.631 / 178.882 / 220.156 |
| Decode | 1110.385 / 768.038 / 2540.633 | 1094.955 / 727.488 / 2540.498 |

| 轮次 | Native mean ms | Formal mean ms | 净节省 ms | 加速比 |
|---:|---:|---:|---:|---:|
| 0 | 2249.962 | 1987.340 | 262.622 | 1.13215 |
| 1 | 1999.549 | 2025.445 | -25.896 | 0.98721 |
| 2 | 2000.838 | 2042.999 | -42.161 | 0.97936 |

E2E_net_saved_ms=64.855，重复轮均值 t(df=2) 95% 区间 [-361.085, 490.795] ms。30/72 个正常配对请求为负收益；详见 paired_requests.json。

| 场景 | 配对净节省 mean/P50/P95 ms |
|---|---|
| 0dae482684ce4cd69a7258f55bc98d73 | 221.156 / 239.627 / 638.030 |
| 0e7ede02718341558414865d5c604745 | -166.479 / -249.112 / 100.113 |
| 112ca771e318478a88cfa692f61ffcac | -54.281 / 8.448 / 438.677 |
| 16be583c31a2403caa6c158bb55ae616 | 259.023 / 253.307 / 594.640 |

## 性能归因

以下 Host wall 与 CUDA event span 可能嵌套、包含排队和同步等待，不能逐项相加。真正净节省仅来自上面的配对完整请求 wall。Cache publication 主机等待不能全部解释为复制。

同 Snapshot 的视觉控制包含首帧及所有正常负收益帧，8 帧×3 次交替：Native 486.792 / 487.254 / 491.940 ms；WSGRR-AR 210.385 / 58.563 / 489.837 ms；M2_visual_saved_ms=276.407。该边界含当前 M2 管理，排除 M1 precheck/read 与下游任务。

| 冻结候选服务分项 | mean/P50/P95 ms |
|---|---|
| visual_cuda_span_ms | 291.614 / 477.107 / 505.937 |
| m1_wall_ms | 175.132 / 145.248 / 329.643 |
| snapshot_wall_ms | 6.082 / 4.596 / 16.469 |
| m2_wall_ms | 284.551 / 474.683 / 488.590 |
| cache_publication_host_ms | 233.268 / 423.278 / 434.123 |
| m3_wall_ms | 1313.674 / 928.703 / 2805.964 |
| prefill_cuda_ms | 187.631 / 178.882 / 220.156 |
| decode_cuda_ms | 1094.955 / 727.488 / 2540.498 |
| CUDA detection | 14.042 / 14.204 / 14.417 |
| CUDA m2_global_state_update | 235.762 / 430.374 / 437.614 |
| CUDA m2_patch_embedding | 1.108 / 1.102 / 1.158 |
| CUDA m2_total | 291.614 / 477.107 / 505.937 |
| CUDA map | 29.702 / 29.818 / 30.525 |

明确归因字段见 latency_attribution.json：`{"M2_visual_saved_ms": 276.4067718138298, "M2_service_visual_span_saved_ms": 185.83872339460584, "M1_state_added_ms": 175.1318944669846, "M2_management_added_ms": 17.725135581309182, "M3_prefill_saved_ms": 4.547631369696717, "M3_decode_per_token_ms": 24.783045893702013, "E2E_net_saved_ms": 64.85489958948983}`。其中 M2_management_added_ms 仅指已观测的 validation+region-scoring Host 段，是管理成本的下界代理，不包含执行入队等未独立分离的管理开销；不能与视觉净收益再次相加。

实际 suffix hooks 验证：三轮 33/72 帧跳过后续 22 个 EVA Block，共 726 次调用；窗口复用、更新集合及原子发布审计见 formal_execution_audit.json，逐帧计数见原始 JSONL。

## M1 同状态因果控制

4 个真实帧×3 次交替回放，输入状态、感知输出和 M2 回执完全一致。关联结果、锚点和导出 Snapshot（仅去除计时字段）全部一致。另将优化前/后整条真实请求在同样 AR4 参数下比较：4 帧的任务输出摘要、自由生成答案及 token IDs 全部相同，见 real_request_equivalence.json；Profiler 时延不参与加速结论。摘要没有跨事务缓存，每次完整绑定全部可变元数据、数组内容和 Tensor 身份/版本；异常和事务一致性检查仍保留。

| M1 CPU 同状态控制 | 优化前 mean/P50/P95 ms | 优化后 mean/P50/P95 ms |
|---|---|---|
| precheck_ms | 154.412 / 194.267 / 208.628 | 148.459 / 189.604 / 192.000 |
| snapshot_ms | 5.348 / 5.722 / 11.188 | 3.008 / 3.859 / 4.321 |
| commit_ms | 26.385 / 24.625 / 53.592 | 21.087 / 22.337 / 29.647 |
| total_ms | 186.146 / 226.084 / 265.882 | 172.554 / 216.484 / 225.968 |

M1_state_added_ms 是正式服务相对无 M1 的新增维护开销，不等于优化节省。可归因的 M1 自身优化节省为 13.592 ms（7.30%）。

优化：来源摘要一次完整树遍历，移除重复 geometry/certificate/descriptor 展开；同一请求共享灰度图；patch 支持中的 float 归一化只做一次。未删除世界状态事件。fork/freeze/完整历史验证在 Profile 中占比较小，保留现有语义。M2 窗口集合输入的完整 Snapshot 对照一致。

## M3 KV 触发、拒绝诊断与 Decode

本轮自然独立任务发生 3 次合法 KV 复用、合计 489 token。 首个案例：场景 112ca771e318478a88cfa692f61ffcac，当前帧 6472a43205b04837bdec2337e4f50100，源帧 f4692d3fbdfc4d16b1f02590ad832b10；输入 578 token，复用 163，实际 suffix prefill 415；age=1、depth=1、risk=0.493241。输入落入既有风险及实测长度范围。 策略与阈值未改，复用情况随本轮自然输入变化，不能归因于调度算法修改。

4 个诊断开发帧中 3 次产生候选、共 41 个候选；决定原因 {'cold_or_invalid_cache': 1, 'no_quality_bounded_net_gain': 3}。逐候选第一拒绝原因 {'input_length_outside_measured_cost_range': 5, 'critical_regressions_in_policy_evidence': 2, 'missing_calibration_or_validation_evidence': 34}。实际复用 0 token，实际 prefill 长度 [578, 571, 575, 579]。

三轮自然独立请求的候选请求数/候选总数：60/963；决定原因 {'cold_or_invalid_cache': 12, 'no_quality_bounded_net_gain': 54, 'task_profile_not_calibrated': 3, 'quality_latency_minimum': 3}。全部候选长度、精确/结构连续长度、实际 prefill、来源及 frame/root/age/depth 保留在 m3_rejection_diagnostics.json 的 heldout_natural 中。

首次请求正常建缓存。后续缓存来源和策略绑定有效，精确前缀较短；近似候选受既有视觉 token 上限、连续因果前缀和校准范围约束。既有 full_visual 策略包含 critical regression，且未提供 approx 证据；4 帧诊断中 map_relations 的输入长度为 571，超出已测范围。object→ego→map 中前部 object 变化，不能跳过它而拼接后面的 map KV。未放宽内部 0.02 策略、没有强制产生非零复用。完整候选、视觉 RMS/最大变化、来源与 root/age/depth 见 m3_rejection_diagnostics.json。自然服务的决定原因及 prefix 长度在各逐问记录中。

固定 Decode：相同实际视觉/语言输入、同一真实 token 序列、同一完整 prefill 初始 KV、相同 FP16、29 步、5 次交替重复；每一步 logits bitwise equal，初始 KV 的版本未变。该控制只有 teacher forcing，没有改变正式自由生成逻辑。

| 固定 token 控制 | 原实现 mean/P50/P95 ms | 优化后 mean/P50/P95 ms |
|---|---|---|
| 单 token CUDA span | 24.793 / 24.772 / 24.918 | 24.783 / 24.773 / 24.927 |
| 固定步数总 wall | 719.042 / 718.999 / 719.957 | 718.764 / 718.867 / 719.711 |

Decode 改动只消除每步对完整 attention mask 的 cumsum/masked_fill，改用请求内 GPU prompt count 与追加 token 数计算同一 position_id。保持 HF EOS、采样、停止条件及生成上限。对照均值差极小且 P95 未改善，**不能宣称可靠的 Decode 加速**。正常自由生成 token 数有差异，所以自由生成 Decode 总时延差不当作单 token 算法加速。

真实 GPU trace 中主要是 FP16 矩阵乘；KV concat、attention、归一化和逐 token kernel 启动也有开销。类别 kernel 累计 ms：`{"other_elementwise": 66.34100000000592, "copy_concat_index": 68.98500000000142, "matrix_multiply": 567.0629999999376, "attention": 3.7210000000000023}`。内核累计与 Host wall 不可相加；采样/同步及全请求 trace 在 before_profile/、after_profile/，固定序列 trace 在 decode_control/。剩余主要开销是逐 token 大模型 forward，而 M1 真实图像验证/关联也继续占用 CPU。

## 单主线、测试及复现

`M1 Precheck → M2 WSGRR → DependencyPlanner → WindowExecutor → Global-State Update → StateCache Commit → Detection/Map → M1 Commit → M3 PrefillExecutor → Decode`。正式路径仍只有这一条；本次新增工具均在 tools/benchmark，未新增备用模型、在线 M2 CostModel 门控或多执行器。

GPU 回归：515 passed, 23 warnings in 7.93s。日志 tests_gpu_final.log，覆盖 tests/worldstream_m1、m2、m3、m123；包括 source tamper、只读 Snapshot、事务、真实 CUDA 缓存及 0.10 包含等号的预算判断。新测试另覆盖 Tensor 替换/原地修改及 padding 下逐步位置和 logits 一致性。

复现入口：`taskset -c 0-3,6-31 bash outputs/m123_latency_quality_20261027_run01/reproduce.sh <新的输出目录>`。所有 Python 使用 /home/qh/miniconda3/envs/OmniDrive/bin/python；CPU 4/5 不使用。GPU 任务串行锁覆盖 checkpoint 加载，GPU 被其他任务占用时等待，未终止其他用户进程。commands.json 和独立诊断日志保留真实失败与退出码；曾并发加载两个本任务模型，基线进程以 -9 结束（疑似内存压力，未读取系统 OOM 证据），未生成有效测量，已保留 *_load_failure 并重跑。

## 仪表变更记录

| 仪表变更 | 用途 | 对正式三轮时延的影响 |
|---|---|---|
| 可选 cProfile + CPU/CUDA Profiler | before/after 第 1 帧诊断 | 正式三轮不启用，诊断时延排除 |
| M1 状态与真实 M3 输入/token 捕获 | 同状态及固定序列控制 | 正式三轮不启用；诊断序列计时外落盘 |
| 原有 Observer suffix hooks / event spans | 验证真实执行及分项 | Native/Formal 统一，原始请求 wall 为最终口径 |
| concrete function Profile 汇总 | 避免同名包装函数嵌套重复计数 | 离线读取 cpu.prof，不影响运行 |
| 进程级 GPU 实验锁 | 避免空闲检查后并发加载 | 只在实验进程启动/等待，不在请求边界内 |

原始 Tensor 捕获、大型 trace、预测文件保留在服务器本目录。raw_artifacts_sha256.json 记录原始文件校验；Git 排除大型 Tensor/trace 和详细状态转储，原始逐帧、逐问指标、评分和源码快照保留。
