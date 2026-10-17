# OmniDrive 跨模块系统优化实验报告

审计与新增实验分开归档。原始模型、LoRA、Transformers 和质量门限保持不变。本报告不把完整回退的正确性称为状态复用成功。

**结论：尚未建立可生产批准的质量约束端到端净加速。** 真实移动输入的视觉跳算和任务世界状态消费以实测计数为准；离线 Prefix/Batch 未满足完整原生输出等价，Auto 保持 Full 回退。生命周期和来源正确性通过不代表已获得状态化计算收益。

本轮正式计时共 760 次方法帧执行，覆盖 76 个唯一真实帧、4 个场景。唯一数量跨方法、预热及重复去重；故障控制和独立 Profiler 不计入。

## 新增真实 GPU 实验

| Run | 方法 | 服务 | 唯一帧/计时帧 | E2E P50/P95 ms | 完整输出等价 | 同服务参考 | 同服务 P50 比值 |
|---|---|---|---:|---:|---|---|---:|
| budget_0 | adaptive | current_output | 4/4 | 9612.74/9975.10 | True | original | 0.959 |
| budget_0 | batch | current_output | 4/4 | 9416.84/9791.12 | True | original | 0.979 |
| budget_0 | m123 | persistent_state | 4/4 | 9278.61/9642.32 | True | 未测同服务基线 | — |
| budget_0 | original | current_output | 4/4 | 9221.16/9593.59 | True | original | 1.000 |
| budget_0 | unified_persistent | persistent_state | 4/4 | 9266.19/9647.02 | True | 未测同服务基线 | — |
| budget_128 | adaptive | current_output | 4/4 | 9610.46/9978.65 | True | original | 0.960 |
| budget_128 | batch | current_output | 4/4 | 9419.85/9786.32 | True | original | 0.979 |
| budget_128 | m123 | persistent_state | 4/4 | 9278.63/9843.21 | True | 未测同服务基线 | — |
| budget_128 | original | current_output | 4/4 | 9224.85/9594.74 | True | original | 1.000 |
| budget_128 | unified_persistent | persistent_state | 4/4 | 9272.83/9820.89 | True | 未测同服务基线 | — |
| language_candidates | adaptive | current_output | 8/16 | 5454.73/9061.76 | False | original | 1.444 |
| language_candidates | batch | current_output | 8/16 | 4497.90/6446.93 | False | original | 1.751 |
| language_candidates | compact | current_output | 8/16 | 7867.07/10573.25 | False | original | 1.001 |
| language_candidates | original | current_output | 8/16 | 7877.12/10717.12 | True | original | 1.000 |
| language_candidates | prefix | current_output | 8/16 | 7301.16/10138.58 | False | original | 1.079 |
| language_candidates | state_full | persistent_state | 8/16 | 8068.99/10993.32 | True | 未测同服务基线 | — |
| language_candidates | state_only | persistent_state | 8/16 | 8038.31/11006.93 | True | 未测同服务基线 | — |
| long_stream_retry | m123 | persistent_state | 64/64 | 3279.57/3402.78 | True | 未测同服务基线 | — |
| long_stream_retry | original | current_output | 64/64 | 3016.08/3081.61 | True | original | 1.000 |
| long_stream_retry | unified_output | current_output | 64/64 | 3015.01/3080.10 | True | original | 1.000 |
| long_stream_retry | unified_persistent | persistent_state | 64/64 | 3287.58/3463.60 | True | 未测同服务基线 | — |
| multi | m1 | persistent_state | 8/16 | 8031.16/10987.93 | True | m1 | 1.000 |
| multi | m12 | persistent_state | 8/16 | 8044.63/10939.83 | True | m1 | 0.998 |
| multi | m123 | persistent_state | 8/16 | 8030.90/10964.72 | True | m1 | 1.000 |
| multi | m13 | persistent_state | 8/16 | 8050.41/10957.31 | True | m1 | 0.998 |
| multi | m2 | current_output | 8/16 | 7880.45/10720.50 | True | original | 1.001 |
| multi | m23 | current_output | 8/16 | 7883.10/10735.44 | True | original | 1.000 |
| multi | m3 | current_output | 8/16 | 7877.78/10727.58 | True | original | 1.001 |
| multi | original | current_output | 8/16 | 7885.32/10725.45 | True | original | 1.000 |
| multi | unified_output | current_output | 8/16 | 7880.93/10731.08 | True | original | 1.001 |
| multi | unified_persistent | persistent_state | 8/16 | 8030.90/10965.27 | True | m1 | 1.000 |
| single | m1 | persistent_state | 8/16 | 3310.58/3371.81 | True | m1 | 1.000 |
| single | m12 | persistent_state | 8/16 | 3294.28/3386.51 | True | m1 | 1.005 |
| single | m123 | persistent_state | 8/16 | 3316.60/3404.14 | True | m1 | 0.998 |
| single | m13 | persistent_state | 8/16 | 3321.89/3381.41 | True | m1 | 0.997 |
| single | m2 | current_output | 8/16 | 3058.96/3064.52 | True | original | 0.999 |
| single | m23 | current_output | 8/16 | 3058.04/3067.26 | True | original | 0.999 |
| single | m3 | current_output | 8/16 | 3057.42/3065.69 | True | original | 1.000 |
| single | original | current_output | 8/16 | 3056.19/3065.44 | True | original | 1.000 |
| single | unified_output | current_output | 8/16 | 3055.92/3087.76 | True | original | 1.000 |
| single | unified_persistent | persistent_state | 8/16 | 3294.69/3360.62 | True | m1 | 1.005 |
| stride_refresh | m1 | persistent_state | 8/8 | 3309.93/3402.89 | True | m1 | 1.000 |
| stride_refresh | m123 | persistent_state | 8/8 | 3304.40/3364.90 | True | m1 | 1.002 |
| stride_refresh | original | current_output | 8/8 | 3057.53/3076.63 | True | original | 1.000 |
| stride_refresh | unified_persistent | persistent_state | 8/8 | 3279.76/3375.03 | True | m1 | 1.009 |

单问题使用原始 Sampling，多问题使用受控 Greedy；两者不合并。所有问题均来自原始数据管线并在实验前固定，320 Token 上限与正常 EOS 保留。完整服务 wall 包含本次输入读取/预处理、collate/scatter、同步、缓存和实际推理；GT 与输出差异计算在计时外。原生输出解析和所有方法同样的 Token 记录在计时内。Original 对照也使用相同的帧记录、生命周期包装与阶段插桩，实际模型计算保持原生；这些绝对耗时对应共同插桩条件，未测量无插桩部署的绝对延迟。组件 CUDA interval 包含 launch gap，不能当作 kernel time；嵌套阶段不相加，组件 P50 不相加。

## 十二项结论

1. 主要瓶颈是原生 LLM Decode；阶段 A 的历史逐帧依据见 workload_audit.md，本轮 GPU trace 的独立 kernel 统计见 latency_breakdown.json。
2. 当前可确认的工程改进是拒绝路径的采样快照。任何 Prefix/Batch 离线速度必须同时看本表输出等价，不能自动归因于世界状态。
3. 有效物理 Anchor、通过验证的观测区域、视觉计算状态、任务语义状态分别记录。历史延长序列任务消费为零，来源帧元数据补齐不创造新的证据。
4. M1 成本只有实际跳算/任务收益才可能支付；前两层 Window 的历史理想节省约 43.6ms，尚未包含认证/缓存，不能支付既有完整 M1 维护。
5. M2 是否发生真实跳算以 actual_skipped_windows/actual_reused_cameras 为准；为零时不声称视觉复用加速。
6. M3 状态消费以 state_activated_questions 为准；零激活或全回退维持 No-Go。
7. Shared Prefix/Batch 的 FP16 shape 敏感性尚无完整原生输出等价修复。Adaptive 分组是独立离线候选，Sampling 在重排前回退。
8. 单问题没有同帧 Prefix 共享收益；完整服务和通用 inference-mode 控制实验分开报告，后者不属于世界状态收益。
9. 多问题质量约束下的观察比值只在完整输出等价时给出；小样本通过不自动发布生产证书。
10. Persistent-State 与 Current-Output 使用不同同服务参考。跳过 M1 只减少服务责任，不能当作同服务算法加速。
11. 保留合法来源、早期拒绝和跨模块失效假设；不支持以物理关联代替 Feature Equality、以默认完整回退证明状态消费的假设。
12. 自回归 Decode 的逐 Token 依赖、全局视觉注意力、未建立的对象到 Carrier 映射、原生 FP16 shape 敏感性，仍是当前推理修改无法保证解决的结构性限制。

## 完整性、限制和复现

本轮执行状态：`{"budget_0": {"status": "completed", "weights_unchanged": true}, "budget_128": {"status": "completed", "weights_unchanged": true}, "language_candidates": {"status": "completed", "weights_unchanged": true}, "lifecycle_validation": {"status": "completed", "weights_unchanged": true}, "long_stream": {"status": "failed", "error": "AssertionError: "}, "long_stream_retry": {"status": "completed", "weights_unchanged": true}, "multi": {"status": "completed", "weights_unchanged": true}, "single": {"status": "completed", "weights_unchanged": true}, "stride_refresh": {"status": "completed", "weights_unchanged": true}}`。
测试记录：`{"resume_cpu_tests.log": {"source": "outputs/system_optimization/logs/resume_cpu_tests.log", "sha256": "71baff7a0920cd21afff0f6a3ca9d7d5bdb8bd03838f4f2dc6ba8954c52e279c", "passed": 245, "skipped": 1, "failed": 0, "errors": 0, "completed_summary_present": true, "caveat": "GPU-visible environment does not mean every individual test executes on CUDA"}, "delivery_gpu_tests.log": {"source": "outputs/system_optimization/logs/delivery_gpu_tests.log", "sha256": "ea7244d178b1ccbda4ab53c8c227a066c6f0ce38423936e594a0bec4d50faafd", "passed": 246, "skipped": 0, "failed": 0, "errors": 0, "completed_summary_present": true, "caveat": "GPU-visible environment does not mean every individual test executes on CUDA"}}`。
正式结果只读取独立运行目录 frames.jsonl；smoke 和失败记录不混入。源代码快照、原始 Token IDs、原生参考张量、逐任务 GT 和完整 trace 均保留。
物理 FAR/关键变化召回的新独立 Oracle、官方检测/地图指标和闭环安全未测，不能由 Token 等价推断。重复计时不增加独立帧或独立场景数量。
详见 commands.sh、tools/system_optimization/run.sh 与 docs/worldstream_system_optimization.md。
插桩清单：worldstream/execution.py（嵌套时间与来源）；worldstream/runtime.py（统一入口与请求级包装）；petr3d.py（可选统一入口）；tools/system_optimization/experiment.py（真实模型、数据、输出与 Profiler）。不配置 stream_config 时保持原入口。

## 本轮关键实测值

- single/m1：M1 P50=247.82 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 16, 'total': 16}。
- single/m123：M1 P50=247.68 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 16, 'total': 16}。
- single/original：M1 P50=0.00 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 16, 'total': 16}。
- single 原生 Planning：16/16 可解析；ADE P50=0.4954475704137763 m，FDE P50=1.2323511752044682 m；原始 QA 文本标注可用 0/16，平均词级 F1=None。未标注问题不填补答案。
- single/unified_output：M1 P50=0.00 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 16, 'total': 16}。
- single/unified_persistent：M1 P50=221.04 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 16, 'total': 16}。
- multi/m1：M1 P50=236.28 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 80, 'total': 80}。
- multi/m123：M1 P50=208.21 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 80, 'total': 80}。
- multi/original：M1 P50=0.00 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 80, 'total': 80}。
- multi 原生 Planning：16/16 可解析；ADE P50=0.5773387138175681 m，FDE P50=1.386864611248929 m；原始 QA 文本标注可用 48/80，平均词级 F1=0.4335555556159379。未标注问题不填补答案。
- multi/unified_output：M1 P50=0.00 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 80, 'total': 80}。
- multi/unified_persistent：M1 P50=215.93 ms；实际 Camera 复用=0，Window 跳算=0；问题完整 Token 等价={'equal': 80, 'total': 80}。
- 离线 adaptive：完整 Token 等价={'equal': 72, 'total': 80}；状态查询激活=0/0；源计算 KV 复用=26784 tokens；仍无生产证书。
- 离线 batch：完整 Token 等价={'equal': 68, 'total': 80}；状态查询激活=0/0；源计算 KV 复用=35712 tokens；仍无生产证书。
- 离线 compact：完整 Token 等价={'equal': 70, 'total': 80}；状态查询激活=0/0；源计算 KV 复用=0 tokens；仍无生产证书。
- 离线 original：完整 Token 等价={'equal': 80, 'total': 80}；状态查询激活=0/0；源计算 KV 复用=0 tokens；仍无生产证书。
- 离线 prefix：完整 Token 等价={'equal': 70, 'total': 80}；状态查询激活=0/0；源计算 KV 复用=44640 tokens；仍无生产证书。
- 离线 state_full：完整 Token 等价={'equal': 80, 'total': 80}；状态查询激活=0/80；源计算 KV 复用=0 tokens；仍无生产证书。
- 离线 state_only：完整 Token 等价={'equal': 80, 'total': 80}；状态查询激活=0/80；源计算 KV 复用=0 tokens；仍无生产证书。

## 长序列与生命周期

以下为新场景的连续真实帧；跨方法重复不增加独立样本数。场景运动、类别和实际帧间隔见 sequence_coverage.json。

- m123：2 场景、64 唯一帧；完整输出等价=True；实际相机复用=0；保留缓存峰值=94.09 MiB，GPU allocated 峰值=15.18 GiB。
- original：2 场景、64 唯一帧；完整输出等价=True；实际相机复用=0；保留缓存峰值=0.00 MiB，GPU allocated 峰值=15.12 GiB。
- unified_output：2 场景、64 唯一帧；完整输出等价=True；实际相机复用=0；保留缓存峰值=0.00 MiB，GPU allocated 峰值=15.12 GiB。
- unified_persistent：2 场景、64 唯一帧；完整输出等价=True；实际相机复用=0；保留缓存峰值=28.46 MiB，GPU allocated 峰值=15.12 GiB。
- 单独受控复用/故障恢复：实际复用 6 个相机；清空后的恢复与干净原生完整结果相同=True。详见 lifecycle_validation/lifecycle_probe/results.json；受控重复不计入上表速度或复用覆盖率。
- 同一 Runtime 的真实场景切换：触发 ['scene_change']；完整输出/特征/时序 memory 与干净原生结果相同=True。
- 原生单问题 no_grad / inference_mode 专项：`{"no_grad": {"wall_ms": {"count": 5, "p50": 2341.7702661827207, "p95": 2348.292750492692}, "exact_token_equality": true}, "inference_mode": {"wall_ms": {"count": 5, "p50": 2338.118427200243, "p95": 2344.8020324576646}, "exact_token_equality": true}}`。单个真实输入的通用执行上下文控制，不是世界状态收益或独立后端准入。

cold 表示场景第一帧，后续帧标记 warm 仅表示序列位置，不保证缓存已填充或命中；零预算没有保留缓存。这不是进程冷启动。模型加载、每序列的配置构造和 GC 不在逐帧服务计时内。早期 single 运行的细分缓存计数只覆盖 M1 捕获张量，未含全部 CPU 历史数组；其 CPU RSS/GPU peak 是实际测量。后续运行 m1_payload_accounted=True 时包含历史图像、几何、掩码和证据。各运行源快照明确保留该诊断差异。

## 缓存预算、刷新与复现边界

budget_0 / budget_128 设置每个计算模块的缓存上限为 0 / 128 MiB，统一 Runtime 的总保留预算为该值的两倍；默认对应 512 / 1024 MiB。S7 仅比较相同帧、原始问题和计时轮次的离线 Batch，标注完整 Token 是否相同。预算为零时持久历史会显式失效，不能把其耗时降低解释为同等历史维护服务的加速。
stride_refresh 同时设置 frame_stride=2、full_refresh_interval=2，与默认间隔为联合敏感性对照；这不能独立估计两项设置各自的因果贡献。真实帧间隔保存在 sequence_coverage.json。
源码改动清单和补丁见 instrumentation_changelog.md、stage_BC.patch、stage_DE.patch；逐轮源快照与最终交付差异见 delivery_source_audit.json。

## 实验图

| 图 | 内容 | 文件 |
|---|---|---|
| S1 | Complete VLM measured components | [PNG](figures/S1.png) / [PDF](figures/S1.pdf) |
| S2 | Verified physical anchors vs maintenance cost (not task quality) | [PNG](figures/S2.png) / [PDF](figures/S2.pdf) |
| S3 | Actual visual reuse on real moving inputs | [PNG](figures/S3.png) / [PDF](figures/S3.pdf) |
| S4 | M1 to M2 to M3: actual state consumption | [PNG](figures/S4.png) / [PDF](figures/S4.pdf) |
| S5 | Offline language quality vs speed (no Auto certificate) | [PNG](figures/S5.png) / [PDF](figures/S5.pdf) |
| S6 | Single/multi service wall measured separately | [PNG](figures/S6.png) / [PDF](figures/S6.pdf) |
| S7 | Offline Batch: M3 cache budgets on matched real frames | [PNG](figures/S7.png) / [PDF](figures/S7.pdf) |
| S8 | Long-stream latency and physical state validity | [PNG](figures/S8.png) / [PDF](figures/S8.pdf) |
| S9 | Cross-module ablation; service semantics are labelled | [PNG](figures/S9.png) / [PDF](figures/S9.pdf) |
| S10 | Observed output-equal service ratios (not deployment approval) | [PNG](figures/S10.png) / [PDF](figures/S10.pdf) |
