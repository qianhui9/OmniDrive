# OmniDrive 第二轮系统优化实验报告

只统计本目录已完成的真实运行；旧结果保持原样。单问题保留原生 Sampling（temperature=0.1、top_p=0.75、max_new_tokens=320）；多问题明确采用 Greedy。
新增正式测量 784 method-frames，168 个唯一帧，14 个独立场景。探索性 smoke 不计入正式置信区间。
CPU 硬件范围：初始批次未绑定 CPU，完成产物后以 139 退出；用户随后确认 CPU 4/5 故障。仅 cost_refresh/cost_refresh_auto 为排除故障核后的实测，两个范围单独列出，初始时延不作为健康核环境的性能保证。
输入准备、完整推理、同步、原生输出写入和共同 Token-ID 审计拷贝计入 E2E；停止计时后的 feature/output/temporal-memory NumPy 拷贝、独立物理 Oracle、M1 状态摘要/快照、Profiler 与 Logits 拷贝均不计入性能样本。早期 protocol 的 audit-copy 表述过宽，准确范围见 timing_scope_correction.json。

| Run | 方法 | 服务 | 帧数 | E2E P50/P95 ms | 完整 Token 等价帧 |
|---|---|---|---:|---:|---:|
| auto_validation | m1 | persistent_state | 8 | 3249.12/3426.75 | 8/8 |
| auto_validation | unified_auto | persistent_state | 8 | 3270.99/3407.08 | 8/8 |
| cost_refresh | m1 | persistent_state | 32 | 3020.37/3480.49 | 32/32 |
| cost_refresh | opt_serial | persistent_state | 32 | 3017.72/3449.05 | 32/32 |
| cost_refresh | overlap | persistent_state | 32 | 3005.89/3198.91 | 32/32 |
| cost_refresh_auto | m1 | persistent_state | 8 | 3274.28/3398.94 | 8/8 |
| cost_refresh_auto | unified_auto | persistent_state | 8 | 3271.22/3406.59 | 8/8 |
| long_stream | m1 | persistent_state | 128 | 3304.79/3604.41 | 128/128 |
| long_stream | original | current_output | 128 | 2832.32/3086.43 | 128/128 |
| long_stream | overlap | persistent_state | 128 | 3111.54/3309.76 | 128/128 |
| multi | adaptive | current_output | 8 | 5954.80/7255.46 | 4/8 |
| multi | batch | current_output | 8 | 4836.51/6084.35 | 5/8 |
| multi | m123 | persistent_state | 8 | 8611.62/9983.37 | 8/8 |
| multi | original | current_output | 8 | 8464.08/9777.25 | 8/8 |
| multi | original_m3 | current_output | 8 | 8465.56/9770.99 | 8/8 |
| multi | prefix | current_output | 8 | 7904.44/9194.71 | 7/8 |
| multi | state_full | persistent_state | 8 | 8631.13/10004.95 | 8/8 |
| multi | state_only | persistent_state | 8 | 8624.64/10003.64 | 8/8 |
| multi | unified_auto | persistent_state | 8 | 8620.05/10000.93 | 8/8 |
| single | m1 | persistent_state | 16 | 3080.62/3312.20 | 16/16 |
| single | m12 | persistent_state | 16 | 3068.25/3336.13 | 16/16 |
| single | m123 | persistent_state | 16 | 3069.17/3296.09 | 16/16 |
| single | native_decode | current_output | 16 | 2947.22/3063.16 | 16/16 |
| single | opt_m12 | persistent_state | 16 | 3072.17/3326.42 | 16/16 |
| single | opt_serial | persistent_state | 16 | 3076.42/3334.67 | 16/16 |
| single | original | current_output | 16 | 2954.90/3073.66 | 16/16 |
| single | original_m3 | current_output | 16 | 2954.28/3074.13 | 16/16 |
| single | overlap | persistent_state | 16 | 3008.23/3143.58 | 16/16 |
| single | overlap_decode | persistent_state | 16 | 3002.16/3132.49 | 16/16 |
| single | unified_auto | persistent_state | 16 | 3093.87/3316.97 | 16/16 |
| smoke | m1 | persistent_state | 4 | 3035.91/3270.14 | 4/4 |
| smoke | native_decode | current_output | 4 | 2856.64/3046.97 | 4/4 |
| smoke | opt_serial | persistent_state | 4 | 3014.61/3269.49 | 4/4 |
| smoke | original | current_output | 4 | 2861.77/3057.14 | 4/4 |
| smoke | overlap | persistent_state | 4 | 2940.37/3154.35 | 4/4 |
| smoke | overlap_decode | persistent_state | 4 | 2928.76/3144.46 | 4/4 |
| stride_refresh | m1 | persistent_state | 8 | 3098.75/3223.95 | 8/8 |
| stride_refresh | original | current_output | 8 | 3017.02/3055.83 | 8/8 |
| stride_refresh | overlap | persistent_state | 8 | 3062.41/3091.30 | 8/8 |

## 配对场景结果

以相同服务的原生实现为基线，独立场景 bootstrap 10,000 次；帧/重复不是独立样本，不相加组件 P50。
| Run / 方法 | 配对场景数 | 平均场景增量 ms | 95% CI ms | 同服务时间比 |
|---|---:|---:|---|---:|
| auto_validation / unified_auto | 2 | 6.65 | [-3.6133495741523802, 16.906629840377718] | 0.9980 |
| cost_refresh / opt_serial | 8 | -8.60 | [-18.364264848059975, 3.3653178874374094] | 1.0028 |
| cost_refresh / overlap | 8 | -134.54 | [-161.34669606253738, -113.79942140047206] | 1.0456 |
| cost_refresh_auto / unified_auto | 2 | 8.51 | [3.5270045627839863, 13.496284140273929] | 0.9974 |
| long_stream / overlap | 4 | -195.46 | [-231.97304431414523, -163.3982772618765] | 1.0649 |
| multi / adaptive | 4 | -2350.78 | [-2491.016783111263, -2170.3902689914685] | 1.3850 |
| multi / batch | 4 | -3655.24 | [-4173.10503416229, -3137.3741456191055] | 1.7613 |
| multi / original_m3 | 4 | 0.64 | [-2.6207957416772842, 3.895892412401736] | 0.9999 |
| multi / prefix | 4 | -682.70 | [-917.2949355561286, -562.7321159117855] | 1.0878 |
| single / m12 | 4 | 4.58 | [-6.091759307309985, 15.24596509989351] | 0.9985 |
| single / m123 | 4 | 0.55 | [-13.914926908910275, 15.00520936679095] | 0.9998 |
| single / native_decode | 4 | -7.67 | [-9.73023075493984, -5.614203459117562] | 1.0027 |
| single / opt_m12 | 4 | -9.08 | [-17.680225195363164, -2.940651829703711] | 1.0030 |
| single / opt_serial | 4 | -2.95 | [-12.604140923940577, 7.268300279974937] | 1.0010 |
| single / original_m3 | 4 | 1.32 | [0.47698343405500054, 2.291617216542363] | 0.9995 |
| single / overlap | 4 | -132.04 | [-149.09052643633913, -114.18130925449077] | 1.0451 |
| single / overlap_decode | 4 | -140.10 | [-157.5111053243745, -121.1553920438746] | 1.0480 |
| single / unified_auto | 4 | 11.45 | [2.040811494225636, 20.851521869190037] | 0.9963 |
| stride_refresh / overlap | 2 | -58.66 | [-79.5517637161538, -37.774797063320875] | 1.0191 |

## 独立物理证据

独立离线 Oracle 在 172 帧上检查 15885 个历史 Anchor，其中可判定 4259；已接受且可判定的 79 个中，物理错误 29 个，False Acceptance=36.71%，Valid State Recovery=2.79%。该描述性汇总包含显式标记的 smoke；正式 calibration/heldout 的独立统计见 by_formal_split。其余身份/地图证据保持 Unknown，不进入已验证分母。
Critical Change 区域召回=99.80%；新可见目标覆盖=99.31%。区域失效不是像素遮挡真值，也不是闭环安全认证；按场景和目标相关性的限制见 physical_oracle_results.json。
全局原问题的 Task-Consumable State Recall 无法由局部身份标注认证，保持未测；实际 Query Hit 则按真实 LLM 请求统计。现有物理误接受证据不足以批准状态替代视觉输入。M1 重叠只保证与原串行维护完全一致，不提高原认证规则的物理可信度。

## 十二项验收回答

1. 时间倒退来自统计索引错误：Dataset 全局按 timestamp 排序，原始 pkl 未同序。17 个历史序列按 scene/token/timestamp 联查均执行连续；只修覆盖统计，保留原时延。详见 sequence_identity_audit.md。
2. 正式配对 352/352 全部字段通过；实际 kernel 与 Worker 的重叠下界为 [222.5095] ms。 Anchor/Mask/Feature Provenance、感知和时序记忆等价性见 overlap_correctness.json；没有用两个时段之差冒充重叠。
3. 长序列原串行 P50/P95=3304.79/3604.41 ms，重叠=3111.54/3309.76 ms；配对场景平均变化 -195.46 ms，95% CI=[-231.97304431414523, -163.3982772618765]。 初始长序列未绑定 CPU；健康核复测（8 场景、每场景 4 帧）的串行/重叠 P50 分别为 3020.37/3005.89 ms，场景平均配对增量 -134.54 ms，95% CI=[-161.34669606253738, -113.79942140047206]。完整 P95 和服务范围见 pinned_revision_results.json。 Current-Output 只作不同服务对照。
4. 同输入独立 Profiler 的 Decode kernel sum 为 1723.5679999981635 → 1709.5389999981596 ms；每个 Token 为 24.622399999973766 → 24.421985714259424 ms。单次 Profiler 不是独立场景的性能证据，完整帧配对区间另见 single_question_results.json。 候选不改变 GEMM、权重或原始生成合同；完整 IDs、同历史 raw Logits、RNG、ADE/FDE、显存和独立 LM 计时见 decode_backend_results.json。Mask 检查的主机同步也计费。
5. 有：cost_refresh/overlap, long_stream/overlap, single/native_decode, single/opt_m12, single/overlap, single/overlap_decode, stride_refresh/overlap，其同服务场景 bootstrap 时间比下界大于 1；适用范围与生产批准仍独立判断。 逐项质量约束时间比见 quality_constrained_speedup.json。
6. 离线候选在真实流上的实际任务查询 200，实际状态输入激活 0（含长历史检查点）。这是实际 LLM 输入消费，不是生产 Auto 批准；独立任务标注质量见 long_state_supply/task_quality.json，物理 Anchor 通过不等于任务消费。
7. 正式移动场景合法 Camera 复用 0、Window 跳算 0；无命中则如实 No-Go。
8. 对既有真实失败 carrier 运行原生逐层公共前缀/缓存后缀诊断，并比较 2/3/5 问题、Batch/Adaptive 的首次不同 Token 前同历史 Logits；见 batch_probes。只作有界诊断，不降低质量门槛。
9. adaptive 完整问题 Token 等价 36/40，严格质量 No-Go；batch 完整问题 Token 等价 37/40，严格质量 No-Go；prefix 完整问题 Token 等价 39/40，严格质量 No-Go QA 原标注重叠、Planning ADE/FDE 与场景配对性能见 multi_question_results.json；近似输出收益不算 Exact 加速。
10. 独立校准批准 0 个精确键；新场景实测 8 帧，其中 0 帧启用重叠，其余保持完整串行服务。真实正准入未发生，正分支仅由回归测试验证，不宣称已完成生产加速闭环。 成本键保留精确部署/实现/线程/Shape/问题长度/合同，只允许证实范围内的性能 frame-gap 区间。无证书维持串行 M1 和完整 GPU/LLM。
11. 尚未形成：M2 合法视觉跳算或 M3 实际任务状态消费仍缺失。M1/EVA 的调度收益不等于三模块缓存因果闭环。
12. 原生单问题逐 Token Decode、动态 KV/LoRA 与小矩阵调度，M1 的独立物理证据不足，以及 Batch 形状造成的数值分歧仍构成主要边界。具体 kernel 排名和逐层差异见原始证据，未用降低输出长度或精度规避。

## 工程和边界

服务器硬件约束：用户确认 CPU 4/5 有故障，后续命令统一使用 `taskset -c 0-3,6-31`，所有子进程继承该亲和性。初始正式批次未绑定 CPU，完成全部 GPU 产物后进程退出码为 139；产物完整性复核见 formal_process_failure.json，不能把该进程记作正常退出。初始时延保留为历史测量，硬件故障对其影响不能排除。cost_refresh/cost_refresh_auto 在排除故障核后重新实测，亲和性进入成本键，禁止跨环境继承证书；见 cpu_affinity_revision.json 和 logs/cost_refresh_pinned.log。
默认保持 serial；实验显式选择 experimental，Auto 需要双证书。M1 Worker 只有一个，独占可变 Store；只与完整 EVA 重叠，在 heads/commit/M3 前 join。异常或超时先 drain 再 reset，超时不承诺硬截止。M2 若需要本帧验证则串行。
Original Persistent Serial 使用固定基线提交 b4782ab 的原 simple_test 方法；统一协调/测量层共同使用本轮版本，新增记录开销在各方法 E2E 内，未将历史旧时延直接当本轮基线。权重、嵌入 LoRA、问题、随机种子与停止条件不变。优化串行仅减少重复签名扫描。Batch 预算预检只拒绝不可能容纳的缓存，不给予复用许可。
single/multi/long_stream/stride_refresh 属于初始冻结的协调器版本；成本键策略字段补全在该进程所有 GPU 工作结束后才应用，模型计算未变。cost_refresh 与 cost_refresh_auto 是新版本的独立真实复测，旧证书归档于 initial_cost_certificates。不同版本的逐运行统计保持分开，旧 Auto 实测不冒充当前版本批准。
旧实现使用 marshal.dumps(code) 指纹，常量元组引用计数变化会改变 marshal 引用标记，造成相同代码的成本键碎片化；已通过可复现诊断确认。新实现按字节码、常量和绑定内容稳定编码，并保留执行策略与 CPU 资源约束；不靠放宽实现键解决失配。见 logs/marshal_reference_reproduction.log、worldstream/code_signature.py 和新版本独立校准。
物理 Oracle 仅离线读取独立 nuScenes annotations；2m 源身份门、1m 当前位置门、visibility=4 和传感器支持均预先冻结。未关联身份/地图/全局任务完整性不得算作已验证；投影目标区域不是像素遮挡真值或闭环安全证明。
生成 14 张实际数据图，包含 PNG/PDF；逐帧、源快照、命令、失败日志均保留。正式帧 CPU RSS 是进程高水位；独立资源控制的 20ms resident 采样峰值见 resource_controls.json，包含已加载模型和共同观察线程，不冒充全序列精确增量峰值。
long_state_supply 在真实完整维护的长历史上，于第 8/16/32 帧执行原生五问题 Greedy 与状态候选；非检查点保留单问题 Sampling。该混合服务和附加 LM 请求不合入常规单问题 E2E，也不增加独立场景数。

