# OmniDrive 第三轮系统优化实验报告

本轮实际测量 880 个方法帧，160 个唯一真实帧，28 个场景。全部使用 `taskset -c 0-3,6-31` 与 OmniDrive；不混入第二轮未绑定 CPU 的时延。所有模型/LoRA、原始问题、Sampling/Greedy 合同保持不变。
主实验按冻结场景顺序切分 calibration/heldout；新 Auto 场景不参与校准。长流用于独立持续维护验证，绝不重复同帧增加独立样本数。风险阈值和候选网格在 protocol.json 中先冻结。

主实验的精确问题长度分层各只有 2 个校准/2 个验证场景，未达到冻结的 3/3 支持门槛，初次校准与 Auto 完整回退单独保留。cost_support_protocol.json 在附加 GPU 输出前冻结新场景 136–143 与新 Auto 144–147，不替换原场景或降低门槛。新成本分析同时执行原先冻结的 [.49,.51] 暖帧间隔域；初次零批准结果归档于 initial_cost_calibration。额外性能场景不进入原物理风险选择集。

| Run | 方法 | 服务 | 帧 | P50/P95 ms | 完整输出 Exact |
|---|---|---|---:|---:|---|
| auto | m1 | persistent_state | 16 | 3308.30/3404.04 | True |
| auto | unified_auto | persistent_state | 16 | 3306.46/3408.97 | True |
| auto_supported | m1 | persistent_state | 16 | 3252.18/3371.00 | True |
| auto_supported | unified_auto | persistent_state | 16 | 3229.37/3348.80 | True |
| cost_support | m1 | persistent_state | 32 | 3151.18/3393.16 | True |
| cost_support | overlap | persistent_state | 32 | 3102.29/3196.47 | True |
| long | certified_overlap | persistent_state | 64 | 3123.41/3167.50 | True |
| long | certified_serial | persistent_state | 64 | 3289.09/3360.29 | True |
| long | m1 | persistent_state | 64 | 3296.58/3375.86 | True |
| long | overlap | persistent_state | 64 | 3122.01/3160.09 | True |
| long | refined_overlap | persistent_state | 64 | 3122.26/3165.98 | True |
| main | certified_overlap | persistent_state | 32 | 3102.07/3183.96 | True |
| main | certified_serial | persistent_state | 32 | 3112.96/3361.55 | True |
| main | m1 | persistent_state | 32 | 3124.47/3384.30 | True |
| main | m12 | persistent_state | 32 | 3110.49/3386.68 | True |
| main | m123 | persistent_state | 32 | 3115.46/3393.52 | True |
| main | native_decode | current_output | 32 | 2993.37/3062.95 | True |
| main | original | current_output | 32 | 2982.46/3074.35 | True |
| main | overlap | persistent_state | 32 | 3099.74/3180.07 | True |
| main | refined_overlap | persistent_state | 32 | 3099.70/3180.95 | True |
| main | refined_serial | persistent_state | 32 | 3112.21/3390.30 | True |
| main | task_compact | persistent_state | 32 | 3118.67/3396.74 | True |
| main | task_full | persistent_state | 32 | 3118.09/3394.48 | True |
| multi | adaptive | current_output | 8 | 4806.51/9132.24 | False |
| multi | batch | current_output | 8 | 3587.42/7462.37 | False |
| multi | original | current_output | 8 | 6950.12/10874.24 | True |
| multi | prefix | current_output | 8 | 6349.98/9979.49 | False |
| multi | task_compact | persistent_state | 8 | 7201.44/10933.28 | True |
| multi | task_full | persistent_state | 8 | 7221.92/10930.99 | True |

## 同服务场景配对

场景 bootstrap 10,000 次。相同输出不代表相同历史职责；certified M1 与原 M1 的状态变化另报，任务替代候选不属于 Exact KV。

| Run / 方法 / 参考 | 平均配对增量 ms | 95% CI | 时间比 |
|---|---:|---|---:|
| main / refined_serial / m1 | -6.67 | [-13.358980771408824, -0.13363974721869454] | 1.0021 |
| main / overlap / m1 | -126.22 | [-139.8050559046169, -112.77576466000028] | 1.0422 |
| main / refined_overlap / m1 | -127.49 | [-143.88582415267592, -111.55462551778335] | 1.0427 |
| main / certified_overlap / certified_serial | -124.66 | [-132.34814026382082, -116.7437089523446] | 1.0417 |
| main / m12 / m1 | -2.76 | [-8.927332611528982, 5.0176266422567775] | 1.0009 |
| main / m123 / m1 | 3.94 | [0.4325597110437229, 7.809200085284829] | 0.9987 |
| main / task_full / certified_serial | 5.04 | [-6.21144599354011, 15.19539583550794] | 0.9984 |
| main / task_compact / certified_serial | 7.61 | [-3.7466801222763024, 18.53553202090552] | 0.9976 |
| main / native_decode / original | -7.09 | [-16.728363834045012, 3.9932252911967225] | 1.0025 |
| main / refined_overlap / overlap | -1.27 | [-5.537341741728596, 2.464325283654034] | 1.0004 |
| long / overlap / m1 | -167.06 | [-186.6964226064738, -147.87572187560727] | 1.0536 |
| long / refined_overlap / m1 | -167.44 | [-185.39756441896316, -149.33454153651837] | 1.0537 |
| long / certified_overlap / certified_serial | -158.12 | [-173.71204381197458, -140.05555337280384] | 1.0507 |
| long / refined_overlap / overlap | -0.38 | [-1.803537474188488, 1.0474892624188215] | 1.0001 |
| auto / unified_auto / m1 | 5.62 | [-0.2972240617964417, 11.531213123816997] | 0.9983 |
| multi / prefix / original | -674.62 | [-779.194840870332, -570.0547994347289] | 1.0949 |
| multi / batch / original | -3349.73 | [-3647.2737926233094, -3056.6099195566494] | 1.7554 |
| multi / adaptive / original | -2002.92 | [-2397.885072568897, -1607.9541817307472] | 1.3464 |
| cost_support / overlap / m1 | -128.54 | [-145.01775264852768, -111.908031423809] | 1.0425 |
| auto_supported / unified_auto / m1 | -2.20 | [-19.916381090297364, 14.658160333056003] | 1.0007 |

## 物理风险与覆盖

单侧 95% FAR 上界为描述性 Clopper–Pearson；重复目标/帧相关，另外报告场景支持。缺少可判定身份不算真负，零接受的 FAR 和覆盖不足不能算成功。

| 候选 / split | Accepted scorable | 错误 | FAR | FAR upper95 | Valid recovery | 支持场景 | 风险通过 |
|---|---:|---:|---|---|---|---:|---|
| reference / calibration | 0 | 0 | None | None | 0.0 | 0 | False |
| reference / heldout | 0 | 0 | None | None | 0.0 | 0 | False |
| bidirectional_depth / calibration | 0 | 0 | None | None | 0.0 | 0 | False |
| bidirectional_depth / heldout | 0 | 0 | None | None | 0.0 | 0 | False |
| bidirectional_depth_strict / calibration | 0 | 0 | None | None | 0.0 | 0 | False |
| bidirectional_depth_strict / heldout | 0 | 0 | None | None | 0.0 | 0 | False |

## 十二项验收

1. 旧 29 例误接受的逐例来源/相机/对应点/残差/Oracle 与图片已保存。主要机制分类为 {'historical_source_localization_bias': 29}；属于两个旧场景的探索性诊断。相似纹理不能定位三维中心；Identity Switch/前景归属没有独立证据时保留未证实。
2. 独立风险结论：No-Go: insufficient risk/coverage support。当前批准候选=None，不以零接受充当零风险成功；静态/动态差异、真实分母和附加验证成本见 m1_risk_coverage_results.json。
3. Post-Perception 只核对真实当前 Head 预测与已冻结物理历史，不更新 Precheck/M2 证据。实际候选漏斗={'historical_candidate_count': 4784, 'physical_verified_count': 0, 'task_verified_candidates': 0, 'task_relevant_record_occurrences': 0, 'actually_consumed_record_occurrences': 0}；预测一致不是独立物理真值。
4. 原始任务查询 224，实际激活 0，消费记录出现数 0。不替换原问题以制造成功。
5. Prompt/Prefill 与验证开销逐请求记录。没有实际状态激活时不声称减少 LLM 计算；Direct Query 没有独立认证支持就不执行，数值误差和延迟保持未测。
6. Refined Overlap 对 Current Overlap 的真实净收益与区间见上表。遮挡表向量化复现原逐对几何判断，完整状态摘要、来源、感知、时序记忆与 Token 逐帧验证；新物理门则分别与自己的串行版本比较。
7. 成本校准批准 1 键；新场景实际准入 1/16，重叠 1，严格质量=True，错误质量准入=0。只在 calibration 实际观测的帧间隔范围准入；统计均值界不是实时截止保证。
8. M2 实际 Camera 复用 0、Window 跳算 0；准确 EVA Block 依赖范围与物理掩码的乐观上限见 m2_feasibility_bound.json。没有缩小 Window 或绕过 Global Attention。
9. 新单/多问题真实 Exact 与速度见质量约束表；2/3/5 问题的原失败 Carrier 逐层/同历史 Logits 诊断重新执行。仅几毫秒的通用 Decode 不归因于状态缓存；质量不同仍 No-Go，未添加无证据的大型推理后端。
10. Overlap 的调度贡献与状态消费/视觉跳算贡献分别记录。若物理风险或实际任务消费尚未通过，三个模块仍未形成经质量认证的完整缓存闭环。
11. 原生自回归 Decode 的小 GEMM、动态 KV 和 Launch、EVA Global Attention 的依赖扩散，以及全局语义问题与局部几何状态的输入需求差异，是主要结构边界。RGB 背景一致性无法解决历史定位偏差。
12. 保留已证明有效的 Overlap 与严格依赖执行边界；后续应围绕具有明确可验证输入需求的任务/对象抽象推进，而不是继续优化零命中的 Window Packing 或降低风险阈值。

## 测量与限制

Full Carrier、History+Full、Task Compact 均以原始实际问题运行；不合格输入保持 Full Carrier。Full 回退的输出等价不证明状态替代质量。QA Token F1 是描述性重叠，不当作语义或安全证书。
`certified_serial/certified_overlap` 是开启额外几何拒绝门的实验名称，是否取得物理风险认证以 m1_risk_coverage_results.json 为准；名称本身不表示风险已通过。新几何门的实际有效 Anchor 集与离线 Replay 另作逐帧对照。
性能表的 Exact 字段只表示该固定输入集合上的完整输出检查；多问题的新场景即使一致，也不能抹去原失败 Carrier 的数值/完整 Token 反例。已知反例单列于 multi_question_quality.json，未授予 Prefix/Batch 通用生产许可。
同服务时延包含输入准备、GPU 同步、实际输出与 Token 捕获；定时后的大张量审计和物理快照、独立 Oracle、原始 Logits 与 Profiler 单独运行。Snapshot/观测等基础成本与所有共同插桩均计费；不相加组件 P50。
Cold 指同一个已加载模型上的冷流/缓存首帧，Warm 指该连续流后续帧；checkpoint 加载和实验配置安装不计入逐帧 E2E，也不当作冷进程启动性能。各场景采用固定轮换方法顺序，额外 Native Reference 不增加独立数据数。
CPU RSS 的逐帧字段是进程终生高水位；独立 20ms RSS 采样和线程变化的控制实验见 resource_controls.json，不把共同加载模型的内存当作各优化的净增量。长历史第 8/16 帧的五问题补充请求和 Direct Query 可行性单列，不计入普通 E2E 或新增独立场景。
最坏观测增量、场景均值上界和预先固定分位数独立报告；Auto 缓存只缓存匹配候选，每次重新计入准入开销，模型/执行/CPU/证书内容变化都会改变键。已推进时序 Head 后不进行同帧重放，回退只对后续请求或尚未开始的优化生效。
生成 13 组 PNG/PDF 图；原始 profiler traces、线程亲和性、温度频率和共享进程记录在 traces/、probes/ 与 logs/environment_monitor.jsonl。权重参数版本/类型每个运行均检查，最终全文件哈希另检。默认不启用未获证书的新研究选项。

## 最终验证与准入范围

完整 GPU 流程和启动器均正常退出（exit 0），未出现 segmentation fault。最终 GPU 可见环境回归 281 项通过；18 个第二轮原结果文件与完整 checkpoint SHA256 校验未变，见 delivery_manifest.json 与 logs/final_gpu_tests.log。

独立物理风险选择集只使用 main/long/auto；额外成本支持场景不参与物理候选选择。原方法和新门在该集上的 Accepted Scorable 都为 0，FAR 保持 Unknown，无法认证“降低风险且保留非零可信覆盖”。旧 29 例的拒绝仅为探索性诊断，不冒充独立风险改善。

Auto 的 1 个批准键严格绑定本次实测的模型、实现、插桩/审计钩子、线程、形状、生成合同和帧间隔。普通未插桩入口或其他部署配置不能直接继承；这不是通用生产批准。独立新场景实际启用 1/16 帧，配对降低 137.41 ms；包含全部回退的平均场景增量为 -2.20 ms，95% CI [-19.92, 14.66] ms，尚无稳定的整体净收益证据。未扩大帧间隔或问题长度域来制造命中。
