# OmniDrive M2 冻结与 M3 任务条件化服务

**结论：M2 条件复用 No-Go；M3 直接回答生产 No-Go；Exact Prefix 保留为受限实验接口，没有签发新的 Auto 证书。**

冻结 M1、原始网络与权重不变。新增严格 TaskEvidenceContract 和影子/实验执行入口，默认仍完整运行原始 VLM。拒绝样本始终留在分母。

真实默认规划 profile：prefill P50 176.33 ms，decode P50 2106.16 ms；两者独立计时，未用 P50 相减。513 个视觉 token 全部保留。逐 token 执行与输出长度共同构成 decode 成本，详细 CUDA/wall、token 数、原生全问题分类和短 decode trace 见 vlm_latency_breakdown.json。

七组对照：168 次完整模型前向、336 个查询；原生全问题补充 27 个，受控负例 7 个。全部属于已暴露开发人口 exploratory_reused_population，不能重新命名为独立最终测试。默认任务、原生多问、同帧完全相同请求分别报告。

| 工作负载 | 方法 | 请求 | E2E P50/P95 ms | 冻结质量通过/检查 |
|---|---|---:|---:|---:|
| identical_requests | direct_prefix_fallback | 8 | 5399.00/5729.71 | 8/8 |
| identical_requests | evidence_shadow | 8 | 5573.83/5894.26 | 8/8 |
| identical_requests | exact_prefix | 8 | 5431.11/5709.14 | 8/8 |
| identical_requests | experimental_direct | 8 | 5581.04/5870.84 | 8/8 |
| identical_requests | frozen_exact_m2 | 8 | 5563.22/5909.41 | 8/8 |
| identical_requests | full_maintain | 8 | 5549.51/5879.37 | 8/8 |
| identical_requests | original | 8 | 5235.03/5241.65 | 0/0 |
| native_multi | direct_prefix_fallback | 8 | 8317.91/9189.81 | 5/8 |
| native_multi | evidence_shadow | 8 | 8629.61/9476.30 | 8/8 |
| native_multi | exact_prefix | 8 | 8358.62/9153.80 | 5/8 |
| native_multi | experimental_direct | 8 | 8606.53/9441.17 | 8/8 |
| native_multi | frozen_exact_m2 | 8 | 8590.18/9443.25 | 8/8 |
| native_multi | full_maintain | 8 | 8589.45/9405.97 | 8/8 |
| native_multi | original | 8 | 8108.39/8916.13 | 0/0 |
| natural_planning | direct_prefix_fallback | 8 | 3195.45/3522.08 | 8/8 |
| natural_planning | evidence_shadow | 8 | 3186.88/3514.39 | 8/8 |
| natural_planning | exact_prefix | 8 | 3184.21/3503.91 | 8/8 |
| natural_planning | experimental_direct | 8 | 3193.93/3509.63 | 8/8 |
| natural_planning | frozen_exact_m2 | 8 | 3180.69/3498.77 | 8/8 |
| natural_planning | full_maintain | 8 | 3170.75/3467.24 | 8/8 |
| natural_planning | original | 8 | 2876.17/3062.60 | 0/0 |


原生输入完备且独立质量合格的直接查询为 0；真正免除的 prefill/decode 为 0，节省 decode token 为 0。没有用关键词、语义相似、当前 Head 与历史一致性批准直接回答。没有独立可接受对象指称参考，误拒率和直接回答物理误差保持 Unknown，零接受不等于准确率 100%。受控 fixture 只证明函数执行分支，不作为真实质量证书。

固定 96 帧 M2 新鲜离线 oracle、来源漏斗和恢复实验见 m2_admission_funnel.json、m2_freeze_decision.md。部分 token 支持不会成为完整窗口复用证书。

完整最终 CUDA 回归 664 项，失败 0，跳过 0；固定全部源码后另执行 21 次七臂原模型前向，启动前与结束后哈希逐项一致。前次抽验及快照脚本版本审计保留在 final_source_validation_attempt1。初次 runner OOM 和 Python 3.9 注解失败日志保留。研究性能主实验源码与最后新增的输入提示/输出预算绑定守卫与细分计时分别固定并验证，不把前者重新签发为最终源码生产成本证书。

同服务边际收益与相对 Original 的总收益分列 runtime_cost_benefit.json；Exact Prefix 的正收益若仅出现在受控重复请求，不能外推默认单规划任务。单次重放、两场景与阶段同步插桩只支持描述性比较。所有数值失败均保留并阻止部署准入。原生局部问题的文本示例及离线参考在 task_workload_census.json，仅用于分析；当前显式 ID 原型与现有证据接口的 No-Go 不等于证明所有局部问题在理论上不可由未来状态回答。

下一优先级是原始逐 token decode 的执行分析与优化，继续保持整模型检测、地图、记忆、文本和规划门。完整结果与证据链接见 formal_go_nogo.json、original_model_parity.json；所有旧输出保留。
