# 执行记录

- 用户要求直接完成上传需求的代码、真实 GPU 实验和报告；Python 为 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。
- 用户授权：持续检查 GPU，空闲后自动继续；不停止其他人的进程。
- 起始 HEAD：`f6c06c3d21e29df5697ed4a13456a7b2edb7d9d4`，初始工作区干净。
- 阶段 A 已冻结：`audit_manifest.json`、`workload_audit.md`、历史真实 profiling 和 Go/No-Go 边界。
- 代码：采样点快照替代 Gate 全图 clone；合法 commit 来源；统一 Runtime/嵌套执行记录；生命周期/异常清空 Head memory；显式服务语义；严格成本/质量准入；离线长度分组；Sampling/Hook 回退；M1 payload 预算；未知/GT 来源不进入语义查询。
- 原生 smoke：`smoke_single_fixed`，10 帧执行的 Token、完整感知输出、视觉特征和两个 Head memory 全相同。`smoke_single` 为包装器签名错误的旧失败，保留，不进入正式统计。
- 已完成 GPU 模块测试 230 passed；随后新增了预算、来源和 Hook 边界回归，需交付前再运行全 GPU 测试。
- 正式 shell：`bash tools/system_optimization/run.sh`，exec session **66851**。日志 `logs/full_run.log`；子实验各自有 `logs/<run>.log` 和 `<run>/status.json`。此进程按序运行 single、multi、language_candidates、long_stream、budget_0、budget_128、stride_refresh，最后调用 report。
- 正式 single 已完成（2 场景 × 4 帧，预热 1 轮、计时 2 轮、10 方法），240 方法帧执行 / 160 计时帧执行全部完整输出等价。每方法 16 个计时帧，8 个唯一帧。Original P50=3056.19ms，unified_output=3055.92ms，unified_persistent=3294.69ms，M1 baseline=3310.58ms；无可信净加速声明。单独 native no_grad/inference_mode 的 LLM P50=2341.77/2338.12ms，仅一个真实输入，不作生产批准。
- 正式 multi 已完成：240 方法帧执行 / 160 计时帧，80 个计时问题/方法全部一致；每方法 8 个唯一帧。Original P50=7885.32ms，unified_output=7880.93ms，unified_persistent=8030.90ms，M1 baseline=8031.16ms。Camera/KV 复用均为零，不声明净收益。
- 正式 language_candidates 正在执行。warmup 已观察 prefix/batch/adaptive/compact Token 不一致，不能生产批准；正式计时也已重现，待完成全部重复及 held-out。不要重复启动另一个主批次。
- single 进程启动后补充了内存 payload 计数、保守来源/Hook/元数据校验；各运行均有执行时的 `source_manifest.json`。single 的旧细分 cache 字节不包含全部 M1 CPU 历史，已在报告生成器明确注明；实际 RSS/GPU peak 有记录。后续运行使用完整 payload 计数。默认工作负载不使用新增的成本证书批准分支；所有 approved 仍为空。
- `report.py` 只汇总 completed 正式运行，排除 smoke；不修改旧模块实验。报告脚本可在最后修复/重跑。注意原始完整 trace 每个约 600MB，最后读 JSON 会占数 GB RAM，宜等模型进程退出。
- 已加入 `lifecycle_probe.py`：long_stream 默认额外执行受控真实缓存填充 → 原生 Head 已推进后注入异常 → 验证真实 Head/三模块缓存清空 → 与干净原生完整请求比较。它是单独控制，不计作长流独立样本/速度；尚待真实 GPU 跑到该步骤验证。
- 后续来源修正：M2 全相机复用不能被 M1 commit 重新标为当前计算；新 FeatureRecord.computation_sources、相同来源保留版本、混合/缺失来源显式处理；OOM 保存失败也保留本次输出来源。4 个来源专项及数值设置失效专项已加入测试。主 single/multi 没有实际相机命中，所以该分支未执行，后续长流控制强制合法 Exact Camera Reuse 验证此分支。
- 计时修正：旧 Runtime.instrument 将有 Past KV 的 suffix prefill 误标 decode。新代码按 inputs_embeds 分类并记录形状/forward_tokens。language_candidates 进程已加载旧包装器，因此报告的 measured_stages 使用明确 KV branch 后的首次 model call 对该旧记录重新分类（shared seed 被限制小于所有 prompt，不会把 full-prompt-cache 的 decode 猜成 prefill）；不改写任何原始时间。单/multi 原生 Full 的计时分类不受影响。

待完成：等主批次全部结束，修复任何真实执行错误（失败留档、使用新目录重跑，绝不覆盖成功旧实验）；最后运行全 GPU 测试、`git diff --check`、更新 patch/manifest/source 差异审计、生成并检查报告及 S1–S10 图，核对所有结论的实际 Token/跳算/状态消费。最终答复应链接报告并明确不能把 No-Go 当成真实净加速成功。

续跑记录：
- 原 long_stream 的 64 唯一真实帧 × 4 方法主对照已完成，但后置生命周期控制失败，原目录 status=failed，严格排除正式汇总。
- 失败原因是 probe 使用 startswith('memory_') 同时检查了不可清空的配置 memory_len；已改为 Head 的实际时序字段。不是已证实的 Runtime 缓存残留。旧失败日志和执行源完整保留。
- 首次续跑因自动审批服务额度不足未执行；用户随后要求继续，审批恢复，GPU 24110 MiB 空闲。
- 当前补跑 exec session 72700：resume_after_probe_fix.sh。独立 lifecycle_validation → long_stream_retry（新目录）→ budget_0 → budget_128 → stride_refresh → delivery_gpu_tests → report。
- 不要重复启动主批次；按各 logs 与 status 监控。lifecycle_validation 是控制验证，不进入正式性能与独立样本数量。
- 续跑 CPU：245 passed, 1 skipped, 19 warnings；日志 logs/resume_cpu_tests.log。
- 最后执行 python -m tools.system_optimization.delivery：重建 stage_BC/DE.patch、implementation_manifest、逐运行 source 差异、历史证据哈希和 trace index；该步骤尚未执行。
- 独立 lifecycle_validation 已 completed、weights_unchanged=True：6 相机实际 Exact 复用，backbone 源版本/帧/时间保持；故障前 M1 有 92 个历史 Anchor、6 相机历史、M2 约 68.8 MB、M3 约 306.9 MB；故障后全部实际时序字段及缓存清空。恢复和真实 scene_change 与干净原生输出/特征/memory max_abs 均为 0，完整 Token 一致。结果在 lifecycle_validation/lifecycle_probe/results.json。
- long_stream_retry 第一场景 32 帧 × 4 方法全部 Token 一致；真实相机复用为 0；逐帧配对 Current-Output/native 约 1，Persistent/native 约 0.92，未把服务职责差异作为机制加速。
- long_stream_retry 已 completed、weights_unchanged=True：2 新场景、64 唯一帧、256 方法帧；四方法 exact_full_output_equality 全部 True，实际 Camera/Window 复用全 0。P50 ms：original 3016.08、m123 3279.57、unified_output 3015.01、unified_persistent 3287.58。轻量 long_stream_results.json 已更新，最终 report 将重建全套汇总。

最终结果：
- 全部正式批次完成：single、multi、language_candidates、long_stream_retry、budget_0、budget_128、stride_refresh；独立 lifecycle_validation 通过。正式计时 760 方法帧，76 唯一帧、4 场景；预热与控制不增加独立样本数。
- 0/128 MiB 下五方法各 20/20 问题完整 Token 一致、完整数值输出等价；Batch/Adaptive 因 persistent_prefix_budget 回退，预算前无收益的 seed 开销保留在 E2E。零预算的 M1 历史显式失效。
- 最终 GPU 可见环境回归：246 passed, 19 warnings, no skips；日志 logs/delivery_gpu_tests.log。静态 diff / shell syntax / Python compilation 通过（static_checks.json）。
- 报告与 S1–S10 PNG/PDF 已生成并检查 S5/S7/S8/S10，可读且全部来自实测。大型 trace 在模型退出后解析；保留完整 kernel/copy/alloc/sync 统计，launch gap 归因仍未测，不把 CUDA interval 当 kernel。
- 原模型权重/LoRA、质量门限、旧模块实验未改写；所有正式进程 weights_unchanged=True。旧 long_stream 后置检查错误与 smoke 失败保留并排除；不将失败状态改写为成功。
- 最终结论：严格质量约束下没有生产批准的净加速。真实移动视觉跳算与 M1→M3 任务消费为零；离线 Prefix/Batch/Adaptive/Compact 仍未满足完整原生输出等价。Auto Full 回退保留。新物理 FAR/关键变化召回 Oracle、官方检测/地图与闭环安全未测。
