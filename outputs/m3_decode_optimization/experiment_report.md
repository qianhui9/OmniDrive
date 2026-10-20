# Decode 优化与 Prefix 正确性实验报告

实现与真实 GPU 验证已完成。新增开关默认关闭；所有冻结模块和权重保持不变。结论以以下实测数据为准，不将微测收益替代完整服务收益。

独立 decode trace 的主要成本是 GPU FP16 GEMM。CUDA Graph 只减少原生静态子图的提交开销，未删除矩阵计算；动态 Attention/KV、LoRA、采样与停止条件保持原生。

| 请求人口 | 方法 | E2E P50/P95 ms | 冻结质量/分母 |
|---|---|---:|---:|
| planning | original | 2877.52/2896.90 | 基准 |
| planning | full_maintain | 3096.12/3467.92 | 8/8 |
| planning | frozen_exact_m2 | 3130.96/3504.56 | 8/8 |
| planning | m3_native | 3115.24/3501.66 | 8/8 |
| planning | decode_graph | 3152.68/3530.85 | 8/8 |
| planning | legacy_prefix | 3111.44/3502.04 | 8/8 |
| planning | guarded_prefix | 3105.06/3503.36 | 8/8 |
| native_multi | original | 8224.80/8580.68 | 基准 |
| native_multi | full_maintain | 8521.13/9105.13 | 8/8 |
| native_multi | frozen_exact_m2 | 8527.14/9201.67 | 8/8 |
| native_multi | m3_native | 8508.18/9180.54 | 8/8 |
| native_multi | decode_graph | 8598.54/9285.30 | 8/8 |
| native_multi | legacy_prefix | 8268.65/8941.69 | 6/8 |
| native_multi | guarded_prefix | 8528.85/9217.73 | 8/8 |

单 token Wall P50 加速比：0.9866。同服务与 Original 比较、含冷启动结果见 end_to_end_comparison.json。两场景小样本仅提供探索性证据。

Prefix 输入逐位相等不等于不同形状 FP16 prefill 的 KV 逐位相等。逐个历史失败的复现、首个分歧、完整当步 logits、各层 KV 误差、重复性和生命周期检查见 prefix_parity_analysis.json。未宣称部分 Prefix 的数值一致性已经修复。
保守整段策略通过完整任务质量才计为通过；它对相关问题退回完整 prefill，因此不能冒充部分前缀加速。生产 Auto 仍关闭。

完整 CUDA 工程回归：672/672，失败 0，错误 0，跳过 0。完整模型 service 请求 168，独立完整生成 72。

初始 observer 未保留 forward 签名导致两次采样失败；归因为包装器而非 PEFT。修正后恢复原始 PEFT generate。首版 Graph 有 24 条完成请求，因约 74 秒捕获与较慢重放主动中止，原始记录未删除。最终版去掉每子图全局 GC，并在原生 decoder 消费范围内复用私有输出 workspace；公开 logits/KV 所有权经过 CUDA 测试。

计时：service 使用请求边界同步与延迟 CUDA events；每 forward CPU 字段为提交 Wall，不冒充同步 decode Wall。独立 micro 完整生成使用每次 forward 同步，另报 decode span 与实际 forward Wall/GPU，含观察开销。Trace 只累加叶子 kernels，不与 CPU 算子归属 CUDA 时间相加。

内存为同一进程各请求峰值；CPU RSS high-water 包含模型加载，后续对照也包含常驻且停用的 Graph workspace，不等同于各方法独立进程的部署峰值。

原始采样参数、全部答案和 320 token 预算不变。实际 EOS/预算耗尽与轨迹可解析性独立检查。Detection/Map/Temporal/LLM/Planning 沿用冻结门；离线 GT 仅在计时外用于质量。本轮是相对原模型的冻结质量回归，不冒充完整 nuScenes 官方 mAP/驾驶安全验证。

```json
{
  "decode_bottleneck_identified": true,
  "decode_optimization_implemented": true,
  "decode_quality_pass": true,
  "decode_token_latency_speedup": 0.9865661257886983,
  "decode_e2e_speedup": {
    "planning": 0.9860185301693494,
    "native_multi": 0.989317848465683
  },
  "prefix_parity_fixed": false,
  "prefix_quality_pass": true,
  "prefix_production_enabled": false,
  "full_vlm_e2e_speedup": {
    "planning": 0.8944417915772482,
    "native_multi": 0.9515050557588962
  },
  "remaining_system_bottleneck": "Native dense FP16 GEMM execution; persistent M1 maintenance and full prefill remain charged",
  "next_research_priority": "Kernel-level GEMM profiling under unchanged weights; seek a supported backend with proven full sampling quality and measured net gain",
  "production_decode_enabled": false,
  "formal_go": false,
  "reason": "Default off; no independent production certificate, partial prefix remains unproved. See measured same-service/Original and cold-inclusive ratios separately."
}
```

复现命令见 commands.sh。原始逐请求质量、完整感知/记忆结果、源码快照、环境、权重 SHA 和失败数据均在本目录。旧实验目录未覆盖。
独立历史诊断通过原始 preprocess 重建历史问题文本；帧数不同会改变随机抽取的描述问句，因此历史问题和当前问题分别保留。
一次尚无完成测量的 micro 启动被停止，统一诊断接口并固定源码后重新启动；日志和原快照保留在 micro_startup_before_prefix_input_fix。

96 条单/多 token 微测及 72 次完整生成全部完成。原生 decode-only trace、优化前后 GPU 事件和同步 Wall 均来自实际执行。
追加 optimized CUDA Graph trace 在完成采集后持续超过两分钟停留于 profiler 后处理（GPU 空闲），已主动终止并保留日志；没有把缺失 trace 当作有效结果。后续复现默认跳过这个可选 trace；测量代码未改，源码比对见 micro_postprocessing_audit.json。

收尾将新增 dataclass 配置字段移到末尾，保留冻结 TaskEvidence 位置参数契约。主实验使用的全部关键字配置值、runtime/native_decode 执行代码均一致，见 config_compatibility_audit.json；最终源码另做 3 次完整模型抽验，Graph/保守 Prefix 均通过。新增采样失败在问题起始及首分歧 token 的采样前 RNG 均逐位相同；当步 logits/KV 和 RNG 原始张量见 final_source_confirmation。
原生 trace 将 kernel、memcpy、memset 分别统计；未归属的时间轴间隙没有被直接归因为 kernel launch。CPU 提交开销下降与 GPU/E2E 净收益分别验证。

最终对账发现首版测量器将可清空的生成 ID 收集列表保留为内存 reference，使 quality.jsonl 中部分 token_ids_equal 元数据失真；逐请求 runs.jsonl 在清空前已完整落盘。保留原文件，通过不可变落盘 IDs 按请求键重新核对：Graph 24/24、保守 Prefix 24/24、旧 Prefix 21/24。完整文本/感知/记忆/规划质量与 E2E 计时不受影响。修订仅增加计时结束后的 ID 快照；完整对账见 token_id_audit.json。
