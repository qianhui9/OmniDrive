# M3 数值归因与依赖安全 Microbatch 实验报告

完成真实原模型数值归因、同帧任务依赖人口检查、Sampling Batch CUDA 测量、默认关闭的 RNG 隔离 Microbatch 原型与完整服务回归。

SGVD 最终归因：`native_fp16_execution_shape_rounding`。原生处理协议一致，最早分歧算子为 00/k_proj, 00/q_proj；原生 FP16、温度、Top-k/Top-p 未改动。SGVD 保持冻结。

依赖人口：105/105 个真实问题通过单轮 Prompt/共享完整视觉输入的独立性检查。数据为 exploratory_reused_population，没有独立 Heldout 资格。

Batch=1/2/4/8 单步 GPU 实验是不同真实问题的吞吐测量，不代表完整服务收益。实际原型仅运行固定 Batch=2/4；每个请求独立 RNG、KV、Position、Mask、EOS，完成行退出 Batch，整帧结果按原问题顺序事务提交。

Microbatch 目标分布最大 TV=0.250912786；全部已测 Logits 逐位相同：False。独立 RNG 不消除 FP16 Batch 形状差异，也不保证与原始共享 RNG 在同 Seed 下逐 Token 相同。

| 工作负载 | 方法 | 全帧 E2E P50 / P95 ms | 冻结质量通过 / 检查 |
|---|---|---:|---:|
| native_multi | full_maintain | 24601.84 / 27519.94 | 6 / 6 |
| native_multi | greedy_control | 25764.29 / 27159.77 | 0 / 6 |
| native_multi | isolated_serial | 24440.49 / 25785.73 | 0 / 6 |
| native_multi | m3_native | 24598.65 / 27472.64 | 6 / 6 |
| native_multi | microbatch2 | 17546.40 / 17863.67 | 0 / 6 |
| native_multi | microbatch4 | 14227.64 / 15787.15 | 0 / 6 |
| native_multi | original | 24169.60 / 27453.94 | 0 / 0 |
| planning | full_maintain | 3145.14 / 3434.72 | 4 / 4 |
| planning | m3_native | 3098.02 / 3435.33 | 4 / 4 |
| planning | microbatch4 | 3103.91 / 3450.49 | 4 / 4 |
| planning | original | 2875.10 / 3320.85 | 0 / 0 |

| Native Multi 方法 | 每问题完成 P50 / P95 ms | 规划完成 P50 ms | 有序发布 P95 ms |
|---|---:|---:|---:|
| full_maintain | 10678.59 / 23718.80 | 2335.74 | 27136.62 |
| greedy_control | 10906.75 / 23296.94 | 2307.78 | 26493.79 |
| isolated_serial | 10406.46 / 23234.64 | 2333.96 | 25384.19 |
| m3_native | 10651.43 / 23681.39 | 2333.49 | 27074.50 |
| microbatch2 | 7921.60 / 16375.29 | 2880.31 | 17296.22 |
| microbatch4 | 6940.15 / 14173.96 | 3273.25 | 15223.05 |
| original | 10669.76 / 23713.93 | 2338.77 | 27149.02 |

完整帧请求 58 次；CUDA 回归 719 项，失败 0、错误 0、跳过 0。

主要性能指标是同一帧完成全部真实问题的时间。各方法保留完整视觉输入、问题、生成预算和 M1/Temporal 状态维护；拒绝、单问题回退和所有额外采样/打包成本计入。每问题完成尾延迟、规划请求延迟、输出事务提交延迟、Token 总量与 EOS 分别见 end_to_end_comparison.json。

Original 使用原始共享 RNG；isolated_serial 使用原始 generate 和与 Microbatch 相同的独立请求 Seed，为区分 RNG 与算术差异提供对照。greedy_control 仅为工程对照，不能代表 Sampling 速度或质量；若其内存预算导致串行回退，诊断中完整保留。没有改变原 batched_decode 对 Sampling 的保护。

生产结论：No-Go。实测吞吐或全帧服务收益不能替代严格采样和任务质量资格。单问题 Planning 不使用多问题 Batch 收益。备用方向仅做有限真实 Batch=1 GEMM 与 Commit 关键路径诊断；没有大幅改写 Attention、重新捕获 CUDA Graph 或建立另一套异步状态系统。

Detection/Map/Temporal 和完整文本/Planning 使用原冻结标准。Planning ADE/FDE 仅以合法 GT Mask 离线计算；未提供额外人工文本标注或完整官方 nuScenes 安全评估。模型加载不包含在 state-cold 首帧计时内。

复现命令见 commands.sh，模型/环境/SHA、原始逐帧结果、数值差异张量和日志均保留。

```json
{
  "sgvd_numerical_root_cause": "native_fp16_execution_shape_rounding",
  "sgvd_frozen": true,
  "native_independent_task_population": {
    "requests": 105,
    "frames": 8,
    "population": "exploratory_reused_population"
  },
  "sampling_microbatch_feasible": true,
  "sampling_distribution_qualified": false,
  "microbatch_runtime_implemented": true,
  "microbatch_task_quality_pass": false,
  "microbatch_gpu_speedup": {
    "1": 1.0,
    "2": 1.7336413727843971,
    "4": 3.348603594275857,
    "8": 5.495459378769481
  },
  "microbatch_gpu_speedup_definition": "Aggregate sampled-token throughput ratio from CUDA event intervals, including native processors/private sampling and host gaps; not a kernel-only or complete-service gain.",
  "microbatch_full_service_speedup": {
    "microbatch2": {
      "original": {
        "n": 6,
        "p50": 1.432131860550574,
        "p95": 1.5415100218299933
      },
      "full_maintain": {
        "n": 6,
        "p50": 1.4575444432907503,
        "p95": 1.54523079850226
      },
      "m3_native": {
        "n": 6,
        "p50": 1.4573444857554638,
        "p95": 1.5425753263596746
      },
      "isolated_serial": {
        "n": 6,
        "p50": 1.4204837694445185,
        "p95": 1.4517997548133357
      }
    },
    "microbatch4": {
      "original": {
        "n": 6,
        "p50": 1.7234598149066254,
        "p95": 1.7695271128035372
      },
      "full_maintain": {
        "n": 6,
        "p50": 1.7569766419614916,
        "p95": 1.8163464842351118
      },
      "m3_native": {
        "n": 6,
        "p50": 1.7562939706258736,
        "p95": 1.817952352967482
      },
      "isolated_serial": {
        "n": 6,
        "p50": 1.7107393421960326,
        "p95": 1.7923278136242722
      }
    }
  },
  "performance_qualification": "Actual experimental timing ratios, including all attempts/fallbacks; not quality-qualified acceleration. Generated token totals and frozen quality are reported separately; generation budget was never lowered.",
  "single_request_optimization_result": "Measured original GEMM/LoRA/LM-head diagnostics only; no certified replacement",
  "m1_decode_overlap_opportunity": "bounded_commit_only",
  "production_microbatch_enabled": false,
  "formal_go": false,
  "reason": "No qualified native FP16 sampling distribution and frozen microbatch task quality failed",
  "remaining_system_bottleneck": "Batch=1 FP16 weight traffic/shape efficiency, full prefill and M1 maintenance. Batched arithmetic/RNG semantics remain a formal sampling/quality qualification barrier.",
  "next_research_priority": "Use measured per-question planning latency and source-bound native FP16 probabilities to evaluate any future batching contract; do not expand runtime or enable production under current evidence."
}
```
