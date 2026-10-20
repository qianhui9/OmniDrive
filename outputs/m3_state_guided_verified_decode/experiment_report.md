# M3 SGVD 连续驾驶验证报告

本轮完成真实六摄像头连续 nuScenes 历史答案采集、Teacher Forcing、CUDA 成本测量、采样原语验证、并行/串行目标分布及拒绝后 KV 检查，以及完整 OmniDrive 冻结服务质量回归。

结论：`stop_no_exact_target_certificate`。没有安装或启用 SGVD Generation Loop；原生采样参数、320 Token 预算、M1/M2、任务质量门、Prefix 策略和模型权重保持原样。

历史答案可用 31/129 次请求；所有数据标记为 exploratory_reused_population，没有独立人口资格。
小词表采样原语验证通过：True。真实 FP16 并行与串行处理后目标分布逐位相同：False；最大 TV=0.203965321。数学上的接受/拒绝恒等式不能消除上游目标分布差异。

部分固定长度的离线预计收益为正，但离线原生前缀探测不代表拒绝后随机路径的收益，也未减少正式 Decode Forward。没有将共同前缀、影子节省或单次并行 Forward 数量当作实际加速。

| 工作负载 | 方法 | E2E P50 / P95 ms | 冻结质量通过 / 检查 |
|---|---|---:|---:|
| native_multi | original | 24338.46 / 27521.86 | 0 / 0 |
| native_multi | full_maintain | 24736.25 / 27623.60 | 6 / 6 |
| native_multi | frozen_exact_m2 | 24744.17 / 27634.22 | 6 / 6 |
| native_multi | m3_native | 24734.07 / 27629.39 | 6 / 6 |
| planning | original | 2907.58 / 2957.32 | 0 / 0 |
| planning | full_maintain | 3162.12 / 3493.58 | 12 / 12 |
| planning | frozen_exact_m2 | 3128.91 / 3529.75 | 12 / 12 |
| planning | m3_native | 3123.62 / 3529.06 | 12 / 12 |

完整质量回归 54/54；CUDA 工程测试 699 项，失败 0、错误 0、跳过 0。

备用 KV 分析：实测 aten::cat GPU 归属 1.7277 ms/Token；移除全部 cat 的乐观 Decode 上界为 1.0721 倍。该上界并非 Adapter 实测收益。旧版 Attention 内部直接拼接，纯 Tuple Adapter 无法消除它；预分配视图还会改变 stride，因此没有改写 Attention。

扩展多问题存在同帧任务，但原始生成串行执行，现有 BatchedExecutor 对 do_sample=True 保守回退；没有构造虚假并发，也未用贪心替换原始采样。

完整服务各次都运行 EVA、Detection、Map、Temporal Memory、LLM、Planning。质量为相对原模型的冻结回归；Planning ADE/FDE 只使用合法 GT Mask 离线评价，不冒充完整官方 mAP 或驾驶安全验证。冷启动、第二帧、独立采样 Seed、所有未命中和不同间隔分别保留。这里 cold 表示 Scene/Temporal/M1 状态清空后的首帧；模型权重加载不在请求计时内，不是进程启动延迟。

旧 CUDA Graph 负面基线以来源与 SHA 归档引用，没有重捕获 160 个子图。新实验不覆盖历史目录。实际命令见 commands.sh，逐请求来源/Token/时序见 phase1/requests.jsonl，原始完整模型输出见 phase1/raw_outputs 与 service/raw_outputs。

SGVD 未执行，因此其任务质量、Decode/E2E 收益没有可报告的实测值；正式 No-Go 不用理论收益替代。下一步应聚焦原始 FP16 GEMM 与完整 M1 维护成本，而不是继续新增缓存或启发式解码。

```json
{
  "historical_draft_available": true,
  "draft_acceptance_sufficient": true,
  "verification_profitable": true,
  "verification_profitable_scope": "Exploratory native-prefix expected cost only, not a runtime gain",
  "sampling_distribution_correct": false,
  "categorical_sampling_primitive_correct": true,
  "sgvd_runtime_implemented": false,
  "sgvd_task_quality_pass": false,
  "sgvd_decode_speedup": null,
  "sgvd_e2e_speedup": null,
  "sgvd_production_enabled": false,
  "remaining_bottleneck": "Original FP16 dense GEMM and charged M1 maintenance/full prefill; parallel-vs-serial target arithmetic is an additional exact-distribution barrier",
  "next_research_priority": "Profile original dense GEMM bandwidth/compute and full-maintenance cost under frozen output contracts; require a serial/parallel target-distribution certificate before any future SGVD execution",
  "formal_go": false,
  "decision": "stop_no_exact_target_certificate",
  "independent_heldout": false,
  "unexecuted": "Experimental SGVD and its GPU optimizations stop at the target-distribution gate; no fabricated speedup or quality."
}
```
