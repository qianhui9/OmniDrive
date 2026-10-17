# 系统瓶颈审计（算法修改前）

来源为已有真实 GPU 原始逐帧记录；SHA256 见 audit_manifest.json。这些记录不代表本次新代码的性能。单问题 Sampling 与五问题 Greedy 分开。

| 工作负载 | E2E P50/P95 ms | Decode interval P50 ms | Prefill interval P50 ms |
|---|---:|---:|---:|
| single | 2876.05/2933.54 | 2125.33 | 181.85 |
| multi | 8055.84/9310.32 | 6576.25 | 908.26 |

首要热点为 Decode。完整消除单问题 Prefill 也只是理想上限；同帧只有一个问题，不能从同帧共享 Prefix 获益。理想上限按每帧残余时间计算后汇总，没有相加组件 P50。

EVA 前两层的受控理想节省仅 43.60 ms，尚未包含认证和缓存；真实移动场景 Camera/Window 跳算为零。Camera 上限仅是条件性收益，不能外推到真实场景。

M1 延长历史实验拒绝原因：`{"motion_unknown": 405, "no_reliable_view": 971, "observation_mismatch": 516, "no_historical_observation": 209, "projected_occlusion": 140, "temporal_support_insufficient": 10, "temporal_source_missing": 1}`。任务激活仍为零；M1→M3 No-Go，保留可靠性条件。

Prefix 与 Batch 均未通过冻结的完整 Token 等价标准；Auto 必须回退。不继续堆叠精度补丁或 Window Kernel 优化。优先处理拒绝路径成本、真实来源记录、跨模块失效及公平的单问题后端评估。

新增插桩文件：tools/system_optimization/audit.py（只读历史证据并输出审计）。完整预处理、H2D/D2H、分配及 launch-gap 的本轮数值需新的 GPU trace；未知值不填零。
