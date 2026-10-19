# Implementation / instrumentation changelog

M1 工程接口及配置、M3、原模型参数与网络结构、正式 Exact 成本表均不修改。新增生产配置默认关闭研究路径。

| File | Change type | Purpose / instrumentation | Lines |
|---|---|---|---|
| `worldstream/m2/config.py` | modified | 默认关闭与简单硬门限配置 | 1–46 |
| `worldstream/m2/runtime.py` | modified | 隔离研究缓存与可关闭分支；Exact 默认行为保持 | 1–347 |
| `worldstream/m2/source_alignment.py` | created | 真实点支持/冲突和整 Window 对齐；诊断误差 | 1–115 |
| `worldstream/m2/conditional_prefix.py` | created | 实际 Window 跳算、源版本、当前位置与全局尾部执行 | 1–194 |
| `tools/worldstream_m2/state_aware_cuda_batch.py` | created | 空闲 GPU 等待与本轮回归/实验串行调度 | 1–40 |
| `tools/worldstream_m2/state_aware_probe.py` | created | 96 帧真实来源对齐、分层误差和错误来源对照 | 1–152 |
| `tools/worldstream_m2/state_aware_finish_batch.py` | created | 等待计时结束，执行最终源码回归和归档 | 1–38 |
| `tools/worldstream_m2/state_aware_integration.py` | created | 原模型完整推理、CUDA事件、真实工作量、层/任务/Memory质量对照 | 1–195 |
| `tools/worldstream_m2/state_aware_finalize.py` | created | 原始结果聚合、图表与严谨 No-Go/归档审计 | 1–355 |
| `tools/worldstream_m2/state_aware_prefix_profile.py` | created | 单独补测原始 Block 0/1 GPU/Wall，排除 PE/ABS | 1–35 |
| `tools/worldstream_m2/state_aware_feasibility.py` | created | 修改前实际 EVA GPU 上界/原始 packing 对照 | 1–159 |
| `tools/worldstream_m2/state_aware_execution_profile.py` | created | 实际 prefix 跳算/缓存成本、CUDA kernel trace、权重指纹 | 1–94 |
| `tools/worldstream_m2/state_aware_lineage.py` | created | 逐 Window 的真实根帧/版本/位置与实际计算量审计 | 1–27 |
| `tests/worldstream_m2/test_source_aligned_conditional.py` | created | 配置/测试，无原模型结构改动 | 1–161 |
| `projects/configs/OmniDrive/worldstream_m2_state_aware.py` | created | 配置/测试，无原模型结构改动 | 1–7 |
| `projects/configs/OmniDrive/worldstream_m2_conditional_research.py` | created | 配置/测试，无原模型结构改动 | 1–9 |

原始模型 Forward 的临时事件/钩子只在独立 runner 内安装，并在 finally 中移除。测试 fixture 不用于宣称自然驾驶质量。受控来源干预显式未通过 M1 准入。

生产：`projects/configs/OmniDrive/worldstream_m2_state_aware.py`。研究：`worldstream_m2_conditional_research.py`。`conditional_cost_table` 为保留字段，不授予 Auto 准入；不存在合格 Conditional 成本人口。

Raw artifacts 在本目录下，旧实验目录不写入。Baseline HEAD/source tar/hash 清单保留修改前状态。
