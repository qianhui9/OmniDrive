# OmniDrive

唯一推荐的 M1+M2+M3 研究入口：`projects/configs/OmniDrive/worldstream_m123_auto.py`。

M1 保留普通 A 关联。M2 的成本、来源或质量准入不通过时，更新集扩大到全部窗口，仍通过同一个 `WindowExecutor` 完成全量视觉计算。M3 使用 Quality-Bounded WSDR-KV 的统一校准策略，按任务、世界状态和连续视觉块变化选择精确或有界近似前缀；无质量预算内净收益时，令 `reuse_length=0`，仍通过同一个 `PrefillExecutor` 完成 Prefill 和 Decode。

模块完成执行、实际复用与端到端加速分别测量。M3 当前读取 `outputs/final_route/m3_reuse_policy.json`，旧 M3 批准文件保留为历史记录；M2 的窗口成本/质量机制不变。近似复用可能改变回答，短序列研究校准不能替代完整集任务质量评估。

M3 实现、质量预算和复现方法见 [WSDR-KV 说明](docs/worldstream_m3_wsdr_kv.md)，本轮真实结果见 [质量—时延报告](outputs/m3_quality_bounded_20261026_run01/experiment_report.md)。

运行命令与边界见 [主线说明](docs/worldstream_final_route.md) 和 [benchmark 文档](docs/worldstream_benchmark.md)。本轮结果见 [真实 GPU 执行审计](outputs/m123_execution_audit_20261025_run01/experiment_report.md)，此前代码契约测试见 [2026-10-25 执行契约报告](outputs/m123_execution_contract_20261025/experiment_report.md)。
