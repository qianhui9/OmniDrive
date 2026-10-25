# OmniDrive

唯一推荐的 M1+M2+M3 研究入口：`projects/configs/OmniDrive/worldstream_m123_auto.py`。

M1 保留普通 A 关联。M2 的成本、来源或质量准入不通过时，更新集扩大到全部窗口，仍通过同一个 `WindowExecutor` 完成全量视觉计算。M3 的历史语义或 KV 前缀未获准复用时，保留完整当前视觉输入，仍通过同一个 `PrefillExecutor` 完成 Prefill 和 Decode，不静默调用原生 `lm_head.generate()`。

模块完成执行与产生复用、获得加速分别审计，不能相互代替。四份部署成本/质量批准文件当前均无批准记录；短序列不能批准生产复用或替代完整集精度评估。

运行命令与边界见 [主线说明](docs/worldstream_final_route.md) 和 [benchmark 文档](docs/worldstream_benchmark.md)。本轮结果见 [真实 GPU 执行审计](outputs/m123_execution_audit_20261025_run01/experiment_report.md)，此前代码契约测试见 [2026-10-25 执行契约报告](outputs/m123_execution_contract_20261025/experiment_report.md)。
