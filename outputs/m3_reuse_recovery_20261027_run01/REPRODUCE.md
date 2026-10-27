# 复现说明

必须在项目根目录使用 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，所有进程排除故障 CPU4/5。当前 CUDA/PyTorch/Transformers 环境及真实 `ckpts/iter_10548.pth` 保持；脚本等待 GPU 空闲，不终止其他进程。

完整复现到一个尚不存在的新目录：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.m3_recovery_reproduce --root outputs/m3_reuse_recovery_reproduction_new
```

复制已冻结的主代码前快照、Policy 和三个 Manifest。诊断复现中的 `historical_current` 与 `single_before` 只通过已有 `frozen_worker` 装载任务开始版本；正式代码仍只有一个 Runtime。随后执行当前实现的自然单/多问题流、三轮固定相同 Token 的完整 Native/Zero/Reuse 请求、历史实际 Carriers 回放、相同视觉/初始 KV 的三条件控制、真实数据统计微基准与官方 GT 评价。不会自动改正式 Policy。

`commands.json` 记录外层实际命令、开始/结束和退出码，`control_commands.json` 记录额外诊断子进程。`initial_driver.log`、`experiment_driver.log` 和各任务日志保留。`same_input_controls.jsonl` 是强制 Token 的计算误差诊断，绝不可作为自然回答质量或完整系统时延；自然质量只来自 `*_free_r0_*`。

只重建分析与报告，不运行模型：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.m3_recovery_analysis --root outputs/m3_reuse_recovery_20261027_run01
```

仅执行回归与更新原始文件 SHA 索引：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.m3_recovery_experiment --root outputs/m3_reuse_recovery_20261027_run01 --stage tests
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.artifact_index --root outputs/m3_reuse_recovery_20261027_run01
```

测试阶段会更新本轮测试日志；不可用于覆盖其他历史实验根目录。完整重跑应总是使用新目录。`m3_inputs/*.pt`、较大的完整 M1 状态和 Profiler traces 留在本机并纳入原始 SHA 索引；复现原历史隔离控制还需保留 `outputs/m3_quality_bounded_20261026_run01/native_reference/` 的真实捕获。问题来自原数据，不向模型提供 GT。重复本轮已观察场景不是新的独立验证。

## 追加的语义失败和最终策略验收

`semantic_failure_review.json` 保存原冻结策略新增道路规则断言的六相机核查、原始分数和图像 SHA。`qualified_m3_policy.json` 仅提高该 Map Approx 规则的 critical_regressions，并保留原始自动分数；不调阈值、不修改成本、不新增放行规则。`qualification/` 在冻结收紧策略后使用另四个新场景，按原生标注顺序执行真实问题。最终以 `FINAL_REPORT.md` 和 `qualification/deployment_decision.json` 为准，不能用前一阶段冻结候选的1606 Tokens替代最终策略成绩。

单独复现最终阶段到当前已创建但尚未测量的 qualification 目录：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.m3_recovery_qualification --root outputs/m3_reuse_recovery_20261027_run01/qualification
```

完整 fresh-root 复现会依次运行历史候选和收紧策略验证，不需要把当前正式策略恢复为已知有风险的历史文件。重新使用已观察 Manifest 只是复现，不是新的未见场景证据。
