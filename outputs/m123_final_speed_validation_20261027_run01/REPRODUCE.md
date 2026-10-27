# 复现说明

在项目根目录使用 OmniDrive Python。必须排除有硬件故障的 CPU4/5。执行脚本检查 GPU 空闲并串行装载真实 `ckpts/iter_10548.pth`；不会停止其他进程。

完整复现，输出目录必须尚不存在：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.final_speed_reproduce --root outputs/m123_final_speed_reproduction_new
```

固定复制原 6 场景×8帧 Manifest、已冻结 RMS/Risk0 参数、旧代码快照和 .10 预算。分别运行三轮自然生成与三轮固定 Native Token 序列的完整请求；仅自然生成 round0 进行官方任务质量评价。额外执行真实开发帧的精简等价性控制和新数据真实 Suffix 标签，并运行完整 GPU 测试。不会因为复现结果自动更改正式参数。重复这些已观察场景不能再称为新的未见场景验证。

当前目录的实际子进程 argv、时间和退出码在 `commands.json`；父进程日志在 `experiment_driver.log`。计算等价性和误差标签两个诊断运行含额外 Suffix，不能混入 headline 性能。

重建分析与报告（不执行推理）：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_speed_analysis --root outputs/m123_final_speed_validation_20261027_run01
```

GPU 回归及完整原始文件 SHA 索引：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.artifact_index --root outputs/m123_final_speed_validation_20261027_run01
```

`free_*` 为正常答案与任务质量数据；`fixed_*` 仅测性能，强制答案不提供质量成绩。`fixed_reference.json` 记录 Native 自然 Token 的来源和哈希；逐请求记录实际执行步数、Decode 输入历史验证和处理器开销。固定模式仍使用现有 HF/PrefillExecutor 真实 Forward，没有修改正式生成流程。

采样 trace、大 Tensor 与全量 `m1_state.jsonl` 留在本机并列入全量 SHA 索引，其余任务输出、问题答案、逐帧计时和官方评价保存在本次交付目录。复现时原数据、checkpoint、Conda 环境和前述本机文件应保留。

最终交付 GPU 回归日志为 `tests_gpu_delivery.log`（536 passed）；初次失败日志 `tests_gpu_final.log` 保留。实际命令和修正原因见 `delivery_validation.json`。
