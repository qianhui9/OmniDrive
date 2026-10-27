# 复现

在仓库根目录使用 OmniDrive 环境；CPU4/5 禁用。只使用空闲 GPU，不终止其他任务。

完整复现到新目录（不会覆盖本轮或历史结果；约需数十分钟）：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_risk_reproduce --root outputs/m123_world_risk_reproduction_new --evidence-root outputs/m123_world_risk_optimization_20261027_run01
```

此命令固定复制原开发/独立 Manifest、实验计划、质量预算及 3492c962 的旧代码快照；重新执行真实 checkpoint 教师标注、两档开发候选、新独立质量验收、受控模型边界、三轮交替真实服务、旧代码 Profiler、同状态 M1 控制、固定 Token Decode、完整 GPU 测试和离线诊断。不会因复现结果自动更改正式配置。复现读取相同场景，不能再称作新的未见场景验证。

重建已有结果报告与本地全量索引：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_risk_report --root outputs/m123_world_risk_optimization_20261027_run01
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.artifact_index --root outputs/m123_world_risk_optimization_20261027_run01
```

GPU 测试：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q
```

每次真实运行的完整 argv、时间和返回码在 `commands.json`。附加风险统计微基准命令见上方复现模块；它只使用开发输入，不触碰独立参数。大型 `.pt`、采样 `.prof`/CUDA trace 和全量 `m1_state.jsonl` 仅保留于本机，其余逐帧指标、问题答案、GT 官方评价和输出预测随 Git 提交。`raw_artifacts_sha256.json` 索引所有本机原始文件（包含这些未入 Git 的大文件）。

正式配置仍为 `projects/configs/OmniDrive/worldstream_m123_auto.py` 的 AR2 / RMS 0.7721027880907059。`risk_0.json` 为通过独立子集质量但未证明稳定计算加速的离线候选，不是已部署默认。
