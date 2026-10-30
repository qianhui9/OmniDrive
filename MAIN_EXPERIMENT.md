# 冻结的 M123 主实验

唯一研究配置：[final_research_policy.json](outputs/m123_final_quality_validation_run01/final_research_policy.json)。

- M1：`min_detection_score=0.3`，保留已优化的哈希执行器。
- M2：`ordinary_budget=3`，保留唯一增量视觉执行器。
- M3：`current_only / C-basic`，`min_confidence=0.45`、`max_binding_risk=0.75`。
- 结果基准：`c8ddb66bf90ee2c4d4627751d5f9d3961133f810`。
- 主实验数据与完整审计：[m123_final_quality_validation_run01](outputs/m123_final_quality_validation_run01/)。
- 选择与性能证据依赖：[m123_critical_path_and_joint_tuning_run01](outputs/m123_critical_path_and_joint_tuning_run01/)。

冻结的是研究路线，不是风险认证。保留“残余严重错误、独立 TEST 未完成、生产默认关闭”结论；不把已暴露 DEV 或事后病例升级成独立测试。

## 只重放冻结主路线

以下命令在项目根目录执行。第一条只准备新的输出目录，不使用 GPU；第二条需要空闲 GPU，会持有既有 GPU 锁，不能中止其他进程。

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.freeze_cleanup prepare-replay --output /tmp/omnidrive-r0-main-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_capture main --root /tmp/omnidrive-r0-main-replay
```

只保留 R0 一套参数；A/B/C 是必要的 Native、正常语言路径与语言旁路对照，均不搜索参数。五种负载各 16 帧，完整重放 240 次模型调用。原报告是 208 次新执行加 32 个历史实测复用；重放结果另存，不能混为同一统计总体。

完整历史审计复现见原 [REPRODUCE.md](outputs/m123_final_quality_validation_run01/REPRODUCE.md)。其中 R1/R2/R3 仅属于已封存诊断，当前主路线不再运行这些参数选择阶段。

## 保留范围与清理记录

[冻结清单](outputs/main_experiment_freeze_run01/freeze_manifest.json) 固定主配置、代码提交、全部版本化审计文件、检查点/初始化权重 SHA256 和环境版本。数据集软链接、Conda 环境和 Git 历史均保留。

仅删除清理清单中的无版本旧实验输出、经 SHA 验证的冗余原始 JSONL、可重建临时特征与安装缓存。主请求级证据继续以无损 gzip 保存。历史参数与 NO_GO 文件仍作为当前审计依赖保留，不构成新的活动路线。

清理明细与释放量见 [cleanup_result.json](outputs/main_experiment_freeze_run01/cleanup_result.json)。释放的是磁盘空间，不清空进程内存、GPU 显存或系统页缓存。
