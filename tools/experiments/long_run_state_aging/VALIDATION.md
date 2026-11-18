# 本次交付验证

日期：2026-11-17（Asia/Shanghai）。所有代码、测试和数据预检命令均使用 `taskset -c 0-3,6-31` 与 OmniDrive 解释器。

- 新实验 CPU 测试 15 项通过；与原环境评价语义/变化响应回归共 56 项通过。
- 使用真实 `WorldStateManager.precheck/commit` 连续提交 8 个 CPU 测试帧，跨四个统计阶段的 epoch 不变、Commit 版本从 1 递增至 8。
- 使用实际 `StateStore`、`StateCache`、`GlobalAnchor`、`AnswerMemory` 数据结构和小 CPU 张量验证三个只读采集接口，并核验读取前后缓存内容和 revision 未变。
- 核验 PyTorch 1.13 typed storage 的大小统计与视图去重、年龄分布、Token 接受率合并分母。
- 核验中断半行恢复、跨阶段意外 reset 拒绝、部分运行阶段不重划、部分 attempt 不混入完整配对主表、报表重复生成不修改原始 journal。
- 真实数据预检：主实验最长完整日志 `scene-0962`，scene token `2086743226764f268fe8d4b0b7c19590`，41 帧，19.949757 秒。四阶段 11/10/10/10 帧。
- 预检核验原始六相机文件、几何信息、严格时间序、`next/prev`、官方 scene 首末帧与总帧数、原 QA 文件、权威 GT 和冻结权重 SHA256；不启动模型推理。

**真实模型 GPU Smoke 已完成。** 起初监测到其他任务占用约 17.3 GiB 显存，利用率最高为 98%；等待至显存 15 MiB、利用率 0% 后才启动短 Smoke。没有停止或干预其他任务，也没有启动完整 GPU 实验。

结果目录：`outputs/experiments/long_run_state_aging/smoke_20261117_01/`，状态 `COMPLETED`。Native 与完整 WorldStream 使用独立进程，各完成真实 8 帧、40 个查询，四阶段各 2 帧。两边冻结权重严格兼容检查通过，实际相机、图像/几何摘要、问题 Token、RNG、生成参数、GPU 和精度均通过配对核验。WorldStream 8 帧 OGSM epoch 不变、Commit 从 1 到 8、DAVP publication version 从 1 到 8、QCSE frame_index 从 0 到 7，跨全部三个阶段边界保持状态。

三模块实际采集均成功；完成 summary/stage/per-frame/OGSM/DAVP/QCSE/scored-query CSV/JSON、report.md，以及三张 PDF 和三张 PNG 图，并检查 PNG 可读性。Smoke 中没有达到 DAVP 复用年龄上限；全局锚点每帧实际刷新；QCSE 提出 136 个候选 Token、接受 110 个。Early 未提出 Token，接受率如实记为 N/A。此短序列仅 3.449 秒，不能推断长期稳定性；不能把未执行的正式实验当作完成。

完整 41 帧预检另存 `outputs/experiments/long_run_state_aging/preflight_20261117_final/`，状态 `PREPARED_NO_INFERENCE`；其中有 205 个既定查询、70 个已知 GT 查询、3 个采样事实变化查询。`cpu_validation.xml` 保存 56 项通过结果。

正式运行代码已提供，但需按任务说明先确认后执行 `--run-full`，默认 3 次独立运行。每个方法/重复独立进程，日志内部保留完整状态；阶段边界只有统计含义。缺少精确变化时间，Change Response Delay 与 Stale Duration 为 N/A。

完整配置、可用指标和每个字段的数据结构来源见 README。当前分发目录不能读取 Git Commit（`.git` 无仓库元数据），manifest 记录 null 和原因，并保存实际源码哈希。
