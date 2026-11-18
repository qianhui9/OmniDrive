# 五问、三阶段验证（2026-11-18）

本次将旧四阶段协议升级为 v2。旧 `smoke_20261117_01` 和预检目录的文件保持不变，v2 要求新输出目录，不能将旧收据当作已测量峰值显存的新协议结果。

## 修改与已验证内容

- 配置固定 `other_questions: 2`；每帧强制核验三个原固定事实问题 + 两个原始 QA、顺序和唯一性。缺失问题拒绝运行，不能降成三问或四问。
- Early / Middle / Late 按原时间顺序三等分；完整默认日志 41 帧对应 14/14/13 帧。阶段边界不清状态。
- 新增 `memory.py`，每帧五问期间记录 GPU allocator peak allocated/reserved；阶段取最大值。统计计数重置不触碰缓存。CPU RSS 仅端点采样，进程 ru_maxrss 保留生命周期标签。
- 主表记录 Accuracy、Stale、完整五问 E2E、GPU 峰值；各模块保留平均缓存大小和峰值，新增同帧合计峰值；缓存规模图使用阶段峰值。
- `query_group_summary.csv` 单列固定三问与原始两问的事实得分、已知/UNKNOWN 数及 Stale 分母。不为子集估算时延，不虚构完整生成质量评分。
- OGSM 记录原生关联对用于追溯；无可靠预测 identity 到 GT instance 的对应，错误身份关联率明确 N/A。DAVP 实际漂移/窗口年龄/全局刷新和 QCSE 实际年龄/验证 Token 接受率照常记录。

## 真实数据预检

目录：`outputs/experiments/long_run_state_aging/preflight_v2_20261118_01/`。

- 主实验 18 条完整日志中，最长为 scene-0962 / 2086743226764f268fe8d4b0b7c19590：41 帧、19.949757 秒。
- 每帧严格五问，共 205 个原定查询；六相机/几何/原 QA/官方 scene 首末帧和相邻链接校验通过。
- 三阶段 14/14/13 帧；仅预检，无完整日志模型推理。
- CPU 回归共 **60 项通过**，包含本目录 19 项及原语义/变化响应回归；见该目录 cpu_validation.xml。

## 真实 GPU Smoke

目录：`outputs/experiments/long_run_state_aging/smoke_v2_20261118_01/`。

- 状态 COMPLETED；Native、WorldStream 分别独立进程运行，使用真实连续首 9 帧，各 45 个查询，三阶段各 3 帧。
- 完整模型严格权重加载、两方法实际输入/问题 Token/RNG/精度/GPU/生成参数配对检查通过。
- OGSM epoch 保持，commit 和 DAVP publication version 均从 1 递增至 9，QCSE frame index 从 0 递增至 8，跨两个阶段边界连续。
- 逐帧 GPU 内存峰值、三模块真实结构统计、主表阶段峰值聚合与 E2E 均值、两类问题评分分母通过独立收据复算。
- per_frame_metrics 共 18 行，三个状态/缓存表各 9 行，scored_queries 共 90 行，query_group_summary 共 12 行。
- 六个 PDF/PNG 图文件生成；检查了三阶段缓存峰值图可读性。
- 详细复核结果见该目录 validation_v2.json。

首次在受限沙箱中探测 GPU 失败，没有执行推理；随后在可访问 GPU 的主机环境恢复这份 Smoke 配置并完成。未修改系统驱动，没有干预其他进程。此前启动失败记录保留在 failure.json；最终是否完成以 summary.json 和完整收据验证为准。

**没有启动完整日志 3 次重复实验。** 运行命令见 README。独立重复不会连接成一段长视频。当前数据仅约 20 秒，不能得出小时级运行稳定结论；精确 Change Response Delay / Stale Duration 和完整生成质量仍为 N/A。
