# 验证记录

- 环境：OmniDrive Python；所有执行命令绑定 CPU `0-3,6-31`。
- 单元及真实 CPU 契约测试：**20 passed**；另有依赖库自身的 19 个弃用警告。机器可读记录：`outputs/unit_tests.xml`。
- 覆盖：CUDA Events 延迟解析/实际 current stream、组件不加 fence、空字段、失败调用、Query E2E fence、无插桩对照、逐帧调用汇总、跨轮样本标准差、稀疏路径、拒绝后真实 continuation 判定、原始 AST 语句保持及可恢复代码替换。
- 真实小型 CPU 执行：OGSM 连续三帧事务提交和状态连续性；真实 EVA 前缀/全局执行路径安装全部 15 个插桩边界并恢复原代码。
- 额外真实 QCSE CPU 检查：原项目小型 Llama + 原生多模态输入组装，连续三帧与未插桩原生输出 Token 完全一致；后两帧各实际执行 2 次候选验证 forward。此项用于验证插桩行为，不是正式 checkpoint GPU Smoke 或论文时延数据。记录：`outputs/cpu_qcse_contract.json`。
- 正式负载预检：与既有 manifest 的 18 Scene、723 帧、3615 个查询、seed/权重完全一致；Smoke 为同一正式首 Scene 的前三帧；每个进程在独立 Scene 预热四帧。校验源码/配置 178 项、输入哈希 5816 项。记录：`outputs/gpu_smoke_run01/experiment_config.json`、`outputs/static_validation.json`。
- 原核心/主实验源码：对照已有正式 manifest 检查 117 个文件，**0 个变化**。记录：`outputs/core_source_integrity.json`。
- 完整 5/10 轮实验：**未启动**，按任务说明仅交付配置与运行命令。

真实 GPU Smoke 已于 2026-11-17 13:23（Asia/Shanghai）完成，结果目录为 `outputs/gpu_smoke_run01/`。两对独立进程各测试相同连续三帧：插桩共 6 帧/30 查询、对照共 6 帧/30 查询；每个进程另外预热四帧。所有输入、查询及答案一致；200 条插桩原始记录通过实际调用次数核验，跨帧提交版本 1→2→3，后两帧分别复用 22、8 个窗口。

两对 E2E 均值偏移分别为 -3.923%、-1.750%，平均 -2.837%，在预设 ±5% 检查阈值内，状态 PASS。负偏移不解释为插桩带来性能提升；三帧测量仍受启动、温度等波动影响。记录：`outputs/gpu_smoke_run01/instrumentation_overhead.json`。

全部要求的 CSV 汇总及三组 PDF/320 dpi PNG 已从真实 Smoke 数据生成，图中带 SMOKE ONLY 标记。短负载没有触发全局残差复用、候选验证及其拒绝后 fallback；这些路径调用数为 0，条件时延为 N/A。候选验证的代码路径另有真实小型 CPU Llama 契约检查，但不能替代正式 GPU 路径覆盖。

完成证据：`outputs/gpu_smoke_run01/completion.json`；详细函数/边界及 5/10 轮完整实验命令见 `README.md`。完整实验仍未启动。

## forward_only_v2 纯前向口径修订

当前默认帧级主指标改为 `Frame Forward`，不包含外部加载/预处理、H2D、问题编码，以及 M3 后的文本解码/JSON 导出。详见 [新口径验证](outputs/forward_smoke_run01/结果.md)。24 项测试通过，两对真实 GPU Smoke 通过；完整五轮新口径尚未启动。原有验证记录及旧全流程时延不作为新纯前向结果。

## 三轮正式纯前向测量已完成

`outputs/20261117_191525_562820` 为实际三轮配置，3/3 轮已完成，累计 2169 帧次/10845 查询次。原始记录重新统计、文件哈希、跨轮 ddof=1、真实路径次数和纯前向分母均核验通过，详见 [正式结果](outputs/20261117_191525_562820/结果.md) 和 [核验记录](outputs/20261117_191525_562820/results_report_audit.json)。本条更新此前“完整测量尚未启动”的阶段性状态；实际轮数是三轮，不是五轮。
