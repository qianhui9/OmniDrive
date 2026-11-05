# OmniDrive / Frozen R0

本地可在没有 `.git` 的情况下运行主实验；必要 Native 源码已导出并做 SHA256 校验，详见 [独立源码与溯源说明](docs/provenance/README.md)。GitHub 保留版本历史。

唯一改进研究配置为完整 **M1 + M2 + M3 R0**。入口是 `projects/configs/OmniDrive/worldstream_m123_auto.py`，完整参数来自 `configs/worldstream_r0.json`。Native 原配置保留为配对实验的必要基线；模型、权重和训练任务头保持原样。

执行顺序：M1 Precheck → M2 唯一窗口/全局增量执行器 → Detection/Map → M1 Commit → M3 当前事实准入 → FAST 直答或完整当前视觉 Prefill + K8 目标验证推测解码。FAST 不读取完整历史答案；R0 不持久复用 Prefill KV。

WSDR-KV、视觉前缀压缩/记忆、旧答案复用策略、时序风险/目标完整性候选、离线参数搜索、替代 M2 执行入口和未接入 R0 的旧 TensorRT/EdgeLLM 部署入口已退役。历史结果保留其原始含义，不作为本轮新测量。

- [最终清理验收报告](outputs/r0_route_cleanup_validation_run01/CLEANUP_VERIFICATION_REPORT.md)
- [运行指令](tools/运行文件.md)
- [主线与执行边界](docs/worldstream_final_route.md)
- [主实验及删除验证](docs/worldstream_benchmark.md)
- [冻结主实验说明](R0_MAIN_BENCHMARK.md)

所有执行命令使用 OmniDrive 解释器并以 `taskset -c 0-3,6-31` 启动。GPU 占用时等待，不终止其他用户任务。实验默认不自动部署；固定配置不等于质量或独立泛化认证。
