# R0 实验与清理验证

完整正式实验使用 `tools.worldstream_m123.frozen_main_benchmark`，支持 prepare、pilot、main、verify；五负载 A/C 采用同图像、相机顺序、时间戳、问题、生成设置与请求 RNG。Native 关闭 M123，R0 全部启用。冷启动与无 QA 视觉帧计入完整 E2E。

旧 `benchmark.worker` 只保留主入口需要的配置/模型加载/状态清空工具函数，不再提供 B1/B2/B3 搜索或消融 CLI。完整运行命令在 [tools/运行文件.md](../tools/运行文件.md)。

清理验证目录为 `outputs/r0_route_cleanup_validation_run01/`。before 来自删除前代码的隔离快照；after 使用当前代码。预先固定两条完整 Log、81 个不同帧、全部五负载，每边正式810次完整模型调用，pilot每边8次。该验证不是新的参数选择，也不是独立 TEST。

从项目根目录只执行源码/参数/导入核验：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.r0_cleanup_validation audit
```

before/after 正式运行都完成后比较真实输出、输入摘要、视觉预测、Token 和 FAST 决策：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.r0_cleanup_validation compare
```

审计记录为 `source_audit.json`，前后结果为 `before_after_comparison.json`，回归结果见各 CPU/CUDA 验证目录。墙钟计时受系统状态影响，不要求逐毫秒一致。只有实际比较通过才能声称本子集行为保持；不能把单元测试通过等同于18条Log全部重测。
