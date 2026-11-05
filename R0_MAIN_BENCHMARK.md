# FINAL_MAIN_CONFIGURATION: R0

清理后的完整主实验已通过最终核验。唯一完整参数：[worldstream_r0.json](configs/worldstream_r0.json)；实际运行冻结快照：[frozen_main_config.json](outputs/m123_r0_only_reproduction_run01/frozen_main_config.json)。Native PyTorch 保留作必要配对基线。

本次重新执行 18 Log、18 完整 Scene、723 个不同帧、五种负载，共7,230次正式模型调用和8次 pilot。完整11,226个 A/C 问答与历史正式实验逐项比较，答案、评分、目标召回、FAST决策与请求随机状态零差异；M1/M2/M3主要非计时科学指标一致。没有复用历史收据冒充新执行。

| 负载 | 历史加速比 | 本次加速比 | 本次时延降低 |
| --- | --- | --- | --- |
| Every-4 | 0.9047× | 0.9053× | -10.46% |
| Every-2 | 0.9903× | 0.9930× | -0.70% |
| Every-1 | 1.1772× | 1.1835× | 15.51% |
| Three-per-frame | 1.6836× | 1.6915× | 40.88% |
| Mixed-three | 1.0796× | 1.0818× | 7.56% |

Every-4 变慢，Every-2 的收益区间跨1；高频单问、每帧三问、混合三问的加速比95%区间均高于1。原有负载适用性结论保持。

CPU 352通过、4跳过；真实RTX3090 CUDA 356通过。21个必需交付文件、39个原始权重资产及受保护历史输出校验通过；无损归档后12组科学结果重算一致。测试总数减少来自旧实现测试退役，现行R0测试与数值控制保留。

R0已评分事实平均准确率85.27%，严格非完全正确比例467/3085=15.14%；另有2528/5613个UNKNOWN。原有新增严重错误23次观测、18个唯一帧—问题病例保持；清理没有增加这些错误。数据仍属历史暴露验证数据，不宣称独立TEST或生产风险认证。

最终报告：[FINAL_RESEARCH_REPORT.md](outputs/m123_r0_only_reproduction_run01/FINAL_RESEARCH_REPORT.md)。源码清理验收：[CLEANUP_VERIFICATION_REPORT.md](outputs/r0_route_cleanup_validation_run01/CLEANUP_VERIFICATION_REPORT.md)。逐问对照：[full18_question_comparison.json](outputs/r0_route_cleanup_validation_run01/full18_question_comparison.json)。

从项目根目录只读核验冻结来源和历史证据：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.frozen_main_benchmark verify --root outputs/m123_r0_only_reproduction_run01
```

核验归档中的全部配对收据，并写入 `integrity_validation.json` 验证记录：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.frozen_main_report --root outputs/m123_r0_only_reproduction_run01 --verify
```

重新运行或重新生成报告的完整指令见 [运行文件](tools/运行文件.md)。新实验使用新的不存在目录；上面的已验收目录只用于核查已有证据。CPU 4/5不得使用，GPU被其他任务占用时等待，不停止其他用户进程。本次未训练、未改权重、未搜索参数、未推送Git。
