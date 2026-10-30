# FINAL_MAIN_CONFIGURATION: R0

本轮正式主实验入口：[frozen_main_config.json](outputs/m123_r0_frozen_main_benchmark_run01/frozen_main_config.json)。
原始研究策略与此前冻结输出保持不变；完整快照继承其全部参数，并将 R0 标记为唯一 `FINAL_MAIN_CONFIGURATION`。本轮不运行任何参数选择。

本轮使用 18 条官方验证 Log，各取字典序第一个完整可读 Scene，从官方首帧运行到末帧，共 723 个不同帧。A 为关闭 M1/M2/M3 的原始 Native 路径，C 为完整冻结 R0。五种既有负载已完成 7,230 次真实模型调用；8 次试运行单列，不进入主结果。A/C 顺序交替，所有输入、问题及请求初始 RNG 状态配对。

最终中文报告：[FINAL_RESEARCH_REPORT.md](outputs/m123_r0_frozen_main_benchmark_run01/FINAL_RESEARCH_REPORT.md)。五种负载 A/C 分别为 0.905、0.990、1.177、1.684、1.080；新增严重错误为 23 次观测、18 个唯一帧—问题病例。低频没有明确系统收益，目标覆盖损失与风险结论均保留。CPU 899 项通过、8 项跳过；RTX3090 CUDA 907 项通过；全部归档指标重算一致。

研究参数为 M1 `.3`、M2 budget `3`、M3 `current_only/C-basic/.45/.75`，完整历史 Runtime 默认项也保留。通用部署模板与显式研究配置的差异见 `configuration_alignment_audit.json`；部署模板未改，生产默认仍关闭。所有验证 Log 保守标为历史暴露数据，不能声称独立 TEST。此前残余严重错误结论保留。

从项目根目录只读校验：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.frozen_main_benchmark verify
```

正式 GPU 运行完成后，对原始收据重新计算指标：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.frozen_main_report --verify
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.frozen_main_report
```

主结果见 `outputs/m123_r0_frozen_main_benchmark_run01/main_results_tables.md`；完整执行与自动收尾步骤见 [FULL_REPRODUCE.md](outputs/m123_r0_frozen_main_benchmark_run01/FULL_REPRODUCE.md)，基础复现说明见同目录 `REPRODUCE.md`。只有 `main/run_status.json` 为 `COMPLETED`、完整性校验通过且主报告存在时，才能把该轮结果称为已完成。运行中的进度或单日志结果不能代替最终结论。

环境固定为已有 OmniDrive Conda 与 RTX3090。所有命令排除故障 CPU4/5。GPU 被占用只等待，不停止其他进程；不安装包、不训练、不修改任何模型权重或 M1/M2/M3 算法源码。
