# 开发实验：新阈值候选超出精度预算

四项任务允许的绝对下降：`{"mAP": 0.05, "NDS": 0.05, "DET_l": 0.05, "Token_F1": 0.05}`。新阈值只来自开发集漂移数据；下列候选均为同一 M2 算法，非多条部署路径。

| 间隔 | 漂移阈值 | 后缀复用帧 | 请求 mean ms | mAP 下降 | NDS 下降 | DET_l 下降 | Token F1 下降 | 通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.971169874 | 14 | 3000.51 | 0.044247 | 0.065262 | 0.010383 | -0.016403 | False |
| 3 | 0.971169874 | 16 | 3079.63 | 0.054828 | 0.062926 | 0.028819 | 0.008651 | False |
| 4 | 0.971169874 | 18 | 2963.19 | 0.058048 | 0.066193 | 0.035964 | -0.006043 | False |

新阈值下没有候选通过四项开发精度预算，因此不生成新策略 selection.json，也不使用预留独立场景做反复调参。提高复用率没有保证任务误差在记录的预算内。

[原配置预算重评](previous_budget_reassessment/reassessment.md) 单独列出旧预测在新标准下的精度结论与原有计时。保留原参数，不把预算修改解释为精度或速度改善；回顾性结果不是新阈值的独立证据。

数字/空间/运动/安全逐问代理退化保留在 cal_age*_quality_pairs.json。曲线包含不合格候选：[PNG](quality_latency_curve.png) / [SVG](quality_latency_curve.svg)。新测试日志见 tests_gpu_final.log；运行命令与退出码见 commands.json。

M1/M3 算法、前缀预算与模型权重未改；M3 内部 0.02 策略参数不等于离线任务验收预算。新预算及默认值见 quality_budget.json 与 tools/worldstream_m2/quality_budget.py。

复现：从仓库根目录执行 `taskset -c 0-3,6-31 bash outputs/m2_global_residual_20261026_run02/reproduce.sh <新输出目录>`；没有合格候选时，脚本在开发结果处结束。
