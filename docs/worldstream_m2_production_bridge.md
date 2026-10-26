> 历史成本晋升流程，已退出正式 M2 执行。当前正式路线见 [直接 WSGRR 执行契约](worldstream_m2_direct_wsgrr.md)。以下保留用于解释旧实验，不能再用作正式缓存发布或窗口执行前提。

# M2 完整模型验证与正式准入

这组工具连接现有窗口执行器、完整模型任务评估与正式成本表。M1/M2/M3 核心算法、
默认 `adaptive` 缓存策略、模型权重与正式成本表均保持原样。
候选测量、质量通过、正式表发布和正式路径真实复用分别记录。

## 1. 在运行前固定样本和判定规则

使用未覆盖旧结果的新输出目录，例如 `outputs/NEW_M2_RUN`。准备带稳定 nuScenes
`tokens` 和每帧 `questions` 的连续场景 `sequence.json`，格式与 M3 独立验证相同。
初步调试可以使用开发帧，但不能据此声称独立泛化通过。

复制 `configs/m2_assessment_rules.example.json` 为本轮规则文件，先确定允许的特征
误差与任务损失，再开始模型运行。模板中的两个 `null` 必须改成有独立依据的有限
非负数；模板本身不能运行或生成 passed。数值容差不能从本次候选误差反向放宽。
任务指标的 `max_loss` 为绝对分数差。高分优先指标损失为 native − candidate；低分
优先指标为 candidate − native。Map 路径依据真实 evaluator 字段填写，嵌套字段用 `/`。

`--m2-assessment-rules` 在模型加载前记录规则 SHA-256。汇总工具要求两条路径都绑定
同一份规则；只在评估结束后提供规则会被拒绝。设置最少帧、场景和每 key/count 的
案例数量，避免用重复计时扩大质量样本量。

## 2. 原生参考与候选完整模型执行

两条路径均执行实时 M1、Detection、Map 与原生完整 Prefill 的 VLM，保持相同输入、
问题、seed 和 warmup。候选使用已有离线 D API，不改变正式准入条件。

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.worker --condition B2 --workload language --limit 24 --warmup 3 --seed 0 --sequence-file outputs/NEW_M2_RUN/sequence.json --m2-task-evidence --m2-assessment-rules outputs/NEW_M2_RUN/rules.json --run-root outputs/NEW_M2_RUN --output outputs/NEW_M2_RUN/native

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.worker --condition B2 --workload language --limit 24 --warmup 3 --seed 0 --sequence-file outputs/NEW_M2_RUN/sequence.json --m2-task-evidence --m2-assessment-rules outputs/NEW_M2_RUN/rules.json --offline-wsgrr D --m2-candidate-cost-table outputs/m2_reuse_diagnostic_20261025_run01/visual40/window_costs_provisional.json --m2-profile-evidence --run-root outputs/NEW_M2_RUN --output outputs/NEW_M2_RUN/candidate
```

候选表用于绑定实测身份；其 `quality_certified=False` 不会变为正式批准。
候选刷新策略和普通窗口预算必须与正式配置一致。输入 ID 与生成配置必须配对相同，
视觉 carrier 允许因 M2 复用改变。

`--m2-profile-evidence` 在真实请求之后，从该请求捕获的 M1 快照和不可变源缓存进行
配对视觉路径及 Block 测量（预热 1 次、测量 3 次、交替顺序）。随后恢复真实缓存、
诊断和 RNG，不提交额外 M1 事务。产物为：

- `m2_full_model_validation.json`：真实 frame/scene、特征差异、实际窗口计划、原始计时；
- `m2_window_costs_provisional.json`：本次真实 key/Block/count 的新成本候选，未批准；
- 原有完整模型 `predictions.pkl`、逐帧指标与逐问题回答。

额外 profiling 不计入请求时延；它保留源张量并使用额外工作区，累计峰值显存不能
当作无诊断部署峰值。不能把嵌套 Block、视觉路径和整帧时延相加。性能结论仍需在
关闭成本 profiling 后进行重复配对测量。

## 3. 官方任务评估及质量报告

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.task_quality --run outputs/NEW_M2_RUN/native --output outputs/NEW_M2_RUN/native_eval
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.task_quality --run outputs/NEW_M2_RUN/candidate --output outputs/NEW_M2_RUN/candidate_eval

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.assess_candidate --candidate outputs/NEW_M2_RUN/candidate/m2_window_costs_provisional.json --baseline outputs/NEW_M2_RUN/native --candidate-run outputs/NEW_M2_RUN/candidate --baseline-eval outputs/NEW_M2_RUN/native_eval --candidate-eval outputs/NEW_M2_RUN/candidate_eval --rules outputs/NEW_M2_RUN/rules.json --assessor m2_assess_candidate_v1 --output outputs/NEW_M2_RUN/assessment
```

评估文件绑定实际预测 SHA-256、样本顺序、checkpoint、问题与生成参数。
每 key/count 必须在对应真实帧中执行部分更新，签名和数量必须匹配，且覆盖所有数值
案例。VLM 同时检查总体和该记录覆盖帧的 Token F1，并记录数字/空间/运动/安全代理
新增退化。代理不等同人工语义或安全认证。

`quality_report.json` 为现有 `m2_window_task_quality_v1`；每条记录保留具体拒绝原因。
`partial` 表示只有部分记录通过，晋升只允许其中真正通过的完整 count 组。独立性
指分开的原生与候选完整模型任务评估，不自动宣称与历史阈值选择场景隔离。质量数据
衡量完整实际 M2 策略，不能宣称每个 count 的独立因果收益。

## 4. 检查、生成新表和正式验收

从新记录中明确选择待验证的数量。下面的 `30 34 36` 只是旧候选的示例，须替换成
本次真实测量且质量合格的数量；不能为了触发复用写入未测数量。

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.promote_costs --candidate outputs/NEW_M2_RUN/candidate/m2_window_costs_provisional.json --quality-report outputs/NEW_M2_RUN/assessment/quality_report.json --approved-counts 30 34 36 --check-only --report outputs/NEW_M2_RUN/promotion_check.json
```

check-only 重建当前 GPU/模型/输入几何签名，逐 count 组返回原因，不生成批准表。
只有质量、数值和每个覆盖案例的 Block 及完整视觉路径 wall/GPU 成本均通过，才使用
已有写入命令：增加 `--approved-by <实际研究者身份> --confirm-approval --output <新表路径>`，
移除 `--check-only --report`。保持旧表与旧实验结果不变。

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.worker --condition B3 --workload language --limit 24 --warmup 3 --seed 0 --sequence-file outputs/NEW_M2_RUN/sequence.json --m2-cost-table outputs/NEW_M2_RUN/window_costs_approved.json --m2-task-evidence --m2-acceptance --run-root outputs/NEW_M2_RUN --output outputs/NEW_M2_RUN/formal
```

该命令走真正的 `forward_from_m1()` 和 `adaptive` 缓存策略，没有离线 hook。
`m2_acceptance.json` 验证：先发生正式缓存发布、复用确实经过匹配成本准入、实际执行
ID/窗口数正确且 `reused_windows>0`。无实际复用会明确 FAIL，不删除失败样本。
它是执行验收，不替代质量与收益复测。单独评估正式 M2 时可加 `--m3-strategy native_full`；
启用联合 M3 后须重新验证视觉 carrier 改变对 M3 的影响，旧 M3 质量结果不能直接沿用。

本轮仅提供显式的新表路径覆盖；默认正式配置仍引用原表。新表完成正式验证后，才将
项目配置的 `m2_config.cost_table` 指向该新表，并保存新的配置与源码哈希。
