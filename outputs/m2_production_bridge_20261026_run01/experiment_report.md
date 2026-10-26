# M2 完整模型证据与正式准入桥接代码

本轮完成代码与验证，未修改 M1/M2/M3 核心算法、默认正式配置、模型权重或正式成本表。旧实验结果保留。

新增能力：

- 完整模型 worker 显式加载候选表或正式表，保存真实输入/生成签名以及运行前质量规则哈希。
- 请求计时外使用同一真实 M1 快照与源缓存重放 M2，采集交替顺序的三次 Block/完整视觉成本，恢复缓存和 RNG，不增加 M1 Commit。
- 官方任务评估结果绑定实际预测与指标文件哈希；评估工具生成每个 key/count 的质量报告和具体拒绝原因。
- 质量汇总检查真实窗口数量、实际模型签名、数值案例覆盖、总体及覆盖帧 VLM 分数、关键代理退化和样本量。
- 晋升工具支持 check-only，并允许只选择 partial 报告中真正通过的完整 count 组；正式写入仍检查全部既有条件。
- 正式 B3 验收要求 adaptive 发布历史缓存、实际执行成本准入、匹配记录及非零跳过窗口，不能使用离线 hook 冒充正式复用。

## 测试结果

最终 GPU 回归 **458 passed，0 failed**。新增 23 项测试覆盖完整报告→晋升→CostModel、真实窗口执行器/实时 M1 下正式 adaptive 首帧发布与后续复用、源缓存/RNG恢复、缺失及篡改证据拒绝、错误数量/签名拒绝。所有临时批准证据仅在 pytest 临时目录，标为 synthetic unit fixture。

完整原始日志：`raw_logs/tests_gpu_final.log`。测试命令：

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q
```

## 真实 checkpoint 冒烟

使用上一轮 M2 开发场景前 4 个连续帧，每帧 1 个真实问题，实时执行 M1、Detection、Map 和原生完整 Prefill VLM。它不是独立质量评估，也没有进行参数扫描。

| 路径 | 帧/问题 | 总窗口计算实例 | 实际跳过 | 窗口复用率 |
|---|---:|---:|---:|---:|
| 原生参考 | 4/4 | 432 | 0 | 0% |
| 离线候选完整模型 | 4/4 | 432 | 14 | 3.2407% |

4/4 回答一致，观察到 GT Token F1 损失 0、新增关键代理退化 0。四个开发问题不足以形成独立质量认证。测量段候选缓存版本为 2→3→4→5（预热产生了此前版本），每个实际帧只推进一次，额外重放没有推进真实历史。

新增成本记录为 Block 0/1、更新数 53（各 2 个案例）和 49（各 1 个案例），共 4 条。所有记录仍是 `quality_certified=False`、`production_ready=False`，且均未满足其全部覆盖案例的完整视觉路径 wall/GPU 净收益条件，因此没有晋升或启用正式表。

原始执行命令、时间与返回码见 `smoke_commands.json`，输入见 `smoke_sequence.json`，配对输出见 `smoke_paired_quality.json`。真实新成本/数值证据在 `smoke_candidate/m2_window_costs_provisional.json`、`smoke_candidate/m2_full_model_validation.json`，每个案例保留三次测量原始样本。

## 使用与边界

完整操作说明见 [M2 正式准入流程](../../docs/worldstream_m2_production_bridge.md)。判定规则模板中的特征误差限值故意留空，必须在模型运行前固定；工具会拒绝未绑定规则、修改后的规则、缺失预测、篡改指标、未执行的数量和未覆盖的案例，不从本次误差反向生成通过限值。

正式全 checkpoint 路径尚未被宣称非零复用成功。现在已具备从真实完整模型生成证据、检查并加载合格新表的代码；实际启用仍取决于独立任务质量与净成本是否通过。M2 启用后需要重新验证 M3 联合质量，旧的 M2 全量计算下 M3 质量结论不能直接沿用。

机器汇总：`comparison_summary.json`。源码版本：`source_hashes.json`、`implementation.patch`。受保护文件检查：`protected_verification.json`，全部未变化。
