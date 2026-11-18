# 主实验统计口径统一（2026-11-17）

最新结果：

- [消融结果](../../outputs/worldstream_ablation_full_run01/结果.md)：14 组离线重评分，四张 CSV、指标 JSON 和逐问题评分已同步更新。
- [环境稳定性结果](../environment_stability_eval/outputs/run_20261117_113951_740642/结果.md)：逐问题复核评分一致，保留原准确率、Stale 和 Change Response 数值，统一主 FPS/QPS 分母。

无新增 GPU 推理；原始回答、收据、manifest、冻结推理源码和权重保持原字节。各结果目录的 `protocol_history/20261117_142838/` 保存旧报告、旧指标和旧评分；`main_protocol_v1/audit.json` 保存本次证据。

## 统一定义

1. Accuracy：使用主实验 `continuous_stream_report.grade`，对全部 known 查询取平均事实得分，保留自然 QA 的有限部分分。无可见正目标不新增否定标签，UNKNOWN 排除且计数；完全正确率独立报告。
2. FPS/QPS：帧数或查询数除以累计帧 E2E 秒数。环境实验原处理区间墙钟吞吐保留为辅助值。
3. E2E：保留原始每帧五问的完整计时，不推算其他负载的反事实耗时。
4. OGSM：主表统一为含适配器时延；LLM Prefill/Decode 主表统一为 ms/视觉帧。原 Precheck+Commit、ms/查询独立保留。
5. Stale、稳定/变化分组、MACs 与 Change Response：主报告没有完全相同指标，继续按正目标 GT、实际执行计数与采样视频时间域定义作为补充指标。消融 Stale 统一使用环境实验的同问题前次有效可比 GT 规则。

## 负载边界

|范围|WorldStream|评分分母|
|---|---:|---:|
|历史主实验五负载合并|85.267%|3085|
|历史主实验 S4 三问/帧|86.244%|1214|
|消融共同三问已知 GT|86.244%|1214|
|消融实际五问负载|84.459%|1969|
|环境稳定性实际五问负载|84.459%|1969|

相同评分规则不代表相同负载，不将历史 85.267% 强行填入新表。旧 58.007% 属于含新增无目标问题的另一套二元评分，已从当前主表撤下并归档。

## 生成与验证

`unify_main_protocol.py` 核验源码/输入/权重哈希、逐组收据 SHA256、帧序与问题配对；对所有 14 组离线重评，并断言 Full 的 2169 个核心回答及序列化评分与主实验 S4 完全一致。环境实验的新旧评分逐项一致。

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m tools.worldstream_ablation.unify_main_protocol
```

本次版本已生成，重复执行会拒绝覆盖 `main_protocol_v1`；不同协议需要另行版本化。旧 `summarize_results` 属于历史扩展评分，不用于覆盖当前主表。

部分分/UNKNOWN、FPS/QPS 分母及环境语义共 28 项测试通过。另独立核验全部 14 组得分和、Stale 分母、分场景统计、四个 CSV、Markdown 表及两实验吞吐分母，E2E/MACs 原值均保持一致。
