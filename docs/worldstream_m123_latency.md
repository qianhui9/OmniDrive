# M123 任务预算与推理时延

离线任务预算为 Detection mAP、NDS、Lane DET_l、VLM Token F1 各自绝对下降不超过 **0.10**，边界包含等号。默认常量、CLI、选择和曲线共用 `tools/worldstream_m2/quality_budget.py`。M3 WSDR-KV 内部预算仍为 **0.02**。复现历史 0.05 标准时显式传入 `--quality-budget 0.05`，使用新的输出目录。

M1 来源摘要完整遍历可变状态，避免重复展开同一几何/描述符/证书；每次仍重新检查数组内容、Tensor 身份和版本。图像灰度值只在同一请求内共享，patch 支持中的归一化图像供强度与梯度共同使用。这些变动不删除关联、来源校验、事务或必须刷新事件。`latency_controls --stage m1` 用冻结旧源码及真实捕获状态交替回放，检查关联、锚点和 M2 Snapshot 一致。

M3 `GenerationFacade` 对当前 batch-one HF Decode 按初始有效 token 数与追加长度计算同一位置，消除每步完整 mask 的重复扫描。采样、EOS 和生成上限保持不变。固定真实 token 序列、步数和初始 KV 的 GPU 控制验证了逐步 logits 完全一致，但单 token 时延变化极小，不能宣称可靠的 Decode 加速。

本轮独立自然输入每轮出现一次合法的 163 Token KV 复用（三轮共 489），没有修改规划器阈值。四帧诊断中的 0 复用与完整独立集结果分别记录。

本轮实验与完整记录在 [experiment_report.md](../outputs/m123_latency_quality_20261027_run01/experiment_report.md)。两个新开发场景比较 AR2/3/4，选定 AR3 / drift=0.9120883047580719。四个新独立场景的 mAP 下降 **0.105005685**，超过 0.10，因此该候选不发布，正式默认仍是 **AR2 / drift=0.7721027880907059**。独立失败没有触发其他 Runtime，也没有反向调参。三轮计时继续测量同一冻结候选，仅作诊断。

复现：`taskset -c 0-3,6-31 bash outputs/m123_latency_quality_20261027_run01/reproduce.sh <新输出目录>`。解释拒绝原因用 `tools.worldstream_m123.latency_diagnostics`；完整实验入口为 `tools.worldstream_m123.latency_quality`。可选 Profiler、冻结源码控制与 GPU 任务锁都只属于实验工具，不是额外正式执行路径。
