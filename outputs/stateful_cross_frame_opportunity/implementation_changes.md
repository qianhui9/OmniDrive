# 实现修改与代码边界

基线 HEAD：`c471d5d`。开始时存在未跟踪目录 `outputs/m3_state_guided_verified_decode/`，保留且不读取。IDE 中的 `outputs/m2_freeze_m3_task_serving/experiment_report.md` 在实际工作区不存在，不能作为证据。

新增文件：

- `tools/worldstream_m123/__init__.py`：独立研究工具命名空间，不由原模型导入。
- `tools/worldstream_m123/stateful_opportunity/common.py`：来源上下文检查、互斥漏斗计数、有限值 JSON、流式 SHA、异常安全的临时方法替换。
- `tools/worldstream_m123/stateful_opportunity/gpu_audit.py`：原始权重、原生六相机连续帧、冻结 M1 的 GPU 审计入口。默认 8 场景×12 帧，固定为已曝光 Exploratory 人口；不强制 Greedy，不删改问题。
- `tools/worldstream_m123/stateful_opportunity/report.py`：只接受完整且来源完整的运行，重建机会/依赖/成本/验收结果；未运行阶段写 `Not Run` 和原因。若实测 P0 通过，明确要求后续 P1/P2/P3，禁止自动伪造 No-Go。
- `tests/worldstream_m123/test_stateful_opportunity.py`：来源隔离、真实 ExpandedInput/LCP、真实 PyTorch MHA FIFO 来源、异常/OOM 钩子恢复、HF Sampling 签名与漏斗/成本/缺失证据回归。
- 两个工具包 `__init__.py` 与本目录交付文档、命令、结果。

P0 对 M1 在真实 Precheck 返回、当前 Commit 前冻结历史快照，记录 Anchor/Feature 来源与版本、Camera 时间、Dirty/Unknown。查询复用现有 `freeze_verified_state`、`query_world_state`，不把当前 Head 输出塞回历史状态。

M2 保留既有 EVA Window No-Go。新边界只旁路观察原有 Detection/Map FIFO Memory：真实 Self-Attention 是 PyTorch MHA，逐层 Value 的本地线性投影与含当前位姿/时间的 Key、当前查询/视觉依赖分别统计。原生历史零填充槽不进入来源机会人口。记录前一帧实际投影来源和更早的原始 Memory 来源帧；不重复记账原模型已有时序机制。

M3 使用原生 `expand_input`、`exact_lcp`、`PrefixKVCache`、`execution_signature`、`clone_kv`，缓存仅供审计匹配与旁路成本探针；不替换原生任何输入/KV/输出。LCP 比较完整 Embedding/Token/Position/Mask/签名，逐问题区分真正历史源和同帧源。额外 Full/Suffix Prefill、线性层执行用于 P0 诊断，绝不记作少做的 Forward。每次计时探针验证 CPU/CUDA RNG 不变。

临时拦截局限于当前独立串行审计进程的实例方法和 `torch.nn.functional._in_projection_packed`；后者原样返回原计算结果，观察实际 V 投影。所有替换和 Hook 用退出栈恢复。`functools.wraps` 保留 HF 检查的真实 Forward 签名。它不是并发服务 API，也没有部署到 Runtime。

保持原样：所有 `worldstream/m1/`、`worldstream/m2/`、`worldstream/m3/`、原模型 `projects/`、所有已有配置、Checkpoint/LoRA/Tokenizer、冻结阈值。运行保存这些源码的前后 SHA、参数存储和版本；报告核对实际 HEAD 和数据文件 SHA。

两个冒烟失败保留在 `attempts/`：第一次按实际实现拒绝不支持的 Attention 假设；第二次发现未保留 Forward 签名影响 HF Mask 检查。`gpu_audit01/` 是后续原生诊断 `inf` 序列化失败的部分回放，也完整保留且不进入最终人口。本轮没有停止其他用户任务；准备中断自身旧任务时发现其已经因该诊断失败而退出。正式完整人口仅来自 `gpu_audit02/`。

原生非有限不确定性保留为 `status=Unknown` 与原始非有限值名称；不替换成零。新的数值/成本测量仍要求有限，不能用这一诊断编码绕过数值门槛。后续来源追踪覆盖 Detection 的 Propagated Query→Memory 转移；局部根因探针区分继承行和新 GEMM 行。

P1 不预先实现。是否新增执行机制，以正式 `formal_go_nogo.json` 的 P0 结论为准；P0 不通过时，按任务停止条件保留原 Runtime 和审计工具。
