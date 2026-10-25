# OmniDrive 单一研究主线

研究目标：**World-Referenced Stateful VLM Serving with Dependency-Aligned GPU Execution and Efficient Semantic Consumption**。

唯一正式入口：`projects/configs/OmniDrive/worldstream_m123_auto.py`，直接继承原始 `mask_eva_lane_det_vlm.py`，保持权重、EVA、Detection/Map、LoRA、Tokenizer 和原始 Decode。其他 worldstream 模型配置及替代执行器已退役。原始 OmniDrive 配置保留作独立参考基线。

执行顺序固定为：M1 Precheck → M2 窗口级视觉执行 → 原始 Detection/Map → M1 Commit → M3 可信历史增强与精确 Prefill → 原始 Decode。M2 与 M3 的历史快照均在当前 Commit 之前冻结；M3 在 Commit 后消费冻结副本，不读取新提交状态冒充历史。

M1 只有普通 A 的几何门控、一次全局一对一分配与实际观测更新。正式配置开启 `temporal_visibility`、`local_support`、`source_linked_state`。场景、时间、epoch、执行签名、来源摘要、过期及原子事务检查保留。DQTrack、Shadow/Hint、身份消歧、深度搜索及替代身份历史策略不再提供配置开关。普通关联及观测支持不签发物理身份或几何真值认证。

M2 只有 `window/source_linked/maintain`。当前 WSGRR 使用状态事件、区域漂移、年龄、局部观测约束与窗口依赖闭包决定更新集，细节见 [M2 说明](worldstream_m2_window.md)。缓存未命中、来源失效、强制刷新或成本/质量准入不足，都将同一个 WindowExecutor 的更新集扩大至全部窗口。没有 camera/conditional 缓存执行器，也没有 Exact 绕过模式。Patch Embedding 与 block 2 起的全局依赖部分完整执行。

历史全形状点积策略曾用于数值对照；当前唯一 WindowExecutor 的 v2 策略在部分更新时对选定窗口执行 QKV/Attention/投影，对其有效 Token 执行 MLP，完整更新在该执行器中调用完整原生 Block。GEMM 形状变化需匹配当前策略重新认证，不能套用旧策略的位级一致结论。统计严格区分复用输出窗口、跳过 Attention 与完整 GPU 时间节省，不能把跳过窗口比例作为端到端加速比例。窗口成本表单独维护；旧策略证据不迁移，未获数值、任务质量与成本准入时全量更新。

M3 只有完整当前视觉载体、经过来源与任务质量门控的历史文本补充，以及精确扩展输入前缀/KV 复用。历史不可用于替代当前视觉；缺失、不适用或未获独立质量批准时保留原问题与完整载体，不添加历史文本。前缀未命中或未获准入时 `reuse_length=0`，继续使用同一个 PrefillExecutor；不调用原生 `lm_head.generate()` 作为运行时回退。不支持的模型/生成契约显式报错。原生 generate 仅在关闭扩展的独立基线和等价性验证工具中作为参考。

完整 Prefill 在真实模型与相同种子下通过 token、logits、RNG 对照；完整相同前缀也通过。受控局部前缀的 token 与 RNG 一致，但 logits 超过严格容差，因此当前不授予正式局部前缀复用。历史文本质量与 KV 匹配分别认证；两者不能相互代替。当前部署批准表保持为空。

当前 `outputs/final_route/window_costs.json` 的 `records`、`formal_quality_policy.approved_counts` 均为空；`m3_prefill_costs.json`、`m3_prefill_quality.json`、`m3_state_quality.json` 的 `approved` 均为空。成功生成答案不代表批准缓存复用，前缀匹配不代表实际 KV 复用，历史文本消费不代表答案质量提高。

实际存在的报告：[执行契约测试](../outputs/m123_execution_contract_20261025/experiment_report.md)、[WSGRR 历史修复](../outputs/wsgrr_mainfix_20261023/experiment_report.md)、[本轮 GPU 短序列审计](../outputs/m123_execution_audit_20261025_run01/experiment_report.md)。旧报告保留其原始运行边界，不作为本轮新结果。完整验证集 Detection/Map 和语义精度、当前策略的独立数值/任务质量/净成本证据、历史文本独立质量、正式 KV 复用及端到端加速仍需分别验证。

运行和数值验证命令见 `docs/worldstream_benchmark.md` 与 `docs/worldstream_m2_window.md`。
