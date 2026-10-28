# 自然质量验收：未通过

独立 6 个 VAL 场景/Log、48 连续帧、每帧原始有序 3 个自然问题，共每方法 144 请求。A=Native 原始布局；B=WRCPS-ZeroKV、真实训练 Adapter。自由生成、原始问题与参考答案不变，B 全部 actual_reuse_length=0。C=ExactKV 未进入服务实验。

检测与地图使用项目官方 evaluator 的这个声明子集，Token F1 对自然回答计算；不是完整 nuScenes VAL 指标。A 为原生全计算，B 保留已有 M1/M2 正式路径，因此 A→B 除表示外还包含原有 M1/M2 路径差异，不能用所有差异对 Composer 作单因果归因。

|指标|A Native|B ZeroKV|绝对下降|
|---|---:|---:|---:|
|mAP|0.306739|0.292012|0.014726|
|NDS|0.350864|0.340438|0.010426|
|DET_l|0.084874|0.084183|0.000691|
|Token_F1|0.484386|0.482149|0.002236|

四项下降均在 0.10 预算内。**关键语义门未通过**：自动 proxy 标记 16 条潜在退化，另有 6 条新增高影响主张待复核；两类有重叠，不能相加，也不能把 16 条全称作经视觉证实的错误。crosswalk 被关键词误当作行人运动、漏掉 ahead 等会造成假阳性；显式行人穿越矛盾 parser 检出 0，也不证明无运动错误。

至少以下三条存在明确的参考答案支持的左右翻转：

|Frame|问题实体|参考/Native|ZeroKV|
|---|---|---|---|
|5193a061a8c24adb9663daa3a3c8e2cc|bus stop|右前 / 右|左|
|e2f2f4b0a2424cfe9f709587c7e1ef8d|No Turn on Red sign|右前 / 右前|左前|
|d10b460fd96042d18edb5bb648857d32|pedestrian pathway|右 / 右|左|

这些请求的核心均无效且历史复用为 0，不能归因于陈旧核心或旧 KV。完整原文及代理评分见 zero_quality_pairs.json，逐请求生成 Tokens 和统计见 quality_request_summary.jsonl。

## 限定归因对照

仅重跑包含第一例的同一场景 8 帧/24 请求，诊断选择发生在失败之后，不是新的独立验收。24 条 Zero 回答与原实验全部相同。随后在**同一当前 M1/M2 载体、同一问题采样 CUDA RNG** 上执行冻结 Native Teacher；bus-stop 例 Teacher 回答右、Composer 回答左。当前载体 SHA 为 8546d50178c638ba10174e8825d64421a80c4d882923d9982b3f3b7ae03d190e。这证明新表示足以造成该例空间回答变化，无需历史 KV 或 M2 输入差异；不把其他所有错误也归因于同一机制。

检查了前/前右/前左相机图像：可见右侧大巴，但字面 bus-stop 标识位置未能由这些图像无歧义验证。图中还有参考与两个模型都遗漏的交通灯，说明原始 QA 标注也不完美。因此保留“参考支持的空间回归”及视觉不确定性，不伪称所有 QA 是物理真值。路径及图像 SHA 见 spatial_failure_case.json；同载体证据见 failure_attribution_summary.json 和 failure_diagnostic/teacher_attribution.jsonl。诊断额外 Teacher forward 在计时之外，不能作为效率证据。

最终 VAL 48 帧核心有效数 0、跨帧命中 0/42（target_domain_coverage.json），即便放行标量指标也无法证实 WRCPS 目标收益。结论：不准入 C，不替换正式配置；下次结构/数据修改应使用新的开发 Log 和新的保留验收集。
