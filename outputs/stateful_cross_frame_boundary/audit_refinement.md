# 旧 P0 修正

旧 8 场景/96 帧/88 转换/96 请求是原生 Planning，而分类器返回 global_safety；不是 native_multi VQA。全部已曝光，保持 Exploratory；不与旧 VQA 相加。

Value 与 Prefix 的局部数值差异是已测事实，Sampling 分布是 Not Certified；Stateful Active 完整质量与 E2E 是 Not Run，不改写为实际质量失败。旧 658 项通过记录与全部失败/成功目录保持原样。

report.py 的 P0 不再由任一局部正收益触发。新 opportunity_gate 独立检查真实历史来源、数学闭包、对应 Claim 的状态证据、原生重复工作、覆盖维护成本的上界、正净成本潜力及可执行验收计划。production_go 始终需要后续正式认证，本轮未给予。缺项、未知或非有限测量均 Fail-Closed。

新增性能字段见 optimistic_savings_bounds.json（含旧局部候选）；纯验证/缓存没有隔离测量时保留 null 和具体范围，绝不用零冒充。旧报告重建到新 p0_refinement/，工具拒绝覆盖原始历史研究目录。