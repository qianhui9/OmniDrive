# M2 冻结结论

M1 冻结：True。M2 正式 Exact 算法保留，仅规范诊断字段；条件复用生产开关始终为 false。

固定 8 场景 / 96 帧 / 576 相机的来源链、patch、window 漏斗见 m2_admission_funnel.json；同一窗口分母不变。未形成完整可复用窗口，停止条件复用优化。

新鲜 96 帧 Full Forward oracle 仅在离线用当前真实 block-1 输出补齐未知 token，并运行原始 blocks 2..23 测量误差；既不能在线获得，也不能授权部分窗口跳算。来源 patch 的支持率不是注意力窗口的可跳算率。

研究路径顶层标记 experimental_conditional_forced；即使其中包含同坐标 Exact 子集，也保留 quality_qualified=false、performance_qualified=false。正式缓存和研究缓存隔离。

恢复实验证据见 service/recovery/results.json。仅当前 EVA 完整计算不代表下游记忆已恢复；本轮受控扰动后第 1/2 帧 EVA 误差为 0，检测中心最大差仍分别约 25.25/78.80m。可靠恢复先合法丢弃下游记忆，再从序列起点重放，本轮 3/3 帧恢复冻结目标。
