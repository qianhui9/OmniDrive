# v2 最终问题回答

1. v1 固定分块探针与实际拒绝/裁剪顺序探针共同表明：FP16 批量 K/V 舍入可延续到单 Token 续写，在并列/近并列 logits 上翻转 argmax。v2 针对性重放发现 1 处分歧；因果、位置、pending 与裁剪规则保持正确。另发现并修复了新 M1 快照接口的异步相机/关键帧时间戳匹配错误，旧尝试全部排除。
2. Greedy heldout 原始 QA 一致统计见 greedy_comparison.json；Sampling 明确是近似模式，固定 GT 任务事实变化见 quality_comparison.json，未用 Native 或文本相似度充当真值。完整驾驶质量预算尚未资格化。
3. 三问题完整请求中的 K8 History B Decode wall A/B：自然 1.000×，固定重复 1.929×。完整 Runtime 旧→新节省 38.043 ms（相同真实当前 carriers，三轮）。
4. 原来同一 48 QA 严格候选是 2/48=4.17%；v2 C 检索到 14/48 个兼容语义候选，启动验证 2/48，至少接受两 Token 的覆盖 2/48=4.17%。该开发重放没有逐问题 Native 成本配对，净成本有效覆盖为 NOT_RUN；新自然流已配对的净成本有效覆盖为 0%。不把候选数或单纯接受率当净成本覆盖提升。
5. 新自然/三问 C 兼容候选 27/108，启动验证 3，接受 0 Token；固定重复对应 81 次验证、1032 Token。是否产生有效接受增量，必须与 B 和净成本一起读，不能仅看检索数量。
6. 新自然/三问 C 每问题 world diff 0.0426 ms、scheduler 0.0770 ms；当前已提交快照成本另在模块表中，已计入完整 E2E。
7. B→C 三问 E2E：自然 1.002×，固定 0.973×。三轮与场景 CI 见 abc_comparison.json，M1 独立正收益（两种工作负载均成立）：False。
8. A→C 完整 M123 三问 E2E：自然 0.991×，固定 1.305×。没有把单独 Decode 收益外推为 E2E。
9. P2 自然/固定、单问/三问四个工作负载有完整 A/B/C 配对；一次 P3 主策略修订后重测三问自然/固定。最终策略单问 E2E 未重复，标记 NOT_RUN，不混用 P2 单问结论。源图像始终逐帧更新，概览模板未混入原始 QA。
10. 类别/数量/方位/运动/车道/信号/安全按可靠当前 QA 事实及独立标注记录。缺失样本与规则无法确定的项目保留 UNKNOWN/未评分；不能宣称全部类别都已通过 2pp 质量预算。关键错误逐项保留。
11. 最终资格：NO_GO。M1-guided C is reproducibly slower than history-only B in at least one primary workload; semantic candidates did not establish independent E2E benefit. 即使 History B 可有局部/完整收益，也不等于已经证明 M1 的独立贡献或完成近似部署质量准入。
12. 最主要瓶颈是 M1 对当前 QA 依赖的可信语义覆盖不足，语义候选数与可证据化有效草稿脱节。下一步唯一优先方向：补齐并独立校准 M1 的问题相关当前语义/观测覆盖，再评估相同调度策略；不继续 K 值搜索或制造场景稳定性。
