# v3 最终问题回答

1. v2 的 K4 截断已移除。固定重复 B/C 接受 1098/1122 Token，Target Forward 1278/1192。

2. Exact 保留完整历史 Token。最终固定三问 B→C E2E 1.038×，三轮/三场景均更快，delta scene CI95 [-177.0079591854786, -3.7798583895588913] ms。固定单问 0.986×，CI跨零，不能泛化为所有负载均消除退化。

3. 首 Token 未获得可信显著改善：开发36 QA中原始候选0/2命中，问题条件化候选1/6命中，仅重合3个通用开头Token，没有达到对象事实。最终自然三问没有可执行重构候选，候选内匹配率不可估，全部请求匹配覆盖0%。离线使用当前Native轨迹仅作诊断，绝非在线Oracle。

4. 固定三问筛选跳过7个 Exact 候选请求，其中首 Token 概率为零3次。实际 B→C Target Forward 少 86 次，但输出也少 58 Token；相对各自串行长度的额外 Forward 节省仅 28 次。不能把总差额全归功于筛选。

5. 自然三问 C 至少接受两 Token 请求数 0/108；B 无 Exact 时的额外接受请求逐项见 effective_draft_coverage.json。

6. 自然三问净成本有效覆盖由 v2 的 0% 到 0.00%；该指标包含真实接受与配对 Decode 净成本，输出长度变化仍是限制。

7. 自然三问 C 每问题候选准备 0.0244 ms、World Diff 0.0311 ms；Tokenizer未激活（0个合格构造候选），不能将0当实际编码免费。固定三问筛选CUDA平均每请求 0.2067 ms；host值含Prefill等待，不可当增量成本或重复加总。完整模块和独立事实提取CPU重放见runtime_cost_report.md/fact_extraction_cost.json。

8. B→C Decode / 完整 E2E：自然 1.000× / 0.993×；固定 1.076× / 1.038×。

9. A→C 完整 E2E：自然 0.997×；固定 1.383×。单问结果另表完整记录。

10. 已覆盖事实分数无相对下降，新规则级关键差异 0 项。关键限制：已评分的C固定流中仅 0 条实际启用验证；后续主动推测帧缺当前问题标签，不能据此通过2pp预算。Native既有错误/规则疑点完整保留；不宣称全面安全。

11. 语言开头可复用与世界事实可复用不是同一件事。实际重构候选到达事实内容 0 次；M1 语义覆盖、前缀重合与失败验证成本分别保留统计，不继续堆叠模板。

12. 最终 NO_GO（针对 M1 条件化语言复用主创新）：M1-conditioned language-reuse objective failed: no nonexact proposals or new natural accepted tokens in the final cohort. Fixed-three-QA C is measurably faster than B, but all proposals are ordinary exact history; the gain belongs to target-aware history scheduling and changed sampled lengths, not an identified M1 semantic contribution. 固定三问目标感知历史调度有有限实测收益，可保留为工程组件；完整质量仍不足以资格化。不能将其写成 M1 的独立贡献。
