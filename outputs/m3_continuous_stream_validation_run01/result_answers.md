# 24项实验回答

WITHIN_LOG_EXTENDED_EVALUATION / INDEPENDENT_GENERALIZATION_INCOMPLETE

1. 执行前清单：5553帧未见完整M123语言执行收据；已见466帧。计划不当作执行，分类收据及局限见historical_execution_inventory.json。

2. 149个合法>=16帧片段，包含5545帧。固定元数据规则选104帧，DEV2×16，扩展评价3×24。

3. 新M3语言执行Frame隔离成立，但全部104/104帧此前做过M2纯视觉GPU评价，额外语言重叠=0。Scene/Log和通用模板历史暴露；不是新图像或独立TEST。

4. 能。逐帧真实M1 commit和M2执行已验证，保持24帧顺序、时间戳与位姿；无QA帧仍完整运行视觉系统。

5. 主3Q C的M1含适配器成本：冷启动80.368ms（3帧），稳态230.193ms（60帧）。状态没有按问题到达重置；逐帧commit计数连续。Cold指序列状态空，模型加载在计时外。

6. 主3Q稳态实际复用51/60帧，共跳过1518窗口；可比GPU流视觉跨度C=469.330ms、Native A=483.773ms。这些均值显示有限视觉跨度减少；不能把计划复用或整体ABC差异当作独立M2收益。原生host仅enqueue，不能直接与M2同步host对比。

7. every4: B=862.798, C=803.140ms/视觉帧，1.0743×; every2: B=957.295, C=829.326ms/视觉帧，1.1543×; one: B=1146.045, C=935.334ms/视觉帧，1.2253×; three: B=1811.899, C=1287.076ms/视觉帧，1.4078×

8. 本预设负载最大B→C点估计为three，1.4078×；完整log区间及各log结果保留。

9. 正式C对象三问真实跳过126/216次Prefill和Decode；全阶段总数见gpu_execution_counts.json，所有命中Forward Hook=0且输入展开=0。

10. category: B=0.82, C=0.98，C标签50/72; existence: B=0.7916666666666666, C=1.0，C标签48/72。这是已评分必要事实质量，UNKNOWN未计正确。

11. Position共72问，已评分20、UNKNOWN=52。B正确率0.0%，C-basic为45.0%。Restricted在DEV及扩展评价的支持、准入和答案文本均与Basic一致（扩展评价216/216），未重复GPU测量。实际C-fallback正确率0.0%；同帧配对见position_policy_comparison.json。

12. 本样本没有减少：已评分Position严重错误由C-basic的11增至C-fallback的12，配对新增1、净增1。主C快速Position已评分9/28，正确率1.0、严重错误0；主C未旁路Position严重错误11。正式B→C新增严重输出0/118不代表绝对严重错误为零，也不构成严格2%风险认证。

13. 真实C→伴随策略平均增加159.041ms/视觉帧，log CI=[-14.269276890748491, 261.6116188194913]（跨零）；实际新增28次LLM请求、275次Forward。历史生成上下文随实际执行演化，不是静态替换答案。

14. 主对象3Q C物理对象标签114/216；快速回答分母另列；使用不变的2m/.5m一对一标准。

15. 问题目标标签94/216；匹配对象正确不代表最近目标正确。

16. 主对象3Q答案UNKNOWN=98/216；原因={'no_visible_GT_target': 55, 'multiple_equally_plausible_GT_targets': 3, 'visibility_ambiguous': 37, 'detector_GT_matching_ambiguous': 0, 'question_semantics_ambiguous': 3, 'missing_annotation_support': 0}。Object和Target UNKNOWN另列，未合并分母。

17. 原始自然1Q：解析1/72，正向事实0，准入0，真实跳过0。

18. 预设mixed3Q B→C=1.090836×，log CI=[1.0370611143712019, 1.1537648151073774]；人为1/3结构化插槽，不是自然分布。

19. every4的C无QA帧：M1含适配器=219.767、M2=445.694、E2E=719.693ms; every2的C无QA帧：M1含适配器=208.161、M2=421.856、E2E=684.113ms；没有省略视觉帧。

20. every4: 33.23%; every2: 22.29%; one: 22.81%; three: 43.00%，按有QA帧实际M3墙钟/E2E计算；嵌套计时不重复相加。

21. DEV2pp已评分质量约束下选择C-basic，实测E2E最小；1/2/3pp工作点见position_policy_comparison.json。扩展评价已评分loss=-0.2288135593220339，UNKNOWN阻止全任务质量保证。

22. 证明所报告历史日志内、新连续帧和固定到达模式的真实系统性能/计算省略及条件事实质量；不能证明历史log外泛化、自然应用分布或低罕见严重风险。

23. 是，仍然缺乏与基础模型和历史方法开发完全隔离的独立TEST；历史v3/v4 NO_GO及上一轮INCOMPLETE不变。

24. 可支持与实测范围一致的连续六相机对象事实QA系统论点，性能状态=SYSTEM_PERFORMANCE_VALIDATED_WITHIN_LOG；论文必须保留数据暴露、任务覆盖、UNKNOWN和严重风险统计局限，不能表述成泛化GO。
