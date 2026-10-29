# 24项结果回答

结论：INCOMPLETE_INDEPENDENT_VALIDATION；历史v4 NO_GO不变。

1. 是。单一Runtime保留解析、当前事实、已有绑定门槛、一个Position支持度门槛、事实回答或原K8生成；历史复现分支隔离保留。

2. 是。current_only路径拒绝这两个历史风险开关；历史实验入口保留。

3. 是。current_detection_support及中心投影接口未改变，没有恢复M2全框复用资格对检测事实的过滤。

4. v3新增左右错误dcdbd110...：真实Front-Left目标的已解码候选最高0.286888，被M1 .3过滤，当前旁路选Front-Right。另一个73c335...目标0.309866通过M1但低于M3 .35。v4的2ea0bbd8...在冻结标签下方位错误成立，但新保守实例/可见性评价为UNKNOWN，且正式帧无过滤前raw捕获，无法断言M1漏检、过滤或具体身份误判。逐例新匹配和旧全链路证据见position_risk_audit.json。

5. 当前观测、ROI、几何扇区、已有绑定风险和DEV置信度门槛通过者可候选快速回答；这仍不是可靠最近实例认证，本轮没有新独立TEST证明。

6. 新TEST未运行，不能宣称减少。历史DEV三个CPU反事实的新增严重错误见dev_policy_comparison.json；不作为新实验结论。

7. 历史DEV全部回退反事实多调用2次Position LLM，但这2条配对事实标签均UNKNOWN，因此不能判断方位事实质量提高或下降；不能把更多回退视为改善。新GPU E2E未测。

8. 实现保留；历史DEV每种策略Category候选3/16、Existence候选8/16相同。新完整E2E收益未测。

9. 本轮真实跳过0，因未启动完整模型；v4正式为77/480。CPU反事实命中不算跳过率提升。

10. 见runtime_cost_report.md及cpu_cost.json：真实CPU微测，Position gate嵌套于validity；不能外推GPU E2E。

11. 历史v4对象三问C提案Object标签 41/96；真正快速回答的标签分母另见nearest_target_identity_audit.json。无新独立TEST标签。

12. 历史C提案Question Target标签 31/96；这是实例正确性，不能与回答方位正确率混合。

13. 历史回答标签原41/96，现40/96；新增同扇区/同类别回答标签不建立唯一实例身份，同时更严格地拒绝可见性歧义。UNKNOWN原因及获得/失去标签逐条列出，未放宽2m/.5m实例匹配标准。

14. 是。权重内嵌config和本地原始config都使用nuScenes train及train Conv/VQA。没有逐scene训练执行清单，不声称精确训练曝光可排除。

15. 未找到：合格且历史未使用的官方val日志 0。

16. 只能支持当前事实旁路的工程实现、历史受限对象QA性能和诊断；不能支持新泛化GO或2%罕见严重风险认证。

17. 没有合格的新DEV最优配置。历史实测最佳target_dev_0未改变；三个新策略仅CPU诊断，不冻结为在线策略。

18. 新正式值null；历史v4对象三问1.344391×，不能归给本轮代码或Target Completeness。

19. 新正式值null；历史v4预设混合1.116378×，不代表真实自然QA分布。

20. 新TEST新增严重错误和CI均null，不伪报零。历史标签的配对新增/净变化、场景CI与唯一请求见severe_error_audit.json；历史v4原验收1/41和NO_GO保持不变。

21. 尚不能判定更优，没有新配置真实质量—E2E配对。历史重评分和CPU成本不能证明新Pareto优势。

22. World-Fact-Grounded Selective Language Bypass：利用M1当前已提交结构化事实选择性省略Prefill和Decode；此前真实收益来自Current-Only机制。

23. 缺少独立模型未见TEST、充分最近目标身份标签、严格罕见严重风险样本、新DEV最佳策略及本轮完整E2E测量；不证明Temporal Risk/Completeness独立贡献。

24. 已有工程与限定对象QA的历史性能依据，尚不足作为MobiCom等会议M3核心研究结论的完整证据；需补独立TEST、任务覆盖及场景级质量—时延/严重风险证据。
