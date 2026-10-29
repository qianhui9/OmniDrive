# 20项最终回答

1. 严重Pedestrian案例中，当前正确Front-Left候选原始得分0.286888，被M1 0.3过滤。另选Front-Right有当前观测支持，却不能证明其为真正最近可见目标。全链路和物理/可见GT分别保留在nearest_target_failure_audit.json。

2. 三个旧可评分快速错误：两个目标过滤（分别M1 .3和M3 .35），一个目标完整性不足。不是三个原始检测缺失，也没有证实排序代码按距离排序出错。

3. 最小合法Anchor仅是检测集合中的最近者；集合可缺失低分目标，也可能含不满足问句可见目标要求的候选。相机描述子不证明最近排序完整。

4. 已有解码输出中的更近/接近且会改变类别或方位的低分对象、当前目标分数、相机数及证据来源可提示歧义。未来、GT、历史预测均未用于当前准入。

5. 同一Runtime：问题解析→当前正向Witness→只对相关选择任务计算竞争分数比/距离支配与任务来源条件→准入或原K8生成。竞争候选只提升不确定性。风险不是概率。

6. 实测解析、当前事实提取、完整性检查、快照、M1输出适配器和E2E见runtime_cost_report.md及abc_comparison.json；嵌套时间不重复相加。 补充实测：C-target对象三问实测：[{"stage": "ablation", "condition": "C-target", "workload": "addressable", "question_count": 3, "questions": 96, "snapshot_ms_per_frame": 2.3986779124243185, "extraction_ms_per_question": 0.017079219105653465, "completeness_ms_per_question": 0.018662380171008408, "retained_competition_candidates": 39.90625}]

7. 新TEST同证据核心消融拒绝={}；有益/无益/UNKNOWN按独立标签分开，不能把低分候选本身当作真实肯定事实。

8. 正向存在性可由合法当前单个Witness支持；最近类别/方位按DEV校准的置信度、竞争风险和中心投影相机数准入。中心投影不证明无遮挡/身份/排序，也不是M2复用证书。

9. v3当前检测证据修复保留，未恢复90%全框复用资格过滤。旧Car实际输出回归见prior_regression.json；回归不是新TEST证据。

10. 新增严重输出相对B=1，唯一Frame/Question=1；C-basic与C-target比较见核心消融，所有严重左右/前后案例在severe_error_audit.json。

11. Category/Existence/Position/Motion的A/B/C分母、UNKNOWN、分数和严重错误分别见quality_by_task.json；不存在的可靠Motion标签不计正确。 补充实测：分任务B→C：category: 1.0 → 1.0; existence: 0.5142857142857142 → 0.9428571428571428; position: 0.4444444444444444 → 0.7777777777777778; motion: None → None。未评分显示None；分母与唯一Frame/Question见对应JSON。

12. 正式请求真实跳过77次，零输入展开/Prefill/Decode/验证Forward。分负载实际分母见fast_answer_quality.json；DEV及消融另计。 补充实测：正式C共480个问题，真实跳过77；可用候选和实际准入覆盖在workload_coverage.json分开。

13. 同一新TEST对象三问C-basic命中47/96，C-target命中47/96；不把不同数据流直接当成相对v3的因果提升。

14. 未命中进入同一K8生成器；正常B和C-basic不承担新增竞争读取，C-target承担读取/快照/检查成本。不支持问题不扫描竞争候选。配对E2E与模块时延共同解释，不能归因于单个微秒级解析。

15. 对象三问B→C=1.344391x，场景差CI=[-579.0333341865335, -268.7101450719638]ms。

16. 预设1/3结构化混合三问B→C=1.116378x，差CI=[-390.1508381241001, -126.90427587949671]ms；不代表自然分布。

17. 原始自然1Q=0.996820x，3Q=0.998203x；机会、跳过数和场景CI分别报告，不能从对象负载外推。

18. 1/2/3pp DEV最优候选=[{"budget": 0.01, "index": 0}, {"budget": 0.02, "index": 0}, {"budget": 0.03, "index": 0}]；本轮2pp选择target_dev_0，target enabled=False。研究严重风险边界预先冻结为已评分请求新严重率及场景CI上界≤2pp；部署安全始终不合格。 补充实测：DEV预算工作点：1%: target_dev_0, target=False, confidence=0.45, binding_risk=0.75, E2E=1517.542ms; 2%: target_dev_0, target=False, confidence=0.45, binding_risk=0.75, E2E=1517.542ms; 3%: target_dev_0, target=False, confidence=0.45, binding_risk=0.75, E2E=1517.542ms

19. 最终NO_GO：可信速度收益=True，质量预算=True，研究严重风险=False。UNKNOWN、32帧/4日志和基础模型训练分区暴露限制仍保留。

20. 新完整性模块独立质量/严重风险改善证据=False。仅在相同新数据/生成器下的核心消融能支持这一判断；最终可选更简单Current-Only，不能为了新模块名字牺牲实测最优。历史v3 NO_GO不变。
 补充实测：同数据消融E2E：addressable1Q basic→target=0.987105×，差CI=[-14.335397339891642, 35.938026441726834]ms; addressable3Q basic→target=1.030573×，差CI=[-108.58062694023829, 5.294540387694724]ms; mixed3Q basic→target=1.008495×，差CI=[-33.74117740895599, -5.5440384458051994]ms。质量/严重差的场景区间若接触零，不能宣称严格统计显著；仅记录有限样本中的经验变化。