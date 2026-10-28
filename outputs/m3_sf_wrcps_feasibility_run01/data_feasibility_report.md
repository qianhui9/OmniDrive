# 真实训练数据可行性

结论：数据门 **GO**。全官方TRAIN元数据共28130帧、700场景、50Log；有575个场景满足当前保护Log排除、真实连续32帧、六相机存在及原始前三问要求。没有机械排除全部历史研究TRAIN。历史源索引与Log身份见 training_historical_exposure.json；不称这些数据全历史未见。

真实审计576帧，原6训练Log加8个合法历史正样本Log、原4开发验证Log，各32帧；在线M1/M2按时间因果执行，零LLM Decode。当前资格/开发验证/最终保留Log禁止进入训练；最终6个验收Log未打开，资格SHA和门槛原样保持。audit_sequence.json在输出前冻结，GT在commit之后评分，不用于当前状态。

|窗口|原6训练Log有效帧/延续|全部14训练Log有效帧/延续|开发验证有效帧/延续|
|---|---:|---:|---:|
|8|1/0|45/28|6/5|
|16|2/0|93/53|13/9|
|32|3/0|195/108|22/11|

原48训练帧中47帧没有合格事实。原始快照car首个拒绝原因：{"no_current_consecutive_observation": 181, "insufficient_consecutive_observations": 52, "class_score_below_calibrated_range": 86, "association_residual_above_calibrated_range": 1}。这些是候选首失败计数，不能相加解释为独立帧因果比例。原6Log延长至32帧仍0次延续，说明窗口短不是这些Log的主要可恢复瓶颈；增加有真实正事件的合法Log才获得108次延续。

原始14训练候选Log：448帧、195有效帧、108次延续、87个有效Anchor（其中39个真正持续）、8个正样本Log，有效Anchor平均2.241帧。训练事件富集后296帧，保留全部195有效帧和108次延续。开发验证维持完整128帧，22有效帧、11次延续，来自2个正Log，平均有效Anchor2.000帧。原Anchor寿命上限8帧保持不变，horizon到期也计入刷新，不为提高命中延长上限。

训练富集只选真实有效事件前后各2帧、真实无核心边界和稀疏常态帧；不重复帧、不改时间戳/Track ID、不把未来信息写入快照。稀疏样本之间的缺帧不算连续转换。独立部分Oracle对训练/开发验证延续分别评分{'correct': 108} / {'correct': 11}，未知独立计数，零观测错误不代表总体零风险。原始候选池本身按历史正证据选过，也不是自然部署覆盖率。

逐过滤器计数：{"frames_no_current_car_candidates": 46, "car_observations": 5612, "score_below_threshold": 4207, "residual_above_threshold": 1836, "no_previous_frame_association": 1710, "no_current_observation_support": 615, "previous_core_id_not_currently_observed": 0}。涉及检测分数、残差、上一帧关联缺失、当前支持和ID撤销。Core事件：{"no_core_frames": 359, "core_created_or_replaced": 98, "core_lost": 24, "core_replaced": 67, "core_continued": 119, "new_object_but_core_kept": 119}。物理遮挡、漏检与关联ID变化在缺少可独立验证可见性标注时不能强行区分；报告保留这一不可识别性。每帧候选、合格事实、撤销信息及原始M1诊断在服务器raw审计，精简事件见 core_event_timeline.jsonl。

训练准入定义冻结在experiment_plan.json/profit_gate_definition.json：至少3正Log、6序列、24有效帧、12次已评分正确延续，以及独立开发验证正事件；错误0、未知≤5%不放松。现有wrcps_train.py --preflight-only已实际通过，产物在training_preflight/data_validation.json。数据不是本轮最终停止原因。

## 候选与单核心分母补充

|范围|帧|候选|连续证据eligible|合格事实观察|选入核心事实|延续|
|---|---:|---:|---:|---:|---:|---:|
|all|576|22087|14333|539|217|119|
|train|448|18296|12106|503|195|108|
|validation|128|3791|2227|36|22|11|

此处可评分观察指在线证据eligible，不代表离线GT已评分；候选与合格事实观察数可超过每帧一个核心。逐帧数字见core_candidate_timeline.jsonl，逐场景见candidate_fact_distribution.json。
