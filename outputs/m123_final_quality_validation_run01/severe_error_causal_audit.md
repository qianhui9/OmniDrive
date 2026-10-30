# 严重错误因果审计

只用于 Post-hoc Causal Diagnosis，不参与最终参数选择。四组从历史收据使用的同一合法连续流起点重放全部8帧，至错误帧结束；不是只在错误帧改参数。另有同执行器 Full 控制。完整原始检测、M1提交、M2缓存/刷新、M3候选和请求Forward保存在压缩收据及 severe_error_counterfactual.json。

GT最近可见Pedestrian是前左方，中心约(10.0937,8.1488,1.3329)m；其唯一GT身份通过原有可见性/最近目标协议，未改变GT定义。但多重检测框的独立匹配可能UNKNOWN，不能把附近框都认证为同一GT目标。

|组|参数|实际回答路径及答案|
|---|---|---|
|D00/R0|{} / {}|FAST: The nearest pedestrian is to the front-left.（严重=False）|
|D10/R3|{'min_detection_score': 0.4} / {}|FAST: The nearest pedestrian is to the front-left.（严重=False）|
|D01/R2|{} / {'ordinary_budget': 1}|LLM: The nearest pedestrian is on the left side, walking on the sidewalk parallel to the road.（严重=False）|
|D11/R1|{'min_detection_score': 0.4} / {'ordinary_budget': 1}|FAST: The nearest pedestrian is to the rear-left.（严重=True）|
|Full/Full|{} / {}|FAST: The nearest pedestrian is to the front-left.（严重=False）|

以下针对**历史8帧重置起点**：连续状态中存在M1×M2交互，以及C-basic对可见最近目标覆盖的原有限制。官方Scene起点结论见下节，不能将该8帧条件的预算归因外推到所有起点。

1. **当前帧直接M1过滤**：D11的Front-Left附近解码候选存在，但最高约0.1984，已经低于原始0.3。不是0.4在当前帧直接过滤一个0.3原本会接受的正确目标。
2. **M2 Approximation Effect**：固定M1=.4，只恢复M2预算3，Front-Left附近高分框约0.7638被保留，FAST回答正确方位；预算1对应约0.1984且目标不进入世界状态。该对照是整个合法闭环序列的因果效应，不证明仅错误帧某个特征区域是唯一原因。
3. **M1历史介导效应**：固定M2预算1，只恢复M1=.3，当前检测又有约0.7942的Front-Left框及更近约0.3949框；M3选择更近低分框，未通过.45准入，转LLM。M1阈值会改变前序世界状态及后续M2刷新，不能把M1视为只影响当前帧后处理的独立开关。
4. **M3 Target Selection Effect**：D11正确目标没有进入合法当前候选集合；M3按现有集合距离排序选择后左方真实Pedestrian，其3D匹配约0.386m。没有证据显示距离排序实现把一个已存在、更近、合格的Front-Left目标排在后面。真正局限是把不完整检测集合的最近对象当成问题所要求的最近可见对象。检测置信度/绑定风险不等于该语义正确概率。D01的低分最近框会使高分次近框不能直接回答，这是原选择/准入逻辑，不在本轮修改。
5. **Base Perception Limitation**：不能把本例完全归为Full基础视觉漏检；Full和D00均有高分前左候选并回答Front-Left。模型的一般检测与重复框问题仍存在。
6. **Evaluation Ambiguity**：GT目标唯一、方位远离边界，本例无GT方位歧义；预测框身份匹配的UNKNOWN另行保留。D00/Full答对方位也不等于每个重复框都有已认证身份。

历史R1错误仍是新增严重前后反转。恢复单一参数后本例不再出现反转，只支持事后机制诊断，不能用本例给新配置授予独立TEST通过。历史v3/v4 NO_GO和上一轮R1风险失败均不改写。

## 官方Scene首帧的完整历史补充

历史起点位于实际Scene第8个sample之后。额外从官方first_sample_token重放16帧，五个控制共80次模型调用；配置冻结不变。以下结果与上面的历史8帧起点不合并。

- R0: FAST，The nearest pedestrian is to the front-left.；severe=False。
- R3: FAST，The nearest pedestrian is to the rear-left.；severe=True。
- R2: FAST，The nearest pedestrian is to the front-left.；severe=False。
- R1: FAST，The nearest pedestrian is to the rear-left.；severe=True。
- Full: LLM，The nearest pedestrian is on the left side, walking on the sidewalk parallel to the road.；severe=False。

完整结果及所有前序目标分数、M1状态数、M2刷新/复用见scene_start_severe_error_counterfactual.json；原始GT缓存保留不动，额外历史GT在独立文件。起点敏感性限制因果结论的外推，不能拿其中一个有利起点覆盖另一个实际错误。

真实Scene首帧结果表明：.4阈值在预算3和预算1下均出现Rear-Left；.3在两种预算下均回答Front-Left。失败臂当前GT附近最高分仍仅约.1903/.1980，已低于.3；因此仍是前序M1状态改变经M2及检测头连续记忆传播的效应，不是最终帧0.4直接剔除一个0.3本会保留的候选。预算1不是跨起点的必要原因；M1历史参数影响更稳健，但也不能定位单个先前窗口为唯一原因。Full有高分前左候选，同时较近.3901候选使M3转LLM，说明全量视觉并不自动保证完整QA正确。

## 已定位的M1→M2窗口链路

在真实Scene起点固定预算3的对照中，前15帧的原始检测和M2前缀执行窗口ID均逐帧完全一致；错误帧前一帧，目标附近Pedestrian分数0.3543037474被M1=.3提交、被.4排除。下一帧，该保留锚点在CAM_FRONT_LEFT的观测不确定（uncertainty_exceeds_search），因此合法地要求强制刷新窗口19、22；.4缺少此锚点，两个窗口在M2块0、1均被复用，共少4个实际窗口计算。预算1的固定对照同样缺少这一窗口对。两臂的普通窗口选择相同，全局后缀都完整执行22块，**本帧不是全局后缀复用导致的错误**。真实返回特征及检测仅从本帧开始分岔，高分前左目标下降至约.19，随后M3在残缺候选中选择后左对象。causal_window_chain.json保存前帧独立GT匹配、M1预检查、强制刷新区域和实际执行ID。该证据定位到改变的窗口对，没有声称分别干预单个窗口。

## 冻结候选残余严重错误

R0在真实Scene前序仍有1/39已评分绝对严重错误（48总请求、9 UNKNOWN），为FAST左右方位错误；同一错误在Full中也存在。它不是相对正常LLM B的“新增”计数，因为该补充没有B臂。完整帧、答案、GT、候选及Full对应事件见residual_severe_cases.json。不能因为重点前后错误修复就宣称R0通过严格风险验收。风险元数据在最终策略文件中补充，原始参数冻结文件逐字节归档，所有参数及选择完全不改。
