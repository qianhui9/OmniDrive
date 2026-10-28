# SF-WRCPS 实现与预先声明

基线 `0932a377b199e6249b87e15881c6e3c960f3cf3b`，独立分支 `m3-sf-wrcps-20261028-run01`。不修改 M1/M2 算法、基础权重或 PrefillExecutor，不覆盖旧实验。模型、资格、代码 SHA 与交付提交分别归档。

## 1. 自适应输入合同

同一 M3Runtime、ReusePlanner、PrefixMatcher、PrefixKVCache、PrefillExecutor 内以 `prefix_layout=sf_adaptive` 选择输入布局。无有效核心时直接保留同一个 native carrier tensor，Object256/Ego1/Map256，不运行学习层、转换 dtype、添加 null prefix 或重排。Runtime 不因无核心而创建新模型执行器。仅当核心有效时，必须加载版本匹配、真实训练且绑定基础权重与资格 SHA 的空间 Composer；缺少有效 Adapter 直接拒绝，不能执行随机权重。

两种模式分别为 native_full_visual / stable_core_visual。实际 layout groups、Adapter SHA、布局模式、资格版本、模型执行签名共同绑定缓存范围。布局切换清除旧缓存；同帧、同布局的严格 Exact 仍保留。跨帧失效核心即使产生相同数字也不能继承视觉 KV；普通文本匹配仍由原因果 Matcher 证明。Position/Mask/Embeddings/Cache 来源的原检查保留。

## 2. 原生槽位审计与有限合并

StreamPETRHead 的 VLM carrier 来自最后 decoder 的前 num_extra 查询，Object 输出另外附加 CAN bus/Ego。PETRHeadM 同样从额外学习型查询产生 Map carrier。它们不是实际检测 boxes 或固定空间网格，额外查询的位置会被清零，不能把相邻索引解释成相邻物理区域（见 streampetr_head.py:533–588 与 petr_head_map.py:404–475）。M1 adapter 发布的地图类别统一为 lane_control_points，未发布可供 EQ 直接认证的交通设施/道路拓扑类别。

主方案 K=64：Stable64 / Object224 / Ego1 / Map224，总长513。每个 Object/Map 分支只合并32个槽位。用当前原始特征余弦相似度做二部依赖匹配；奇偶索引仅是候选分区，不是物理邻接。挑选32个最相似源槽位，将其并入各自目标；保留其他原始槽位身份与顺序。只对合并目标应用可学习融合门及零初始化低秩残差，所有被移除输入真实参与目标，未合并槽位逐元素不变。Ego始终直接保留。

这是一种有约束的近似表示，尚不能从特征余弦推断驾驶空间真值。与旧全局自由 Attention Pooling 的差别是有限合并、明确槽位来源、大部分原生特征严格保留；是否修复有效核心下的回答仍须用真实 LLM 独立评估。

保留预先声明的 K=32 接口（Stable32/Object240/Ego1/Map240），只有 K64 在开发集的有效核心请求上出现经确认空间退化时才可进行一次开发比较，不按最终保留集选择 K。单元合同测试不是 K 最优性证据。

稳定编码沿用现有有限事实语义编码，不包含任意 Anchor ID、Ego Pose、位置、速度、运动或瞬时灯态。同一 Anchor 的稳定 embedding 直接复用一个已缓存 tensor；参数版本、内容与 tensor 修改保护保持原样。合并计划只属于当前 Dynamic，不能缓存成 Stable。

## 3. 新资格、数据和停止规则

所有旧 EQ 资格/训练/诊断 Log 排除。新的14个资格 Log：4 TRAIN+3 VAL校准，另4 TRAIN+3 VAL独立验证，各8连续帧。另预留10个新的 TRAIN Log（6训练、4验证）用于条件性适配、6个其他 VAL Log用于条件性最终验收。原拟选24个全新 TRAIN Log，元数据审计只有18个符合连续8帧、原始3问和历史排除条件，故在任何新模型输出之前确定以上18 TRAIN Log布局，没有基于结果修改场景。

官方VAL仅18个 Log，历史M3已暴露17个；保留的6个最终Log中5个存在更早M3研究暴露。因此只声称本轮方法选择保留及 Log 级分离，不能虚称对全部历史研究完全未见。名单、SHA和历史暴露记录在 experiment_plan.json 与 historical_data_audit.json。

旧 car score/residual/连续观察门槛保持数值不变；无观测宽限0，Unknown规则不变。在新数据中重新检查该固定规则，支持不足不得自动继承准入。其他现有对象类别仅可使用3组预先声明的校准分位数规则，选择后冻结，再在独立 Log验证；错误0、Unknown≤5%、可评分边≥30、场景≥3，均不放松。无额外样本扩充或事后降低阈值。

Map 当前只有无类别分数的 lane_control_points，Region没有独立可评分身份/语义生产者。M2刷新并非身份或地图语义证书。因此本轮审计这些来源但不无条件把其预测发布成稳定道路拓扑。在线 GT 始终禁止；离线 nuScenes 部分 Oracle 单独评分，未知不算正确，零错误不作总体零风险保证。

进入适配前必须看到独立验证中的正确持续核心，且独立 VAL域至少有一次真实核心延续。否则不启动训练或正式 ExactKV 性能验收，仍完成无核心合同、旧错误回归、已有真实载体的低成本结构诊断。

## 4. 条件性训练与质量

沿用 wrcps_train.py，冻结所有基础参数、真实 Answer KL+Task CE、现有显存检查点，只更新 Composer。方向增强采用固定0.5权重的答案位置 KL：仅在参考答案方向词处且 Teacher argmax 同意参考时对齐；单独记录 Teacher/参考冲突，不将 Teacher视为物理真值。未合并槽位的保真由结构保证，无额外大批辅助 Loss。

无核心时 Student 就是 Native，没有可训练路径，不执行虚假 optimizer update；仍记录这些真实帧。按有效核心的开发验证问题 Loss选 checkpoint，防止大量无核心恒等请求掩盖有核心压缩损失。数据需要真实冷启动/新建/延续/失效事件；未观察到的状态不能伪造为训练覆盖。若准入满足，优先 K64、rank128、3epoch，绝不按最终保留集选择 Epoch 或规则。

质量先分别评估无核心严格等价、有核心 ZeroKV、自然整体四指标与关键语义。只有全部质量和目标域核心适用性通过，才校准绑定新布局/Adapter/资格的 Exact 策略并做 B/C 同输入、固定 Tokens/Decode步数、三轮交替和场景配对95%CI。任何条件不满足，保持正式配置，不声称加速。
