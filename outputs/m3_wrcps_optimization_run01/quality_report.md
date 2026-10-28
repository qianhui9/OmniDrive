# 质量验收

状态：NOT_EVALUATED_STAGE_ONE_FAILED。没有训练合格的新视觉布局，因此没有把未经适配的重排/压缩输入送入真实任务并宣称质量合格。

|比较|Detection mAP|NDS|Lane DET_l|Token F1|关键语义错误|
|---|---|---|---|---|---|
|Native vs WRCPS-ZeroKV|未测|未测|未测|未测|未检查|
|WRCPS-ZeroKV vs ExactKV|未测|未测|未测|未测|未检查|
|Native vs WRCPS-ExactKV|未测|未测|未测|未测|未检查|

四项绝对下降预算仍为0.10，未放宽。交通灯/道路规则、行人等待/穿越、数量/空间、新增/消失/属性变化需逐项检查，不能用Token F1抵消关键错误。实际审计中的144份原生布局自然回答只证明执行发生，不是上述新布局的质量证据。

合成小模型合同测试验证输入长度、布局、真实Position/Attention/Embedding匹配、同Anchor缓存、不同动态suffix、KV跳过forward和答案等价。这些测试不能代替自然nuScenes质量评估。旧policy与旧近似语义错误证据全部保留；WRCPS不继承旧质量证书。
