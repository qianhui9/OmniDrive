# 首轮真实跨帧近似复用样例

以下仅对比真实输出与数据参考文本，没有进行人工图像语义验收。原始依据为
`runs/r0_native_full/per_question_metrics.jsonl`、
`runs/r0_wsdr_kv/per_question_metrics.jsonl` 及 `paired_questions.jsonl`。

两例来自两个不同的独立场景；每例实际复用连续前缀 163 token，其中严格精确
匹配前缀为 35 token，额外复用视觉部分为 128 token。缓存年龄和近似深度均为 1。
总近似前缀为 326 token，超出严格精确前缀的部分为 256 token。它们是两例唯一
质量样本，重复运行不会增加这个数量。

| 场景 | 当前帧 | 任务 |
|---|---|---|
| bed67ef03c4a4066a74f6c0117d512ee | cd6b9c4674704ef6877790b0f0f95118 | map_relations |
| 2ed0fcbfc214478ca3b3ce013e7723ba | ec718fc75e8148c0afc3aabbcdac9ff5 | map_relations |

## 施工道路布局

问题：How does the construction affect the road layout where you are?

参考：The construction has led to lane closures and diversions, as indicated by
orange traffic cones directly in front of the vehicle. It requires careful
navigation and attentiveness to temporary road signs and markings.

原生和 WSDR-KV 的回答相同：The construction has altered the road layout, with
barriers and cones directing traffic into a narrower path.

回答提到了施工、障碍物、锥桶和道路变窄。参考文本中的 front 未在回答中出现，
这是原生回答已经存在的空间标签遗漏，不能算作 WSDR-KV 新增错误。

## 自行车道

问题：Is there a designated bicycle lane on the street?

参考：Yes, there's a marked bicycle lane on the right, with a bicycle symbol
painted on the road.

原生和 WSDR-KV 的回答相同：Yes, there is a designated bicycle lane on the
street, as indicated by the bicycle lane markings on the road.

回答保留了有自行车道及车道标识的判断，但省略了参考中的 right。这个空间标签
遗漏也已存在于原生回答中。本例没有观察到近似复用新增的文本差异；它不能证明
所有空间关系判断都正确。

## 解释边界

这两例确认了冻结策略在新场景中真实执行跨帧近似 KV 复用。两个回答与原生相同，
只能支持这两例未出现相对原生的新增输出错误。完整 GT 分数、原生既有错误、
所有重复轮次及质量样本量限制应结合总报告阅读。
