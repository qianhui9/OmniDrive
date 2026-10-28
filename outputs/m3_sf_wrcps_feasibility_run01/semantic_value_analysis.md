# Stable Prefix的信息与语义价值

代码审计：WorldPrefixComposer.encode_stable只编码排序后的(kind,label)，opaque Anchor ID只控制缓存寿命。所有有效单事实car状态均对应相同字节 `[["object","car"]]`；无论训练后参数是什么，不同车辆ID和场景的内容编码仍相同。实测两个不同真实Anchor的当前编码torch.equal=True。条件在“有效单car核心”上，语义内容变化熵为0；这不等于声称所有64个query的embedding彼此相同，也不等于有效性/连续观察这一条件本身没有价值。

K64可以表现为一个由可靠car存在条件触发的学习soft prompt，但没有对象位置、速度、具体身份、车道或道路关系。原生视觉已经提供车辆视觉证据，尚未证明这份类别/稳定性条件对驾驶问答有增量价值。64个learned queries并不会凭空增加世界事实种类。对信息容量的结构性审计不依赖随机权重优劣。

有限诊断使用旧真实开发核心连续帧的3个原始自然问题，无历史KV。固定Dynamic449、Stable64布局、LLM和RNG，仅比较当前未训练car编码、Stable64全零反事实、bus类别反事实；Native完整视觉另列。最大prefill最后位置logit变化11.337891，说明改变Stable输入能影响模型。原始全部答案/参考、tokens、KL和关键标签辅助量见 controlled_micro/semantic_diagnostics.json。

这些是明确标记的反事实与未训练结构探针，不是合法bus在线事实、训练后的SF精度、自然场景改进或物理真值证据。若答案/F1发生变化，也不能由随机权重推断训练后有用或无用。Teacher/参考可能有偏差；不将Teacher单独当作物理真值。本轮没有已训练新Adapter，因此“已证明实用语言价值”答案仍为否。

实际案例：第一问灯色参考为green，Native为red，二者本身冲突；未训练car前缀产生重复数字。第三问oncoming traffic，参考与Native均为No，未训练car前缀为“There are several”，bus反事实为Yes并重复文本。这说明当前未经适配的Stable输入影响很大且没有可解释的实用语义证据，不能将变化当作正确car事实帮助。派生数字溢出记录Unknown，原始自然文本和tokens完整保存。收益No-Go独立来自硬件/覆盖估计，未用这些随机输出作为正式质量否决或已训练模型结论。

Map/Region维持前轮审计结论：lane_control_points潜变量缺少低成本独立道路类别/拓扑身份生产者，不新增地图认证或Tracking系统，不写入Stable Core。
