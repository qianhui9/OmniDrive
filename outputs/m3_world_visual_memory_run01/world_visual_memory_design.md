# 单个历史区域视觉记忆

实现：`worldstream/m3/visual_memory.py`，一份VisualMemory与一个VisualMemoryManager。记录身份、类别、相机、读取帧、源几何状态版本、最新有效观测、age、revision、scope、真实特征及内容hash。身份仅作生命周期key，投影层只接收实际1024维特征；同类不同对象可以不同。

本轮选择每个memory token对应一个不同原生网格观测，K64/128/256只作有限预算诊断；少于K个位置拒绝，不做补零、复制或把类别复制成K个token。多于K时按原始row-major位置等距下采样成K个。该选择是本轮最小路线的约束，不是所有可学习resampler的信息论下界。

当前帧首次满足身份与区域条件才创建，后续每帧同时重验当前身份资格和区域支持。保持直接返回相同feature tensor与相同projected embedding tensor，未重新生成近似前缀。新对象、类变化、缺失支持、超过8帧、时间断裂、state回滚及scene/epoch/model/adapter/memory/qualification版本变化撤销或重建；revision防止偶然相同内容继承已撤销的视觉KV。修改保存tensor会fail closed。

历史EVA区域仍含旧的外观、位置和背景。本实现承诺其为固定历史上下文，不承诺当前外观不变。位置/速度/朝向/灯态/新对象必须由完整当前513个Native tokens提供，不从旧框或velocity生成稳定语义。语义/运动质量尚未通过训练后LLM测试，不能据此部署。

唯一具体候选布局为 `[P, historical K, current native513, Q]`，无memory直接返回原Native tensor，指针/dtype/顺序不变。投影为单个1024→4096 Linear，测试参数未训练，仅用于实际张量保持和计算成本；不得加载为正式adapter。成本门NO_GO后未选主训练K、未压缩当前视觉、未扩展M3Runtime正式布局。

成本控制直接调用既有PrefixKVCache/PrefixMatcher/ReusePlanner/PrefillExecutor；没有新执行器。源KV由真实LLM forward产生，严格比较实际embedding/position/mask与live执行签名，stable_anchor元数据绑定合法memory scope与revision，动态suffix仍实际prefill。控制实验的显式reuse_length独立于scheduler准入；正式链路集成与scheduler成本表均未获准发布。撤销边界沿用既有matcher裁掉视觉前缀，单测覆盖偶然同值重建。
