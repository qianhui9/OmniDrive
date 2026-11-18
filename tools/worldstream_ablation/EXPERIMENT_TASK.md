# 实验任务包与固定协议

本目录独立实现推理期消融。原 `worldstream/`、`projects/`、正式 R0 JSON、模型权重及旧实验输出不修改。Full WorldStream 始终运行现有完整 R0，保持正式运行时校验。其他配置在严格权重加载之后挂接本目录的实验适配器，明确记录为消融研究配置。

## 执行矩阵

| ID | OGSM | DAVP | QCSE |
|---|---|---|---|
| native | 关闭 | 关闭，完整原生Backbone | 关闭，原生VLM生成 |
| worldstream | Full | Full | Full |
| without_ogsm | 关闭 | 相邻帧归一化图像差异，保留计算依赖闭包及全局残差执行器 | 禁用世界状态事实直答，保留精确历史候选和当前目标模型验证 |
| without_davp | Full | 关闭，完整原生Backbone | Full |
| without_qcse | Full | Full | 关闭，原生VLM生成 |
| ogsm_tracking | 检测框尺寸、类别、空间距离的一对一Hungarian关联；未匹配历史固定TTL | Full执行器，消费替代状态的区域/有效性依据 | 同一事实语法、置信/风险阈值及K8目标验证，信任Tracking的状态有效性输出 |
| ogsm_geometry | 历史世界框按当前真实位姿/标定投影，覆盖/深度/可见性检查，无光流或外观验证 | 同上 | 同上，信任Geometry的有效性 |
| ogsm_flow | 双向Farneback + 前后向误差≤1px + 5×5局部灰度残差≤0.08；区域支持比例≥0.8，无历史世界锚点 | 同上 | 同上，信任Flow的有效性 |
| davp_periodic | Full | 每摄像头固定相位轮转，每帧预算3窗口，最大年龄8；全局Suffix完整执行 | Full |
| davp_feature_drift | Full | 当前Patch窗口描述符与缓存描述符余弦距离Top-K，同预算/年龄；全局Suffix完整执行 | Full |
| davp_region_mapping | Full | 直接映射变化/不确定区域，各层独立映射，不传播逐层依赖闭包；全局Suffix完整执行 | Full |
| qcse_exact_cache | Full | Full | 相同完整Prompt及查询直接返回历史答案，不检查状态 |
| qcse_semantic_cache | Full | Full | 查询词嵌入均值归一化后余弦检索，阈值0.95，命中直接返回历史答案 |
| qcse_prompt_kv | Full | Full | 相同查询复用首个视觉Token之前的严格相同文本前缀KV；当前视觉后缀重新Prefill，仍执行全部Decode |

总共14个真实配置，四张表共17行；Full OGSM / Full DAVP / Full QCSE引用同一次`worldstream`结果。单组实验自动增加Native作为计算量分母。答案缓存最多256条，KV缓存512MiB，Scene边界全部清空。语义缓存采用冻结目标模型输入词嵌入，是不额外训练的轻量语义表示；它不等同于专门训练的句向量模型。阈值由参数记录，不根据结果自动搜索。

## 公平性和实现边界

所有配置使用同一冻结权重、六相机预处理、GT不可见的当前查询、逐请求相同随机种子、生成参数（sampling / temperature 0.1 / top_p 0.75 / max_new_tokens 320）。每个Scene从真实首帧开始连续处理，场景与配置边界重置模块及原生任务头。正式任务为每个配置启动全新Python进程。Smoke只检查可执行性，复用同一次严格加载的权重、独立重置每个配置，不用于论文时延比较。

Periodic与Feature-Drift不读取OGSM区域、图像有效性Lease或状态TTL来选择窗口，避免把简单刷新变成Full DAVP。两者每摄像头执行相同的**总窗口预算**，并设置相同的最大年龄；在临近期限时预留刷新槽位，剩余槽位再按固定相位/余弦Top-K选择。初始帧完整刷新。若预算×年龄不足以覆盖窗口数，直接报错，不能以无限复用换取收益。

本模型可缓存的局部前缀是两个相同窗口尺寸的独立窗口层。其局部闭包可能与直接映射等价；三个DAVP替代方案还取消全局锚点残差复用，因此这组表比较的是完整设计，**不能独立证明依赖闭包在当前固定拓扑上的收益**。若要单独证明闭包作用，需要另设具有跨窗口/不同尺寸依赖的骨干或独立机制实验。

OGSM替代状态保留真实历史来源帧、来源时间和验证方法；不会把历史验证伪装成当前检测。Geometry允许仅凭当前投影有效来继承历史语义；Flow只保留图像框、历史语义及历史ego事实读数，不保存历史世界坐标框；因此Flow的历史度量位置不会凭空随自车姿态更新。QCSE适配器沿用现有事实语法和阈值读取这些替代有效性判定，未准入时仍由当前目标模型验证历史候选。这是替换OGSM状态接口所必需的适配，收据明确区分真实当前检测和替代验证。

Full R0使用`current_only`事实直答，当前帧重新检测也会更新状态。不能仅凭本表的QA准确率宣称已完整验证跨帧状态有效性机制；应联合查看有效/失效历史状态、过期回答及下游视觉执行变化。所有替代方案是此任务包中的明确实现，未声称复现某个外部论文方法。

## 指标定义

- **Factual QA Accuracy (%)**：支持闭合事实语法且具有当前GT标签的问题中，回答完全正确的比例。UNKNOWN不计正确并报告数量/原因。原始开放式QA的有限事实召回可能为小数，另列辅助分数，不混入核心准确率。对官方GT中不存在当前可见目标且无在视野内遮挡歧义的情况，允许明确否定答案；不能把“无法判断”计为正确。
- **Stale Answer Rate (%)**：同一Scene、同一问题的相邻观测中，前后GT答案语义均明确且发生变化的机会数作分母；当前回答对新GT错误、对前次GT完全正确的次数作分子。该定义是“仍输出旧事实”的行为判断，不假定错误一定来自缓存。另列实际答案缓存来源与当前GT发生变化的过期回答率及其分母。没有变化机会时输出null/N/A，绝不填0%。
- **OGSM Latency (ms)**：Precheck+Commit主机实测平均耗时，另保留包含输入/输出适配传输的总耗时。不将检测/地图头时间加到此项。
- **Visual Backbone Latency (ms)**：实际视觉执行路径的CUDA Event跨度，DAVP包含调度、窗口缓存及全局Suffix，Native包含完整Backbone；另列P95。CUDA流跨度可能包含主机调度间隙，不能解释为纯kernel busy time。
- **Visual / LLM Compute Reduction (%)**：`100 × (1 − 本配置实际执行MACs总和 / Native实际执行MACs总和)`，以完全相同帧和问题配对。计数覆盖真实调用的Linear、Conv2d和Attention QK/AV乘积，包括函数式EVA QKV/Proj、FlashAttention及LoRA；保留Patch Embed、全局Suffix、候选验证的实际执行量。逐元素运算、访存、Sampling不属于此MAC口径。1 MAC=2算术FLOPs，但不是硬件计数器读数。不同生成长度的实际工作量差异不被抹平；负收益按实报告。
- **LLM Prefill / Decode Latency (ms)**：对应实际目标模型Forward的CUDA跨度总和，除以全部查询数。事实直答/答案缓存命中贡献零，KV缓存仍有当前后缀Prefill及Decode。Python选Token、历史检索与缓存管理开销仍保留在E2E内，不混入Forward时延。
- **E2E Latency (ms)**：每个视觉帧及其全部问题的同步主机时间；包含图像读取/预处理、问题编码、H2D、完整模型执行和答案导出，包含场景冷启动帧；排除权重加载、GPU排队、离线GT评分及收据序列化。
- **Direct Answer Rate**：状态事实直答数/全部查询数；答案缓存命中单独记录，不混作事实直答。
- **Candidate Token Acceptance Rate**：实际目标验证接受的历史候选Token数/实际提出验证的候选Token数；没有候选时为N/A。

## 场景分析与结果留存

默认固定既有18个主实验Scene，不根据消融结果筛样本。离线GT自动标记`vehicle_motion`（匹配实例的世界位置变化>0.5m）、`object_disappearance`（原可见对象标注消失或退出相机视野）、`local_occlusion`（仍在视野但当前可见性不足）、`ego_motion`（自车平移>0.5m或转角>2°）、`stable_other`及首帧。标签允许重叠，各组报告Accuracy、Stale及分母；没有覆盖的场景不能推断结论。

新实验从原始图像重新执行，不使用旧实验回答充当新输出。manifest绑定源文件、权重、info、官方标注和所有选中图像/问题文件的SHA256。缺帧、非连续时间流、推理失败或源码/数据变化均报错；只有全部配置真实完成且评分通过后才写`completion.json`。脚本不训练、不修改权重、不自动调参，也不把已暴露val数据称为独立TEST。
