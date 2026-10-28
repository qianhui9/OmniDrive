# EQ-SSC / WRCPS 实现设计

本轮从 `e9e232c501da481ad5ccc8b9d2c376a2b302484c` 创建独立分支。上一轮WRCPS、物理认证失败证据和正式配置全部保留。

## 资格范围与数据边界

`worldstream/m3/semantic_core.py` 定义 `SemanticContinuityEvidence` 与独立的 `SemanticQualifier`。它们只声明“具有实证支持的关联身份与当前观测类别连续性”，不声明物理身份真实性、坐标、速度、方向、交通灯、行人穿越、当前可见性、完整场景数量或缺失对象不存在。它不是 `PhysicalCertificationResult`，不修改 `identity_link_qualified` 或物理认证接口。

M1关联主算法、原有距离/类别/歧义门槛和匹配分配均不变。仅导出已经计算的选中边残差、歧义间隔、是否存在竞争边，并在现有 `association_history` 中记录来源帧、是否为紧邻帧和类别。没有新检测器、第二套Tracking或新的世界状态池。

在线候选严格绑定当前已完成的M1 Commit：scene/frame/time/epoch/state-version/execution。只提取实际当前检测、当前观察支持和同类紧邻关联。旧观测、失观测后的直接重现、缺少当前支持、竞争歧义和不获准语义类型均不能延续核心。遮挡宽限为0帧，因为本轮未校准任何失观测宽限。

在线代码不读取GT。离线 `semantic_labels_offline.jsonl` 使用项目已有唯一类别/尺寸/2米关联匹配器评价来源与目标实例，记录correct/wrong/unknown。Unknown明确保留，不能算正确。GT在模型执行完成后由benchmark observer写入独立文件，未进入M1、Selector、Composer或PrefixMatcher。

## 有限校准和独立验证

预先固定3组规则：score分位数 `.50/.75/.90`，残差分位数 `.75/.50/.25`，连续观测数 `2/3/3`。分位数只来自校准Log。每个语义类别只选择一个规则，选择目标先为校准Anchor连续命中，再为可用关联数量。校准准入要求已知错误0、unknown不超过5%、可评分边至少30、场景至少3。这些是本轮有限实证支持要求，不是经过物理认证的概率界，也没有套用原物理认证的59个以上独立场景零错误门槛。

固定规则存入 `calibration_frozen_rules.json` 后独立验证不再调参。初次验证不足时，只允许一次按元数据排序选择12个未使用Log的固定样本扩充；冻结文件SHA不变，初次失败结果存入 `initial_qualification/`。扩充是序贯数据收集，所有区间均作为描述性统计，不作总体物理风险保证。

校准和资格验证各使用不同的官方train场景/Log；另留官方val中的6个Log用于后续语言/系统验收。资格验证相对于本轮EQ-SSC拟合和未来Adapter数据独立；这些官方train场景可能在原始固定OmniDrive感知模型训练中出现过，不能声称是原始M1感知模型从未见过的数据。真实服务泛化仍须由另留的val场景质量和性能实验验证。旧M3工作已覆盖绝大多数val Log，因此不虚称“对所有历史研究完全未见”。

## 不可变稳定核心

`StableAnchorManager` 仍只有一个Anchor。初始选择至多一个已获准事实，按当前分数与观测历史确定优先级。该小核心规模事先固定，没有按验证结果扩大到全部对象。

只要原核心事实是当前合格事实集的子集，就保留同一个Anchor ID、核心Digest、来源帧、Stable Tensor与对应Cache作用域。新对象即使优先级更高，也不能替换仍然有效的旧核心；新信息通过全部当前视觉载体进入Dynamic Suffix。核心成员消失、类别/身份矛盾、不再满足资格、超龄、时间不单调、场景/epoch/执行版本或资格版本变化时，在同一路径生成新核心。

Digest只含承诺事实；Frame、时间、State Version和Ego Pose不进入稳定语义内容。来源、单调性和回滚仍独立校验。实体ID用于成员匹配和生命周期；不作为语言语义编码输入。

## Composer 与唯一M3执行路径

显式布局版本为 `wrcps_semantic_core_v2`，不能加载上一版Adapter。仍保留K=128和总预算513：128稳定+385动态。Stable Encoder只编码排序后的实体类型与类别，不编码任意Anchor ID。Dynamic Resampler读取全部当前Object/Map tokens，保留当前Ego。增加原始隐藏维度的注意力池化值通道，低秩模块仅学习残差，避免将全部当前证据先压缩进rank=128值空间。该改变仍必须经过实际训练和ZeroKV质量验证，不能以结构或梯度覆盖宣称语义无损。

正式实验WRCPS入口要求独立验证的语义资格文件；拒绝后不会切换到物理认证或其他资格算法。旧物理导出仅保留供历史审计。仍使用唯一 `M3Runtime → ReusePlanner → PrefixMatcher → PrefixKVCache → PrefillExecutor`。执行器本身未修改。

资格负责构造稳定内容；Exact执行仍逐token检查真实Embeddings、Position、Attention Mask、视觉Group Spans、布局、模型/Adapter/资格版本与Cache来源。不存在非连续跳跃KV、Approx改名Exact或忽略嵌入差异。失效/不同Anchor的跨帧复用上限止于第一个视觉token之前：固定文本仍可严格匹配，任何旧视觉KV都不得继续消费，即使两个空核心恰好产生完全相同的Tensor。同帧当前视觉的真实Exact复用仍合法。该边界由两个Matcher和Cache revision共同校验，仍由同一Planner/Executor处理，不新增回退执行路径。模型/Adapter SHA、Composer实现及精度、语义资格SHA参与绑定，旧WSDR policy不能直接复用。

## 条件性训练与后续门槛

只有独立资格与实际核心保持通过后，实验脚本才允许捕获Adapter训练数据。训练/Adapter验证/语义校准/资格验证/最终验收使用不同Log。真实冷启动空核心帧也应进入训练；它们从不被算作稳定命中。

冻结原始全部权重，只训练Composer，使用Answer KL + Task CE，关键问题CE加权。梯度需穿过冻结LLM；训练代码支持逐层梯度检查点，冻结LoRA/dropout子模块保持eval，未改动权重。离线载体训练时将无需执行的感知模块移至CPU；第一次真实反向传播记录峰值显存，OOM保存失败而非假定24GB足够。

先评估Native与WRCPS-ZeroKV四项系统质量及关键语义，再验证ExactKV及完整请求性能。没有通过前一阶段，就不启动后一阶段，不修改既有正式配置。最终结果及未执行阶段以本目录其他报告和 `deployment_decision.json` 为准。
