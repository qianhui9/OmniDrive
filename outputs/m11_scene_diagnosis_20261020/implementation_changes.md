# M1.1 实施清单

基准 HEAD：`e8632ab7f3947c927a8bfcc7025eb796fa16dbe8`。全部修改均为 M1、实验/验收工具或必要的最小接线；M2/M3 源码、模型参数、LoRA、Planning 任务及原认证风险门限不变。

| 文件 | 修改及目的 |
|---|---|
| `tools/worldstream_m1/effective_acceptance.py` | 首先重建实验组织：每个场景/重复独立创建管理器，分目录运行；保存确定性选择、缺失检测、截断、未执行清单和状态账本；兼容分析旧混合日志，但明确不能当作独立执行证据。 |
| `tools/worldstream_m1/scene_protocol.py` | 按 scene_token 分组、时间排序、去重；提供全目标场景模式；分开计算 Micro 原始计数比率和 Macro 场景分布，不将 null 当零。 |
| `tools/worldstream_m1/scene_metrics.py` | 独立核验每场景的源帧、Epoch、执行签名、版本及提交；记录真实 keyframe 间隔、缺帧、长间隔、候选、提案/Unknown/认证、拒绝原因、计时原样本及 P50/P95。 |
| `tools/worldstream_m1/effective_gpu_validation.py` | 每个场景启动独立模型子进程，复用原 model_smoke 入口；配置全部场景和明确抽样均支持，记录未执行清单；分别验证输出、语言及源码一致性，逐场景聚合。没有新增推理执行器。 |
| `tools/worldstream_m1/model_smoke.py` | 单场景帧按真实时间排序；接入可信非标注来源上下文；记录逐帧完整模型 wall/process CPU 耗时；保留原始模型与生成设置。 |
| `tools/worldstream_m1/dataset.py` | 原图回放从真实 scene/sample 表获取日志来源；只抽取来源字段，不传递 GT 或实例标注。 |
| `worldstream/m1/source_context.py` | 不可变 scene→log、sample→scene 上下文及摘要；拒绝重复/缺失/冲突映射；同日志多场景合法，但不虚构独立日志。在线无上下文且无日志时明确 missing。 |
| `worldstream/m1/omnidrive_adapter.py` | frame_from_model 接收可信上下文/真实数据集日志字段；日志可附实验计时。 |
| `projects/mmdet3d_plugin/models/detectors/petr3d.py` | 仅向 M1 输入适配器传递运行所有者设置的 m1_source_context，不改变模型执行或生成路径。 |
| `worldstream/m1/state_manager.py` | 同场景已绑定日志消失/变化或绑定冲突时拒绝并重置；启用时输出逐 Anchor/相机只读漏斗；区分排除前和排除后的局部支持点计数。 |
| `worldstream/m1/physical_certifier.py` | 资格审查补充 log_source 场景/日志绑定检查；5% 风险与 95% 单侧界要求保留。 |
| `worldstream/m1/provenance.py` | 提交审计增加真实日志及预测类别/中心，供事后独立身份评分；在线不接收标注。 |
| `worldstream/m1/observation_diagnostics.py` | 记录来源、历史裁剪重建、投影、残差、实际轨迹、几何/光度支持、唯一性、空间及时间检查；只诊断，不修改原 valid 判定。前景、身份、静止、度量与神经等价声明继续分列。 |
| `worldstream/m1/config.py` | 新增默认关闭的 observation_diagnostics，避免旧运行默认增加诊断开销。 |
| `configs/worldstream_m11_diagnostics.yaml` | 继承本地有效化配置全部门限，仅开启 observation/identity 诊断。仍 observe_only、shadow、未捕获 Tensor、未提供资格文件。 |
| `projects/configs/OmniDrive/worldstream_m11_diagnostics.py` | 原始完整 Planning 继承链，显式 M2/M3 关闭。 |
| `tools/worldstream_m1/observation_diagnostics.py` | 生成确定性实际前后帧局部对应可视化，并保存对应 Anchor/相机原始诊断。 |
| `tools/worldstream_m1/scene_diagnosis.py` | 导出逐帧/Anchor/相机 CSV；分别分析全部历史视图与可投影候选视图，分开静态先验、动态未知和 map；保留实际可见性点数/比例/象限门及几何拒绝原因。 |
| `tools/worldstream_m1/offline_identity_scoring.py` | 在线日志提交完成后才读取 nuScenes 标注；冻结独立日志校准/验证划分；分别评分候选、Unknown 和正式认证，不可判定接受不会变成成功；始终不生成资格文件。 |
| `tools/worldstream_m1/identity_cases.py` | 绘制有实际测量轨迹的候选/Unknown 案例；明确不是正式身份或像素前景认证；没有相应案例则如实记 Not Available。 |
| `tools/worldstream_m1/scene_revalidate.py` | 独立账本要求历史状态版本严格等于先前实际提交的对应版本；验证相机证据版本、真实日志来源、保留版本与计算提交边界，拒绝未来/伪造来源。已在 12 条 CPU 重复轨迹及 2 条模型轨迹逐一复核。 |
| `tests/worldstream_m1/test_scene_diagnostics.py` | 31 项新测试覆盖来源合法/缺失/重复/冲突、同日志多场景、场景切换全历史清空、连续 keyframe 选择、缺帧、Micro/Macro/null、诊断不改变接受、离线 Unknown/风险与日志泄漏。 |
| `tests/worldstream_m1/test_identity_surface.py` | 更新既有测试夹具，使未来帧继续携带真实测试日志；日志消失现在是明确的来源契约失败。 |

# 生命周期与统计约定

每场景独立运行是主评估。单管理器跨场景 reset 仅作为额外回归测试，不参与主实验拼接。GPU 子进程同时隔离原生模型 Memory；CPU 每个场景及每次重复均使用新 Store、历史相机和身份历史。共享内容仅为只读权重/数据/来源表。

每个场景的第一个 precheck 不允许引用任何历史来源。核查器要求历史源版本来自该场景此前真实 commit；提交后的认证与提交前观测保持独立。新日志契约包含在原有 frame_digest 元数据绑定中。

历史投影候选是投影几何候选，不等于通过不确定性门；动态未知运动的投影可以是 candidate，但仍须拒绝静态观测支持。局部 LK/几何点、排除后的支持点、稳定像素、有限身份候选、正式资格和神经特征是不同人口。

Micro 从原始计数与分母重新计算；Macro 跳过无定义值并报告有效场景数、均值、中位数、P05/P95、范围及标准差。两种坐标系/数据集分开汇总。计时只使用第一重复，不把重复帧当作新场景；CPU 进程工作量与 wall latency 不相加。

所有神经特征仍未捕获，不能复用。真实来源补齐也不能代替独立资格。本轮没有修改任何认证风险门限或生成资格文件。
