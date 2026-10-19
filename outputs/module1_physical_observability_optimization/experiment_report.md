# M1 基于可辨识物理证据的覆盖率优化

本轮完成 **nuScenes + OmniDrive 原模型 + M1** 的实际编码、20 个已曝光场景的 Development、消融、身份人口分析、全量 nuScenes 可用性审计、517 项 CUDA 回归及 8 帧原模型感知一致性验证。正式结论为 **No-Go**。新路径减少错误接受并恢复关键 ROI 刷新，但付出的正确覆盖损失很大，没有达到原有风险置信度和接受人口要求，也没有超过 Conservative。

## 实验边界和冻结

实际基线 HEAD：`7aeedc95d87c63c69a2e49259bb15af1370b1e21`。旧报告和历史结果未覆盖，9 项历史产物 SHA256 校验一致。仅修改 M1、M1 配置、M1 工具/测试及本轮输出；M2/M3、原模型、Checkpoint、训练与输入预处理未修改。环境为 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，所有执行使用 `taskset -c 0-3,6-31`。GPU 空闲检查后运行，没有停止其他任务。

本轮所有主要结果来自同一组 20 Scene / 20 Log / 240 Frame / 1,440 Camera，含启动和周期强制刷新；区域参考分母 130,627 个可评分 Annotation-owned LiDAR 点、236,540 个关键点、3,250 个事件。身份原始几何分母固定 5,798，扩展合法边 13,314。全部标记 `exploratory_reused_population`。旧工具内部历史名称 calibration 仅指原开发样本加载入口，**本轮不是正式 Calibration**。

`observability_diagnostic_freeze.json` 固定方法修改前的 690 个 Consensus 和194 个 Local Plane 接受点，合并为780个实际参考位置；`development_source_freeze.json` 在新 Development 评分前固定算法、配置和消融。没有阈值网格搜索或事后按正确性挑选在线人口。完整命令在 [commands.sh](commands.sh)，源码/数据/模型/原始轨迹哈希分别在 source_manifest.json、data_manifest.json。正式方法选择始终 null。

## 可辨识性：同源一致为何不足

旧 Consensus 正确494/错误196中，同双帧 RGB 上重新运行 LK 且与 DIS 相差≤0.5px 的分别为 385/150；真实额外时间观测支持 62/23；实际测得互不相容双帧替代解释 118/66。额外时间支持本身仍包含23个错误，不能因多帧就称为独立物理认证。该冻结人口没有跨相机正支持，后续全候选实验确实测得少量跨相机像素支持，但没有可评分跨相机接受点。

| 测量分布 [Q25, 中位数, Q75] | 正确494 | 错误196 |
|---|---|---|
| fb | [0.143619, 0.206613, 0.275191] | [0.158007, 0.213516, 0.268742] |
| photo | [0.002328, 0.004529, 0.008701] | [0.002578, 0.004994, 0.008521] |
| gradient | [0.002993, 0.005022, 0.007247] | [0.002884, 0.004514, 0.006918] |
| texture | [2.2e-05, 4.6e-05, 0.000177] | [2.1e-05, 4.7e-05, 0.000255] |
| jacobian | [1.01261, 1.028343, 1.050143] | [1.008893, 1.021334, 1.044719] |
| parallax_deg | [1.588503, 1.896772, 2.724245] | [0.867922, 0.978424, 2.072734] |


每个点保存实际 RGB 源像素、独立离线参考源像素、姿态/内参、FB/光度/纹理/Jacobian、实际视差、Consensus 单元/多平面、重新跟踪、有限替代源像素和不同采集观测。**196 个错误点的具体物理根因均保留 Unknown Root Cause**：独立参考确认对应错误，不据光流自洽或参考运动自动声称遮挡/前景混淆。Patch 不匹配是未获支持，不能当作已证明遮挡。示例在 visualizations/correspondence_cases、true_false_controls；所有原始测量在 surface_ambiguity_analysis.json。

## 新机制与接口

显式启用 `configs/worldstream_m1_physical_observability.yaml`，`dense_region_policy: surface_constraints`。原有 FB、纹理、光度、射线、完整邻域和动态/遮挡排除保留，半径2沿用旧 Consensus 的实验设置；另以 observation_base 消融披露其覆盖和风险。新机制不拟合更多平面，不作残差加权或最佳假设择优：

1. 对每个实际候选射线三角化，并直接在源 RGB 中验证局部深度的投影 Patch。
2. 要求至少一个**真实不同采集**支持：已 Commit 的更早图像，或同步误差≤10ms且真实投影共同可见的其他相机。时间顺序、标定、坐标、正深度与视差不足时 Unknown；不使用未来帧。
3. 在已有±4px搜索孔径内以0.5px采样极线对应；原解释与实际支持的>2px不相容源解释共存时拒绝为 Ambiguous。有限搜索没有发现替代解释不等于全局唯一。
4. 对具备附加支持的实测假设执行源像素深度排序冲突检查；这是拒绝冲突假设，不是确定真实遮挡者或物理表面所有权。

OnlineVisibilityEvidence 提供 observation_supported_mask、visibility_consistent_mask、ambiguity_rejected_mask、stable_candidate_mask、refresh_required_mask；最后一项包含周期/重置强制刷新。稳定候选不扩张原测量基底。不同采集仍有相关性；所有输出为条件性表面观测支持，**无 Transformer Feature Reuse / 物理前景所有权认证**。正常在线函数不接收 LiDAR/Annotation。局部前平行 Patch 是条件性探针，也可能拒绝真实斜面/非刚性表面；不能把该模型不适配的失败都归为“Camera-only 不可辨识”。

## 主要区域方法对照

| 方法 | Effective Stable 面积 | 正确/错误参考点 | Sparse FAR | Event Recall | Point Recall | 最差 ROI Refresh | 接受 Scene/Log |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_refresh | 0.000000% | 0/0 | N/A | 100.000000% | 100.000000% | 100.000000% | 0/0 |
| conservative | 0.331562% | 297/9 | 2.941176% | 99.969231% | 99.980976% | 87.782805% | 2/2 |
| dense_original | 4.406813% | 2458/481 | 16.366111% | 99.046154% | 99.228460% | 23.529412% | 6/6 |
| local_r12 | 1.248860% | 1167/169 | 12.649701% | 99.938462% | 99.802993% | 23.529412% | 3/3 |
| surface_visibility | 0.001666% | 0/0 | N/A | 100.000000% | 100.000000% | 100.000000% | 0/0 |
| surface_local_plane | 0.017963% | 155/39 | 20.103093% | 100.000000% | 99.996618% | 99.833224% | 1/1 |
| surface_consensus | 0.325636% | 494/196 | 28.405797% | 99.876923% | 99.873172% | 61.111111% | 2/2 |
| surface_constraints | 0.003654% | 5/0 | 0.000000% | 100.000000% | 100.000000% | 99.970238% | 1/1 |


历史基线是相同输入及未变评分协议的 SHA256 固定结果，明确为 archived same-input baseline，不冒充新独立实验。新方法实际接受5个正确、0个已知错误，只来自 **1 Scene / 1 Log**；Sparse IID 描述性上界为 45.071973%，Log 事件上界 95.000000%。Log 聚类 bootstrap 的零支持重采样353/1000，条件 FAR 区间 [0,0] 没有覆盖最坏未观测风险，不能据此认证。相对 Conservative 面积减少 98.897952%，面积差 -0.327908 个百分点。

同一参考点配对：旧494个正确仅保留 1，损失 493；旧196个错误全部阻止；另新增4个正确。旧 Local Plane 的39个错误全部阻止，155个正确只保留2个，另有3个新正确。**不能将这一巨大正确损失隐藏在“错误清零”中**。

## 消融与证据实际贡献

| 消融 | 面积 | 正确/错误 | FAR | 最差 ROI Refresh |
|---|---:|---:|---:|---:|
| constraints_observation_base | 1.372150% | 1314/230 | 14.896373% | 23.529412% |
| constraints_two_view_only | 0.016668% | 84/35 | 29.411765% | 99.583333% |
| constraints_temporal_only | 0.003598% | 5/0 | 0.000000% | 99.970238% |
| constraints_camera_only | 0.000077% | 0/0 | N/A | 100.000000% |
| constraints_no_ambiguity | 0.010237% | 51/14 | 21.538462% | 99.861111% |
| constraints_no_depth_order | 0.003661% | 5/0 | 0.000000% | 99.970238% |
| surface_constraints | 0.003654% | 5/0 | 0.000000% | 99.970238% |


相同真实采集和 Commit 状态上使用预声明逻辑消融。独立的 two_view_only manager 在支持场景完整重放，连同 ROI 主路径重放核对共288个相机记录，面积/Stable 无差异，详见 independent_ablation_replay_check.json。主路径的区域掩码不会改变原模型完整输出或 Shadow 身份关联，所以共享观测消融没有跨方法状态反馈。

可评分漏斗：1544点（1314正确/230错误）→源投影支持257（177/80）→真实附加采集65（51/14）→歧义拒绝后5（5/0）。歧义检查单独排除14个错误，同时损失46个正确。源探针本身损失1137个正确，附加采集约束再损失126个；这些含未观测和模型假设不适配，不能全称为物理矛盾。深度排序在本人口仅减少少量像素，没有新增可评分错误剔除收益，不能包装成已证实的主要质量改进。

全候选、强制刷新前：时间支持35,001个像素；跨相机支持570个像素。实际跨相机几何可用探针19,794次；不是默认六摄像头共视。跨相机独立消融有效像素255，**可评分接受0，FAR=null**。探针计数可能在多个视角重复；独立风险单位始终是Log。帧/像素支持去重详见 surface_evidence_provenance.json。

离线 Oracle Upper Bound 在 observation_base 的1544个候选参考中知道1314个正确，主路径仅恢复5个，剩余1309个反映条件探针覆盖缺口。此为**稀疏候选上界**，没有稠密物理表面真值，dense area、完整 ROI Oracle 指标保持 null；不据 sparse 点外推全图真值。Oracle 从未进入在线决策或资格签发。

## Critical ROI 与精确掩码

| 帧/相机 | 事件 | 旧精确刷新 | 新精确刷新 | 旧 Tile 包络刷新 | 活跃预测动态扫掠覆盖 |
|---|---|---:|---:|---:|---:|
| ef7ef6f1/CAM_BACK_RIGHT | pedestrian: moving_object | 88.382353% | 100.000000% | 100.000000% | 0.000000% |
| 8efb06a0/CAM_FRONT | traffic_cone: newly_visible | 61.111111% | 100.000000% | 100.000000% | 0.000000% |
| 112816ea/CAM_FRONT | pedestrian: moving_object | 89.583333% | 100.000000% | 100.000000% | 0.000000% |
| ab711125/CAM_FRONT | car: moving_object | 89.073227% | 100.000000% | 100.000000% | 1.315789% |


四个历史失败事件均恢复到100%精确刷新，全体3,250事件的最差刷新 99.970238%。原最差交通锥 ROI 只有108个整数像素，其中42个被旧 Stable 占用，导致61.11%刷新。预测扫掠基于活跃私有 Precheck 状态重新核对，不能用已过期公开旧状态虚构保护范围。动态预测保护未覆盖一些真实事件；这并不足以证明具体物理漏检原因，仍保留 Unknown。

精确掩码与保守32px Tile包络是不同输出，旧四事件的 Tile 包络均为100%刷新；没有发现精确掩码到包络丢失覆盖的映射错误。所有主要统计坚持评分 exact mask，未将保守包络替换为更好看的面积/ROI数值。GT ROI只在在线 Precheck后评分与画图；没有 GT触发刷新。见 roi_exact_mask_audit.json 和 visualizations/roi_before_after。

## 身份：保留现有生产器，分析在线人口

| 身份方法 | Proposal | 正确 | 错误 | 不可评分 | 唯一正确实例 | 完整可评分接受 Log | Log 风险上界 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hungarian_geometry | 5798 | 2655 | 20 | 3123 | 458 | 0 | N/A |
| selective_identity | 175 | 144 | 0 | 31 | 33 | 0 | N/A |
| temporal_identity | 284 | 213 | 0 | 71 | 43 | 1 | 95.000000% |
| temporal_shadow | 284 | 213 | 0 | 71 | 43 | 1 | 95.000000% |
| temporal_object | 288 | 217 | 0 | 71 | 45 | 1 | 95.000000% |
| object_surface_only | 4 | 4 | 0 | 0 | 4 | 3 | 63.159685% |


| 在线定义人口（非互斥） | 原始几何人口成员/5798 | 评价边 | Proposal | 正确/错误/不可评分 | 唯一正确实例 | 完整可评分接受 Log |
|---|---:|---:|---:|---:|---:|---:|
| all_measured_legal_edges | 5798 | 13314 | 288 | 217/0/71 | 45 | 1 |
| has_supported_crop | 583 | 840 | 288 | 217/0/71 | 45 | 1 |
| multiple_supported_temporal_camera_crops | 29 | 29 | 28 | 20/0/8 | 3 | 0 |
| measured_object_surface_and_background | 7 | 8 | 4 | 4/0/0 | 4 | 3 |
| background_depth_separable | 13 | 14 | 4 | 4/0/0 | 4 | 3 |
| unresolved_identity_competition | 292 | 549 | 0 | 0/0/0 | 0 | 0 |
| resolved_identity_competition | 8 | 8 | 8 | 5/0/3 | 1 | 0 |
| short_gap_source | 134 | 1318 | 1 | 1/0/0 | 1 | 1 |
| predicted_displacement_over_05m | 645 | 6214 | 1 | 0/0/1 | 0 | 0 |


所有人口按实际在线输入定义，无 GT正确性筛选；非互斥人口不能相加。展开边的比例与原5,798人口成员比例分开报告，避免用13,314边代替原分母。完整预测类别、距离、位移和源年龄分层在 identity_population_analysis.json；短间隔人口只说明来源恢复，**不声称已确认真实遮挡**。多相机 crop 支持不等于同步可三角化的前景共视。

本轮逐边核对13,314边与现有 Temporal Object 的 candidate/correct/path/short-gap，差异0。现有217正确、71不可评分、45唯一实例保持；没有新生产器收益，保留 Temporal Object 默认 Shadow。Object Surface 人口仅4个正确候选、3个完整接受Log，Log上界约63.16%；完整Hybrid只有1个完整可评分接受Log，上界95%。保留全部71个不可评分接受，不用GT选出可评分人口发证。唯一 Claim 仍是 object_identity_continuity，不扩展为 stationarity/metric velocity/foreground ownership/box surface/feature或LLM复用。

## nuScenes 独立评价条件

逐帧核对Train 28,130帧/700 Scene/50 Log，Val 6,019帧/150 Scene/18 Log；实际六路 RGB 文件204,894个全部存在，LiDAR 34,149帧全部存在。非空Annotation对应Train 28,124帧、Val 6,016帧；空标注帧不是额外独立Log。实际内参/传感器与Ego Pose数值核对，使用帧的1,680个图像/LiDAR文件、元数据、预测缓存及Checkpoint均保存SHA256。

68个Trainval Log均有历史真实状态/评分轨迹的具体路径及SHA256见 historical_exposure_audit.json；没有把仅出现在可用性清单中的token算作曝光。两份真实原模型输出缓存共600个不同样本，缺少缓存不等于缺少原始数据或标注。Test有15个未曝光Log，但本地 sample_annotation.json 为0条，不能评价身份或物理对应；Mini的8个Log也已曝光。本轮未引入其他数据集、输入结构或跨数据集 Adapter。保留的文件名 independent_data_adapter_audit.json 内容仅为nuScenes同数据集可用性与曝光审计。

Calibration / Heldout / Final：**Not Run**。独立已标注新Log=0；qualified_certifications=0，qualified_FAR=null，Identity Selection=null，Region Selection=null。当前统一实验中的按Log探索性交叉验证仅作固定方法逐Log报告，19/20折无可评分接受，全部仍为 exploratory_reused_population，不能擦除开发曝光。采用95%单侧、每Log任意错误为失败的零错误风险上界，要≤5%至少需59个完整可评分、独立、实际接受Log；即使20个开发Log全是全新且零错误，上界仍有 13.910834%。重复场景、帧或对象不能补足Log单位。

## 六项 Gate

| Gate | 结论 | 实测依据 |
|---|---|---|
| 1 Engineering | Pass | CUDA 517/517；实际240次本轮Commit无重复历史ID，身份13,314边行为一致；8帧原感知/骨干/记忆摘要全部相等；旧产物哈希一致 |
| 2 Surface Physical Observability | Partial | 测得不同采集和冲突解释可排除错误；歧义边际14错误/46正确损失；没有物理所有权认证 |
| 3 Region Feasibility | No-Go | 非零5正确/0已知错，四项点估计门槛通过，但面积极低、仅1接受Log/Scene、点风险上界45.07%，原资格门槛未通过 |
| 4 Identity Feasibility | No new improvement | 保留217正确/45唯一实例及Shadow；人口分析未产生新增收益或认证 |
| 5 Independent Quality | No-Go | 新独立已标注Log=0；Calibration/Heldout/Final Not Run；正式选择null |
| 6 Scientific Feasibility | Boundary identified | 相关RGB证据可自洽；附加采集仍可歧义，条件Patch模型有额外覆盖损失；独立数据限制另行报告 |

## 下一轮最小必要证据

先在**同一nuScenes六摄像头输入与原模型**内验证：对实际共同可见、纹理可辨且具有足够非共线基线的第三次采集，直接测量可唯一排除竞争源像素的对应，而非继续增加同一DIS链上的平面或评分项。必须明确记录该观测真的可访问、时间可靠、未被遮挡与足够视差；本轮不同采集正支持仍有错误，尚未证明有限Patch搜索能提供这样的唯一性。前平行探针拒绝的真点应单独检查斜面/光照/遮挡等可实测模型条件，不能先给根因标签。

只靠当前弱重叠视角和两帧自洽不能支持已观测歧义区域的物理结论；此类区域保持Unknown/Refresh。物理资格方面，需要后续合法新增且未曝光的nuScenes acquisition Log及完整离线参考；本轮本地不能满足。算法继续可在复用人口研发，研究收益与独立认证应分开结论，不改变数据集、模型输入或目标来凑样本。

主要源码：[surface_visibility_constraints.py](../../worldstream/m1/surface_visibility_constraints.py)、[surface_observability.py](../../worldstream/m1/surface_observability.py)、state_manager.py/config.py/state_types.py/novelty_detector.py/dense_support.py；新配置 worldstream_m1_physical_observability.yaml。实现说明见 implementation_changes.md；可视化入口 region_risk_coverage.png、roi_and_paired_support.png、identity_population_recovery.png 及真实RGB病例目录。当前所有未通过指标和完整风险统计保存在 formal_go_nogo.json、region_method_comparison.json、identity_population_analysis.json。
