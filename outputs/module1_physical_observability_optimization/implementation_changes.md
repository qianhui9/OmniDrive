# 实现与复现说明

本轮统一环境：nuScenes 六摄像头连续场景、原 OmniDrive、M1。没有新增其他数据集 Adapter。所有正式物理资格仍为0，正式 Identity/Region Selection 为 null。

## 在线代码

- `worldstream/m1/surface_observability.py`：实际射线三角化、局部深度投影 Patch、有限极线替代解释、真实额外采集的测量。区分同源一致、未观测和实际支持冲突；不接收离线参考。
- `surface_visibility_constraints.py`：唯一主要新机制；实际源支持 + 不同采集正支持 + 实测歧义拒绝。条件深度排序仅用于拒绝，不证明遮挡者/物理所有权。提供五个显式证据掩码和可审计漏斗。
- `state_manager.py`：仅新配置分支在所有实际相机证据建立后应用约束；为新分支维护既有有限实际图像历史。原Precheck/Commit原子性、来源/签名/版本绑定、资格签发、默认Shadow与原模型执行保持。
- `dense_support.py`：新分支使用既有完整邻域基底，跳过旧 Surface Visibility/Consensus 的平面路径。旧路径保留。
- `config.py`：增加合法显式 `surface_constraints` 策略；默认值不变。
- `state_types.py`：新增可选 observation_supported/visibility_consistent/ambiguity_rejected/stable_candidate/refresh_required 掩码与纯测量诊断字段。旧分支字段为None。
- `novelty_detector.py`：新分支证据的 refresh_required_mask 同步最终精确刷新（含强制刷新）；精确mask与保守Tile包络原定义保持。
- `configs/worldstream_m1_physical_observability.yaml`：研究配置，identity_method=temporal_object、identity_tracking_policy=shadow；无独立Qualification。

没有修改 M2/M3、原模型、Checkpoint、原预处理、coverage_oracle.py、evidence_oracle.py、identity_temporal.py 或 identity_object.py。没有新增Feature Reuse标记，没有使用预测Box冒充真实表面。

## 工具与证据

- `observability_diagnostics.py`：改动前690/194接受点的原图和观测可辨识性诊断。成功重放1320个非启动相机评分记录与历史差异0；原始诊断已冻结，不覆盖。
- `observability_experiments.py`：20场景的实际Development、新路径、6项预声明逻辑消融、逐点决策与13,314条身份边的行为核对。所有场景保留原始JSONL。
- `observability_nuscenes_audit.py`：逐帧检查Train/Val实际六摄像头与LiDAR、标注范围、68个Log历史具体轨迹证据、原预测缓存和数据/模型SHA256。只扫描nuScenes。
- `observability_replay_checks.py`：四个历史失败ROI的实际双方法重放、精确mask/Tile包络/活跃动态扫掠审计；另外独立manager重放two_view_only消融。
- `observability_results.py`：原始轨迹生成配对误差/正确损失、在线定义人口、Log聚类区间、探索性逐Log报告、稀疏Oracle候选上界、No-Go及来源清单。
- `observability_report.py`：从真实产物生成中文实验报告与物理可辨识性审计。

消融共享相同已计算观测，原模型每帧完整运行且掩码不影响Commit来源；独立manager重放核对288个相机记录，结果一致。Online人口定义包含旧Selective正证据的reason与新Temporal/Object的status，避免漏掉合法旧路径；不可评分接受全部保留。

ROI离线诊断使用Precheck私有已裁剪状态中的活跃来源，防止将公开旧状态中已经过期的Anchor误算成保护。四个旧失败ROI新路径均100%精确刷新；没有将100%的Tile包络当精确ROI刷新成绩。

## 验证与限制

完整CUDA回归517/517通过（原506项+新增11项观测约束测试）。8帧原模型EVA/检测/地图输出及时间记忆摘要完全一致；仅感知验证，没有执行LLM。GPU空闲时运行，未停止任何其他进程。原9个历史文件SHA256一致。

新方法5个可评分正确点、0个已发现错误，仅1 Scene/Log；面积0.003654%，点风险上界45.07%、Log上界95%，没有正式风险资格。旧494正确只保留1个，损失493个，另增4个正确；旧196错误均阻止。不能将这些数字表述为可信面积增益或物理认证成功。

局部前平行深度Patch是条件探针，既可能缺少视野/纹理，也可能不适配真实表面；失败时Unknown，不将失败一概解释为遮挡或物理不可辨识。有限替代搜索不证明全局唯一。深度排序没有可评分错误剔除收益，不将其作为实证主要贡献。

正式Calibration/Heldout/Final Not Run：68个Trainval Log全部曝光；Test的新Log本地无质量标注。探索性交叉验证不创造独立Log；至少59个完整可评分、独立、零错误接受Log才可能满足既有5%风险上界。没有改变风险门槛、原模型输入或数据集以凑样本。

`commands.sh`记录实际执行命令；在仓库根目录、指定CPU亲和性下执行。诊断和Development会拒绝覆盖已有冻结/场景目录。复现实验须先保留现有整轮输出，使用全新工作副本或新输出位置；派生分析与报告可直接从已保存原始轨迹重建。
