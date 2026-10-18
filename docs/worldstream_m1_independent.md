# M1 独立状态与区域更新接口

本次仅修改 M1 及其测试/工具，不修改权重或 M2/M3 算法。需要 OmniDrive Python 环境、SciPy/OpenCV；执行命令始终以 `taskset -c 0-3,6-31` 启动。

## 验证语义

`precheck(frame)` 的输入仅为当前 RGB、位姿、标定及执行上下文。当前模型检测在 `commit` 才进入 M1，因此 `association_candidate_ids` 是具有历史投影的候选，不是与当前检测匹配的结果。真正的当前几何关联见 commit 的 `geometric_candidate_pairs`、`association_pairs` 和 `rejected_ambiguous_associations`。

`valid_anchor_ids` 保持兼容，等于 `observation_supported_anchor_ids`，表示已有观测/几何/时间门限支持。它不能证明同一物理实例、目标静止、前景归属或三维位置正确。`physically_certified_anchor_ids` 专供严格物理结果；当前适配器缺少独立前景、身份、定位和运动误差生产者，所以该列表为空。逐对象拒绝原因见 `diagnostics.anchors[id].physical_certification`。这属于 Physical Certification No-Go，不能将空接受写作 FAR=0。

`history_observations` 和 `association_count` 是候选关联计数。`certified_observation_count` 只允许真正的独立认证推进，当前为 0。`source_frame_id`/`source_timestamp` 始终表示最新实际计算来源；track birth 不随新观测推进。`last_verified_source_frame` 说明验证针对哪个实际历史来源，避免将旧几何的验证误当成新提交几何的验证。`anchor_provenance` 提供这些字段和有界的关联历史快照；特征 `provenance` 接口保持原结构。

## 生命周期与原子发布

一个 manager 由单一写入调用者管理。普通 Precheck 在私有 store 元数据副本中执行 prune 和 verification，公开的 `manager.store` 始终指向上次成功提交的快照。Scene/时间/模型签名/相机布局变化则立即失效历史，这是安全重置操作。

Commit 绑定 scene、frame、时间、执行签名、state epoch，以及图像、位姿、标定、相机顺序、可选支持掩码的 SHA256。修改已验证 Frame、超时、旧结果或重复提交会被拒绝。全部 anchor、descriptor、特征副本和 provenance 在私有工作对象构造完成后才发布；异常不会留下半写入状态。直接调用 store 的内部写方法不提供多步事务保证。

模型异常后可调用 `abort_pending()` 释放临时状态；忘记调用时，下一次 Precheck 也会丢弃未提交事务并强制刷新。丢弃帧不会替换 `_previous_cameras` 的实际来源，也不会推进已提交验证时间或版本。pending 超时按单调墙钟计算，不延长 source 的有效时间。

## 关联与容量

合法预测先按分数、类型/类别、中心与角点确定性选择至 `max_anchors`，然后执行最大合法匹配数、再最小距离的 Hungarian 分配。分数仅用于容量优先级。SciPy 是必要依赖，不再静默退化为不等价的 greedy。

最优/次优候选在任一端发生竞争时拒绝延续；默认只拒绝精确 tie。正 margin 是可配置的开发参数，尚未校准，不提供身份概率保证。缺失一个实际观测帧后重新出现的 object 建立新候选 track，并记录拒绝原因；这些有意的保守断轨应在 Oracle 中计为 fragmentation。普通几何延续始终不增加认证计数。

`selected_prediction_indices` 和 `committed_anchor_ids` 按位置对应；association 的 prediction index 指向输入 `FullOutputs.anchors` 的原始顺序。离线工具必须使用该映射，不能在排序/截断后按位置猜测原 detection index。

## 四个精确 Mask

各相机 `invalidation_regions` 中输出压缩编码的 `observation_stable_mask`、`observed_changed_mask`、`unknown_mask` 和 `refresh_required_mask`；用 `worldstream.m1.mask_codec.decode_mask` 还原。前 3 类互斥且覆盖该相机图像。Refresh 是 Changed|Unknown，加上强制刷新策略。强制刷新不改变观测分类；`invalidation_area_ratio` 直接取最终 Refresh 的均值。

`verified_stable_area_ratio` 只描述 RGB 局部观测；`effective_stable_area_ratio` 描述最终无需刷新面积。框级描述子失配不能定位真实变化，故保留 Unknown；Changed 仅来自已建立对应后的局部变化证据。`unknown_uncovered_area_ratio` 保留兼容，旧语义为 Changed|Unknown，明确同名别名是 `observation_refresh_area_ratio`。

`exact_invalidation_mask` 是 Refresh 的兼容字段，所有模式均提供。Tile regions 为精确 Refresh 的保守包络。每相机独立保留 image_shape、coordinate_space、raw_to_image；mask 已在当前 image space 中，不能再次应用 raw_to_image。640×360 回放与 640×640 实际模型输入的面积不可直接比较。

## 几何与证据边界

旋转 box membership 使用八角点恢复并验证的局部正交坐标，支持角点重排和完整刚体旋转；非法 cuboid 拒绝，不回退到宽松 AABB。但落在预测旋转框内仍不证明点属于目标实例。双向几何继续是拒绝条件；静态类别只是运动先验，动态对象的静态三角化被拒绝。检测头速度保留预测数值但不标为可靠运动；旧 `trust_detection_velocity` 设置不再将有限速度升级为独立证据。

`geometry_uncertainty=0` 不表示零真实误差；`geometry_uncertainty_calibrated=False` 明确其状态。当前没有适用于全部目标的校准误差界。2m GT 关联后得到的截断残差不能用于宣称全群体覆盖。

## 复现与结果范围

运行命令和记录见 `outputs/module1_independent_refinement/commands.sh`、冻结协议、实验报告。真实检测回放、离线 annotation/LiDAR Oracle、受控异常注入与原始模型感知数值对照分别报告。旧 150 个验证场景均有本地记录曝光，不能把本次再次使用包装成新的独立测试。

GT 只由 tools 下离线评分函数读取；在线 M1 不导入 dataset/annotation/Oracle，metadata 中显式 GT/instance token 字段被拒绝。此边界不能辨别调用者将任意外部数组伪装为普通图像或支持掩码的行为；严格物理认证也不会因此接受。
