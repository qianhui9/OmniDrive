# M1 实测证据与局部拒绝实验

本轮复用原状态事务与 Mask 接口；默认行为仍为保守模式。原始模型、Checkpoint、LoRA 与感知预处理/推理参数不变，未修改 M2/M3。

## 证据接口与声明

`PhysicalEvidence` 保存 anchor、scene/source/current frame、epoch、源/当前 state version、图像 digest、执行签名、实际测量 JSON 和执行回执。`PhysicalCertificationResult` 返回具体 `certified_claim`、拒绝原因和数值来源。它们不能靠 `identity_verified=True` 或检测分数生成成功结果。

`foreground_correspondence.propose_foreground_correspondence` 是实验生产器，实际检查双向测量点、相机时间、三角化/视差/重投影、旋转预测框、空间分布和邻域环深度分离；环内点也只是候选背景。静态类别不批准静止，目标和背景不可区分时返回 Unknown。框内光流不会被整体填为前景 Mask。

测量回执由一个 manager 内的 issuer 生成并认证，阻止普通数据接口伪造/替换测量。它是受信执行路径的进程内边界，不能对抗任意 Python 代码访问私有密钥，也不是传感器真值证明。真正允许声明还需要与生产器协议、人口、场景/log 独立性一致的离线质量资格；本轮没有发放任何批准资格。

唯一已实现的声明是 `static_candidate_identity_continuity_with_local_ray_consistency`，即限定对象连续性和局部射线一致性；不批准三维中心、距离或速度。完整数值声明必须另有适用误差界，当前明确拒绝。即使将来限定连续性批准，也不会自动批准其他属性。

## 时序与版本

当前 Detection 只在完整感知后的 Commit 使用。Precheck 的结果不接受当前检测或本轮事后认证的回写；`post_perception_physical_certifications` 明确记录事后阶段。只要新检测替换几何，就生成新的 `state_version`，旧 `physical_certification` 和 `last_certified_timestamp` 不沿用。历史认证次数可保留为统计，但不代表当前几何已认证。

公开历史几何、描述子和灰度图采用自有不可写 bytes-backed 数组；fork 复制 Geometry/Observation 包装和可变元数据。保留的大 tensor 在 fork/acquisition 边界隔离，默认仍不捕获大 tensor。Commit 检查 Precheck 时的源状态 digest，防止 pending 期间替换历史源数据。Frame Digest、超时、epoch 和原子发布保持原契约。

## 研究配置

`configs/worldstream_m1_evidence.yaml` 显式打开实验性前景测量；仍使用原保守区域策略。`physical_qualification_path`/`physical_qualification_sha256` 默认为 null，故不足证据和资格时均拒绝。哈希针对资格 JSON 的 canonical serialization（sort_keys、紧凑分隔符），不是缩进文本文件；资格必须来自独立审计，不能手填 flag 代替质量证据。

资格协议还绑定生产器源码和全部证据相关配置，包括关联、观测门限、模型执行标识和图像坐标空间。输出路径、日志、线程数和资格文件本身不参与指纹。改变证据算法或门限后必须重新独立验证，不能沿用旧资格。

`dense_region_policy=local_visibility` 只增加局部拒绝，使用旋转补偿视差梯度、双向流、折叠和显著光度残差作为未知区域种子。实验用较小支持邻域配合局部拒绝半径；历史动态扫过保护保持。该方案必须在 Calibration 通过约束后才有资格考虑推广，不能只看稳定面积增长；默认 `uniform` 保持不变。

四类 mask 沿用原语义：Stable/Changed/Unknown 互斥覆盖，Refresh 为 Changed|Unknown 加强制策略。新增局部拒绝进入 Unknown，不伪造 Changed；Force Refresh 保留观测分类。

## 评价人口

已曝光的 val 场景只作开发/复现。本轮另冻结 38 个对本地记录的 M1 开发新的 train scene/log；它们仍属于 OmniDrive 模型训练人口，不是模型独立测试。先采集原模型预测，再用互斥 Calibration/Heldout/Final 的离线 GT 评分。

Foreground 的 LiDAR/GT volume 对照是稀疏几何参考；体积内点不等于实例像素真值。Association Oracle 保留 2m/竞争门限，将超距、竞争和无支持 GT 分开报告；不能把未评分样本视作正确。20-scene 的有限条件误差区间仍受该筛选截断，不能应用于全部在线对象。

完整命令、协议、原始 trace、风险/覆盖、失败像素、成本与四项决策见 `outputs/module1_evidence_optimization/`。未批准部分保持 No-Go，不开展以牺牲验证换取的加速。
