# M1 编码前审计

已阅读需求列出的全部 18 个核心文件、默认配置、现有测试与三轮 M1 报告，并解析 v2–v5 的物理 Oracle、risk/coverage、task fact、前景诊断和方向校准文件。原始 M1 源码与 SHA256 保存于 baseline_source/ 和 baseline_manifest.json，后续结果不覆盖旧目录。

## 已正确实现的机制

世界坐标 SE3、gravity center、相机时间位姿、近裁剪平面拒绝、来源 epoch、特征 computation version、当前帧 commit 门禁及 detached tensor 预算已有基础。双向 LK/DIS、光度/梯度/纹理验证提供 RGB 对应支持；不将整个目标框填成可靠像素的默认行为正确。task_facts 和 fact_certifier 已明确拒绝缺少独立身份、前景和误差界的物理事实。保留这些机制，不重写模型或 M2/M3。

## 已确认的实现风险

1. Precheck 原位 prune、更新 anchor 和 feature 的 last_verified_timestamp；模型失败或丢弃当前帧后仍留下这些变化。pending tuple 只能防止重复 commit，不能实现事务隔离。
2. Commit 在逐目标、逐特征写入前消费 pending；descriptor、tensor clone 或后续特征写入失败会保留部分新状态。需要在私有工作副本完成全部构建后一次发布。
3. Assignment 在 predictions 全集执行，之后仅消费前 max_anchors 项；被排除项可能先抢占合法历史匹配。应先确定性筛选，再关联，并追踪原始 detection 索引。
4. SciPy 缺失时 greedy 并不保证最大合法匹配数（例如 [[1,1.9],[1.5,inf]]）。选择明确依赖 SciPy 并失败，避免静默变更算法。
5. 相机字典顺序/集合改变、Frame 同 key 内容被替换、错误 camera_id 未统一约束；依赖模型签名不能保护手工 Frame/replay 调用。
6. 历史观测次数递增只说明候选关联，缺少歧义/缺帧记录和独立认证计数，不能作为身份可信等级。
7. 双向三角化使用世界轴对齐 envelope 判断 box 内点，对旋转框过松。应在有序角点建立的局部正交坐标中检测，非法/不支持的角点结构拒绝。
8. 默认 geometry_uncertainty=0 没有校准含义；需要显式标记 Uncalibrated，检测头有限速度也不是独立运动证据。

## 区域统计的实际版本情况

需求描述的 ratio 在强制刷新前计算确实存在，但当前 checkout 的 invalidation_area_ratio 已使用最终 invalid.mean()，并未复现这一具体字段错误；unknown_uncovered_area_ratio 实为 changed|unknown，命名含糊。不能声称修复不存在的旧错误。仍需增加四个精确 mask、三类互斥覆盖、独立 full refresh 原因，去掉将单纯 ROI 描述失配称作实际变化的路径。强制刷新必须保留原观测分区。

## 证据不足，与程序缺陷分开

几何合法匹配不等于同一物理实体；RGB 稳定不等于对象前景归属；三角化点落入预测框不等于真实实例认证；低残差也不能消除历史中心固定偏差。适配器只有 RGB、检测框、预测速度、模型特征和相机/自车位姿，无独立实例前景、可信身份轨迹或校准的定位/运动误差界。

历史物理风险：v2 可评分接受 79，错误 29（36.7%）；旧 coverage final 仅 3 条可评分接受，2 条错误。v5 72 帧独立前景 mask 为 0；方向校准只有 5 个 scene，半径 null，xy 误差 p95≈5.36m，且受 2m GT 关联截断影响，不能外推全体。保守拒绝全部也不能证明 FAR=0。

## 分阶段决定

先做 Gate 1 的事务、接口和 mask，再做 Gate 2 的确定性/歧义拒绝与独立 Oracle。Gate 3 允许明确 No-Go：目前没有可信前景生产者，不能用布尔 flag、框内光流或 GT 填补。Gate 4 使用既有 LiDAR/annotation 离线工具，未知背景不可作为静态负例。质量未通过前只测成本，不开展 P1 性能改进。GPU 可用性和新独立 scene 的可用性需在运行协议中如实记录。
