# M1 实现与验证范围

基线为 `d95011a`。新实验入口为 `configs/worldstream_m1_correspondence_recovery.yaml`，继续使用 `dense_region_policy: surface_constraints`、`identity_method: temporal_object` 和 `identity_tracking_policy: shadow`。原配置的默认行为保留。

`geometry_adaptive_patch.py` 只增加一种在线几何模型：中心逆深度固定，四个局部角点的实际 RGB Flow 对应拟合两个逆深度斜率，四个分离的轴向点留出验证。所有八点都必须具备原始候选支持、正深度、既定视差及重投影约束，拟合与留出误差均不得超过原0.5px几何容差。随后直接在源图像或实际第三图像采样投影 Patch，使用原光度、纹理、梯度阈值。保留完全相同的 Frontoparallel 路径。通过局部平面不代表完整物理表面归属；每个最终接受像素都独立执行检查。

`surface_observability.py` 的 Patch 投影增加可选逆深度斜率参数，未提供时维持旧运算路径。`surface_visibility_constraints.py` 在同一测量人口上保留全部旧掩码，并额外形成自适应几何、联合消歧及完整方法的掩码。旧验证不会被新模型失败覆盖；实际测到的竞争解释不会因局部 Flow 拟合而直接消失。

联合消歧不使用残差排名。第三图像需要 H0 正支持，以及直接 Current→Third 的双向 LK、分离空间留出、几何三角化、回到原 Source 的闭环和已经验证的局部几何。只有竞争假设落入实际观测的局部范围且与实测射线存在超过既定对应容差的矛盾时，才允许排除。任何实际视角仍支持竞争解释时保留 Unknown；离屏、纹理不足、跟踪失败和 Patch 不匹配均不构成独立负证据。不同 RGB 采集仍可能相关，未声明统计独立或物理认证。

两个配置字段均显式校验：`patch_geometry_model` 默认为 `frontoparallel`；`joint_observation_disambiguation` 默认为 false。独立实验配置启用 `local_inverse_depth` 和 true。没有修改身份生产器、Physical Certifier、StateStore、State Manager、原模型或 Checkpoint。

`correspondence_diagnostics.py` 在算法改动前复现固定的1314正确/230错误人口及177/80源验证结果，并冻结所有记录。`correspondence_patch_archive.py` 补充逐点实际 RGB 样本及投影有效位。未知物理原因保持 Unknown；视差、纹理等数值检查失败只说明测量不充分，不能替代物理归因。

`correspondence_experiments.py` 回放20个已有Development场景，正常 Precheck/Commit 使用真实六摄像头RGB、标定、姿态和原始 OmniDrive 完整预测缓存；LiDAR/Annotation 仅在 Precheck 之后评分。并非每帧重新运行完整神经模型。实际原模型新推理另外在8帧 Perception Parity 中执行。旧 Surface Constraints 在同一回放逐相机重算；其他历史方法按相同输入的已固定结果引用。

`correspondence_validation.py` 使用独立 manager 完整重放两个预声明消融：4px Conservative 邻域基底、关闭预测运动扫掠。关闭扫掠仍保留其他既有排除与变化保护，不能解释为完全移除所有变化证据。所有共享测量消融依赖相同已 Commit 状态，区域 Mask 不反馈原模型和 Shadow 身份关联。

最终统计、完整配对、风险上界、Log分层、ROI精确刷新与七项Gate由 `correspondence_results.py` 从原始JSONL生成。未确认的稠密正确/错误面积、不可评分面积以及替代假设物理真值不从稀疏点数外推。正式选择始终为 null，没有重扫描或重定义已有68个Log的独立性。
