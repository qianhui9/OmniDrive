# 第一轮失效诊断（算法修改前生成）

依据：第一轮原始逐 Anchor/Camera 日志，以及相同预测、真实图像与离线 GT 的重新回放。GT 只在 precheck 后用于诊断。

## replay_strict

| 按现有判定顺序的约束 | 剩余 | 重叠失败 | 唯一阻塞 | 首失败 |
|---|---:|---:|---:|---:|
| all | 468 | — | — | — |
| within_horizon | 468 | 0 | 0 | 0 |
| motion_known | 363 | 105 | 0 | 105 |
| calibration | 363 | 0 | 0 | 0 |
| reliable_view | 362 | 106 | 0 | 1 |
| historical_description | 362 | 106 | 0 | 0 |
| observation | 327 | 141 | 0 | 35 |
| visibility | 0 | 468 | 327 | 327 |

Anchor 原因：`{"no_reliable_view": 106, "motion_unknown": 105, "visibility_unknown": 331, "observation_mismatch": 35}`。
Camera 原因：`{"incomplete_depth": 1507, "uncertainty_exceeds_search": 98, "no_historical_observation": 817, "visibility_unknown": 347, "observation_mismatch": 35, "outside_camera_horizon": 4}`。

- Physical Oracle 历史身份已知：229
- Physical Oracle 同一实例仍存在：229
- Physical Oracle 传感器支持：226
- Physical Oracle Visibility=4：202
- Physical Oracle 中心误差≤1m：160
- gt_displacement_m P50/P95：[0.0, 0.082]
- gt_speed_mps P50/P95：[0.0, 0.164]
- center_error_m P50/P95：[0.524, 1.508]
- 投影所需半径 P50/P95 px：[2.0, 71.35]

## replay_appearance_ablation

| 按现有判定顺序的约束 | 剩余 | 重叠失败 | 唯一阻塞 | 首失败 |
|---|---:|---:|---:|---:|
| all | 468 | — | — | — |
| within_horizon | 468 | 0 | 0 | 0 |
| motion_known | 363 | 105 | 0 | 105 |
| calibration | 363 | 0 | 0 | 0 |
| reliable_view | 362 | 106 | 0 | 1 |
| historical_description | 362 | 106 | 0 | 0 |
| observation | 327 | 141 | 4 | 35 |
| visibility | 117 | 347 | 210 | 210 |

Anchor 原因：`{"no_reliable_view": 106, "motion_unknown": 105, "projected_occlusion": 212, "observation_mismatch": 35}`。
Camera 原因：`{"incomplete_depth": 1507, "uncertainty_exceeds_search": 98, "no_historical_observation": 817, "ok": 132, "projected_occlusion": 215, "observation_mismatch": 35, "outside_camera_horizon": 4}`。

## replay_second_scene

| 按现有判定顺序的约束 | 剩余 | 重叠失败 | 唯一阻塞 | 首失败 |
|---|---:|---:|---:|---:|
| all | 247 | — | — | — |
| within_horizon | 247 | 0 | 0 | 0 |
| motion_known | 12 | 235 | 0 | 235 |
| calibration | 12 | 0 | 0 | 0 |
| reliable_view | 4 | 243 | 0 | 8 |
| historical_description | 3 | 244 | 0 | 1 |
| observation | 0 | 247 | 0 | 3 |
| visibility | 0 | 247 | 0 | 0 |

Anchor 原因：`{"no_reliable_view": 243, "motion_unknown": 235, "no_historical_observation": 1, "observation_mismatch": 3}`。
Camera 原因：`{"incomplete_depth": 833, "uncertainty_exceeds_search": 239, "no_historical_observation": 404, "insufficient_coverage": 1, "observation_mismatch": 3, "tiny_region": 2}`。

- Physical Oracle 历史身份已知：146
- Physical Oracle 同一实例仍存在：140
- Physical Oracle 传感器支持：125
- Physical Oracle Visibility=4：34
- Physical Oracle 中心误差≤1m：1
- gt_displacement_m P50/P95：[0.086, 5.631]
- gt_speed_mps P50/P95：[0.169, 7.423]
- center_error_m P50/P95：[0.99, 6.191]
- 投影所需半径 P50/P95 px：[14.5, 35.65]

## replay_interval_1s

| 按现有判定顺序的约束 | 剩余 | 重叠失败 | 唯一阻塞 | 首失败 |
|---|---:|---:|---:|---:|
| all | 309 | — | — | — |
| within_horizon | 309 | 0 | 0 | 0 |
| motion_known | 236 | 73 | 0 | 73 |
| calibration | 236 | 0 | 0 | 0 |
| reliable_view | 234 | 75 | 0 | 2 |
| historical_description | 227 | 82 | 0 | 7 |
| observation | 112 | 197 | 0 | 115 |
| visibility | 0 | 309 | 112 | 112 |

Anchor 原因：`{"no_reliable_view": 75, "motion_unknown": 73, "visibility_unknown": 116, "observation_mismatch": 120, "no_historical_observation": 7}`。
Camera 原因：`{"incomplete_depth": 935, "uncertainty_exceeds_search": 64, "no_historical_observation": 485, "visibility_unknown": 120, "observation_mismatch": 127, "outside_camera_horizon": 122, "insufficient_coverage": 1}`。

- Physical Oracle 历史身份已知：161
- Physical Oracle 同一实例仍存在：158
- Physical Oracle 传感器支持：153
- Physical Oracle Visibility=4：138
- Physical Oracle 中心误差≤1m：112
- gt_displacement_m P50/P95：[0.056, 0.216]
- gt_speed_mps P50/P95：[0.056, 0.236]
- center_error_m P50/P95：[0.475, 1.748]
- 投影所需半径 P50/P95 px：[2.0, 123.8]

## 根因及统计边界

1. 所有回放 CameraFrame 都未提供 visibility_mask/geometry_support_mask。旧严格认证没有可用可见性证据；取消该约束的消融不能成为正式方案。
2. 动态类别没有可靠速度，因此 motion_known=false；即使外观相似也不会被正式接受。0.5 秒运动误差和静态预测误差已单独保存。搜索半径由速度不确定性推导，常超过4px；不能仅扩大搜索就声称运动可靠。
3. 旧 novelty 将未提供支持的像素全部置为 Unknown；所有回放的 Verified Stable Pixels=0、Unknown/Unsupported=100%。旧 changed_regions 将 inf（包括未测量）也算变化，不能据此报告 Observed Changed Pixels；该指标不可辨识，须修正。
4. 周期刷新和 reset 只增加更新需求，不能解释其余帧仍然全图失效。新实现必须保留局部证据，即使其他区域未知。
5. 多约束统计依赖现有短路日志；sole_blocker 是移除单一布尔约束的结果，不能把未执行的图像验证当作独立观测。完整逐相机原因及离线运动记录见同目录 JSON。

下一步：采用校准相机约束的双向稀疏 LK、逐像素局部残差支持和保守遮挡排除；不填充 Box，不把图像证据等同于 Transformer 依赖认证。

补充：为解除旧短路日志对遮挡/外观条件的遮蔽，完成了[逐约束独立测量](diagnosis_independent.md)。首场景加入独立遮挡检查后327→117，再由缺失可见性证据117→0；未可测条件以null保留。
