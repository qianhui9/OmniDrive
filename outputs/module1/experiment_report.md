# M1 实测交付记录

关联/有效性实验使用真实 nuScenes 多视图关键帧及历史完整检测回放，Map 回放使用 OpenLane 标注辅助 commit。另外已执行新的 GPU 完整模型对照。M1 没有跳过任何计算。

## 验证

- 单元测试：51 项，失败 0，错误 0，跳过 0。
- 真实 CPU 图像 Pipeline：passed_actual_CPU_image_pipeline_and_adapter，Tensor shape=[1, 6, 3, 640, 640]，输入未被适配器修改。
- 最终预处理投影与项目 lidar2img 最大绝对差 4.57e-11 px（双精度、同一标定路径数值交叉验证，不代表实际标定误差）。
- RTX 3090 完整模型 3 帧 Baseline/Observe-Only：通过；Backbone、Detection/Map 输出及 Temporal Memory 数值摘要完全一致。

## 首场景物理关联

| 方法 | Precision（可评分对） | Recall | ID Switch | 可评分/全部提议 | 重投影误差 px | 回放计时 ms |
|---|---:|---:|---:|---:|---:|---:|
| fixed_image | 96.50% | 90.19% | 24 | 200/427 | 1.279 | 476.538 |
| appearance | 94.42% | 86.92% | 30 | 197/426 | 1.221 | 466.047 |
| world | 96.62% | 93.46% | 16 | 207/432 | 1.433 | 18.623 |

未知身份未充当误匹配。ID Switch 使用本回放定义，不是官方 tracking benchmark。关联回放时间受到 tracemalloc 影响；性能结论用下方独立 profile。

## 首场景状态有效性

| 配置/方法 | 候选率（全部历史试验） | 物理 Oracle 错误接受率 | 有效状态恢复率 | 被接受且有标签 |
|---|---:|---:|---:|---:|
| replay_strict/geometry_only | 77.35% | 29.27% | 72.50% | 164 |
| replay_strict/appearance_only | 95.09% | 27.91% | 96.88% | 215 |
| replay_strict/geometry_observation | 0.00% | N/A | 0.00% | 0 |
| replay_appearance_ablation/geometry_observation | 25.00% | 15.19% | 41.88% | 79 |

首场景共 468 个历史状态试验，229 个有可用 Oracle 身份。严格认证没有独立可见性证据，因此没有接受状态；其 False-Reuse 是 N/A，不能写成 0%。消融仅取消可见性要求，其错误率说明不能将它当作安全缓存规则。Oracle 只验证物理中心/身份/可见性，不验证深层特征。

## 真实场景覆盖

| 运行 | 帧数 | 实际帧间隔范围 s | 历史状态试验 | 新入相机视野标注区域覆盖 | 未知背景更新率 |
|---|---:|---:|---:|---:|---:|
| replay_strict | 8 | 0.4993–0.5004 | 468 | 8/8 | 100.00% |
| replay_second_scene | 12 | 0.3998–0.5010 | 247 | 78/78 | 100.00% |
| replay_interval_1s | 8 | 0.8997–1.0009 | 309 | 33/33 | 100.00% |

全图未解释时的覆盖召回为平凡结果，意味着请求完整更新，不意味着语义新物体检测成功。开放世界 semantic FPR 缺少可靠负例，保持 null。

## CPU 性能与内存

| Anchor 数 | 描述 | Projection ms | Observation ms | Unexplained ms | Association ms | State Update ms | Total ms | Traced peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | normalized_gray | 9.279 | 6.348 | 2.115 | 0.379 | 10.502 | 29.180 | 1.778 |
| 8 | gradient | 3.285 | 2.636 | 0.810 | 0.144 | 4.305 | 11.381 | 1.785 |
| 32 | normalized_gray | 13.521 | 18.931 | 1.635 | 1.477 | 16.478 | 52.720 | 2.128 |
| 32 | gradient | 13.616 | 21.707 | 1.628 | 1.479 | 17.090 | 56.193 | 2.145 |
| 64 | normalized_gray | 27.227 | 39.145 | 2.765 | 6.657 | 32.757 | 109.770 | 2.573 |
| 64 | gradient | 27.279 | 45.206 | 2.777 | 6.617 | 34.360 | 117.506 | 2.630 |
| 79 | normalized_gray | 33.577 | 46.879 | 3.323 | 8.831 | 34.785 | 128.879 | 2.793 |
| 79 | gradient | 33.738 | 54.516 | 3.328 | 8.937 | 36.413 | 138.481 | 2.906 |

真实两帧微基准，warmup=1、repeats=5；分配跟踪另测。以上不含完整模型、离线 Oracle、IO/可视化或在线 GPU→CPU 输入拷贝。默认未保留任何大型 Tensor，retained_tensor_bytes=0；Anchor 的小描述仍占用 CPU 内存。该 CPU 微基准不测 GPU 时间/显存；完整模型指标单列。进程 RSS 含输入表，不用作 M1 净内存。

## 轻量描述对照与校准

| 描述 | 固定 crop 外观错误接受率 | 人工亮度扰动平均残差 |
|---|---:|---:|
| normalized_gray | 27.91% | 0.0099 |
| gradient | 27.57% | 0.0106 |

亮度对照是在真实当前图像上施加 0.7×I+25，再比较固定区域描述，不代表跨天气泛化。当前结果不足以选择所谓最优阈值；默认灰度实现保持 provisional。

train-only 校准状态：insufficient_evidence，独立 Anchor 可用试验 6，threshold=None。未将评估集用于阈值调节，未写入不满足风险支持的阈值。

## Map 与可视化

真实 OpenLane 标注辅助 Map 回放 6 帧：每帧地图线数 [19, 19, 19, 19, 19, 19]，后续几何匹配数 [19, 19, 19, 19, 19]。未声称完成新的 Map Head 推理或官方 Map ID 评价。

逐帧六视图见各运行目录 visualizations/*.jpg，精确 mask 见 masks/*.npz；contact_sheet.jpg 为连续关键帧总览。figures/*.png 和 *.pdf 为可导出图表。颜色区分历史 crop、预测投影、认证结果、当前完整输出和失效/未解释区域。

M2 接口实例见 m2_example.json，字段来自真实回放的最后一次 precheck；valid_feature_candidates 只代表物理/观测候选，Transformer dependency 未认证。完整接口、已验证命令、文件职责、坐标与限制见 ../../docs/worldstream_m1.md。

## 新执行的 RTX 3090 完整模型对照

| 模式 | 帧数 | CUDA Event 总时长 ms（含 host 间隙） | 峰值 allocated GiB |
|---|---:|---:|---:|
| baseline | 3 | 10656.197 | 15.119 |
| observe_only | 3 | 10040.815 | 15.119 |

单次、未充分预热的完整执行检查，包含输入处理、数值摘要拷贝与 LLM 生成，不用于推导加速比。验证结论是开启 M1 未改变这 3 帧的视觉/感知/历史 Memory 数值；不要求采样文本逐字一致。

M1 实际在线 CPU wall overhead（包含输入/输出适配与必要同步拷贝）：[86.717, 398.203, 306.014] ms；均值 263.645 ms。

实际捕获的形状元数据（默认不保留大型 Tensor）：

| 特征 | Shape |
|---|---|
| backbone | [1, 6, 1024, 40, 40] |
| object_carrier | [1, 257, 4096] |
| map_carrier | [1, 256, 4096] |
