# M1 World-Anchored State Continuity

M1 已实现为独立、可测试的 Observe-Only 扩展。它维护物理世界锚点、低成本原始观测证据、计算版本和来源；不跳过 EVA、Detection、Map 或 LLM，不替换 Carrier，不修改 Prompt，不构造 Transformer 精确依赖图。默认关闭原模型接入，专用配置显式开启。

本机环境是 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，无依赖升级。宿主机 NVIDIA 驱动及 RTX 3090 正常，OmniDrive CUDA 张量计算已验证。初次检查发生在未暴露 `/dev/nvidia*` 的受限执行环境中，不能据此推断宿主驱动故障。真实图像、标定、既有完整检测输出回放、OpenLane 标注辅助 Map 实验和 CPU 图像 Pipeline 已运行；新的完整 GPU 模型对照结果见 `outputs/module1/experiment_report.md` 和 `model_smoke/status.json`。不要将回放或测试替身解释为新模型推理。

## 文件与理论对应

| 部分 | 文件 | 作用 |
|---|---|---|
| World-Anchored Representation | `worldstream/m1/state_types.py` | WorldAnchor、Geometry、FeatureRecord、ComputationalProvenance、Frame、M1Result 的类型与严格 JSON |
| 有界持久存储 | `state_store.py` | Scene/epoch、过期清理、最大 Anchor 数、detach/clone、Tensor 字节预算和计算版本 |
| Motion-Compensated Correspondence | `geometry.py`, `association.py` | SE(3)、完整 Box 角点、逐相机时间预测与投影、曲线采样、类别/距离/大小/方向门控和一对一匹配 |
| Observation Certification | `observation_validator.py` | 亮度归一化灰度或梯度描述、纹理检查、几何约束搜索、覆盖与可见性证据、二元判断 |
| Unexplained Observations | `novelty_detector.py` | 几何支持区域以外一律失效，局部匹配失败/变化区域、精确 mask 与保守 image region set |
| Versioned Propagation | `state_manager.py` | initialize/precheck/commit/reset/export_m2_inputs/get_candidate_feature |
| 来源与集成适配 | `provenance.py`, `omnidrive_adapter.py` | 分支级来源、实际 shape/dtype/device、去归一化图像副本、Ego/World/Camera 转换与日志 |
| 配置 | `configs/worldstream_m1.yaml` | 所有阈值、单位、捕获、日志、可视化、存储、周期刷新和校准文件 |
| 推理配置 | `projects/configs/OmniDrive/worldstream_m1_observe.py` | 继承本地有权重的 EVA 1024-channel / 4096-dim LLM 配置，保留 cam_infos |
| 原模型唯一修改 | `projects/mmdet3d_plugin/models/detectors/petr3d.py` | 可选初始化；simple_test Backbone 前 precheck；simple_test_pts 感知后只读 observer；原始返回结构不变 |
| 审查 | `tools/worldstream_m1/inspect_pipeline.py`, `pipeline_smoke.py` | 环境/原项目投影交叉验证、真实 CPU 图像预处理和适配器冒烟 |
| 实验 | `run_streaming.py`, `dataset.py`, `io_utils.py`, `evaluate.py` | 连续真实关键帧、流式 JSON、离线身份与物理 Oracle、关联/有效性/变化区域统计 |
| 校准 | `calibrate.py` | train-only、最小样本数、候选阈值多重比较修正的二项风险上界试验 |
| 性能/图形 | `profile.py`, `visualize.py`, `plot_results.py`, `map_demo.py` | 独立 CPU 计时/分配跟踪、六视图图像、mask、PNG/PDF、标注辅助地图关联 |
| 完整模型冒烟 | `model_smoke.py` | 有 CUDA 时限制帧数，记录原始感知/Backbone/Memory 摘要，比较关闭和开启 M1 |
| 测试 | `tests/worldstream_m1/` | 几何、关联、观测、存储、实际入口、工具与数据读取回归测试 |

逐文件清单可由 `rg --files worldstream tools/worldstream_m1 tests/worldstream_m1` 查看。根 README 和用户已有文件未覆盖。没有改 Head 的 pre_update_memory/post_update_memory，也未修改 TensorRT Engine。

## 坐标与时间契约

这里的真实语义比变量名更重要：`CustomNuScenesDataset` 的 `ego_pose` 是 Ego→Global；`update_coords.py` 已将 GT box/velocity 转为 Ego，Custom Dataset exporter 注释掉 LiDAR→Ego。`LiDARInstance3DBoxes` 在这个本地路径只是容器类型，检测值和 Map control points 实际在 Ego 下。Box 通过真实 `.corners` 和 `.gravity_center` 转 World，避免 bottom-center 的高度偏差。

`extrinsics = inv(T_ego_from_camera)`，`lidar2img` 在该路径实际为 `K_processed @ T_camera_from_ego`。两个 resize/crop 变换都会修改 K；Pad 不改变 K。原始像素到预处理像素的可靠映射是 `K_processed @ inv(K_raw)`。`pipeline_smoke.py` 调用实际 Load/ResizeCropFlip/Resize/Normalize/Pad 并交叉验证最终投影。

M1 World 状态采用 float64 SE(3) 列向量约定，米和绝对秒。投影使用每台相机的 `ego2global @ sensor2ego` 及实际 image timestamp；原模型仍保留原来的近似同步投影。没有 cam_infos 时只允许 `max_camera_skew_s` 范围内的同步近似，否则 Invalid。相机顺序从实际 metadata/dict 顺序读取，图像大小逐相机处理。

Head 的 memory_timestamp/sample_time 会先减绝对时间再加下一帧绝对时间，在使用时是相对年龄，不能作为 M1 的源时间。其 Reference Point 在 pre/post 更新之间也改变坐标系。M1 不依赖这些可变内部对象。

Box 的 8 个角点保留完整 3D 姿态；穿越近裁面、过小、过度截断、非法标定或超出不确定性搜索预算的投影均不能认证。Map 使用世界 polyline/control-point geometry，弧长采样后比较反向/正向距离，并先检查方向，不融合多个加权分数。

## 生命周期与 M2 接口

```python
from worldstream.m1 import WorldStateManager

manager = WorldStateManager('configs/worldstream_m1.yaml')
# frame 是 Frame；full_outputs 是实际感知完成后构造的 FullOutputs。
manager.initialize(frame0, full_outputs0)
check = manager.precheck(frame1)  # 参数中没有 GT、当前检测或当前 EVA 特征
m2 = manager.export_m2_inputs()

valid_anchors = m2.valid_anchor_ids
candidates = m2.valid_feature_candidates
regions = m2.invalidation_regions
provenance = m2.provenance
versions = m2.feature_versions
for item in candidates:
    record = manager.get_candidate_feature(item['feature_id'], item['computation_version'])
    # 默认 tensor=None，仅元数据；开启 capture_tensors 后可在预算内持有 detach 后的副本。
    # 获取历史记录必须在 commit 前进行；旧版本被替换后不能冒充仍可访问。

# M1 的当前版本继续完整执行原有视觉/语言路径。
manager.commit(frame1, actual_full_outputs1, check)
```

`M1Result` 输出 scene/frame/time、valid/invalid Anchor IDs、候选 Feature IDs、逐相机失效区域、provenance、历史 feature_versions、full_refresh_hint 和诊断。`result.to_dict()` 可写为严格 JSON，NaN/Inf 诊断值变为 null。完整 binary masks 在 `manager.last_masks`，实验保存为 `.npz`。坐标空间和 raw_to_image 映射随每个相机输出；未验证的特征层映射明确标记 unknown，没有 Token Index。

候选的 evidence 是 `physical_and_observation_only`，`transformer_dependency_certified=False`。当前只认证来源完整且当前相机已验证的局部低成本 observation descriptor；Object/Map Carrier 与 Backbone 仍是 unknown/conservative，不会因为某个物体匹配成功就认证整条分支。Object Carrier 不与框一对一，Map Carrier 不与地图线一对一。已知的 Backbone→两条分支计算边可记录当前版本，其余时间 Memory、attention/global dependency 不伪造。

precheck 只更新 last_verified_timestamp；computation_version/computation_timestamp 只在对应计算实际发生并 commit 时更新。返回结果保存验证时的历史版本快照，commit 不偷偷替换快照。Scene、非递增时间、大跳帧、模型/图像执行签名变化会清空状态；epoch 防止同一 Scene 重置后的 ID/版本混淆。仅验证而未 commit 的连续 precheck 也检查时间倒退。B>1 显式报错。

模型 reload hook 会清 M1 状态，逐帧参数版本、dtype、相机顺序、K/Extrinsics、图像尺寸/归一化共同构成执行签名。手工用 `.data` 绕过 PyTorch 版本计数修改权重时，调用方须显式 reset 并更新 model_id。Tensor 捕获默认关闭，Backbone 单独开关，预算只涵盖被捕获 Tensor；小型 Anchor/descriptor 元数据通过 max_anchors/age 有界。reset 释放持有引用，不调用影响整个进程的 CUDA empty_cache。

## 保守验证与可见性

二元判断是 geometry ∧ observation ∧ visibility/coverage ∧ horizon，没有加权总分。动态目标只有明确标为可靠的速度才外推；默认不把有限的网络速度当作可靠证据。静态 barrier/traffic_cone、Map 按静态世界几何处理。几何膨胀若超出 search_radius_px，直接 Invalid，不悄悄截短误差预算。

默认 `require_visibility_evidence=true`。没有独立可见性支持时，即使灰度匹配且框位于图像内部也 Invalid；投影到更近历史几何上的潜在遮挡同样拒绝。`CameraFrame.visibility_mask` 是可选、当前可得的独立证据，生产调用方须保证其语义，不能用当前 GT/完整检测来填在线输入。本次相机回放没有此证据，因此严格模式候选率为 0。这是当前保守限制，不能称已证明系统能安全缓存深层特征。

Novelty 只在已认证局部几何上比较描述残差；不对整张三维场景做单平面 Homography。解释像素还要求独立 `geometry_support_mask`（可靠表面支持，不等于 Box 内部或 visibility_mask）；没有它则整张图为未知更新请求。局部描述异常、未覆盖区域与周期刷新都形成 invalidation。没有旧 Anchor 绝不作为当前无物体的证据。

`--experimental-appearance` 是单因素消融：只取消独立可见性要求，速度信任/误差预算/残差阈值不变。它只用于量化风险，不能替代默认严格认证。normalized_gray 是简单主实现；gradient 是对照。亮度扰动评估是对真实 patch 施加已知变换，不伪装为真实天气变化。两者阈值都未宣称最优。

## 已验证命令

从项目根目录运行。`conda activate OmniDrive` 后可使用 `python`；下面使用实际环境绝对路径避免 shell 激活差异。回放拒绝覆盖已有 state.jsonl，完整模型对照也拒绝覆盖已有结果或复用旧 language 生成目录；复跑时指定新的 output 目录。

```bash
M1_PY=/home/qh/miniconda3/envs/OmniDrive/bin/python
M1_PRED=test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json

$M1_PY -m pytest -q tests/worldstream_m1 --disable-warnings
$M1_PY -m tools.worldstream_m1.inspect_pipeline
$M1_PY -m tools.worldstream_m1.pipeline_smoke

$M1_PY -m tools.worldstream_m1.run_streaming --predictions "$M1_PRED" --frames 8 --output outputs/module1/replay_new
$M1_PY -m tools.worldstream_m1.run_streaming --predictions "$M1_PRED" --frames 8 --experimental-appearance --output outputs/module1/appearance_new
$M1_PY -m tools.worldstream_m1.run_streaming --predictions "$M1_PRED" --frames 8 --descriptor gradient --output outputs/module1/gradient_new
$M1_PY -m tools.worldstream_m1.run_streaming --predictions "$M1_PRED" --frames 8 --stride 2 --output outputs/module1/interval_new
$M1_PY -m tools.worldstream_m1.map_demo --frames 6

$M1_PY -m tools.worldstream_m1.run_streaming --infos data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl --source annotations --frames 8 --split calibration --output outputs/module1/calibration_new
$M1_PY -m tools.worldstream_m1.calibrate --run-dir outputs/module1/calibration_train --output outputs/module1/calibration.json
$M1_PY -m tools.worldstream_m1.evaluate --run-dir outputs/module1/replay_strict
$M1_PY -m tools.worldstream_m1.visualize --run-dir outputs/module1/replay_strict
$M1_PY -m tools.worldstream_m1.profile --predictions "$M1_PRED" --repeats 5
$M1_PY -m tools.worldstream_m1.plot_results
```

以下 GPU 命令应在能够访问 NVIDIA 设备节点的宿主终端运行。Codex 受限执行环境可能隐藏这些设备，需获准在隔离环境外执行；不需要重装驱动或已满足的 Python 依赖。脚本将当前进程不可见与宿主驱动故障区分，并记录运行状态：

```bash
$M1_PY -m tools.worldstream_m1.model_smoke --frames 3 --compare --output outputs/module1/model_smoke_new
# 专用 config 也可沿用原项目分布式入口（本次验证使用上面的受限帧数脚本）：
bash tools/dist_test.sh projects/configs/OmniDrive/worldstream_m1_observe.py ckpts/iter_10548.pth 1 --format-only
```

model_smoke 固定随机种子，分别用独立 language 输出目录，保存 Backbone、Detection、Map 和 Temporal Memory 数值摘要。文本不纳入逐字数值对比；生成机制及采样参数未修改。CUDA Event+同步用于完整执行耗时/显存，M1 CPU 计时包含在线适配器的同步拷贝耗时。若确定性条件不足导致摘要不同，脚本报错，不把不同文本或 GPU 非确定性冒充一致。

## 实验与统计定义

具体实测数值及图片见 `outputs/module1/experiment_report.md`、`profile.json`、各运行目录的 manifest/state/oracle_records/metrics、`figures/` 和 `visualizations/`。

真实连续输入为 nuScenes keyframes，manifest 保存每个 timestamp、相机顺序和间隔；约 0.5s 的关键帧不是高频视频。stride=2 的约 1s 间隔可能触发 max_frame_gap_s 重置，统计明确保留这种行为。已有 results_nusc.json 是历史完整模型输出，回放不再执行 EVA/Head/LLM。Map 历史完整结果未保存，因此另用真实 OpenLane 标注在 commit 后建立 Map 状态；不冒充模型 Map 推理。

本地 sample.json 的 anns 列表被过滤过：scene-0101 首样本只列 3 项，但 sample_annotation.json 按 sample_token 反查有全部 43 项，与 ego Info 一致。Oracle 使用反查关系，身份来自真实 instance_token。单元测试覆盖该异常，避免错把未知身份的预测当作误匹配。关联 precision 仅统计两端可评分的提议并单独列出 unscorable；recall 分母为两帧都能可靠匹配到的 GT 实例交集。ID Switch 是这些观测实例被分配到另一历史 track 的次数，属于本回放定义，不等于官方 nuScenes tracking 指标。

物理 Oracle：上一完整输出按同类别、中心距离≤2m 一对一匹配 GT；当前存在同一 instance、有传感器支持、GT visibility=4 且历史几何外推中心误差≤1m 才算物理有效。上一身份未知的试验不进入 False-Reuse 分母。这个判据没有验证深层特征误差、精确 orientation/shape 或 Transformer 依赖；报告中的 False-Reuse 是物理状态候选的错误接受率，不能推导 Feature 精确可复用。接受数为 0 时错误率为 null，不是 0%。

Geometry Only、固定历史 crop 的 Appearance Only、Geometry+Observation 统一在离线 Oracle 上评分。固定图像坐标、描述相似性和 World Association 是独立的关联对照。不能将未知背景视为负例：新入相机视野的标注 Box 评估失效 Mask 的覆盖召回；无可靠稳定背景/开放世界负标注时 semantic FPR 为 null，报告 background update ratio。严格模式全图未知会有高更新率，召回高不代表具备语义新物体检测能力。

校准强制 train infos 和 split=calibration；测试/评估运行不能用于调阈值。按 scene/anchor 去重，使用最小接受数和阈值多重比较修正；单场景相关性仍限制二项上界解释。当前训练子集不足以支持设定风险，输出 threshold=null、insufficient_evidence，不回填配置。即使未来生成阈值，M1 也检查采样/图像预处理 protocol；raw-resized 校准不能直接套到模型 cropped/stretched 输入。Geometry/coverage 等阈值仍 provisional。

profile.json 是真实两帧输入的重复 CPU 微基准，分别测 Projection、Association、Observation、Unexplained、State Update 和 Total；分配跟踪另开一轮，不混入主计时。tracemalloc 只覆盖其可见的 Python/NumPy 分配，不能替代全部 C 库分配或 GPU 显存；进程峰值 RSS 包含加载的 Info/JSON，不能冒充纯 State Store 内存。回放 timings.json 在 tracemalloc 开启下测量，应优先用独立 profile 评估耗时。

完整模型 GPU 数值回归状态见交付报告。目前仍缺：独立的当前可见性/表面证据、跨更多场景/天气的可靠阈值、深层特征 Oracle、完整 M2 依赖闭包和任何实际 GPU 跳算。相似纹理、纹理不足、遮挡、几何预测误差、新物体未覆盖和长时间外推均可能导致漏检或错误候选，因此保持周期刷新提示与保守失效。M1 本次没有测得或报告端到端模型加速。

2026-10-15 宿主 GPU 验证已完成：RTX 3090，驱动 580.173.02，PyTorch 1.13.1+cu117。3 帧完整模型 Baseline/Observe-Only 的 Backbone、Detection/Map 输出与 Temporal Memory 数值摘要一致，峰值 PyTorch allocated 显存均约 15.119 GiB；详见 model_smoke/results.json。此单次检查未用于宣称速度提升。

## 第二轮完善

新增同源RGB时序可见性证据、局部支持和精确失效掩码导出，默认通过独立配置开启。接口、复现与边界见 [worldstream_m1_refinement.md](worldstream_m1_refinement.md)，实际风险/覆盖/成本结论见 [第二轮报告](../outputs/module1_refinement/refinement_report.md)。原配置继续复现第一轮严格行为。
