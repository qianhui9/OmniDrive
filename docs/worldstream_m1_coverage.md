# M1 稠密观测支持与覆盖率验证

推荐的实验入口为 `projects/configs/OmniDrive/worldstream_m1_coverage_observe.py`，配置为 `configs/worldstream_m1_coverage.yaml`。它保留完整 GPU Forward 和原模型权重，只增加 CPU 观测验证。结果见 [最终报告](../outputs/module1_refinement/refinement_report.md)。默认 M1 配置仍关闭稠密支持，因此旧配置行为保持兼容。

## 机制与证据边界

原严格模式没有外部 visibility / geometry support mask，因此无法认证锚点与背景；全图刷新是其失败关闭行为。后续 sparse M1 使用双向 LK 与 7×7 实测 patch，但支持分散，并且单点附近使用相同位移可能跨越运动/遮挡边界。

新增 `worldstream/m1/dense_support.py`：

1. DIS FAST 分别计算前向、后向光流，只作为对应提议。
2. 在当前每个整数像素读取历史坐标，要求前后向误差 ≤0.35px，坐标在界内，局部 Jacobian 在 [0.5,2]，防止折叠和过度压缩。
3. 使用已知相机位姿和内参，向量化解两条视线的深度；要求条件良好、双侧正深度 0.5–200m、双向重投影误差 ≤0.5px。平移小于 2cm 时使用受限旋转投影，不能从中推导独立深度。
4. 每个像素实际采样比较：灰度误差 ≤0.02、梯度误差 ≤0.04；双图局部最小特征值须支持二维纹理。没有自动亮度补偿，也不填充无纹理背景。
5. 完整邻域均通过才保留中心像素。邻域保护使用腐蚀，仅删除证据，绝不将稀疏支持膨胀成稳定区域。
6. 历史动态对象排除方向性运动扫过范围及已有运动不确定性。采用原始、未裁剪的投影角点，最后才与图像求交；预测速度仅用于扩大拒绝区域，不认证动态状态。

稠密路径替代局部 stable mask，不与弱 sparse mask 简单取并集。原稀疏锚点验证和稠密像素拒绝范围分别保存，避免局部像素保护无意更改物理锚点候选的统计分母。场景切换、历史过期、标定不可靠、纹理不足、边界和无充分证据的区域均继续刷新。没有增加多级状态分类或加权分数。

最终计算范围优化将排除区域计算移入相机工作线程。对已确定刷新的像素省去射线求交，仅为未排除像素及其完整 guard 邻域计算几何；整个相机都被排除时跳过 DIS。这里对计算范围保留邻域，不是膨胀稳定支持。`roi_verification.json` 保存最终在线代码 SHA256，以及 768 帧次实际回放的精确输出一致性；`source_before_roi/` 保留优化前源码。此优化主要省去无用的几何计算，通常仍须执行整幅 DIS，不保证显著加速。

`OnlineVisibilityEvidence.dense_source_map` 是当前像素到最近一次真实 commit 图像的浮点坐标；只有 `~last_masks[camera]` 为真的位置才可视为观测候选。未支持位置仍可包含光流提议，不能直接复用。每次 precheck 替换该数组，不积累跨帧地图；六相机 640×360 坐标映射约 10.55 MiB，640×640 约 18.75 MiB，另有光流计算临时数组。它们是 CPU 观测数据，不计入旧的深层 tensor retention 配额。

## 新配置

| 配置 | 默认 | 最终实验值 |
|---|---:|---:|
| `dense_local_support` | false | true |
| `dense_guard_radius_px` | 4 | 8 |
| `dense_motion_guard` | false | true |
| `temporal_workers` | 1 | 6 |

启用稠密支持需要同时开启 `temporal_visibility` 和 `local_support`。灰度、几何与双向对应门限比旧 sparse 路径更严格；没有为了提高覆盖率放宽门限。线程数来自此前同机实验，其他机器应重新测量，防止嵌套线程过量。

## 实验隔离与结果位置

- `coverage_round3/`：8 个校准场景、16 个初始验证场景和第一轮 GPU/特征实验。首选 guard=2 在初始验证上出现关键 LiDAR 点漏检增加，保留为失败候选。
- `coverage_confirmation/`：各方向扩张运动范围的开发尝试，因过度刷新在进入确认场景前停止。
- `coverage_final/`：方向性保护的 24 场景开发集和新 16 场景确认集。`selection.json` 在查看确认结果前冻结 guard=8。
- `coverage_final/holdout_before_scope_fix/`：发现像素保护污染锚点证据作用范围后的原记录；修复不改变任何像素 mask，也没有改变选择参数。
- `coverage_final/holdout/`：作用范围修复后重新实际执行的全部确认场景。`scope_fix_verification.json` 审计 960 帧次，mask 完全一致，锚点判断与已有 sparse 路径完全一致。
- `coverage_final/gpu/`：计算范围优化前的完整 GPU 对照和特征误差。
- `coverage_final/gpu_roi/`：最终代码重新执行的完整 GPU 对照和特征误差。计时不包含 feature oracle 的大规模拷贝开销。
- `coverage_final/roi_timings.json` 与 `warmed_latency.json`：最终代码的全确认集回放耗时、预加载交替次序计时；优化前预加载结果保存在 `warmed_latency_before_roi.json`。
- `coverage_final/figures/`：Risk–Coverage、Invalidation–Recall、Runtime Breakdown、真实图像 mask 和实际特征误差 PNG/PDF。

原始严格 M1、Geometry Only、Appearance Only 的锚点判定在相同历史状态上对比；后两者只是消融，不具有可信像素面积。物理状态 Oracle 沿用同实例、visibility=4、传感器支持、中心误差 ≤1m，没有降低要求。

新增 LiDAR Oracle 独立于在线 RGB：在 precheck 完成后读取当前/历史点云和标注，通过世界坐标与实例刚体运动建立参考对应。当前/历史深度差 ≤0.5m 才进入可评分对应集合；误差 >2px 或实际移动却接受为静态，计为错误。关键点与关键物体区域分别统计召回。点云稀疏、不同传感器时间偏差及未标注点静态假设都限制了它的效力；不能将采样错误率外推为全像素保证。

## 运行

```bash
conda activate OmniDrive
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/omnidrive-mpl
python -m pytest tests/worldstream_m1 -q
```

重新执行冻结的覆盖率实验（使用新目录）：

```bash
python -m tools.worldstream_m1.refine_coverage \
  --root outputs/module1_refinement/reproduction_coverage \
  --confirmation
```

这是在已经公开的场景上复现，不构成新的独立确认。程序拒绝覆盖已存在的 protocol。当前代码已修复作用范围问题，复现无需再运行 `verify_coverage_scope.py`；后者专用于审计本轮保存的中间实现。

完整 GPU 对照及特征采样：

```bash
python -m tools.worldstream_m1.gpu_refinement \
  --output outputs/module1_refinement/reproduction_coverage/gpu \
  --config configs/worldstream_m1_coverage.yaml \
  --reference-root outputs/module1_refinement/coverage_round3 \
  --reference-config configs/worldstream_m1_balanced.yaml \
  --frames 8 --warmup 1 --repeats 2
```

基线源代码快照经过 SHA256 校验，确保旧源码不会被当前实现替换。每轮新的语言输出目录保证 LLM 实际执行。`gpu_refinement` 中 baseline 表示不开 M1，reference 表示已有 sparse M1，balanced 表示传入的覆盖率配置；名称沿用工具，不代表本轮做了加权评分。

汇总报告和图表：

```bash
python -m tools.worldstream_m1.report_coverage \
  --root outputs/module1_refinement/coverage_final
```

## 后续 M2 接口

`verified_stable_area_ratio` 表示在线规则支持的观测面积；扣除 `full_refresh_hint` 后真正的图像候选面积是 `1 - invalidation_area_ratio`。二者不能混用。完整 16×16 图像块用于衡量连贯空间支持，与实际预处理后的 EVA token 并非一一对应。

尽管实际完整 Forward 的特征误差可以很小，空间重采样、patch 足迹、窗口/全局注意力和 carrier 身份仍未认证。因此本轮始终输出 `transformer_dependency_certified=false`，精确深层 token 复用认证数量为 0，也没有实际 GPU 跳算。M2 必须另外验证所选层的依赖和误差预算，再衡量节省量是否超过 M1 的额外开销。
