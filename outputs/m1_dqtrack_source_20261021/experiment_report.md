# DQTrack 源码接入与 M1 真实配对验证（2026-10-21）

本轮接续初轮桥接实现。用户提供了 `/data/qh/phdCode/work4/DQTrack-main` 源码后，已完成 DETR3D 源码模型构建、checkpoint 严格加载、官方数据流水线 CPU 校验、两帧真实 DQTrack 推理、同帧真实 OmniDrive 检测采集及 off/shadow/hint 配对回放。没有训练、安装旧依赖、修改原始源码或权重，也没有覆盖之前报告、提交或推送 Git。

**真实结果：22 条轨迹中 14 条成功对齐（63.64%），但此样本的 hint 没有改变最终配对，也没有降低部分 GT 指标下的错误身份延续。** 本报告是开发 smoke，不构成独立验证或风险保证。

## 本轮文件

| 文件 | 作用 |
|---|---|
| `worldstream/m1/dqtrack_export.py` | 官方 decoded tracklet 输出适配；LiDAR→ego→world 转换，尺寸/朝向/速度与 ID 校验，保留源索引 |
| `tools/worldstream_m1/run_dqtrack_source.py` | 独立进程选择导入原 DETR3D 源码；严格权重加载、CPU 数据检查、GPU 等待和小规模标准结果导出 |
| `tools/worldstream_m1/audit_dqtrack_source.py` | 五组新源码配置与 checkpoint 内嵌配置差异，以及源码哈希清单 |
| `tools/worldstream_m1/evaluate_dqtrack.py` | 保留 JSON 候选入口，新增互斥的可信本地 `--candidate-pickle`；复用真实 FullOutputs，明确 scene_count |
| `tests/worldstream_m1/test_dqtrack_export.py` | 13 项导出、坐标、索引保持、非法输入、受限权重加载、GPU 忙等待和导出→M1 测试 |
| `docs/worldstream_m1_dqtrack_source.md` | 源码路线、限制、真实结果与可复现命令 |
| `docs/worldstream_m1_dqtrack.md` | 更新当前状态并链接本轮续报；保留初轮历史记录 |
| 本目录 `source_config_audit.json` | 原始源码文件哈希、五种配置的逐字段差异 |
| `check_cpu_v4/run_report.json` | 最终受限加载器的 DETR3D 严格 CPU 加载证明 |
| `cpu_data_check_v2/run_report.json` | 两帧官方数据流水线校验，包括实际输入形状和 sample tokens |
| `gpu_smoke/normalized_tracking.json`、`gpu_smoke/run_report.json` | 真实模型导出及逐帧推理/导出耗时，非替身结果 |
| 本目录 `normalized_tracking.json` | 真实结果经 sample.json 再次校验、独立 run_id 规范化后的 M1 输入 |
| `omnidrive_candidates/` | 冻结两帧选择清单、原 OmniDrive checkpoint 的真实检测候选与来源记录 |
| `paired_evaluation_unique_run.json` | 最终三模式配对结果；`paired_evaluation.json` 保留首次结果 |
| `runtime_environment.json`、`source_manifest.json`、测试和运行日志 | 环境、最终实现哈希、验证及复现证据 |

初轮核心 M1 修改、外部 ID 与物理认证分离及默认 off 语义保留；本轮没有放宽原几何门限或身份拒绝条件。用户原有其他输出目录保留。

## 实际源码与权重核对

新源码的注册类名为 `DETR3DTrackerDQ / UVTRTrackerDQ / PETRTrackerDQ / StereoTrackerDQ`；已有 checkpoint 配置中的类名为旧的 `*TrackerDualQuery`。不能仅改名作为兼容证明。`source_config_audit.json` 记录了五组 model 配置的 8/3/9/7/5 处差异：包括类型名、旧字段、DETR3D 的 eval_det_only、PETR 的阈值和 grid mask 等。

实际使用：

- 配置 `projects/configs/dqtrack/detr3d_r101_size_900_1600_dq.py`，模型 `DETR3DTrackerDQ`，tracking 模式 `eval_det_only=False`。
- checkpoint `/data/qh/phdCode/work4/models/DQTrack/model_val/detr3d_res101_size_900_1600_finetune_amota_36_7.pth`。
- SHA256 `93213fa9896ba1d161976c26fd512dc9e25f45c70c5d63c2dade31e90a8ec3d4`，与初轮清单一致。
- **982 个 state_dict 条目经 `strict=True` 加载成功**。没有手工别名、丢弃未知参数或 non-strict 回退；MMCV 执行了内置 DCN state 版本升级。

原始插件 `__init__` 全量导入时，即使选择 DETR3D 也会导入 stereo 并因 `voxel_pooling_ext` 缺失失败。独立入口只导入经检查的 DETR3D registry 路径，未编译或安装无关扩展，未修改用户源码。

原模型构造函数把无参数的类别查表硬编码到 CUDA；入口在 CPU 构建阶段只更改其分配设备，并在真实运行前将该查表移至 CUDA。模型参数不变。配置读取使用源码自包含的 model/test pipeline/fp16 字段，显式提供 infos 和 data root，没有凭空补造缺失的 mmdetection3d 基类配置。

该 checkpoint 的四类 pickle GLOBAL 已静态核对。Torch 1.13 weights_only 无法读取其中 BINFLOAT 元数据，且该版本 torch.load 会覆盖传入的 pickle_module；最终入口直接用受限 Unpickler 和 tensor archive reader，只允许已核对的 tensor/OrderedDict globals，拒绝动态 global/扩展和 TorchScript，无无限制回退。单元测试验证了可执行 global 被拒绝。中间失败/旧尝试目录保留，以 v4、CPU data v2 和 GPU smoke 的最终记录为准。

其他四种 tracker 的配置已审计，但**没有声称其权重严格加载或真实推理已通过**。

## 环境与数据契约

实际 GPU 入口 `/tmp/dqtrack_source_env_20261021/bin/python` 是 `--system-site-packages --without-pip` 虚拟环境，只读继承现有依赖：torch 1.13.1+cu117、mmcv-full 1.6.2、mmdet 2.28.2、mmdet3d 1.0.0rc6。没有安装新包或更改 OmniDrive 环境。它提供独立解释器/插件命名空间，依赖尚非完全独立锁定；不能将其称为原 README 旧版本环境的全面复现。

官方源码明确 `sample_idx=info['token']`，结果按对应 data_infos token 导出。真实 infos 为 `/data/qh/phdCode/work4/models/DQTrack/infos/track_cat_10_infos_val.pkl`（6019 个条目，metadata.version=v1.0-trainval）。只选了一个 scene 起点开始的两帧连续前缀；逐帧与 sample.json 检查 token、scene_token、timestamp 和 prev 链，没有读取 GT 进入在线 tracker/M1。

- 场景：`01452fbfbf4543af8acdfd3e8a1ee806`。
- 帧：`d5324133e1d241658a7b763831b3af2e`、`0f5cbb909e9544dd861dc2ff63ac4d13`。
- 官方 CPU pipeline 形状均为 `[1, 1, 6, 3, 928, 1600]`。
- decoded LiDAR 框使用 gravity_center、dx/dy/dz 和 yaw；输出经过 lidar2ego、ego2global 转换为 global/world，标准 wlh 尺寸、wxyz 四元数和 world XY 速度。M1 不再次变换这些结果。
- 官方 formatter 先 range-filter boxes 再按过滤后的序号取原 track_ids/track_scores，存在索引错配路径；新适配器逐源索引选择，回归测试覆盖“第一个远框被过滤，第二个框仍保留正确 ID/分数”。
- 原始结果命名空间为该次 smoke 名；最终标准文件通过真实 sample.json 规范化，赋予本轮独立 run_id。后续运行入口默认生成 UUID，避免仅用目录 basename 跨运行冲突。

## 真实推理与三模式结果

等待其他 GPU 任务释放资源后才运行 DQTrack，未停止任何其他进程。真实 DQTrack 推理使用已有权重；随后释放后使用原 OmniDrive `ckpts/iter_10548.pth` 的现有采集入口，对同样两帧采集实际检测候选。采集关闭 M1/M2/M3 与语言生成，只获取原检测器输出；没有更换检测器权重，不代表完整 VLM 端到端时延。

| 帧 | DQTrack 轨迹 | 对齐成功 | 竞争拒绝提议 | OmniDrive 对象候选 |
|---|---:|---:|---:|---:|
| d532… | 13 | 7 | 3 | 94 |
| 0f5c… | 9 | 7 | 2 | 78 |
| 合计 | 22 | 14 | 5 | 172 |

22 条轨迹中 8 条未对齐：5 条竞争、3 条无合法类别/距离/尺寸边；非法行拒绝 0。对齐率 **63.64%**。6 个跨帧外部身份关系可比较，6 个一致、0 个冲突。这只是外部关联诊断，不是物理身份概率。

| 模式 | 错误延续 / 可评分延续 | 身份切换（部分 GT oracle） | 与 off 配对一致 | 与 off 认证一致 | 两帧 M1 replay 墙钟 |
|---|---:|---:|---|---|---:|
| off | 3 / 27 | 7 | — | — | 306.01 ms |
| shadow | 3 / 27 | 7 | 是 | 是 | 361.33 ms |
| hint | 3 / 27 | 7 | 是 | 是 | 368.07 ms |

GT 匹配只在 commit 后离线执行，要求类别兼容、传感器支持和 2m 门限；未知匹配不计作正确。这是部分 oracle 诊断，**不是官方 AMOTA/IDS，也不是对真实错误率的完整估计**。2 帧之间高度相关，仅 1 场景；统一标注 exposed development。没有独立留出集、显著性或风险保证。

当前 hint 只记录几何已接受且外部关系唯一的未认证支持，或拒绝明确冲突。本样本没有冲突，因此配对未改变，错误延续也未减少。没有通过扩大门限或强行解除竞争来追求更好指标。两帧实际物理认证数量均为 0。

成本：DQTrack 推理及导出两帧为 **803.29 / 182.93 ms**；OmniDrive 检测采集为 **918.32 / 545.80 ms**（各自首帧有初始化影响）。上述计时不包括 checkpoint 加载和全部图像预加载。最终 replay 内 shadow 桥接累计 3.84 ms、hint 累计 6.89 ms；固定模式顺序、缓存、CPU 调度和小样本限制使这些耗时不能作为稳定成本比较或加速结论。额外 DQTrack 前向成本必须单独计入系统预算。

## 实际命令与验证

所有 Python、测试和实验均以 `taskset -c 0-3,6-31` 启动，避开故障 CPU 4/5。宿主 GPU 检查与两次 GPU 工作通过宿主权限运行；检测到其他进程占用时只等待。

独立入口实际运行参数：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m venv --without-pip --system-site-packages /tmp/dqtrack_source_env_20261021

taskset -c 0-3,6-31 env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/omnidrive-dqtrack-mpl \
 /tmp/dqtrack_source_env_20261021/bin/python tools/worldstream_m1/run_dqtrack_source.py \
 --source-root /data/qh/phdCode/work4/DQTrack-main \
 --checkpoint /data/qh/phdCode/work4/models/DQTrack/model_val/detr3d_res101_size_900_1600_finetune_amota_36_7.pth \
 --output outputs/m1_dqtrack_source_20261021/gpu_smoke --run \
 --infos /data/qh/phdCode/work4/models/DQTrack/infos/track_cat_10_infos_val.pkl \
 --data-root /data/qh/phdCode/work4/OmniDrive/data/nuscenes \
 --scene 01452fbfbf4543af8acdfd3e8a1ee806 --frames 2
```

上述输出目录已存在，复现时必须换新目录。CPU 严格加载及输入检查的实际参数见相应 run_report.arguments。

原检测候选采集实际调用：

```bash
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/omnidrive-dqtrack-mpl \
 /home/qh/miniconda3/envs/OmniDrive/bin/python -c "from tools.worldstream_m1.run_dqtrack_source import wait_for_gpu; wait_for_gpu(60); from tools.worldstream_m1.collect_evidence_observations import main; main('outputs/m1_dqtrack_source_20261021/omnidrive_candidates')"
```

规范化、配对回放的输入参数见 `paired_evaluation_unique_run.json.inputs`；可直接复现命令在 `docs/worldstream_m1_dqtrack_source.md`，输出应另选新路径。规范化工具本轮已用于真实结果，不再仅限测试替身。

实际验证：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1/test_dqtrack_export.py tests/worldstream_m1/test_dqtrack_bridge.py -q
# 35 passed（13 项新增 + 初轮 22 项）

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 -q
# 570 passed, 19 warnings in 132.54s

taskset -c 0-3,6-31 git diff --check
# passed
```

19 条警告是既有依赖弃用提示。配置 balanced/coverage/shadow/hint 加载与两种工具 CLI 检查通过；原始配置保持 off。受限加载器测试包含可执行 global 的负例，临时标记文件未产生。

## 当前使用与剩余工作

M1 指向 `outputs/m1_dqtrack_source_20261021/normalized_tracking.json`，设置 dqtrack_mode=shadow 或 hint 即可在对应帧使用；默认 off 不变，文件仅覆盖本轮两帧，其余帧回退。详细运行命令、字段约定和配置加载方式见源码接入文档。

源码缺失阻塞已解除；DETR3D 的实际两帧推理和配对评估已完成。剩余包括其他四种源码/权重路线的独立装配验证（stereo 仍涉及扩展）、官方全量 tracking 评测、更多场景的成本与错误分析、严格独立的依赖锁定环境，以及冻结未暴露场景后的风险/有效召回研究。当前数据不支持扩大 hint 的接收范围或宣称关联准确率提升。未训练、未执行长时/全量实验，未实现深层特征复用或 GPU 跳算。
