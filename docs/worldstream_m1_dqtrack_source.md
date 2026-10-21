# DQTrack 源码接入与两帧真实验证

源码现已位于 `/data/qh/phdCode/work4/DQTrack-main`。本轮完成了 DETR3D 源码模型构建、982 项 state_dict 严格加载、真实连续两帧推理、坐标转换导出以及同帧 OmniDrive 候选的三模式回放。完整数据见 `outputs/m1_dqtrack_source_20261021/experiment_report.md`；初轮“缺少源码、未运行真实 DQTrack”的记录保留为历史记录。

## 提取与复用方式

`worldstream/m1/dqtrack_export.py` 实现依赖轻的 decoded-result 适配，依据官方 `NuScenesTrackDataset` 的 gravity center、dims、yaw、LiDAR→ego→global 和 tracking ID 约定，独立实现坐标转换。它接收 `tracklet2results()` 的真实输出，保持源数组索引；过滤远框时同步选择 ID、分数和类别。没有搬入模型依赖或将跟踪查询当作 M1 物理状态。

`tools/worldstream_m1/run_dqtrack_source.py` 是独立进程入口，直接导入用户提供的原始源码文件，不修改原始项目。当前只支持已实际验证的 DETR3D 路线。它选择导入所需 registry，避开无关 stereo 模块的全量导入；后者要求本机尚未编译的 `voxel_pooling_ext`。第三方源代码继续使用原目录的 LICENSE。

checkpoint 内的类名为 `DETR3DTrackerDualQuery`，新源码为 `DETR3DTrackerDQ`。入口使用新源码配置构建模型，用严格加载证明参数兼容，没有注册旧类名别名，没有忽略参数。MMCV 自身执行了已有 DCN 权重版本升级。CPU 构建阶段仅把原源码硬编码 CUDA 的无参数类别查表改为 CPU 分配；真正运行前移到 CUDA，参数和数值运算不变。该替换只存在于独立进程内。

PyTorch 1.13 的 weights_only 加载器不支持此 checkpoint 元数据中的 BINFLOAT；其 torch.load 还会覆盖显式 pickle_module。入口使用 PyTorch tensor archive reader，静态检查 pickle GLOBAL，只允许这份权重实际需要的 OrderedDict、FloatStorage、LongStorage、_rebuild_tensor_v2 四类 global，再使用受限 Unpickler；拒绝动态 global/扩展指令和 TorchScript。没有无限制 pickle 回退。该路径有拒绝可执行 global 的测试。

## 独立运行环境

实际 GPU 推理使用 `/tmp/dqtrack_source_env_20261021/bin/python`：从 OmniDrive Python 创建 `--system-site-packages --without-pip` 虚拟环境，只读继承当前已安装的 torch/MMCV 等包；没有安装旧依赖或更改 OmniDrive 环境。它具有独立解释器入口和插件命名空间，但依赖版本没有完全独立锁定。版本记录见 `runtime_environment.json`。若未来需要安装官方旧版本，应另建完全独立环境。

源码 README 要求 mmdet3d rc3 / mmcv-full 1.4.0 / mmdet 2.24.0；实际此次验证使用 torch 1.13.1+cu117 / mmcv-full 1.6.2 / mmdet 2.28.2 / mmdet3d rc6。验证结论仅覆盖当前 DETR3D 两帧路径，不代表其他模型或官方完整评测兼容。

## 已执行与复现

所有代码执行都使用 `taskset -c 0-3,6-31`。需要宿主 GPU 访问；入口在分配 GPU 前检查物理 GPU 0 的其他 compute processes，最多等待 `--gpu-wait-seconds`（默认 60 秒），超时不启动推理，不停止其他进程。

创建轻量隔离入口（首次使用新的路径）：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m venv --without-pip --system-site-packages /tmp/dqtrack_source_repeat_env
```

验证与导出（`--output` 必须不存在）：

```bash
taskset -c 0-3,6-31 /tmp/dqtrack_source_repeat_env/bin/python \
 tools/worldstream_m1/run_dqtrack_source.py \
 --source-root /data/qh/phdCode/work4/DQTrack-main \
 --checkpoint /data/qh/phdCode/work4/models/DQTrack/model_val/detr3d_res101_size_900_1600_finetune_amota_36_7.pth \
 --output /tmp/dqtrack_repeat_export --run \
 --infos /data/qh/phdCode/work4/models/DQTrack/infos/track_cat_10_infos_val.pkl \
 --data-root /data/qh/phdCode/work4/OmniDrive/data/nuscenes \
 --scene 01452fbfbf4543af8acdfd3e8a1ee806 --frames 2
```

省略 `--run` 且添加 `--check-data` 时只做 CPU 严格加载、元数据和真实图像预处理，不进行前向。只检查模型时省略 `--run`、`--check-data` 和数据相关参数。

入口不依赖下载源码包中缺失的 mmdetection3d `_base_` 配置；它读取该官方配置自包含的 model、test pipeline、fp16 等字段，显式指定数据路径。只运行从 scene 起点开始的 1–40 帧连续前缀，逐帧校验 sample.json 的 scene/token/timestamp/prev。当前只支持 >=1.0 MMDetection3D 的 LiDAR box 约定。原始 infos 和 checkpoint 不修改，导出 JSON 已处于 world 坐标，无需额外 ego transform。

本轮真实标准结果为：

```text
outputs/m1_dqtrack_source_20261021/normalized_tracking.json
```

该文件经过 `prepare_dqtrack_results` 再次校验 sample.json，并设置独立 run_id。后续入口默认用 UUID 命名运行；不要跨运行复用 ID 命名空间。此文件仅包含上述两帧，其他帧安全降级。

使用本轮真实 OmniDrive 候选重放：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
 -m tools.worldstream_m1.evaluate_dqtrack \
 --infos data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl --data-root data/nuscenes \
 --candidate-pickle outputs/m1_dqtrack_source_20261021/omnidrive_candidates/model_observations.pkl \
 --trajectories outputs/m1_dqtrack_source_20261021/normalized_tracking.json \
 --scene 01452fbfbf4543af8acdfd3e8a1ee806 --frames 2 --with-gt \
 --output /tmp/dqtrack_repeat_paired.json
```

`--candidate-pickle` 只接受可信本地采集文件，沿用项目已有 FullOutputs pickle；原 `--detections` 标准 JSON 入口保持兼容，两者互斥。GT 在 commit 后只用于离线评价。

M1 启用设置：在现有 balanced/coverage 配置或新增 hint/shadow YAML 中设置 `dqtrack_mode: shadow` / `hint`，并将 `dqtrack_results_path` 指向上述文件。默认 off 不变，三种模式及认证限制仍遵守主文档。

## 结果边界

1 场景、2 个连续关键帧；DQTrack 导出 13+9=22 条轨迹，14 条对齐成功（63.64%）。5 个 Hungarian 提议因竞争拒绝，另有无合法几何边的轨迹；未对齐共 8 条。6 个可比较跨帧外部关联一致，0 个外部 ID 冲突。shadow 和 hint 的最终配对与认证均与 off 相同。

部分 GT 匹配指标下，各模式均为 27 个可评分延续中的 3 个错误、7 次实例对应 anchor ID 变换。本样本没有测得 hint 的身份改善；它是有传感器支持、2m 门限的部分 GT oracle，不是官方 AMOTA/IDS。轨迹 ID 一致性不能代替真实身份正确性。

此场景已经过本轮查看，统一标为 exposed development。两帧时序相关，不能宣称独立验证或风险保证。仍未验证 stereo、UVTR、PETRv2 的真实推理，也未运行官方全量评测、训练或全量 nuScenes 推理。
