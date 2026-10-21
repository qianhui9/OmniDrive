# M1 DQTrack 桥接实现记录（2026-10-21）

已实现可选的 commit 阶段轨迹桥接、严格帧/坐标/命名空间校验、保守的一对一对齐、未认证的外部身份支持及冲突拒绝。默认 off，不替换 M1 anchor ID，不增加物理认证或认证观测计数，不产生可靠速度，不改变现有模型或 GPU 执行。未运行真实 DQTrack 推理，未证明关联风险或任务收益。

## 代码与文件

| 文件 | 作用 |
|---|---|
| `worldstream/m1/dqtrack_bridge.py` | 标准 JSON/显式 manifest、惰性缓存、字段/帧/世界坐标契约、原几何 Hungarian 对齐、竞争拒绝、唯一外部身份支持及冲突诊断 |
| `worldstream/m1/config.py` | 末尾追加 mode/path，默认 off，保留位置参数兼容 |
| `worldstream/m1/state_types.py` | WorldAnchor 末尾追加独立 external_tracking；不改变 PerceptionAnchor/FullOutputs 的生产者接口 |
| `worldstream/m1/state_manager.py` | 只在实际感知可用的 commit 内读取跟踪，shadow 只诊断，hint 可支持/拒绝未认证延续 |
| `worldstream/m1/state_store.py` | 外部元数据事务 fork 深复制；非空数据纳入既有状态 digest，空值保持旧 digest |
| `worldstream/m1/provenance.py` | 仅存在外部元数据时加入防御性审计导出 |
| `configs/worldstream_m1_dqtrack_shadow.yaml` | balanced 参数、shadow、独立日志路径 |
| `configs/worldstream_m1_dqtrack_hint.yaml` | balanced 参数、hint、独立日志路径 |
| `tools/worldstream_m1/inspect_dqtrack.py` | 全文件大小/SHA256；静态 pickle opcode 检查 checkpoint 字符串键及配置，不执行 pickle |
| `tools/worldstream_m1/prepare_dqtrack_results.py` | 官方标准结果与 sample.json 合成显式帧契约；不是模型推理 |
| `tools/worldstream_m1/evaluate_dqtrack.py` | 同批真实 OmniDrive 候选的三模式 M1 replay；GT 仅离线读取，报告相关样本及有限指标范围 |
| `tests/worldstream_m1/test_dqtrack_bridge.py` | 22 项桥接、模式、认证边界、事务隔离、工具与硬门限测试 |
| `docs/worldstream_m1_dqtrack.md` | 完整格式、启用方法、复现命令和边界 |
| 本目录 `checkpoint_inventory.json` | 15 个原始文件的完整路径、大小、SHA256、静态参数键和内嵌配置线索 |
| 本目录 `architecture_summary.json`、`checkpoint_configs/*.py` | 从 checkpoint 元数据提取的 7 份完整配置及架构摘要；仅文本提取，未执行配置 |
| 本目录 `check_off_baseline.py`、`original_audit.json`、`off_audit.json` | 修改前 HEAD 与当前 off 的完整状态审计等价检查（去计时） |
| 本目录 `real_fallback_smoke.json` | 已暴露真实场景 2 帧缺失轨迹降级 smoke；不是 DQTrack 对齐结果 |

保留了用户原有未跟踪目录 `outputs/m3_state_guided_verified_decode/`，未覆盖已有实验目录、报告或校准记录。没有提交或推送 Git。

## 实际检查

已读取用户指定 M1 核心源码、balanced/coverage 配置、optimization/coverage 文档、M1 总览及测试；追踪 `Petr3D` 初始化、precheck、实际感知适配、commit 和 nuScenes 数据集元数据。仓库文件检索未找到可用的论文正文；发现的 LaTeX 文件为既有图表 include。

`nuscenes_dataset.py` 的实际实现明确：sample_idx=info['token']，scene_token=info['scene_token']，timestamp=info['timestamp']/1e6。M1 Frame 的 frame_id 来自该 sample_idx；不是 frame_idx。规范化工具从 nuScenes sample.json 的 token/scene_token/timestamp 生成映射。其他调用方必须传入它自己经过验证的映射，不能根据整数索引推断。

原始目录 `/data/qh/phdCode/work4/models/DQTrack/` 共 15 文件：9 个 .pth、6 个 .pkl infos，全部计算了 SHA256。infos 只读字节哈希，**没有反序列化**。checkpoint 使用 zip 内 data.pkl 的静态 opcode 字符串提取，**没有执行 torch.load/pickle.load**。完整文件清单与参数键在 inventory。

以下架构来自内嵌 `model = dict(type=...)`，不是根据名称猜测：

| checkpoint basename | 内嵌架构 |
|---|---|
| bevstereo_r101_fb3_size_512_1408_finetune_amota_40_7.pth | StereoTrackerDualQuery |
| bevstereo_r50_fb3_size_512_1408_finetune_amota_36_7.pth | StereoTrackerDualQuery |
| detr3d_res101_size_900_1600_finetune_amota_36_7.pth | DETR3DTrackerDualQuery |
| petrv2_vov99_fb3_320_800_finetune_amota_44_6.pth | PETRTrackerDualQuery |
| uvtr_r101_size_900_1600_finetune_amota_39_6.pth | UVTRTrackerDualQuery |
| bevstero_r101_fb3_size_512_1408.pth | StereoTrackerDualQuery，train_det_only=True |
| bevstero_r50_fb3_size_512_1408.pth | StereoTrackerDualQuery，train_det_only=True |
| uvtr_c_r101_h11_convert.pth / uvtr_c_r50_h11_convert.pth | 有参数键但无内嵌模型配置；精确架构/配置未确认，不能据文件名用于运行 |

五份 model_val 配置具有 NuScenesTrackDataset、tracker_cfg、track_frame=3、训练 pipeline、模型与测试 pipeline 等明确线索。配置全文和参数键均已保留。没有执行配置或检查 state_dict 与某个官方模型实例 strict-load 的兼容性，因此不声称已完成可运行的模型装配。

搜索 `/data/qh/phdCode`、`/home/qh` 的 DQTrack 相关目录/文件，并列出 `/home/qh/miniconda3/envs`；仅找到权重目录及本轮新增桥接文件，没有找到匹配的官方实现、官方测试入口或已验证的独立 DQTrack 环境。其他任意命名目录/环境是否藏有实现尚无法排除。当前缺少上述自定义 TrackerDualQuery 实现的匹配来源；不能仅凭 checkpoint 和 OmniDrive 环境运行官方推理。没有下载权重、安装旧 MMCV/MMDetection3D 或训练模型。

## 数据与运行边界

世界系 gravity center、wlh 尺寸、wxyz quaternion、世界 XY 速度；桥接保留完整 3D 旋转，不重复应用 ego pose。manifest 绑定 run_id、模型配置、M1 frame → sample_token、scene_id、绝对秒时间，场景严格一致且时间差≤1微秒。未知坐标契约整体拒绝。ID 命名空间为 run+scene。速度只校验，不提升可靠性。

对齐沿用原类别/中心距离/尺寸门限和 Hungarian；有任一额外合法竞争边即拒绝外部对齐。hint 只支持原来已接受且外部关系唯一的边，不能救回几何歧义、超距、过期、跨场景或重现边。已对齐 ID 明确冲突可拒绝**未认证**延续；独立物理认证有优先权。缺失/错误数据退回原行为。没有使用 GT 在线判断。

启用：原 balanced/coverage 配置不变即 off。使用新增 shadow/hint YAML，并设置 dqtrack_results_path 到真实规范化结果。实际模型配置沿用 `model=dict(m1_config='configs/worldstream_m1_dqtrack_shadow.yaml')`（或 hint）。完整 schema 与命令见 `docs/worldstream_m1_dqtrack.md`。

## 实际运行的验证

所有执行均使用 `taskset -c 0-3,6-31`；Python 为 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。未使用故障 CPU 4/5，未停止其他 GPU 进程。

- 执行 `python -m tools.worldstream_m1.inspect_dqtrack --root /data/qh/phdCode/work4/models/DQTrack --output /tmp/dqtrack_inventory_20261021.json`，然后将新产生的 inventory 存入本目录。
- 新增测试 `python -m pytest tests/worldstream_m1/test_dqtrack_bridge.py -q`：22 passed。
- 完整测试最终结果见本目录 `full_tests.log`（后续在验证完成时追加结果）。早期完整运行发现配置位置参数兼容问题，已修正；随后中间版本完整测试通过。最终代码测试独立重跑，日志单独保存。
- `git diff --check`：通过。
- M1Config.load 读取 balanced、coverage、新 shadow、新 hint 四个 YAML：通过，模式依次 off/off/shadow/hint。
- 两个工具模块 `--help` 入口：通过。规范化和三模式评估函数已由新增测试实际执行。
- 修改前 HEAD 的 worldstream 在 `/tmp/omnidrive_m1_dqtrack_original_20261021` 只读源快照加载，与当前 off 跑同一组合成两帧，完整审计剔除计时后 `diff -u` 无差异（含 source_digest、状态版本、配对和认证）。检查脚本及双方结果已保存。

真实数据只进行了**缺失 DQTrack 文件**时的回退验证：

- 场景 `08110c8640454affb5167801442a85fd`，2 个既有关键帧，来源 `outputs/module1_evidence_optimization/model_observations.pkl`，infos `data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl`；真实 RGB/位姿/内参与历史完整模型检测。
- off/shadow/hint 各处理同样 2 帧；可用 DQTrack 轨迹 0、对齐 0、对齐率 null。shadow/hint 各记录 2 次文件不可用；双方配对与认证均与 off 完全相同。
- 本次三模式 replay 耗时分别 318.61/216.16/154.70 ms（2 帧总墙钟，含 M1，不含图像预加载）。固定次序/首次初始化/CPU 共享会影响计时，不表示 hint 加速。
- 同场景相邻帧相关，场景曾被查看；仅开发 smoke，不是独立确认。没有读取 GT，本轮错误延续/identity switches 为 null，没有错误率/有效召回/风险保证。
- 没有运行新 OmniDrive GPU 前向或真实 DQTrack 前向；GPU 可用不解决源码和匹配环境缺失。

## 尚未完成与下一步

真实 DQTrack 官方推理、真实轨迹对齐率、身份错误改善、有效召回与端到端计算成本尚未测量。阻塞为匹配的官方源码/独立运行环境及真实标准 tracking 输出缺失。两份 converted UVTR 预训练文件精确配置也仍未知。首先应取得与提取配置一致的官方源码 revision，在独立环境检查依赖并 strict-load 已有 checkpoint；保留原始权重，不下载重复权重、不训练。

随后使用官方环境的**实际测试入口**对连续帧导出 nuScenes tracking JSON。由于本机没有该入口，本轮不编造其命令。已提供的后处理/评估命令可立即用于真实结果：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.prepare_dqtrack_results \
 --results /path/to/official_tracking.json --sample-json data/nuscenes/v1.0-trainval/sample.json \
 --model-config /path/to/verified_config.py --checkpoint-sha256 ACTUAL_SHA256 \
 --run-id UNIQUE_RUN_ID --confirmed-nuscenes-world-output --output /tmp/new_normalized_tracks.json

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.evaluate_dqtrack \
 --infos data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl --data-root data/nuscenes \
 --detections /path/to/omnidrive_results_nusc.json --trajectories /tmp/new_normalized_tracks.json \
 --scene SCENE_TOKEN --frames 8 --with-gt --output /tmp/new_dqtrack_eval.json
```

GT 只能进入离线评价，工具指标是部分 GT 匹配而非官方 tracking 分数。使用既有场景须继续标注 exposed development；要主张独立验证，必须另外冻结从未查看的场景清单，按场景/实例处理依赖性。不得将本轮单元测试替身或缺失文件 smoke 解释为真实 DQTrack 效果。

## 最终验证结果

最终代码完整测试：**557 passed, 19 warnings in 129.72s**，无失败；警告为现有依赖弃用提示。新增测试 **22 passed in 0.22s**。日志分别为 `full_tests.log`、`new_tests.log`。`git diff --check` 通过，四份 YAML 配置加载通过，源码哈希与 `source_manifest.json` 一致。
