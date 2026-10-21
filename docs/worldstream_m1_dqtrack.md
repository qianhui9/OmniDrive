# DQTrack → M1 可选关联桥接

默认 `off`。模型入口 `Petr3D` 在初始化创建 `WorldStateManager`，前向前建立 Frame 并 precheck；真实检测与地图输出经 `outputs_from_model()` 变成世界坐标候选，随后 commit。桥接仅在 `_commit()` 完成候选合法性筛选之后执行，不在适配器、precheck 或历史证据生产器中读当前跟踪结果。原检测器、Head memory、VLM、M2/M3 执行策略不变。

初轮检查记录保留在 `outputs/m1_dqtrack_bridge_20261021/experiment_report.md`。源码已接入并完成两帧真实 DQTrack 推理，续报见 [源码接入文档](worldstream_m1_dqtrack_source.md) 和 `outputs/m1_dqtrack_source_20261021/experiment_report.md`；仍无独立风险收益结论。

## 模式与关联边界

- `off`：不访问结果路径；原关联、认证及默认审计结构保持原行为。新增空字段不进入旧事务 digest。
- `shadow`：读取和校验当前帧，计算对齐、竞争和拒绝诊断，写入 `last_commit_diagnostics['dqtrack']`；不写外部 ID 到状态，不改变提交匹配或认证。
- `hint`：仅将无竞争的一对一对齐结果写入对象锚点 `external_tracking`。跨帧原有几何匹配与原有拒绝门限全部通过、外部 ID 在历史状态中唯一且一致时，未认证关联明确记录为 `uncertified_dqtrack_corroborated_geometry`。同运行/场景中已对齐 ID 冲突可拒绝未认证延续；不存在可比 ID 时回退几何流程。已有独立物理认证保留优先权。

当前版本不尝试用 ID 解除几何歧义，也不重新分配被原关联拒绝的边。它提供保守的支持/冲突拒绝，而非扩大关联召回。外部 score 仅保留用于诊断；不参与身份概率、认证、速度可靠性或物理计数。DQTrack 速度不写入对象运动状态。地图始终不接收跟踪提示。场景/长时间间隔 reset、过期、重现拒绝和一对一事务检查保留。

外部元数据在事务 fork 和审计导出时防御性复制，并由状态 digest 绑定。ID 命名空间为 `[run_id, scene_id]`；M1 `anchor_id`、epoch、几何版本和认证来源仍由原系统管理。

## 数据契约

采用 nuScenes tracking JSON 的 `results: {sample_token: [row, ...]}`，每行必须提供：

```json
{"sample_token":"sample-token", "tracking_id":"17", "tracking_name":"car",
 "tracking_score":0.8, "translation":[0,0,1], "size":[2,4,1.5],
 "rotation":[1,0,0,0], "velocity":[0,0]}
```

`translation` 为世界/global 坐标中的 gravity center（米），`size` 为 `width,length,height`，`rotation` 为单位 quaternion `w,x,y,z`，速度可缺省，提供时必须为有限的世界 XY m/s。桥接保留完整三维旋转，尺寸转换为 OmniDrive/M1 的 `length,width,height`；绝不再次应用 ego pose。无法证明是世界坐标的输出必须先在独立生产环境中验证/转换；桥接拒绝 ego/未知约定，不猜测变换。

文件必须包含显式 `m1_manifest`：

```json
{"coordinate_system":"world", "size_order":"wlh", "timestamp_unit":"seconds",
 "run_id":"unique-tracker-run", "model_config":"verified/config/path.py",
 "checkpoint_sha256":"checkpoint digest recorded by exporter",
 "frames":{"M1-frame-id":{"sample_token":"sample-token", "scene_id":"scene-token",
                           "timestamp":100.0}}}
```

`frames` 是确定的帧映射，不能用数字 frame_idx 冒充 token。当前 OmniDrive 数据集在 `nuscenes_dataset.py` 明确执行 `sample_idx=info['token']`、`scene_token=info['scene_token']`、`timestamp=info['timestamp']/1e6`；适配器用 sample_idx 创建 M1 frame_id。因此本仓库实际模型入口可用 sample token 作为 frame_id。其他回放入口须提供自己的明确映射。映射 token 必须唯一、场景严格相等、时间差不超过 1 微秒；缺失轨迹帧、重复 ID、缺失字段和非法数值均拒绝。非法单行不影响其他合法行；帧契约错误使该帧整体降级。

对齐复用原 `associate()` 的类别、三维中心距离、尺寸比和 Hungarian。任一端存在另一条合法边即拒绝该对齐，比分差规则更保守，不放宽距离门限。几何拒绝记录为 `no_legal_class_distance_size_edge`，竞争记录为 `competing_legal_candidates`。不保证该阈值已校准身份风险。

结果文件在第一次 commit 时惰性读取，按路径/mtime_ns/大小缓存最多两份 JSON；同进程修改结果文件应产生新的 mtime。该版本面向离线文件交换，不是实时 DQTrack 调度器。大规模文件的首次解析与内存成本应另行实测。

## 配置

原 balanced/coverage 无需增加字段，仍为 off。显式关闭可设置 `dqtrack_mode: off`。新增示例配置：

```text
configs/worldstream_m1_dqtrack_shadow.yaml
configs/worldstream_m1_dqtrack_hint.yaml
```

两者继承 balanced 参数语义，结果路径默认为 `outputs/dqtrack/normalized_tracking.json`，日志路径各自独立。按需修改：

```yaml
dqtrack_mode: shadow  # 或 hint
dqtrack_results_path: /absolute/path/normalized_tracking.json
```

模型配置按现有 `projects/configs/OmniDrive/worldstream_m1_balanced_observe.py` 继承 observe 配置并设置 `model = dict(m1_config='configs/worldstream_m1_dqtrack_shadow.yaml')`（hint 则替换为对应路径）。不要复用已生成语言回答的目录作为 GPU 成本实验。

## 独立结果规范化与评估

没有匹配的官方源码/运行环境时不提供假推理入口。`prepare_dqtrack_results` 只规范化**已由官方环境导出的标准结果**，不会运行模型。调用者必须先确认官方结果的世界坐标约定，显式传入确认标志；工具使用 sample.json 元数据建立帧映射，不读取 annotation。`model_config` 与 checkpoint digest 是生产者来源声明，不是代码或 checkpoint 的真实性认证。

下面为待获得真实输出之后的命令，未在本轮执行真实导出：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m1.prepare_dqtrack_results \
  --results /path/to/official_tracking.json \
  --sample-json data/nuscenes/v1.0-trainval/sample.json \
  --model-config /path/to/verified_official_config.py \
  --checkpoint-sha256 CHECKPOINT_SHA256 --run-id UNIQUE_RUN_ID \
  --confirmed-nuscenes-world-output --output /tmp/normalized_tracking.json
```

原始权重 inventory 的复现（输出必须尚不存在）：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m1.inspect_dqtrack \
  --root /data/qh/phdCode/work4/models/DQTrack --output /tmp/new_dqtrack_inventory.json
```

离线评估对同一组**真实 OmniDrive 检测**按 off/shadow/hint 分别重放实际 M1。只接受一个场景的 1–40 帧小规模入口，输出拒绝覆盖。infos 使用项目现有 pickle loader，仅用于可信本地文件。示例命令尚未用于真实 DQTrack 结果：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m1.evaluate_dqtrack \
  --infos data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl \
  --data-root data/nuscenes --detections /path/to/omnidrive_results_nusc.json \
  --trajectories /tmp/normalized_tracking.json --scene SCENE_TOKEN \
  --frames 8 --config configs/worldstream_m1_balanced.yaml \
  --output /tmp/new_dqtrack_evaluation.json
```

可选 `--with-gt` 在在线 transaction 完成后离线匹配标注，统计可评分的错误延续和实例 ID 变换；这是有 2m 门限、传感器支持约束的部分 GT 匹配指标，**不是官方 nuScenes tracking metric**。未知标注对应不会计作正确。没有 GT 时错误延续/identity switches 为 null，不能用外部 ID 一致代替真实正确性。输出含帧列表、场景数、可用轨迹、对齐率、非法拒绝、未对齐原因、竞争、外部 ID 一致性、配对/认证基线一致性和墙钟。时序预热与固定模式顺序使单轮 elapsed 仅为 smoke 耗时，不能宣称速度优势。

既有校准、确认和诊断场景全部属于已暴露开发数据；默认报告明确为 `exposed_development_replay`，同场景相邻帧及同实例不是独立样本。不存在新的独立验证集或风险保证。

测试：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1/test_dqtrack_bridge.py -q
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 -q
taskset -c 0-3,6-31 git diff --check
```
