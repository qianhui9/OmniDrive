# M1 关联与 M1→M2 实际接入：本轮交付

结论：代码和原生调用链已完成。本轮只恢复了保守 Hint 的 1 次正确延续，3 次错误未解决；自然图像没有复用，受控重复图像实际复用，但正常 Auto 无端到端收益。因此新关联开关保持默认关闭，不将控制实验作为自然场景加速证据。

## 代码差异

| 文件 | 实际作用 |
|---|---|
| `worldstream/m1/association_observation.py`（新增） | 复用实际图像描述子，在原始合法几何连通分量中处理竞争；只有完整、唯一、一对一且正向视觉支持的匹配才改配。缺失是未知。 |
| `worldstream/m1/config.py` | 新增默认 false 的 `association_observation_guard`，原距离、尺寸和状态期限未改。 |
| `worldstream/m1/dqtrack_bridge.py` | 新开关下，冲突需绑定最新历史帧、有效期及连续对齐历史；缺失、新运行命名空间、短历史不构成矛盾。原保守路径不变。 |
| `worldstream/m1/state_manager.py` | Commit 接入视觉提议及有 TTL 的外部身份；改配显式为 uncertified；扩展既有 `export_m2_inputs(expected_context)` 读取当前 pending 事务，核对场景/帧/时间/签名/epoch/来源摘要并防御性复制。 |
| `worldstream/m2/runtime.py` | `forward_from_m1` 真正读取上述快照；读取失败清缓存并执行完整 EVA。有效结果进入原计算来源、lease、mask、独立精确输入验证及成本门。 |
| `projects/mmdet3d_plugin/models/detectors/petr3d.py` | 原生 `extract_img_feat` 使用 `forward_from_m1`；保留已有注入 runtime 的 forward 协议兼容。 |
| `tools/worldstream_m1/dqtrack_policy_eval.py` | 保留原默认四方法，新增可选择的 guard_geometry/guard_hint 配对重放与视觉诊断。 |
| `tests/worldstream_m1/test_association_observation.py`（新增） | 5 项：非跟踪类别视觉消歧、非法/未知/缺失回退、默认行为、成熟有效身份冲突、实际 Commit 改配且不增加认证。 |
| `tests/worldstream_m2/test_live_m1_snapshot.py`（新增） | 7 项：真实 precheck→EVA→commit、变化像素重算、缺失/过期/epoch/aborted/来源变更回退、关闭行为。 |

## M1 实测：相同候选的 4 场景 161 帧配对

直接重放上一轮真实 OmniDrive 候选和原图；不是重新运行检测器或 DQTrack。场景为 scene-0101/0906/0269/0800，40/41/40/40 帧，均为 exposed development。GT 仅在 commit 后评分；同一场景内帧依赖，非独立验证。原 geometry/conservative 804/802 正确、3 错误的记录已复现。

| 方法 | 接受 | 正确 | 错误 | 未知 | IDS | 碎片 | 连续性断裂 | 覆盖率 | 条件有效召回 | P50/P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry | 3224 | 804 | 3 | 2417 | 258 | 141 | 165 | 69.39% | 82.97% | 93.42/244.52 |
| conservative | 3219 | 802 | 3 | 2414 | 260 | 141 | 167 | 69.29% | 82.77% | 98.02/263.83 |
| guard_geometry | 3224 | 804 | 3 | 2417 | 258 | 141 | 165 | 69.39% | 82.97% | 94.88/251.95 |
| guard_hint | 3221 | 803 | 3 | 2415 | 259 | 141 | 166 | 69.33% | 82.87% | 93.68/305.57 |

正确/错误采用既有唯一类别/尺寸/1m 的部分 GT 映射；未知不参与错误率分母，不能当作正确。计算有 CPU 并行任务与固定模式顺序，M1 时间只作本次观测。所有方法物理认证计数均为 0。
视觉策略在真实数据上改配数为 0；合成的实际像素交换回归可触发合法改配，不作为真实收益。guard_hint 相比 conservative 多接受 2 次：1 次可评分 pedestrian 正确恢复，1 次 car 未知；净 IDS/断裂各少 1。直接恢复记录见 `restored_associations.json`，anchor_id 在不同方法之间不可直接等同。
按类别：traffic_cone 所有方法均 230 正确/2 错误/831 未知；bicycle 均 30/1/82。原两个锥桶和一个自行车错误均未解决。锥桶不属于 DQTrack 跟踪类别；当前视觉证据无法形成完整唯一排他关系，不能强行改配。没有继续在这些场景调参。

## 原生 M1→M2 实验

1 个已暴露场景 scene-0101，连续 3 帧。每组 5 个执行模式，各组 15 次完整原生模型请求，真实语言生成；模型 `ckpts/iter_10548.pth`，不替换 detector/VLM，不训练。自然图像与受控重复图像分别记录，3 请求/模式、无预热和重复轮次，P95 只是小样本插值。
调用顺序：Petr3D.simple_test → M1.precheck → extract_img_feat → M2.forward_from_m1 → M1.export_m2_inputs → 原 computation_admission/CameraExecutor → 当前 heads/VLM → M1.commit(真实计算 receipt)。世界快照中的对象 ID、几何版本、来源帧/时间、epoch、新鲜度、期限、不确定性、关联依据及身份资格用于区分状态；对象身份本身没有像素复用许可。实际复用仍需合法计算来源和输入验证。

| 输入 | 路径 | E2E P50/P95 ms | fps | 实际复用相机数 | 峰值 allocated GiB |
|---|---|---:|---:|---:|---:|
| 自然 | original | 2460.80/3172.10 | 0.367 | 0 | 15.28 |
| 自然 | full_maintain | 5259.96/5532.82 | 0.224 | 0 | 15.28 |
| 自然 | source_linked_exact | 5189.24/5840.76 | 0.219 | 0 | 15.35 |
| 受控重复 | original | 2460.16/2465.66 | 0.406 | 0 | 15.36 |
| 受控重复 | full_maintain | 5207.53/5773.18 | 0.220 | 0 | 15.31 |
| 受控重复 | source_linked_exact | 5265.85/5396.22 | 0.227 | 6 | 15.38 |

original 不维护 M1；full_maintain 维护相同 M1、完整执行视觉；source_linked_exact 是现有 camera 边界 Auto（名称 exact 不表示跳过成本门）。比较不同服务成本时不能把原始无状态路径当成同一服务。每个 source_linked 请求的 m1_snapshot_read 均 true。自然图像 18/18 相机重算、0 复用；受控重复图像为 12 重算/6 复用：第二帧全部相机复用，第三帧计算来源 age=1.000767s 超过既有 1s lease，六路回退。验证时间前进不会延长原始计算寿命。
重复实验先运行现有 zero-update 实验 probe；其输出与完整模型一致且测得成本后，才生成本次目录内的局部成本记录供 Auto 使用；没有修改任何已有成本表或扩大到部分相机资格。probe 的 P50 为 4941.66ms，但不是默认模式收益。
两组各有 12 次非原始模式对照，均通过：特征、输出、memory、执行到的 EVA 中间层 max_abs 全为 0，文本和 planning 相同。自然图像逐帧仍执行当前检测和语言头；没有做额外全量 GT 漏检率统计，不把三帧输出一致性扩大为风险保证。额外诊断 CPU copy 和真实 M1/M2 成本包含在 E2E 内；模型加载/数据 scatter 不包含。正常 Auto 未显示净加速。
自然实验启动后补了一个仅针对旧注入 runtime 的兼容 fallback；实际运行 IncrementalVisionRuntime 所走分支相同，已归档 `m2_natural/petr3d_executed.py` 并与原 source_manifest 的 SHA 核对。受控实验采用最终源码，全部 runtime SHA 一致。

## 验证和可用配置

- 针对性：15 passed（12 项新增及 3 项既有 service 回归）。
- 完整 M1+M2：731 passed, 1 skipped, 19 warnings，113.54s；skip 是普通沙箱 CUDA 不可见的既有测试，已在 GPU 通道单独补跑：1 passed，3.12s。初次回归发现注入 runtime 不支持新方法，已修复兼容分支，未降低/删除原断言；初次日志也保留。
- 5 份 M1 YAML、M2 enabled/legacy/source_linked 配置加载通过；git diff --check 通过。
- 原 balanced/coverage、DQ off/shadow/conservative、新开关 false 的基线均保留。`association_observation_guard=True` 只用于本次显式离线研究；缺少降低错误延续的证据，应保持 false。原 calibrated_tiebreak 资格要求未改变。
- M2 可继续使用既有 `enabled=True,state_interface="source_linked",reuse_boundary="camera"` 和 source_linked M1 配置；未通过来源/新鲜度/独立输入验证/成本门就完整重算。`enabled=False` 是关闭路径；`legacy` 原行为保留。不开启 conditional/近似复用。
- 未改 checkpoint、物理认证器或 M3；无 Git commit/push；旧实验目录及用户文件未覆盖。

## 实际执行与复现命令

复现改用新 output 目录；所有命令都使用 OmniDrive 解释器与 CPU 亲和性。GPU 命令启动前确认无其他任务占用。

```bash
 taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests/worldstream_m1 tests/worldstream_m2
 taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests/worldstream_m2/test_execution.py::test_cuda_real_eva
 taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.dqtrack_policy_eval --protocol outputs/m1_dqtrack_policy_20261021/protocol.json --candidates outputs/m1_dqtrack_policy_20261021/omnidrive/model_observations.pkl --dqtrack-root outputs/m1_dqtrack_policy_20261021/dqtrack --output /tmp/m1_m2_replay_fresh --methods geometry conservative guard_geometry guard_hint
 taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.state_interface_integration --frames 3 --scenes 1 --repeats 1 --warmup 0 --case natural --output /tmp/m1_m2_natural_fresh
 taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.state_interface_integration --frames 3 --scenes 1 --repeats 1 --warmup 0 --case repeat --qualify-zero-camera-cost --output /tmp/m1_m2_repeat_fresh
 taskset -c 0-3,6-31 git diff --check
```

原始证据：`association/*/*.json`、两个 m2 目录的 runs/quality/status/source_manifest、memory、language 结果；精简汇总在 `m2_natural_summary.json` / `m2_repeat_summary.json`。实现源文件 SHA 见 `implementation_manifest.json`，测试和执行日志同目录。
