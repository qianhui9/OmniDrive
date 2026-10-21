> **2026-10-21 当前代码说明：** 以下正文保留历史实验记录。生产管理器现已拒绝 `hint` 和非 conservative 策略，默认仍为 `off`，旧文中的在线启用方式与 ResearchManager 不再适用。最新三组研究使用独立离线入口 `tools.worldstream_m1.dqtrack_three_way_eval`，复用缓存和事务，不授予任何在线资格；用法见 [三组离线实验](worldstream_m1_dqtrack_three_way.md)。

# DQTrack–M1 身份辅助策略开发验证

当前结论：**维持默认 off，保留原 conservative Hint；不发布 calibrated_tiebreak 校准资格。** 四个已暴露开发场景、161 帧真实数据中，研究消歧策略未产生满足全部约束的新关联，不能减少错误延续；保守 Hint 相比几何基线还损失了两次正确延续。完整证据与场景/指标表见 `outputs/m1_dqtrack_policy_20261021/experiment_report.md`。

## 行为与资格边界

新增 `dqtrack_association_policy`：默认 `conservative`，原 off/shadow/hint 行为保留。`calibrated_tiebreak` 是预留的受资格限制接口；缺少 `dqtrack_policy_qualification_path` / SHA256 时，commit 记录 `independent_qualification_missing` 并退回保守流程。本轮没有生成合格文件，不能通过只改枚举启用在线消歧。

资格解析要求固定 policy 源码、几何配置、至少四个开发/四个不重叠独立场景及其曝光审计、错误延续减少、正确延续不减、碎片不增和 M1 成本预算。资格文件是待独立研究审计的产物，不是操作者手填布尔开关。当前没有通过这些条件的产物，也没有声称部署校准已完成。

离线 `ResearchManager` 仅注入预先固定的数值 profile，复用实际 M1 提交事务；输出明确标记 `OFFLINE_RESEARCH_ONLY_NOT_QUALIFIED`。本轮 profile 固定为 score≥0.5、至少3次连续对齐、最近提示年龄≤1秒；这些是未校准的研究参数，不是身份正确概率。没有根据结果迭代放宽参数。

`dqtrack_policy.propose()` 只读原 `associate()` 生成的有限合法边：

- 外部 ID 必须在 run+scene 命名空间内唯一，当前帧严格对齐且历史成熟、未过期。
- 原类别、距离、尺寸、状态年龄、最近实际观察、场景/时间约束不变。
- 对相互竞争的几何连通分量，所有端点均须有唯一可用外部身份，且整个分量的身份对应形成合法的一对一映射。未知竞争者、身份丢失/可能切换、低经验分数、新生轨迹或非法边均保留拒绝选项。
- 身份关系作为独立的优先依据，不把 tracking_score 加进几何代价或解释为概率。剩余候选不会因为删去竞争者而自动获准。
- 由此新建的延续写为 `uncertified_dqtrack_identity_tiebreak`，不增加物理认证、认证观察数或可靠速度。与物理/实验身份策略同时启用时，该研究策略保持 inactive。

只有该研究/资格路径会保留带 timestamp/expires_at 的 recent_only 元数据。缺少当前对齐时，它不参与当前身份佐证、冲突拒绝或消歧。独立回归测试覆盖这一点；开发中发现并修复过一次旧冲突函数误用 recent_only 的问题，修复前结果保存在 `paired/`，最终相同阈值重跑在 `paired_final/`。

`dqtrack_geometry.py` 提供离线凸框交集的真实三维 IoU。100 个按时间顺序截取的对齐样本另测 IoU 和外部四元数变化；未将这些观测事后转为门限。没有独立验证的速度残差，相关字段为 null。

## 场景协议与数据独立性

`dqtrack_scene_protocol` 只读 scene/sample 元数据和既有记录，在新目录冻结清单与 SHA256，不读 GT。它保守地将记录中出现的候选场景也视为 potentially exposed，因此本轮150个验证场景全部被命中。**这不是证明150场景均被调参，而是无法据现有历史证明其中任何场景独立**。保留清单为空，不能把已有验证场景重新包装为独立集。

源模型入口新增 `--scene-list protocol.json`，只运行 development 清单；最多8场景、320帧、每场景80帧，保留原 `--scene --frames 1..40`。每个场景单独导出，显式重置 DQTrack 的 detector/tracker/scene/max_track_id，检查冻结 token 列表与官方 infos、sample.json 的 scene/timestamp/prev 链。OmniDrive 采集入口同样检查场景和连续性，逐场景 reset temporal memory。M1 评估每场景创建新 manager，不跨场景携带状态。

冻结新协议（不覆盖当前协议）：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
 -m tools.worldstream_m1.dqtrack_scene_protocol --output /tmp/dq_policy_new_protocol
```

四场景 DQTrack 导出实际命令，复现时更换 output：

```bash
taskset -c 0-3,6-31 /tmp/dqtrack_source_env_20261021/bin/python \
 tools/worldstream_m1/run_dqtrack_source.py \
 --source-root /data/qh/phdCode/work4/DQTrack-main \
 --checkpoint /data/qh/phdCode/work4/models/DQTrack/model_val/detr3d_res101_size_900_1600_finetune_amota_36_7.pth \
 --output /tmp/dq_policy_repeat_tracks --run \
 --infos /data/qh/phdCode/work4/models/DQTrack/infos/track_cat_10_infos_val.pkl \
 --data-root /data/qh/phdCode/work4/OmniDrive/data/nuscenes \
 --scene-list outputs/m1_dqtrack_policy_20261021/protocol.json
```

GPU 忙时等待，不停止其他任务。独立环境创建方式仍见源码接入文档；motmetrics 1.1.3 仅安装在 `/tmp` 该虚拟环境，未修改 OmniDrive 包版本。

## 配对评估与官方算法子集诊断

重用本轮同帧真实候选和跟踪结果，不重新运行 GPU：

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.dqtrack_policy_eval \
 --protocol outputs/m1_dqtrack_policy_20261021/protocol.json \
 --candidates outputs/m1_dqtrack_policy_20261021/omnidrive/model_observations.pkl \
 --dqtrack-root outputs/m1_dqtrack_policy_20261021/dqtrack --output /tmp/dq_policy_repeat_paired
```

工具比较 geometry、shadow、conservative 和研究 tiebreak。GT 在 commit 后只用于标注：类别/尺寸兼容、传感器支持、中心≤1m且双向唯一的部分 oracle。未知匹配不计作正确，也不假定错误率为零；有效召回分母是相邻帧共同可唯一匹配的候选 GT 实例，不是所有 GT 对象。identity switches/fragmentation 是部分 oracle 下的诊断，不是官方 tracking IDS/FRAG。旧两帧报告使用2m oracle，已单独按原代码复现；不要直接混合两种定义。

每帧 trace、分场景/类别/GT密度/visibility/GT运动状态结果及按场景 bootstrap 描述性区间均导出。frame trials 相关，同实例重复出现；只有4个场景，区间不是独立风险保证。

官方算法的完整场景子集诊断：

```bash
taskset -c 0-3,6-31 /tmp/dqtrack_source_env_20261021/bin/python \
 -m tools.worldstream_m1.dqtrack_official_subset \
 --protocol outputs/m1_dqtrack_policy_20261021/protocol.json \
 --dqtrack-root outputs/m1_dqtrack_policy_20261021/dqtrack --output /tmp/dq_policy_repeat_official
```

该工具在当前子进程中仅把 SDK val 场景成员限定为四个完整场景，不改变指标、匹配、阈值扫描和过滤逻辑。结果始终标为 `OFFICIAL_ALGORITHM_SUBSET_DIAGNOSTIC_NOT_FULL_VAL`。本轮没有运行全量验证推理或完整验证评分。

完整模型/VLM 小规模计时入口 `dqtrack_vlm_smoke` 实际生成语言回答，使用新目录，比较默认基线与现有保守 Hint。只允许2–4帧，计时包裹不改变权重、张量或生成参数。原生完整模型耗时包含 M1 和 Planning，但不包含外部 DQTrack 前向；完整 DQTrack+VLM 联合调度服务时延尚未测量，不能用两个独立实验之和冒充实测。

## 回归与采用决定

新增23项策略/评估测试，涵盖合法消歧、非法边、新生/低分/过期/丢失/重复 ID、未知竞争、场景/重现、最近提示不作当前证据、事务与物理认证隔离、序列校验、子集完整性和三维 IoU。

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1/test_dqtrack_policy.py tests/worldstream_m1/test_dqtrack_bridge.py tests/worldstream_m1/test_dqtrack_export.py -q
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 -q
taskset -c 0-3,6-31 git diff --check
```

对应58项和593项通过。没有独立资格、开发收益或完整联合链路成本证据，所以不采用新策略。后续应先补充真正未暴露的场景和稳定可评分身份对应，研究当前帧对齐覆盖及合法几何边内的有效机会；不要在本轮场景上继续反复调参以获得正向结果。

最终所有权复验在 `paired_owned/`，指标与 `paired_final/` 完全一致；最终汇总为 `analysis_summary_owned.json`。原 `paired/`、`paired_final/` 保留，不覆盖。最近提示深复制确保修改诊断不会修改状态元数据。
