# DQTrack–M1 身份辅助策略：开发验证与不采用决定

**结论：不采用新策略，保持默认 `dqtrack_mode=off`，保留已有 conservative Hint。** 已完成4场景161帧真实 DQTrack/OmniDrive 配对实验、官方算法的完整场景子集诊断，以及两帧实际 VLM 完整模型对照。研究策略没有产生满足全部约束的新关联，没有降低错误身份延续；独立验证资格也不足，没有生成校准资格文件。

这是开发研究记录，不是独立确认、完整 nuScenes 验证分数、物理身份认证或端到端加速结果。原始 checkpoint、DQTrack 源码、之前的报告及用户其他输出均保留，未训练、未提交或推送 Git。

## 修改文件

| 文件 | 本轮作用 |
|---|---|
| `worldstream/m1/config.py` | 追加 conservative / calibrated_tiebreak 策略和资格文件路径/SHA256；默认 conservative，保留旧位置参数 |
| `worldstream/m1/state_manager.py` | 仅在 commit 内应用资格检查/合法身份提议；缺少资格退回保守行为；新关联明确未认证 |
| `worldstream/m1/dqtrack_bridge.py` | recent_only 不能作为当前身份佐证或冲突证据，修复开发中发现的错误拒绝；原保守路径不产生该状态 |
| `worldstream/m1/dqtrack_policy.py` | 唯一成熟身份、TTL、完整合法竞争分量、一对一提议、拒绝诊断、资格解析；不读取 GT |
| `worldstream/m1/dqtrack_geometry.py` | 离线诊断真实旋转三维框 IoU，数值未知返回 null；不是新增在线门限 |
| `tools/worldstream_m1/run_dqtrack_source.py` | 场景清单、每场景独立导出与显式状态重置、冻结 token 校验、forward 时延/显存；保留单场景入口及规模限制 |
| `tools/worldstream_m1/collect_evidence_observations.py` | 同帧连续候选采集校验、逐场景 reset、显存记录；修正训练场景成员资格的原硬编码元数据 |
| `tools/worldstream_m1/dqtrack_scene_protocol.py` | 不读 GT 的曝光审计和固定场景清单/哈希，生成候选采集选择文件，检查前缀连续性 |
| `tools/worldstream_m1/dqtrack_policy_eval.py` | 四种真实 M1 配对回放；离线部分 oracle、分层指标、轨迹断裂、逐帧 trace、场景 bootstrap |
| `tools/worldstream_m1/dqtrack_official_subset.py` | 使用官方 nuScenes tracking 指标算法，仅限定四个完整场景；醒目标记子集诊断 |
| `tools/worldstream_m1/dqtrack_vlm_smoke.py` | 实际生成语言的完整模型计时/基线对照，使用全新输出目录，计时包裹不改变计算结果 |
| `tools/worldstream_m1/dqtrack_auxiliary_probe.py` | 100个按时间截取对齐样本的 IoU/四元数变化诊断，不使用 GT，不事后转成策略门限 |
| `tools/worldstream_m1/summarize_dqtrack_policy.py` | 聚合场景、类别、密度、遮挡、运动、失败案例、阶段时延/显存和描述性区间 |
| `tests/worldstream_m1/test_dqtrack_policy.py` | 22项策略、拒绝、事务/认证边界、评估、序列和 IoU 测试 |
| `docs/worldstream_m1_dqtrack_policy.md` | 使用、实验边界、资格限制和复现命令 |
| `docs/worldstream_m1_dqtrack.md` | 链接本轮续报，保留默认和历史证据边界 |

没有重写 association.py 或物理认证器。原 off/shadow/conservative 行为有回归测试；默认配置没有改成新策略。

## 基线、协议与场景

首先用修改前实际入口重放旧两帧候选和轨迹，`baseline_recheck.json` / `baseline_consistency.json` 确认旧报告的14/22对齐、错误延续/身份切换及配对/认证一致性均可复现。该旧报告使用2m部分 oracle；本轮更严格的1m唯一匹配指标单独定义，不能跨定义直接比较数值。

`protocol.json` 和 `protocol.sha256` 在新结果产生前冻结，场景按固定 SHA256 顺序选择；没有用新 GT 难度/结果挑场景。`metric_definitions.json` 预先固定 score≥0.5、至少3次连续对齐、提示≤1秒的单一研究 profile；没有在已查看结果上反复调参或生成“校准成功”产物。

| 场景 | scene_token | 连续帧 |
|---|---|---:|
| scene-0101 | 01452fbfbf4543af8acdfd3e8a1ee806 | 40 |
| scene-0906 | dce6f3f2bf6b4859abcf3268581969d3 | 41 |
| scene-0269 | 9f1f69646d644e35be4fe0122a8b91ef | 40 |
| scene-0800 | 07aed9dae37340a997535ad99138e243 | 40 |
| 合计 | 4场景 | 161 |

每场景从起点到终点，验证 token/scene/timestamp/prev，DQTrack、OmniDrive temporal memory 和 M1 均在场景间 reset，单独导出，无跨场景身份携带。

曝光审计不仅发现候选名单：已有 `outputs/module1_physical_observability_optimization/nuscenes_scene_inventory.json` 明确将150个官方 val 场景全部标为 historically_exposed=True。原历史标记按原样保留，补充证据见 `independence_audit_note.json`。保守的候选名单审计本身不等于证明每场景都调过参，但现有记录不足以证明任何 val 场景独立，因此 `holdout_reserved=[]`，没有冒称新的独立验证集。

四场景共有5875个 GT 对象观察事件，2593个 visibility=1/2/3 事件、467个 fast 事件；这些是相关的重复对象/帧事件。GT密度覆盖 sparse/moderate/dense，运动覆盖 static/moving/fast。近距离反向运动仅是 crossing proxy，尚未人工核验真实交叉事件，不能把 proxy 当作已完成语义交叉标注。

## 数据来源与运行

- DQTrack：用户源码的 DETR3DTrackerDQ、既有 DETR3D checkpoint，SHA256 `93213fa9896ba1d161976c26fd512dc9e25f45c70c5d63c2dade31e90a8ec3d4`；严格加载原982项 state_dict。
- 官方 infos：`/data/qh/phdCode/work4/models/DQTrack/infos/track_cat_10_infos_val.pkl`；世界系导出和 sample.json 一致，无重复 ego transform。
- OmniDrive：原 `ckpts/iter_10548.pth`，相同161帧真实检测；候选位于 `omnidrive/model_observations.pkl`，所有策略读取同一文件、同一序列。
- GT：只在各次 commit 完成后用于离线评价。没有传入 Frame、关联、资格阈值计算或物理认证。
- 虚拟环境 `/tmp/dqtrack_source_env_20261021` 只读继承原有 Torch/MMCV。官方指标依赖 motmetrics 1.1.3 仅安装在该 `/tmp` 环境，OmniDrive 环境未安装旧依赖。pip先被沙箱网络阻止，宿主批准后安装成功。
- GPU 空闲后执行，CPU亲和性均为0–3、6–31，未停止其他任务。没有启动全量6019帧推理或模型训练。

已有采集工具曾把 model_train_scenes 无条件写为 True；这不能证明 checkpoint 的训练成员。未来输出已改为 unknown，并按官方场景表单独写 split；本轮原始产物不覆盖，用 `candidate_provenance_addendum.json` 明确四场景属于官方 val、checkpoint实际训练成员未核验。不能混淆“官方split”和“模型从未训练/查看”。

## 新策略的实现边界

在线 `calibrated_tiebreak` 预留接口需要固定资格产物；缺少资格时保留 conservative。当前没有通过资格的产物，不提供可启用的新策略配置。资格解析要求源码/几何绑定、足够的不重叠开发/独立场景、曝光审计，以及错误减少、正确延续不减、碎片不增和成本预算的联合证据；本轮两项根本前提——开发收益和独立场景——均不满足。

离线研究入口用 ResearchManager 仅注入固定数值 profile，不注入 GT。提议仅限原 associate() 有限合法边；不改变类别、2m距离、尺寸比、过期、场景/时间、重现等硬条件。竞争连通分量中所有端点均须有唯一、成熟、当前有效的外部身份，整个身份映射都必须合法且一对一。未知竞争、低经验分数、新生/丢失/过期、可能ID切换和非法几何均保留拒绝。score不是概率，也不加权进几何距离。

新关联写为 uncertified_dqtrack_identity_tiebreak，DQTrack不替换 anchor_id、不增加认证观察数或可靠速度；与物理/实验身份策略并用时保持 inactive。合成的合法方形交叉用例能完成消歧，事务/认证测试证明机制有效，但不能用单元替身宣称真实数据收益。

研究中发现 recent_only 元数据误触发旧冲突检查，造成额外拒绝；已修复并单独回归。`paired/` 原失败结果保留；最终 `paired_final/` 用相同场景、阈值和定义重跑，`final_run_definition.json` 固定有效决策源码哈希。修复不是参数调优。

辅助特征仅诊断：100个时间顺序对齐样本的真实旋转3D IoU中位数0.5603、P95 0.7978，数值失败返回unknown；外部四元数变化已记录。没有独立验证的速度残差（null）。本轮没有把观察到的特征分布事后变成接受门限。

## M1 三策略对照

部分 oracle：类别/尺寸兼容、传感器支持、中心≤1m且两端都唯一。未知匹配不当作正确。有效召回分母969是相邻帧共同可唯一匹配的候选 GT 实例事件，**不是全部 GT 目标的召回分母**。覆盖率分母4646为排除场景首帧后的当前候选事件，包含新生目标；拒绝率不能解释为纯错误拒绝率。

| 策略 | 正确延续 | 错误/可评分延续 | 未知关联 | 身份切换 | 碎片 | 连续性断裂 | 关联覆盖 | 有效召回 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 几何 | 804 | 3/807 | 2417 | 258 | 141 | 165 | 69.39% | 82.97% |
| 当前 conservative Hint | 802 | 3/805 | 2414 | 260 | 141 | 167 | 69.29% | 82.77% |
| 新研究 tiebreak | 802 | 3/805 | 2414 | 260 | 141 | 167 | 69.29% | 82.77% |

Shadow与几何完全一致。研究策略新增合格消歧提议为0；所有模式的物理认证数为0。identity switches/fragmentation 是上述部分 oracle 的诊断，不是官方IDS/FRAG；碎片指可评分实例中断后再出现，连续性断裂另统计共同可匹配实例未正确延续，避免以多拒绝“改善”错误率。

可评分错误关联率几何0.3717%、保守/研究0.3727%；大量未知关联仍未可靠评分，不能外推全体真实错误率。拒绝率分别30.61%/30.71%/30.71%。按场景bootstrap（2000次，4个场景）保守错误率描述区间0–0.9434%，有效召回区间62.87%–88.25%；小样本、重复实例和历史曝光使它们不构成独立风险保证。

### 对齐诊断与失败原因

原始可用轨迹2270条，对齐982条（43.26%）；1288条未对齐，其中620条存在合法竞争、668条没有合法类别/距离/尺寸边。577是 Hungarian 选中提议中的歧义数量，620还包含未被选中的竞争轨迹，两者不能混用。非法数据拒绝0。

严格部分GT下，584个可评分对齐均正确（584/584），398个对齐仍未知；在两侧均可唯一匹配同一GT的639个机会中召回584个（91.39%）。这个条件精度不能解释为跟踪ID跨帧正确概率。

`analysis_summary.json` 含所有类别、密度、visibility和运动分层。例如保守策略：car正确325/eligible483，truck80/83，traffic_cone230/231，bicycle30/30另有1次错误，pedestrian113/118。fast分组115个条件机会、0个正确延续，显示当前关联覆盖瓶颈；本轮没有用GT速度补在线预测或放宽原距离门限。

3个错误在所有策略中均保留，详细 token/anchor/GT实例/中心见 `failure_case_details.json`：

- scene-0101 的 `717c45682e15432f9862e47775b9b80f` 与 `2e743b1fb9a34383b88bec3e7792ad2a`：traffic_cone 身份往返错误；锥桶不属于官方DQTrack的7类 tracking classes。
- scene-0269 的 `6e147994a5e3493d86a928a612ff5791`：bicycle，缺少可用外部身份历史。

研究拒绝记录包括 external_history_missing 4209、状态过期/重现754、新生/短历史698、丢失/遮挡/过期150、低当前经验分数56、外部身份非唯一/丢失/可能切换47、低历史经验分数15、非法几何5、未知竞争1。计数是相关anchor-frame事件，不是独立对象数。这解释了代码原型有效但真实样本没有安全消歧机会。

## DQTrack 官方算法：仅四场景子集

使用原SDK matching、过滤、阈值扫描和 metric；当前子进程只限定 val 的场景成员为这四个完整场景，未修改SDK文件。`official_subset/SUBSET_DIAGNOSTIC.json` 醒目标记 `OFFICIAL_ALGORITHM_SUBSET_DIAGNOSTIC_NOT_FULL_VAL`。

| 指标 | 子集结果 |
|---|---:|
| AMOTA | 0.352217 |
| AMOTP | 1.407314 |
| Recall | 0.487726 |
| IDS | 50 |
| FRAG | 71 |
| MOTA | 0.340960 |
| TP / FP / FN | 1490 / 342 / 1173 |

这些不等于官方完整val分数，不能直接与论文全val结果比较。没有未经确认运行全量推理。

## 耗时、吞吐与额外 GPU 成本

CPU回放使用640宽 resized_raw_image，原生模型使用既有预处理；不能直接混合两种像素空间的M1成本。161帧，未隐藏首帧；计时范围以原始JSON为准，checkpoint/全部batch IO不计入forward。固定模式顺序、缓存和CPU共享使本轮不是加速benchmark。

| 阶段 | 样本 | P50 / P95 ms | 按总时间吞吐 fps |
|---|---:|---:|---:|
| DQTrack forward | 161 | 177.18 / 198.50 | 5.43 |
| DQTrack forward+export | 161 | 178.26 / 201.27 | 5.38 |
| OmniDrive 原检测采集（无语言） | 161 | 521.61 / 523.89 | 1.91 |
| M1 几何回放 | 161 | 108.38 / 382.41 | 6.52 |
| M1 conservative回放 | 161 | 107.30 / 379.65 | 6.51 |
| M1研究回放 | 161 | 98.64 / 367.27 | 6.48 |
| conservative对齐桥接（M1内子范围） | 161 | 1.89 / 5.26 | 不作为系统吞吐 |
| 研究路径对齐桥接（M1内子范围） | 161 | 1.71 / 6.91 | 不作为系统吞吐 |

M1吞吐按平均总墙钟计算，不能用P50倒数替代；研究路径P50较低而总吞吐并未提高，不支持加速结论。桥接子范围是读取/对齐，不含全部M1状态与策略工作。

额外DQTrack forward累计 **29.67秒 GPU计算计时**（161帧）；独立测得峰值 allocated：DQTrack 1,458,531,328字节（约1.36GiB），OmniDrive检测1,937,199,104字节（约1.80GiB）。不是零成本身份提示，未测联合驻留峰值。

另外实际运行2帧/模式的原生完整模型，四次语言生成均执行，全新语言目录，回答字节及输出/memory/backbone一致：

| 原生完整模型 | full call P50/P95 ms | generate P50/P95 ms | full-call fps |
|---|---:|---:|---:|
| 无M1基线 | 2684.00 / 2854.27 | 1959.73 / 1968.46 | 0.373 |
| conservative Hint | 2800.76 / 2895.17 | 1950.19 / 1953.29 | 0.357 |

两者峰值 allocated 均16,216,610,816字节（约15.10GiB）。full call含原生OmniDrive/M1/Planning，不含DQTrack forward和batch IO；generate是其内部阶段，不可相加。每模式只有2帧，P95只是小样本插值，不是稳定尾延迟保证。原工程加载器出现既有RoPE cache兼容性提示，未修改原始权重或加载路径，模式间结果一致。

**DQTrack+M1+VLM联合调度的完整服务时延/吞吐/显存尚未实测**，在汇总中为null；不能将独立实验的时间或显存相加后冒充实测，也不宣称GPU跳算、深层特征复用或端到端加速。

## 测试、产物与复现

- 新增策略测试22项；桥接/导出/策略合计57项通过。
- 完整 `tests/worldstream_m1`：**592 passed, 19 warnings in 129.46s**；专项 **57 passed in 0.57s**。日志为 `full_tests.log` 与 `targeted_tests.log`。
- balanced/coverage/shadow/hint 四份配置加载通过，策略默认均conservative，DQ模式依次off/off/shadow/hint。
- `git diff --check` 通过；决策源码与final_run_definition哈希、协议哈希已核对一致。

复现指令和资格边界详见 `docs/worldstream_m1_dqtrack_policy.md`。原目录拒绝覆盖，复现应换新输出。关键实际入口：

```bash
# 冻结新协议（不读GT）
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.dqtrack_scene_protocol --output /tmp/dq_policy_new_protocol

# 真实连续DQTrack，使用本轮固定清单；新output
taskset -c 0-3,6-31 /tmp/dqtrack_source_env_20261021/bin/python tools/worldstream_m1/run_dqtrack_source.py \
 --source-root /data/qh/phdCode/work4/DQTrack-main \
 --checkpoint /data/qh/phdCode/work4/models/DQTrack/model_val/detr3d_res101_size_900_1600_finetune_amota_36_7.pth \
 --output /tmp/dq_policy_repeat_tracks --run \
 --infos /data/qh/phdCode/work4/models/DQTrack/infos/track_cat_10_infos_val.pkl \
 --data-root /data/qh/phdCode/work4/OmniDrive/data/nuscenes \
 --scene-list outputs/m1_dqtrack_policy_20261021/protocol.json

# 同一真实候选的三策略+shadow重放；GT只离线
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python \
 -m tools.worldstream_m1.dqtrack_policy_eval --protocol outputs/m1_dqtrack_policy_20261021/protocol.json \
 --candidates outputs/m1_dqtrack_policy_20261021/omnidrive/model_observations.pkl \
 --dqtrack-root outputs/m1_dqtrack_policy_20261021/dqtrack --output /tmp/dq_policy_repeat_paired

# 官方算法子集，不是全val
taskset -c 0-3,6-31 /tmp/dqtrack_source_env_20261021/bin/python -m tools.worldstream_m1.dqtrack_official_subset \
 --protocol outputs/m1_dqtrack_policy_20261021/protocol.json \
 --dqtrack-root outputs/m1_dqtrack_policy_20261021/dqtrack --output /tmp/dq_policy_repeat_official

# 原生完整VLM对照，实际生成，2帧/模式
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.dqtrack_vlm_smoke \
 --output /tmp/dq_policy_repeat_vlm --scene 01452fbfbf4543af8acdfd3e8a1ee806 \
 --trajectories outputs/m1_dqtrack_policy_20261021/dqtrack/01452fbfbf4543af8acdfd3e8a1ee806/normalized_tracking.json --frames 2
```

OmniDrive实际采集复用 `collect_evidence_observations.main('outputs/m1_dqtrack_policy_20261021/omnidrive')`，运行前 `wait_for_gpu(60)`；选择清单在该目录，已有文件不可覆盖。复现到新协议目录时使用其自动生成的omnidrive/data_selection.json。所有命令必须保持CPU亲和性约束。

原始证据：`dqtrack/*/normalized_tracking.json`、逐场景run_report、`omnidrive/model_observations.pkl`及来源、`paired_final/*/{method}.json`、`analysis_summary.json`、`failure_case_details.json`、`auxiliary_geometry_probe.json`、官方指标原始文件、VLM实际回答/计时/字节一致性、协议/源码哈希和日志。`paired/`保留修复前额外拒绝的失败结果，不替换或隐藏。

## 未完成与下一步

仍缺真正独立场景、独立开发校准和重复独立确认；当前没有新策略采用依据。也未完成真实交叉事件人工核验、完整val官方评估、更多重复/交替顺序成本测量、联合DQTrack+VLM服务链路测量，以及未评分关联的可靠GT覆盖。本轮没有对已查看场景继续调参。

下一步应先解决可追溯的未暴露数据和评估覆盖，再研究当前帧唯一对齐率与成熟身份历史瓶颈；原3次错误中两次属于DQTrack不追踪的锥桶，不能期待身份消歧自动修复。新策略没有开发收益且缺独立证据，故本轮不发布资格文件、不扩大使用范围。
