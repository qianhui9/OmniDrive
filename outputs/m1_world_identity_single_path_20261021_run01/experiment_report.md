# M1 单主线世界关联重构交付

**结构重构完成；质量验收 No-Go。** 最终代码与物理/事务/M2 接口相关的 M1 测试全部通过（598 项）。唯一一次真实缓存回放发现跨缺帧身份误合并；已直接修复统一预测的时间误差预算，但遵守“一次主方法实验”限制，没有再运行真实数据。下列实测数值属于修复前冻结版，不能当作最终交付版的验证结果。

## 1. 精简前后与唯一入口

修改前实际链路：历史预测 → `associate(..., None, dormant_rows)` → 再次分配/歧义后处理 → 可选观测 refine → tracks refine 图像测量/候选扩展/组件覆盖 → 搜索运动 → 时序提议替换 → DQTrack veto/tiebreak → 身份更新。

最终链路：

`StateStore 活跃锚点/有界 dormant 元数据 → association_tracks.predict（每对象一次） → associate 内唯一类别/尺寸/时间/来源/几何矩阵 → 分配前歧义门 → unmatched_assignment 一次全局分配 → accepted_pairs 冻结为 tuple → 实际状态更新/可选独立物理验证 → 原子提交/有限历史清理`。

`WorldStateManager._commit()` 只有一次 `associate()` 调用和一次 `accepted_pairs` 赋值。计算好的 `association_priors` 直接传入该入口，创新残差也复用该预测，不再重复预测。同一匹配结果用于活跃和 dormant 身份，没有后置修正分配器。

主实验配置 `configs/worldstream_m1_world_identity.yaml` 明确启用历史、关闭额外身份/搜索/外部 hint 分支。`association_track_prior` 默认 False，含义仅是是否维护/使用有限轨迹历史；**所有配置仍走同一个新分配器**，不是默认保留一套旧分配算法。

## 2. 修改、删除与保留

| 文件/功能 | 最终处理 |
|---|---|
| `worldstream/m1/association.py` | 一个原始候选距离矩阵；候选源约束在优化前检查；未解决的端点竞争先拒绝；加入带私有虚拟未匹配列的全局最小代价分配。保留地图/区域曲线方向、距离等类型专属几何。 |
| `worldstream/m1/association_tracks.py` | 精简为来源绑定、单一预测、候选元数据读取、提交后历史维护、数量/时间/内存清理。移除 refine、逐边图像测量、外观组件覆盖和无效 appearance 兼容函数；轨迹不保存 ROI、描述子或图像。 |
| `worldstream/m1/state_manager.py` | 移除所有匹配改写与二次分配；减少索引扫描、重复预测及逐边序列化。保留原事务 worker/fork/发布/失败回滚。 |
| `association_alignment.py`、`association_diagnostics.py`、`association_observation.py` | 检查调用后删除：最后的生产调用者都是已移除的身份修正链；其对应实验测试迁移或删除。 |
| `dqtrack_policy.py` | 删除已无生产调用者的外部身份 tiebreak 和策略资格代码；DQTrack bridge 只保留来源验证与只读对齐，删除 corroboration/veto helpers。 |
| `search_motion.py` | 删除独立三帧候选搜索预测器及 search_edges；保留仍被独立物理测量引用的图像初始化投影 helper，不分配世界 ID。 |
| `identity_temporal.py` | 保留独立物理测量、证据签发和验证。移除无调用者的关联专用对应缓存扩展，恢复原测量实现。提议可以被检验、记录，但 `applied_to_tracking` 恒为 False，不能改写 ID。 |
| `config.py` 与相关 YAML | 删除大规模逐边测量/诊断参数。旧 hint YAML 迁移 shadow，旧搜索 YAML 关闭身份搜索；其他已退休开关在 manager 构造时明确报错，不静默忽略。 |
| `tools/worldstream_m1/world_identity_eval.py` | 保留单方法、原评分及历史结果读取；去除逐边在线探测，保留快速组/恢复与简短聚合。修正补充恢复评分不能拿更早 GT 标签填补最近源观测的 unknown。 |
| 相关旧实验入口/报告工具 | 移除 ResearchManager/tiebreak 调用，入口只提供 geometry/read-only shadow；旧诊断工具使用 shadow，旧字段读取迁移。没有执行这些旧方法。 |
| 测试 | 删除对退休分支成功匹配的期待；保留并补充单次分配、门控、未匹配、历史、dormant、事务、资格隔离和非物体几何测试。 |

运行时代码 `worldstream/m1` 相对本轮开始共增加约 235 行、删除约 1016 行，净减少约 781 行。物理资格系统、`world_motion.py`、`StateStore`、EVA、Detection、Map、LLM、M2 Exact 和 M3 均未重写。

### 唯一预测与未匹配规则

来源有效、时间连续且至少三次观测时，使用成对斜率中位数、历史残差和速度不稳定性。历史不足时沿用最后实际世界中心；已有独立合格物理运动仅通过同一函数读取。坏版本、时间异常、失效或超预算返回 None，不启动候选搜索、光流或第二个预测器。不读取单帧检测速度来启动身份运动。

类别/尺寸/时效/Scene/epoch/执行签名/实际感知来源均为必要门；物体残差加误差预算不得超过原 2 米门。地图/区域保留原类型规则。全局分配用距离除类型门预算作为代价，未匹配列代价为一个门预算；没有最大基数奖励或贪心补救。离线 Oracle 仍使用其原 `legal_assignment`，没有因此改变 GT 评分。

### 统一历史与资格隔离

有限历史最多 8 个样本、256 条轨迹、3 秒、4 MiB；每条只保存中心/时间/帧/版本及最新几何/来源快照，不保存旧特征、验证期限或认证。普通匹配不成为物理资格。原物理运动历史是独立认证声明，不参与另一套身份分配。

dormant 与活跃对象走相同预测/门/全局分配；恢复 ID 时新建当前实际观测状态，旧认证、验证时间与旧特征不继承。M2 仍读原 precheck 冻结状态和自己的来源/期限门。原最后一帧灰度观测仍用于既有 precheck；删除的是身份轨迹额外建立的多帧图像历史，不是清除 M2 所需观测。

## 3. 测试结果与迁移

- 最终交付代码完整 M1 回归：`final_m1_tests.log/xml`，**598 passed，19 个既有依赖警告，106.57 秒**。
- 最终聚焦测试：`delivery_targeted.log/xml`，**35 passed，0.20 秒**。数量与全集重叠，不相加。
- 唯一真实回放前冻结版完整回归：`m1_tests_final.log/xml`，594 passed。较早失败日志也保留，最终通过记录如上。

测试覆盖连续 ID、类别/尺寸/距离拒绝、一对一、歧义新建 ID、历史有效/过期/来源不匹配、同一预测被唯一分配器读取、相同 dormant 门控、提交历史与预算、提交异常/篡改回滚、独立物理认证和 M2 来源接口、地图匹配。新增时间预算测试同时覆盖短历史和成熟历史；恢复评分测试证明不可用源标签保持 unknown。

旧标量优化参考也迁移到相同年龄预算，保留 1e-14 精度的矩阵一致性断言；不再期待最大基数匹配。旧歧义测试改为验证分配前拒绝，未恢复旧的匹配后修正路径。已删除分支的测试减少，不代表跳过仍存在的安全功能。

以下配置会在 `WorldStateManager` 构造时明确报错：`association_observation_guard=True`、`search_dynamic_motion=True`、非 shadow 的 identity_tracking_policy、DQTrack hint/calibrated_tiebreak 或身份策略资格文件。独立物理测量的 multi-hypothesis 工具并未因此变成身份分配器。

## 4. 唯一一次主方法实验与版本边界

仅执行一次 `tools.worldstream_m1.world_identity_eval`，正常退出 0。四个原开发 Scene、161 帧、相同缓存检测/图像/协议；总候选 **4646**、有效延续机会 **969**，逐 Scene 分母一致。没有新模型推理、GPU 冒烟、DQTrack 推理、基线运行、参数搜索或消融。

原 `strict_labels`、`rates`、`percentiles` AST 未变；unknown 不计正确、不进入 scored 错误率分母。旧、新补充连续性指标均仅读取保存关系离线评分，GT 从未进入在线预测或状态。

### 修复前冻结版实测（不是最终交付版）

| 指标 | 原保存参考 | 上轮复杂候选保存值 | 本轮冻结版唯一回放 |
|---|---:|---:|---:|
| 正确延续 | 804 | 797 | **799** |
| 相邻帧错误延续 | 3 | 3 | **3，同原关系** |
| 有效召回率 | 82.9721% | 82.2497% | **82.4561%** |
| 关联覆盖率 | 69.3930% | 69.1563% | **75.8071%** |
| 旧 identity_switches | 258 | 266 | **246** |
| 连续性断裂 | 165 | 172 | **170** |
| legacy_gt_gap_count | 141 | 141 | 141 |
| predicted_track_fragments | 42 | 42 | **33** |
| continuous_identity_switches | 165 | 172 | **170** |
| 快速运动正确延续 | 0/115 | 0/115 | **0/115** |

相对原结果丢失 7 个正确关系、新增 2 个，净减 5；约降低 0.516 个百分点。逐关系清单在 `relation_differences.json`。accepted=3522、scored=802、unknown=2720；覆盖增长中包含大量未知，不能当作身份正确率提高。

旧 IDS 跨可评分间隔比较，新 continuous IDS 只比较相邻可评分帧。predicted_track_fragments 只计 GT 持续可观测期间预测中断后以不同 ID 返回；GT 缺帧/Oracle 歧义中断统计，立即换 ID 不重复算碎片。各字段定义沿用已有修正口径，不能将旧 GT 缺帧数 141 当作预测碎片。

### 必须单独报告的 dormant 风险

统一门控冻结版实际重新匹配旧 ID **159 次**。按**最近一次实际源观测与当前观测均可评分**核对：

- 正确 13；错误 **20**；unknown 126。
- 可评分动作 33，错误率 20/33≈60.61%。这说明只看相邻帧的 3 次错误会漏掉严重跨缺帧错误，**不能据此通过质量验收**。
- 原实验日志按“最近一次可评分标签”得到 13/21/125，其中 1 次源观测实际不可评分。原日志保留；`exact_source_recovery.json` 用已有逐帧身份映射重新核对，没有重跑关联。最终离线评分入口也已修正此问题。

另有 TTL 内 118 个可评分重出现间隔代理：77 个检测/部分 Oracle 间断、41 个传感器支持间断。其中 13 个机会记录应用 dormant，57 个记录身份保持；这与全部 159 个已发生动作不是同一分母，不能混算恢复成功率。元数据间断不等同于已目视证实的真实遮挡。

### 回放后直接修复主线

冻结版错误根因：未认证历史的误差预算只含已有定位误差及拟合残差/速度变化；一个看似稳定的历史可以在缺帧后继续保持近零预算。它把“过去拟合好”当成了“未来误差不增长”，使旧 ID 获得了不合理的长时匹配机会。

最终在**同一个 predict 函数**中对所有普通对象加入已有配置的年龄误差项：

`预算 = 声明定位误差 + velocity_uncertainty_mps × 观测年龄 +（有历史时）拟合残差与速度不稳定性`

它对短历史、成熟历史、活跃及 dormant 均适用，没有 dormant 专属恢复算法。使用既有 2 m/s 配置、原 2 米几何门，不搜索新参数；负/非有限速度误差预算明确拒绝。独立合格物理运动仍遵守原认证/协方差规则。预算超限直接不匹配，例如普通源年龄 1.1 秒时仅年龄项已为 2.2 米。该修复可能牺牲更多召回；**未再次真实回放，不能宣称已经消除 20 次错误或保留冻结版精度**。

`frozen_source/` 是唯一实测代码；`delivery_source/` 是最终代码；`post_replay_changes.patch`、`source_revision_note.json` 明确列出回放后的预测预算、参数验证及补充评分改动。旧结果没有覆盖。`comparison.csv` 的最终交付行明确为 NOT_RUN，没有填入冻结版数值。

## 5. 真实耗时、内存与日志

以下同类 CPU M1 回放值来自不同轮保存结果，没有重新跑基线，**不是配对性能结论**，也不是完整 OmniDrive/GPU E2E 加速结论。

| 版本 | M1 P50 / P95 ms |
|---|---:|
| 原保存参考 | 100.867 / 365.905 |
| 上轮复杂候选保存值 | 173.097 / 1196.911 |
| 本轮修复前冻结版 | **98.157 / 294.332** |
| 最终年龄预算修复后 | **未实测** |

冻结版最大轨迹数 141，轨迹保守内存计数 954,683 字节，序列化历史 231,429 字节，均低于 4 MiB。轨迹数量和关联结果也变化了，不能把全部内存差异都归因为数据结构精简。普通关联额外多帧图像历史为 0；独立 precheck 上一帧观测仍保留。

聚合诊断序列化体积 P50/P95/max 为 **274 / 496 / 804 字节/帧**，相比此前约 65 KiB/帧大幅减少数据量；没有单独测日志 CPU 时间，不给虚构耗时占比。原 ROI 多边测量已不在身份决策链中。

## 6. 验收结论、限制与复现

**单主线结构、可维护性与安全机制测试验收通过；真实关联质量验收 No-Go。** 冻结版虽只损失 5 个相邻帧正确关系、旧 IDS 下降且耗时较低，但新增跨缺帧错误严重，不能用覆盖、碎片或相邻帧 3 次错误掩盖。最终已修复统一误差预算并通过全部测试，但没有第二次真实实验验证。

剩余限制：几何近邻仍不是物理身份证明；短历史无法启动高位移对象；未校准误差预算会拒绝困难对象；长时遮挡很可能创建新 ID。最终源版本的真实召回、跨缺帧错误和时延仍未验证，因此不能宣称满足完整稳定性验收，也不能作为生产启用依据。

所有执行使用 `taskset -c 0-3,6-31` 与 `/home/qh/miniconda3/envs/OmniDrive/bin/python`；未使用 CPU 4/5、升级依赖或终止他人任务。已存在的 `outputs/m3_state_guided_verified_decode/` 未触碰。

复现入口（仅说明，本任务没有第二次执行）：

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.world_identity_eval --output outputs/m1_single_path_future_unique
```

原始数据与版本证据：`main_method/summary.json`、四个 Scene JSON、`main_method.log`、`before_manifest.json`、`frozen_manifest.json`、`delivery_changed_files.json`、`analysis.json`。`analyze_delivery.py` 仅读取保存结果，不执行任何模型或关联器。本轮只证明上述已暴露开发协议下的代码行为，不能外推独立测试集。
