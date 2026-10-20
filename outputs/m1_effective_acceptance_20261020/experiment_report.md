# OmniDrive M1 有效化与独立验收（2026-10-20）

已直接实施代码修复。**Gate A PASS（运行与状态维护），Gate B BLOCKED（身份/物理资格），Gate C PASS（状态接口）**。M1 不满足进入 M2 的全部前置条件，未开始 M2。

基准 HEAD `169c2969a5daa566990583db72bf0288845774f0`。实际仓库 `/data/qh/phdCode/work4/OmniDrive`。Python `/home/qh/miniconda3/envs/OmniDrive/bin/python`；执行亲和性 `0-3,6-31`。最终 GPU 验证使用 RTX 3090，完整权重、Tokenizer、配置、数据摘要及亲和性见 `full_model_verified/provenance.json`。运行前检查 GPU 空闲，未停止其他任务。

## 实现与实际链路

19 个代码/配置/测试文件的逐项原因、生命周期及接口语义见 [implementation_changes.md](implementation_changes.md)，补丁见 `implementation.patch`。

`forward_test → simple_test → frame_from_model → precheck（输入/来源/历史观测/实际 LK 与 DIS 测量）→ 完整 EVA → 完整 Detection/Map → m1_observer → outputs_from_model → commit（关联/证据/资格/新版本/单点发布）→ 原生 Planning/LLM`。

采用独立配置 `configs/worldstream_m1_effective.yaml`，保持 `observe_only`；开启真实时间观测、前景与有限身份测量，身份跟踪为 shadow；M2/M3 关闭。没有通过修改开关宣称身份有效。`valid_anchor_ids` 仍只表示历史观测支持，当前提交认证通过独立 API 输出。新检测不能继承旧版本证书。

事务绑定精确图像、位姿、相机、时间、元数据、执行签名、Epoch、Store 来源以及历史相机快照。同 Epoch 内非相邻重复帧也拒绝；场景/布局/执行变化及时间异常建立新 Epoch。普通异常回滚到最后可信提交；外部来源改写后丢弃被污染状态和历史图像，保留 aborted/source_state_discarded 原因。模型执行故障及时终止 pending 事务。当前 Feature 全部明确为未捕获 Tensor，不能当作可执行缓存。

## 本轮真实测量

| 实测量 | 完整模型 | CPU 实际图像回放 |
|---|---:|---:|
| 唯一场景/帧/相机图像 | 1 / 8 / 48 | 2 / 24 / 144 |
| 当前实际提交 Anchor 事件 | 451 | 655 |
| 当前图像具有有效裁剪观测的 Anchor 事件 | 238 | 581 |
| 历史投影候选事件 | 489 | 649 |
| 历史 Anchor 观测有效事件 | 0 | 0 |
| 实测双向对应点（跨帧/相机累计） | 11045 | 9944 |
| 通过局部图像/几何支持的对应点 | 3258 | 1258 |
| 稳定观测像素 / 总像素 | 15 / 19660800 | 0 / 33177600 |
| 身份/物理正式认证 Anchor | 0 | 0 |
| 神经 Tensor 存储字节 | 0 | 0 |

事件是跨帧累计，重复运行不增加独立样本数。CPU 回放使用历史实际模型检测和真实连续图像，无在线 GT；它使用训练集中既有曝光场景、640px 原图坐标，不是完整模型执行。GPU 使用原始配置数据及预处理坐标，按探索性人口报告，两者覆盖率不能混用。逐帧来源与完整分母见 `real_stream_metrics.json`。

完整模型 Anchor 观测覆盖率为 0/489；稳定观测像素比例 0.00007629%。身份认证覆盖为 0，绝不能因此报告零错误接受风险。实测 FAR 及其置信上界：**Not Run / 未定义**，原因是没有合法认证和独立可评分接受人口。大量真实对应点并不证明前景归属或身份；低覆盖现状没有被阈值调整掩盖。

完整模型证书拒绝原因分布：`{"measured_foreground_ownership_missing": 84, "measured_identity_correspondence_missing": 84, "trusted_measured_identity_evidence_missing": 58}`。前景生产拒绝：`{"no_distinguishable_foreground_view": 84, "static_ray_hypothesis_contradicted_or_unresolved": 27, "identity_ambiguous": 38, "actual_source_frame_unavailable": 26}`；身份生产拒绝：`{"measured_motion_identity_observation_missing": 170}`。具体相机原因、CPU 回放分布、帧级分母见 `real_stream_metrics.json`。证书尝试与候选边、Anchor 事件是不同分母，未合并。

最终完整模型开启/关闭 M1 的 backbone、检测/地图输出及原生 temporal memory 摘要完全一致；8 个实际生成的 Planning 文本文件也逐字节一致。源码前后摘要一致。证据：`full_model_verified/final_verification.json`、`results.json`、`language_comparison.json`。前两轮相同 8 帧的完整模型对照记录保留在 `full_model/` 和 `full_model_final/`；最终报告只使用 `full_model_verified/` 的冻结源码。没有将配对一致性宣称为完整数据集任务质量认证或加速。

## 独立资格为何阻塞

本地 nuScenes 验证元数据有 6,019 帧、150 场景，但仅 **18 个独立日志**。固定可行性划分按日志隔离：3 个日志用于校准、15 个用于验证。即使这 15 个单位真实零错误，95% 单侧精确上界也为 **18.104%**；把全部 18 个日志都用于验证也仍为 **15.332%**。现有门限为 ≤5%，零错误时至少需 59 个独立单位；重复帧或同日志不同场景不能凑数。这些是样本量可行性上界，**不是实测零错误结果**。

标注表存在，但缺少当前生产者/协议独立评分、足量互斥场景/日志，以及可判定的真实接受人口；历史已曝光结果没有被转成新的资格。正式在线认证还需真实评估日志来源，缺失则继续拒绝。没有生成资格文件，保留风险门和摘要固定要求。后续必须提供足量合法独立日志及校准/验证数据、真实评分和可追溯资格工件后才能重新验收 Gate B。

## 测试实际结果

最终命令为 `commands.sh` 中的 pytest 命令，在可访问 CUDA 的环境运行。**716 passed，0 failed，0 skipped**：M1 403、M2 127、M3 131、跨模块 55。其中本轮新增 38 项，其余 678 项为既有或因更严格来源契约而更新的测试。详见 `regression_verified.xml` 与 `.log`。

覆盖正常提交、真实入口、场景/时间/相机/执行变化、过期、事务超时及失败、低纹理/遮挡/动态/背景歧义、竞争/重复点、版本/签名/摘要篡改、缺失资格、日志/场景泄漏和风险失败。独立验收器还测试了未来来源拒绝。合成几何与 60 个零错误单位的夹具仅用于软件契约，未用于资格或实测统计。

修改前实际 M1/M2/M3 回归为 622 passed、1 skipped（沙箱 CUDA 不可见），见 `baseline_test_record.json` 的工具结果转录；它不含跨模块测试或本轮新增项。旧报告 658 等数字属于历史执行，不与本轮相加。最终使用授权 CUDA 环境消除了跳过；冻结质量门限与原模型计算路径均保持。

## 门结论和复现

- **Gate A — PASS，限操作性。** 完整模型真实调用、原子提交、来源/生命周期/回退验证和实际测量可重复。有效 Anchor 历史观测仍为 0；15 个稳定像素不代表实用覆盖或可信世界身份。
- **Gate B — BLOCKED。** 无当前合法独立资格，现有日志规模达不到门限，真实候选全部保守 Unknown，没有可报告的合格 FAR 或置信界。
- **Gate C — PASS，限状态接口。** 状态版本和来源可提交、核验、失效及防御性导出；未捕获神经特征明确不可用。未声称 M2 跳算或 GPU 加速。

`formal_gates.json` 分别保存结论；`m1_prerequisites_for_m2=false`。复现命令见 [commands.sh](commands.sh)，需指定全新输出目录。资格盘点不需要 GPU；完整模型验证需要空闲 GPU。本轮神经网络跳算、KV 复用与加速量均为 0；官方全量任务质量评估和独立身份资格评分为 Not Run，不计通过。

首个 CPU 回放因错误读取验证集索引、与训练集历史检测不相交而失败，记录保留在 `rgb_replay.log`；修复为从原检测选择文件读取真实 info_file 后重跑。来源检查强化期间出现的旧测试预期差异已更新并通过最终回归。`full_model_release/` 是被最后的来源损坏重置修复取代、仅停止本会话自有进程的未完成执行，保留中止原因，未计通过。所有历史实验和原先未跟踪的 M3 目录均保留。
