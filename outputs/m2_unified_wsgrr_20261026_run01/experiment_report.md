# OmniDrive M2：正式 WSGRR 唯一执行主线

日期：2026-10-26。代码、GPU 回归、真实 checkpoint 40 帧配对和两个真实场景切换均完成。全部代码/实验命令使用 `taskset -c 0-3,6-31`，Python 为 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，GPU 为 RTX 3090。

## 直接回答

| 问题 | 结果 |
|---|---|
| 正式 M2 是否不再依赖成本表和质量批准？ | 是。Runtime 不导入或实例化 CostModel，M2Config 和正式配置均删除成本/发布策略字段。 |
| 第一帧能否建立缓存？ | 是。重放首帧计算 108/108 个窗口并发布版本 1；完整模型测量首帧发布版本 2，原因是预热已产生版本 1。 |
| 正式入口是否真实非零复用？ | 是。同历史 M1 重放连续帧为 516/4212=12.2507%；完整实时 M1+M2+M3 为 350/4320=8.1019%（含首帧）。 |
| 与旧 D3 在相同条件下是否一致？ | 是。40/40 帧的计划窗口 ID、执行窗口 ID、缓存版本和实际复用数量全部一致。 |
| GPU 视觉和端到端是否加速？ | 否。完整模型视觉 CUDA span 平均 479.91→519.59 ms；端到端平均 2980.10→3342.97 ms。没有重新引入成本门控。 |
| 完整任务质量变化多少？ | 本单场景 40 帧子集 Detection mAP 0.288546→0.288720，Lane DET_l 0.232000→0.231822，VLM GT Token F1 0.359270→0.369617；2/40 问题出现关键数字代理退化。不能宣称全验证集或语义质量保持。 |

## 代码修改

- `worldstream/m2/runtime.py`：直接读取当前 source-linked M1 快照；Block 0 输入计算一次；WSGRR 选择后通过真实 `DependencyPlanner.plan_windows()` 闭包执行；后续全局依赖 Block 原样完整计算。删除 adaptive→capture_eligible→has_beneficial 发布链及 assessment/beneficial 引起的全窗口扩张。发布成功后才返回已完成诊断；容量不足、非有限值、OOM、非法上下文和不兼容缓存直接抛异常，不清空缓存伪装成功、不重试另一套执行器。
- 首帧、M1 明确强制刷新和新场景使用同一个 `WindowExecutor` 更新全部窗口；同场景非法状态报错。`StateCache.commit()` 的原子提交、数据所有权和容量约束保持原实现。
- `m1_adapter.py`：保留计算来源、租约、epoch、时间与掩码检查；去除重复时间判断，强制刷新不再提前绕过源和当前状态检查。真实来源过期仍刷新相应依赖，不减少必须窗口。
- `M2Config` 和 `worldstream_m123_auto.py`：删除 `cost_table`、`cache_capture`，旧字段现在明确作为未知参数拒绝；保持 ordinary_budget=3、max_reuse_frames=8、M1/M3 开关、模型权重和 M3 wsdr_kv/quality_budget=0.02/策略文件。
- `tools/worldstream_m2/offline_variants.py`：A/B/C/NoM1 选择消融移出正式算法；D3 为正式核心的直接别名。GPU 计算不复制，缓存签名和版本不区分 D3/正式。
- `CostModel`、晋升和历史成本分析工具保留供离线使用；正式 worker 的可选成本参数只绑定分析元数据，不注入运行时配置。正式 `--m2-acceptance` 不要求成本文件，检查已发布历史、真实执行计数和依赖计划一致性。
- 运行时增加汇总 `total/planned/executed/reused_windows`、每相机/Block、必须/普通窗口、cache_saved/version、full_refresh_reason 及 M2 执行/发布/总时间；删除执行成本批准和发布拒绝诊断。不会把“发生复用”标为“质量已认证”。

当前执行契约见 [说明](../../docs/worldstream_m2_direct_wsgrr.md)，完整改动见 [implementation.patch](implementation.patch)。M1/M3 算法、权重和旧正式策略文件均未改动。

## 测试与真实模型身份

最终全套 GPU 回归 **462 passed，0 failed，0 skipped**，见 [tests_gpu_final.log](tests_gpu_final.log)。后续仅清理工具层无用参数/导入及旧注释，针对工具再跑 **23 passed**，见 [tests_tool_cleanup.log](tests_tool_cleanup.log)。

覆盖：真实 EVA WindowExecutor、首帧发布、连续复用、版本推进、依赖闭包、实时 M1 snapshot、来源/版本/相机/时间错误、缓存篡改、nonfinite、容量/OOM/发布错误、场景重置、完整 M1→M2→Detection/Map→M1 Commit→M3 链。旧测试中要求“非法状态静默全量运行”及“未批准时零复用”的断言已改成新契约；历史成本工具仍有独立回归。

新增回归将 CostModel 构造和查询全部替换为抛异常，正式 M2 仍完成首次缓存及后续复用；强制全刷也不能绕过错误来源/非法掩码。

真实权重：`ckpts/iter_10548.pth`，SHA256 `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`。所有完整模型运行均严格校验 learned keys/shapes 并通过；没有用随机模型替代真实实验。历史 EVA 指纹、M1 回放预测哈希及数值策略与旧 D3 一致。

[provenance_verification.json](provenance_verification.json) 核对正式模型、场景切换和最终重放的核心源码与当前文件一致。原生 B0 的源码快照只差未启用的 M2 文件（当时尚未完成非法缓存检查顺序修复）；B0 的 M2 开关为 False，原生 EVA/任务头/LLM 没有该差异。最终重放已在修复后重新测量；初次日志另行保留，没有覆盖历史输出。

## 同输入和历史 M1 状态：正式入口与旧 D3 配对

使用旧 D3 相同 40 帧、预处理、checkpoint、M1 真实历史 Detection 回放和 WSGRR 参数。本次每个测量调用直接进入 `forward_from_m1()`，不调用 `evaluate_offline()`。每帧执行前保留同一不可变源缓存，重复测量不推进真实 M1 历史；每帧只提交一次 M1。

| 范围 | 总窗口 | 实际执行 | 实际复用 | 窗口复用率 | 发生复用的帧 |
|---|---:|---:|---:|---:|---:|
| 全部 40 帧（含初始化） | 4320 | 3804 | 516 | 11.9444% | 34/40 = 85.00% |
| 后续连续 39 帧 | 4212 | 3696 | 516 | 12.2507% | 34/39 = 87.18% |
| 连续帧 Block 0 | 2106 | 1848 | 258 | 12.2507% | — |
| 连续帧 Block 1 | 2106 | 1848 | 258 | 12.2507% | — |

40/40 帧发布缓存，版本 1→40。每帧缓存转换审计检查选中窗口年龄归零、未执行窗口来源/输出不变、未执行年龄推进、普通预算、必须闭包和实际计数。40/40 帧与旧 D3 的计划、执行、缓存版本及复用数量完全一致，见 [d3_pairing.json](d3_pairing.json)。

原始 [visual40/frames.jsonl](visual40/frames.jsonl) 包含所有执行 ID、每 Block/相机、窗口年龄、真实计算来源、强制刷新原因、缓存审计和 GPU/wall 原始样本；[results.json](visual40/results.json) 保存完整数值结果。

配对视觉测量每帧 warmup=1、repeats=3，以下是每帧中位数再跨 40 帧取均值：

| 测量 | 原生 EVA | 正式 M2 完整视觉运行时 | 加速比（原生/增量） |
|---|---:|---:|---:|
| 同步 Wall | 487.91 ms | 512.87 ms | 0.9513× |
| CUDA event span | 487.89 ms | 512.85 ms | 0.9513× |

视觉特征跨帧平均绝对差为 0.065007、最大绝对差为 78.954414；这不是精度认证，需看下游实际任务。

## 实时完整模型：M1+M2+M3

同一 40 帧自然连续图像、相同 checkpoint/预处理/问题选择/种子/原生采样与生成上限。B0 为原生完整计算，B3 开启正式 M1/M2/M3，没有离线 hook、成本表覆盖或 M3 参数修改。每帧一个真实有标注问题，共 40 问；两个条件顺序测量，各预热一帧，未进行大规模基线或消融。

| 范围 | 总窗口 | 执行 | 复用 | 窗口复用率 | 发生复用的帧 |
|---|---:|---:|---:|---:|---:|
| 实时 40 帧 | 4320 | 3970 | 350 | 8.1019% | 29/40 = 72.50% |
| 实时连续 39 帧 | 4212 | 3862 | 350 | 8.3096% | 29/39 = 74.36% |
| 实时 40 帧每个 Block | 2160 | 1985 | 175 | 8.1019% | — |

40/40 缓存发布，版本 2→41；预热后的 reset 清空内容但保留单调版本。首测量帧执行 108 个窗口发布版本 2，第二帧执行 106、实际复用 2、发布版本 3；M1 source-linked snapshot 读取和实际 attention batch 计数逐帧验证。正式验收 [formal40/m2_acceptance.json](formal40/m2_acceptance.json) 为 PASS，M1 场景账本完整性为 PASS。

实时 M1 与旧重放并非相同状态输入。28 帧复用数量不同；例如第二帧历史重放有 66 个区域、必须窗口 36，而实时有 87 个区域、必须窗口 40，并新增 `lane_control_points` 事件。实时状态由当前 Detection/Map 输出提交、会受增量特征及历史演化影响；旧回放由历史 Detection 预测维护。差异见 [replay_vs_live_m1.json](replay_vs_live_m1.json)，不是正式/离线算法分叉。没有删掉车道、对象或必须刷新窗口来追求相同复用率。

M1 强制刷新诊断：scene_change=1、periodic=4、no_history=1，这些事件可在同一帧重叠，不能相加当作刷新帧数。实际上 M1 整帧刷新 4 帧、依赖/WSGRR要求全部窗口的额外帧 7 帧，剩余 29 帧实际复用。

| 完整服务时延/资源 | 原生 B0 | 正式 B3 |
|---|---:|---:|
| 完整视觉 CUDA span 平均 | 479.91 ms | 519.59 ms |
| Prefill forward CUDA spans 平均 | 189.25 ms | 184.45 ms |
| Decode forward CUDA spans 平均 | 1885.58 ms | 1823.82 ms |
| 端到端平均 | 2980.10 ms | 3342.97 ms |
| 端到端 P50 | 1867.13 ms | 2471.74 ms |
| 端到端 P95 | 9217.60 ms | 8569.61 ms |
| 峰值 allocated GPU 显存 | 15.353 GiB | 16.012 GiB |
| 峰值 reserved GPU 显存 | 15.742 GiB | 16.822 GiB |

完整视觉加速比 **0.9236×**，端到端均值加速比 **0.8915×**，均低于 1。P95 有降低不代表平均服务获得加速。CPU host 提交时间不等于同步 GPU 执行时间；运行时 M2_execution_ms 等字段保留轻量 CPU 计时，实际性能以 CUDA span/同步 wall 及请求总时延评价。

M3 本次复用 token=0：1 问 cold_or_invalid_cache，39 问 no_quality_bounded_net_gain；现有 M3 策略按原逻辑执行，没有更改质量预算。生成 token 总数为 3022→2926，因此不能把 Decode 平均耗时降低归因于 KV 复用。Prefill/Decode 表使用两个条件同一 Observer 的 forward CUDA spans，Decode 不包含采样间隙；它们不能与嵌套总 span 相加。

## 下游质量与数值差异

使用同一预测文件运行未修改的 nuScenes/OpenLane 评估函数，对声明的 40 帧 GT 子集评分。不是 6019 帧完整验证集 mAP/NDS。Map 报告关注实际 lane head 的 DET_l；未学习的 traffic-element/topology 占位分数不能当作真实能力。

| 质量指标（原始比例） | 原生 | 正式 | 正式−原生 |
|---|---:|---:|---:|
| Detection mAP | 0.288546 | 0.288720 | +0.000174 |
| Detection NDS | 0.341442 | 0.341355 | -0.000087 |
| Lane DET_l | 0.232000 | 0.231822 | -0.000178 |
| Lane 3D F-score | 0.067281 | 0.068502 | +0.001221 |
| VLM GT Token F1 | 0.359270 | 0.369617 | +0.010346 |

30/40 回答相同，10/40 变化。平均 F1 上升约 1.035 个百分点，但 **2/40 问题存在关键数字代理退化，共 4 项检查退化**；不是零损失。逐问题证据在 [paired_vlm_quality.json](paired_vlm_quality.json)，包括 `88a6b5e35eb44deba6b1065d74b0a3c4` 的道路布局问题（数字 exact=True→False），以及 `fd612e8660e74196af7da7e6487e5d53` 的交通元素问题（数字 precision/recall 下降）。这些为可复现词法/数字代理，不能当作人工语义判断。

实际输出按类别/Hungarian 匹配，score≥0.3 的检测中心平均位移 0.180011 m，车道平均点位移 0.087493 m，见 [task_differences.json](task_differences.json)。该匹配统计不是 AP，独立子集 AP/NDS/DET_l 已在上表给出。

原始评估：[原生](native40_official_subset/summary.json)、[正式](formal40_official_subset/summary.json)。原始预测与 QA 分别在 native40/formal40 的 predictions.pkl 和 per_question_metrics.jsonl。

**本次视觉特征已不同于旧 M3 在完整 M2 下的输入。40 帧联合结果已测量，但旧 M3 独立精度结论不能继承；仍需独立多场景联合质量验证。**

## 场景切换

额外运行两个真实场景各 2 帧，完整 M1/M2/M3 每帧均完成：

| 帧 | 场景 | 缓存版本 | 执行/复用窗口 | 结果 |
|---|---|---:|---:|---|
| 0 | 01452f… | 2 | 108/0 | 首帧全更新，来源全部为本帧 |
| 1 | 01452f… | 3 | 106/2 | 连续帧增量 |
| 2 | 905cfe… | 4 | 108/0 | 切换场景，来源全部为新场景本帧 |
| 3 | 905cfe… | 5 | 104/4 | 新场景连续帧增量 |

M1 epoch 从 6→7，版本单调推进。场景初始化没有复用旧场景窗口，也没有切换备用全量执行器，见 [scene_switch_summary.json](scene_switch_summary.json)。

## 剩余性能瓶颈与边界

- 24 个 EVA Block 只有前 2 个可缓存。完整服务中前两块 CUDA span 平均合计约 36.19 ms，后 22 块仍需完整执行，约 435.02 ms；仅在此前缀复用 8.10% 窗口，节约上限很小。没有关闭真实全局依赖。
- 窗口 gather/scatter、按窗口形状的算子、M1 快照读取和状态检查/打分有开销。正式 M2 验证（含打分）CPU 均值 19.20 ms，其中打分 7.54 ms；M1 含适配器/提交平均 313.11 ms。这些开销足以覆盖局部节省。
- 缓存发布计时 430.98 ms 含前面异步 GPU 计算在有限值 `.item()` 检查处的等待，**不能把它解释为缓存 clone 本身耗时**。独立 clone/有限值检查的纯 GPU 成本尚未隔离。本次完整 CUDA span 和同步 wall 的变慢是真实测量，不受这一归因限制影响。
- 只测一个开发场景、每帧一个问题、每条件一次端到端 pass；自然 Decode 长度和 GPU 状态会影响尾时延。没有通过本结果宣称统计显著加速或全验证集精度认证。
- Runtime 保持直接 WSGRR，未因变慢再引入成本/质量准入或全量替代路径。后续优化应降低真实执行和状态管理开销，并继续测量联合质量。

## 复现与文件

完整实际命令/退出码见 [commands.json](commands.json)。基于当前代码、现有数据与权重，使用新输出目录重新运行：

```bash
taskset -c 0-3,6-31 bash outputs/m2_unified_wsgrr_20261026_run01/reproduce.sh outputs/m2_direct_reproduction
```

脚本顺序执行全部测试、40 帧正式重放、原生/正式完整模型、场景切换、GT 子集评估和统计。不会覆盖本轮或历史结果；输出目录需为新路径。`CUDA` 运行需要与本轮相同的 GPU 设备访问权限；沙箱内 CUDA 不可见的初次尝试保存在 visual40_sandbox_unavailable，仅作为失败环境日志，不计入实验结果。

机器汇总：[comparison_summary.json](comparison_summary.json)。环境：[environment.json](environment.json)。源码：[source_hashes.json](source_hashes.json)。保护项：[protected_verification.json](protected_verification.json)。完整服务逐帧执行与计时：[formal40/per_frame_metrics.jsonl](formal40/per_frame_metrics.jsonl)。
