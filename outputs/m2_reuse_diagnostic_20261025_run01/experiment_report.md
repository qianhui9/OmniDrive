# M2 D3 窗口复用根因诊断

日期：2026-10-25。本轮只修改 M2 诊断、必要的缓存发布来源校验、既有 window_validate 的结果记录和针对性测试。**真实 checkpoint 的 40 帧 D3 离线验证已完成，34 个合格帧全部实际复用；生产复用没有批准，也没有运行生产 B3。**

## 零复用的直接根因

此前正式 B3 的 0% 来自生产准入和缓存发布策略：

`window_costs.json.records=[]` → `has_beneficial()` 为 False → `capture_eligible=False` → 正式 adaptive 主动不发布缓存 → 后续帧继续 `no_valid_cache`。

同时，缺少与当前模型、WSGRR 策略、实际更新数量匹配的成本/数值/质量/批准证据，即使已有有效候选缓存，生产 `beneficial()` 仍会拒绝部分更新。**缓存不存在和成本证据不足是不同事件；首次完整计算成功也不能批准下一帧复用。** 原代码将有效空 JSON 表标成 ready，且未保留主动不发布的根因，导致现象看起来只是普通缓存 miss。

本轮不将 adaptive 改为 always，也不放开空成本表时的生产发布或复用。既有离线 D 入口本来就允许隔离的候选缓存初始化；通过它收集证据即可。生产与离线缓存的 `production` / `offline_D` 签名隔离保持，已有隔离回归通过。

## 修改文件与诊断语义

| 文件 | 修改目的 |
|---|---|
| `worldstream/m2/cost_model.py` | 空记录表明确为 empty_cost_table；增加针对确切计划数量的 assessment，保留 beneficial 的所有原有判定，仅细化缺失/无效证据的原因。 |
| `worldstream/m2/runtime.py` | 增加缓存发布决定、前次拒绝根因、成本记录/键/数值/质量/批准/收益状态、原始 selector 计划与成本前后计划、逐相机执行统计、必刷闭包和 ID 一致性诊断。 |
| `tools/worldstream_m2/window_validate.py` | main-only 固定单场景连续序列；沿用正式 M2 配置的 adaptive 与成本表，通过现有离线 API 执行 D3；保存新增诊断、原始 block 计时样本、合格帧分母和更完整的缓存审计。 |
| `tests/worldstream_m2/test_reuse_diagnostics.py` | 新增 14 个实例，覆盖真实 M1 连续事务初始化/复用、成本缺失、原始计划与扩展执行区别、不同窗口大小的闭包、刷新/来源/事务、预算与非有限发布失败。 |

`cache_capture_decision` 分开记录 policy_allowed、attempted、allowed、published。主动不发布的 rejection_reasons 与实际提交失败的 publication_failure_reason 分开；预算不足、非有限特征和 OOM 保留具体原因。allowed 最终为 False 的发布失败仍会显示 attempted=True，不能误记为没有尝试。

每个缓存 Block 记录 cost_record_found、cost_key_matched、numerical_verified、quality_certified、approval_valid、beneficial、cost_rejection_reason。空表无需计算昂贵的模型成本指纹即可确定没有匹配记录，不为诊断新增该 D2H 开销。离线记录的这些状态表示“若走生产准入时的证据状况”；cost_admission_applied=False，不能解释为已获得生产批准。

selector_planned_windows/IDs 保存成本扩展之前的原始选择；planned_windows/IDs 保存进入当前 Block 成本判定前的计划；executed_windows/IDs 与 reused_windows 来自实际执行。在前层成本拒绝导致后层依赖扩展时，后层原始 selector 计划和扩展后的计划仍可分辨。region_refresh 与逐 Block 的 per_camera 均保留必须、普通、实际执行及复用计数；通过已验证的 CPU 依赖足迹映射不同窗口大小的必刷闭包，不复制特征。

额外修复一处发布边界：缓存生命周期不匹配时，原逻辑可能先记录缓存失配、跳过当前 M1 来源检查，再发布全量结果。现在用空候选缓存检查当前 M1 上下文，避免解引用损坏缓存，也避免 scene mismatch 掩盖 aborted 等无效来源。来源拒绝会阻止发布；当前有效来源仍可在完整重算后建立新缓存。这项修改收紧发布安全性，没有降低复用准入。

`state_cache.py` 的有限性、预算、所有权和原子发布未改；`region_refresh.py` 的评分、事件、阈值与预算未改；`promote_costs.py` 的独立质量和成本批准流程未改。M1/M3、正式配置和四份生产成本/质量文件均未修改。普通推理未新增 `torch.cuda.synchronize()`；诊断复用已有小型窗口 ID 与 CPU 元数据。

## 实际测试

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m pytest -q -rs tests/worldstream_m2

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m pytest -q -rs tests/worldstream_m123
```

运行时通过父进程设置 OMP_NUM_THREADS=4、MKL_NUM_THREADS=4、OPENBLAS_NUM_THREADS=1；CPU 亲和性及子进程继承均避开故障 CPU 4/5。

| 本轮运行 | 通过 | 失败 | 跳过 | 证据 |
|---|---:|---:|---:|---|
| 中间 CPU 回归 | 181 | 1 | 2 | `tests_cpu.log`：不同 window_size 的测试夹具未同步更新 RoPE，修复夹具后通过。 |
| 最终 CPU 回归 | 182 | 0 | 2 | `tests_cpu_final.log`，2.96 s；沙箱 CUDA 不可见，两项专用测试跳过。 |
| GPU 可见环境 M2 回归 | **184** | **0** | **0** | `tests_gpu.log`，4.00 s；两项 CUDA 测试均实际执行。 |
| 既有正式入口/审计兼容测试 | **54** | **0** | **0** | `tests_entry_compatibility.log`，2.66 s；单元测试，不是生产 B3 GPU 验证。 |

开发早期小集还发现一次成本测试夹具状态不一致：手动替换 records 后仍保留 empty 状态，断言失败；现已改为实际解析临时、未批准的候选 JSON，覆盖真实非空表缺少确切更新数量的场景。该失败只出现在本轮终端记录，没有将其声称为生产推理故障。

非零复用回归使用真实 WorldStateManager 的 Precheck → export → D3 → Commit，不修改全量刷新标志、租约或生产批准。局部输入变化仍由真实来源链和选择器处理；受控 CPU 输出按既有 3e-6 atol/rtol 检查。生产空表测试连续两帧均全量执行，第二帧明确记录 cache_miss_root_cause=empty_cost_table；确切数量缺失测试验证计划 6 → 实际 18，并保留下游原始计划 6 与依赖扩展计划 18 的区别。

## 真实 GPU 验证命令与环境

实际执行命令也保存在 [invocation.json](invocation.json)：

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -u \
  -m tools.worldstream_m2.window_validate --wsgrr --main-only \
  --frames 40 --repeats 3 --warmup 1 --budgets 3 \
  --checkpoint ckpts/iter_10548.pth \
  --m1-predictions outputs/wsgrr_mainfix_20261023/references/wsgrrr_20261022_followup/Native_full/official_eval/detection/pts_bbox/results_nusc.json \
  --profile-block-costs --output outputs/m2_reuse_diagnostic_20261025_run01/visual40
```

RTX 3090；OmniDrive Python 3.9 环境，PyTorch 1.13.1+cu117 / CUDA build 11.7。真实 EVA 从正式 checkpoint 严格加载，沿用正式图像预处理、v2 数值策略及 D3 每相机普通预算 3。未运行其他预算、消融、完整验证集或额外模型基线。Native 只是现有工具对相同输入的配对视觉成本参考。

同一场景 `01452fbfbf4543af8acdfd3e8a1ee806` 连续 40 帧，跨度 19.5 s，无帧数缩减。每帧预热 1 次、测量 3 次，重复执行从同一不可变来源快照开始，不把重复计时当作新历史帧。首帧实际初始化成功，候选缓存共发布 40/40 次，版本依次 1–40。后续强制刷新照常执行，不清除必须刷新事件。

M1 回放是已有 Native 全模型真实 Detection 预测，不是标注，也不是本轮候选下游预测。预测文件 SHA256 为 `4bc5de624719fe6763baa0e2ad1eb6f6b89b13af57c0834a7c08db105d913d9b`。完整 checkpoint 再次计算 SHA256，与此前正式记录一致：`6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`。

## 实际复用与一致性

| 分母 | 实际复用帧 | 帧复用率 | 实际复用窗口 / 总窗口 | 窗口复用率 |
|---|---:|---:|---:|---:|
| 全部 40 帧，含初始化 | 34/40 | 85.00% | 516/4320 | 11.9444% |
| 后续 39 个连续帧 | 34/39 | 87.1795% | 516/4212 | **12.2507%** |
| 来源有效、无全量刷新且 selector 允许部分更新的合格帧 | **34/34** | **100%** | 516/3672 | 14.0523% |

合格帧定义与实际帧 token 均记录，不把缓存缺失、强制刷新或全部窗口确实需要更新的帧放进非零复用验收分母。全部 40 帧中，m1_full_refresh=4 帧（含初始化），all_windows_required=2 帧，共 6 帧合法全量执行。后续 39 帧则为 5 帧全量。

| 缓存 Block，统计全部 40 帧 | 原始计划 / 成本前计划 | 实际执行 | 实际复用 | 总窗口 |
|---|---:|---:|---:|---:|
| 0 | 1902 / 1902 | 1902 | 258 | 2160 |
| 1 | 1902 / 1902 | 1902 | 258 | 2160 |

本轮离线没有成本拒绝扩展，因此两种计划与执行数一致；生产拒绝扩展由单元测试单独验证。所有必刷窗口均执行，来源版本、缓存复用输出不变、年龄、描述符和普通预算等 **812 项逐帧检查全部通过**；所有新增 selection_consistency 也为 True。逐相机统计和具体窗口 ID 均见 `frames.jsonl`，没有重复保存大型特征。

## 数值差异和耗时是否合格

结论分开：执行与来源一致性通过；非零离线复用通过；**生产数值/下游质量未获批准；整体视觉净成本没有通过。**

真实流采用既有 wsgrr_quantitative_v1：对当前原生全量输出量化差异，不用旧全形状策略的位级一致要求替代，也不自行放宽独立质量阈值。后续 39 帧的逐帧 MAE 平均 **0.06667375**，全序列最大绝对差 **78.954414**。Block 成本微测中，Block 0 输出对照最大差 16.946598，Block 1 为 0.0016823；Block 0 对照包含复用的旧窗口输出，不能将其全部解释为选定窗口 GEMM 的舍入误差。

当前没有与本轮候选哈希绑定的独立 feature_error_limits 或 Detection/Map/VLM 质量报告，因而不能判定这些真实流误差达到生产质量标准。候选中的 numerical_verified 沿用既有“局部差异有限”的必要条件，新增 numerical_validation_scope 明确这一范围；它不是独立质量通过标志，也不使 quality_certified 变为 True。

以下为后续 39 帧全部 117 个重复样本的池化 wall-clock，避免与逐帧中位数的汇总口径混淆：

| 完整视觉路径 | 平均 ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| D3：Embedding、准入、评分、前缀与全局后缀、发布及诊断 | 495.644 | 496.070 | 505.218 |
| 同轮 Native 视觉参考 | 490.379 | 491.286 | 497.177 |

合格 34 帧的 102 个重复样本，D3 P50=495.605 ms，Native P50=491.501 ms，同样没有整体净收益。工具原有“逐帧中位数再取中位数”分别为 D3 497.034 ms、Native 492.425 ms，保存在 `results.json.aggregate.D3`，不可与池化值混用。

同一逐帧汇总口径的 CUDA stream span 中位数为 D3 497.017 ms、Native 492.412 ms，完整原始 GPU 样本也已保留。

计时使用既有请求边界 CUDA 同步与 CUDA Event，预热不计入样本。stream span 包含主机提交空隙，不等于活跃 kernel 时间；M1 Precheck/read 单列，不计入上述视觉路径。独立 block 和缓存发布成本在测量外额外采集，其样本不与视觉路径嵌套相加。跳过窗口比例不代表端到端加速比例，本轮没有测完整 B3 请求加速。

## 候选与后续正式批准流程

输出了 **30 条 key/count 成本候选、78 条逐案例 Block 观测**，并保留原始 wall/GPU 样本与差异。6/30 条候选在本次其全部覆盖案例上同时满足 block 和完整视觉管线的 wall/GPU 更快，涉及两个 Block 的更新数 30、34、36；这不代表全策略或生产收益通过。它们仍缺独立数值限值、下游质量及更充分的稳定成本证据，全部 quality_certified=False、production_ready=False、approved_counts=[]。

接下来应沿用现有流程，而不是修改表格制造通过：

1. 对具体候选 key/count，在隔离的现有离线全模型 worker 中取得候选真实 Detection/Map/VLM 输出；同输入参考可使用经过身份核验的已有 Native 输出。回放预测只维护 M1，不能当作候选任务预测。
2. 使用现有 `tools.worldstream_m2.task_quality` 对候选实际预测进行正式 evaluator 评分，并对真实语言输出独立评估；按 `m2_window_task_quality_v1` 提交候选 SHA256、assessor、每个 key/count 的 covered_cases、独立 feature_error_limits，以及 detection/map/vlm 的真实指标、evaluator 与证据文件哈希。不得用本轮有限误差标志自动生成“passed”。
3. 只有这些证据均通过，且每个覆盖案例满足 block 和净视觉管线 wall/GPU 成本门槛，才由研究者明确选择批准数量与身份，调用既有 promote_costs.py。它会重新加载当前正式模型、检查完整策略键、数值/任务/证据哈希与成本，本轮没有运行晋升。

未来满足条件后的命令形式（本轮未执行）：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m2.promote_costs \
  --candidate <本轮window_costs_provisional.json> \
  --quality-report <独立评估的m2_window_task_quality_v1.json> \
  --checkpoint ckpts/iter_10548.pth --approved-counts <明确通过的数量> \
  --approved-by <研究者身份> --confirm-approval --output <明确选择的批准表路径>
```

**只有正式批准后再运行生产 B3 验证。** 本轮没有独立批准，故不宣称生产实际复用率达标，也没有启动 B3、完整任务评估或新的大规模基线。

## 证据与最终边界

- [专项统计](diagnostic_summary.json)、[完整结果](visual40/results.json)、[逐帧 JSONL](visual40/frames.jsonl)、[候选成本](visual40/window_costs_provisional.json)、[连续序列身份](visual40/sequence.json)
- [GPU 原始日志](gpu_validation.log)、[M2 GPU 测试](tests_gpu.log)、[正式入口兼容测试](tests_entry_compatibility.log)
- [权重、38 个受保护文件及 41 个实际测量源码的哈希核对](provenance_verification.json)、[语法及 diff 检查](static_checks.json)

最终交付只包含三个必要的既有 M2 代码文件、一个新增 M2 测试文件和本轮独立输出目录。生产成本/质量文件保持原样，未删除旧结果。未解决事项是当前策略的独立数值与任务质量、稳定净成本收益和正式批准后的真实 B3 复用/端到端效果；这些不能由离线非零复用率替代。
