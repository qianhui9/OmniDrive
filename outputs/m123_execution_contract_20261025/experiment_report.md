# M1/M2/M3 正式请求执行契约审计与修复

日期：2026-10-25。唯一正式配置：`projects/configs/OmniDrive/worldstream_m123_auto.py`。

本次完成代码审计、最小控制流修复、逐请求诊断及轻量测试；没有执行 6019 帧验证集、加载生产模型权重或开展 GPU 性能实验。所有 Python 执行及测试均使用 OmniDrive 解释器，并由 `taskset -c 0-3,6-31` 限定 CPU 亲和性。

## 1. 真实调用链及审计证据

`tools/test.py` → `custom_multi_gpu_test`（`projects/mmdet3d_plugin/core/apis/test.py`）或既有 benchmark worker → `model(return_loss=False, ...)` → `Petr3D.forward` → `forward_test` → `simple_test`。

正式请求在 `simple_test` 内的顺序：

1. 原始输入适配、执行签名绑定 → `WorldStateManager.precheck_with_observation(frame, True)`。
2. 在当前感知 Commit 之前冻结 M3 历史快照，向 M2 绑定当前 frame/time/cameras/epoch/signature。
3. `extract_img_feat` → `IncrementalVisionRuntime.forward_from_m1` → `_from_m1` → `manager.export_m2_inputs` → `forward` → `_forward` → `_execute` → 唯一 `WindowExecutor.execute`。
4. `simple_test_pts` → 原始 ROI/Detection/Map 流程 → `_m1_observer` → `outputs_from_model` → `WorldStateManager.commit`。
5. `M3Runtime.generate` → `query_world_state` → 独立 `Scheduler.state_allowed` → 可选历史文本插入 → `expand_input` → 精确前缀匹配 → `Scheduler.prefix_allowed` → 唯一 `PrefillExecutor.generate` → `GenerationFacade` 使用已安装 HF 生成循环，并委托原模型完成 Decode。

已有满足要求的实现，无需重写：

- M1 的 Precheck 不依赖 M2 缓存或 M3 历史可用性。`state_manager.py` 保留私有事务副本、输入摘要、版本、epoch、超时、历史来源摘要和 Commit 校验；本次没有修改它。
- M2 `_forward` 的来源、缓存或上下文拒绝会设置 `plans=None`，仍调用 `_execute`。`_execute` 的成本/数值/质量拒绝将更新集扩展到全部窗口，并闭合后续前缀依赖。完整窗口在同一个 `WindowExecutor` 内调用原生完整 block，后续全局依赖后缀及 Detection/Map 保持执行。缓存发布失败清理缓存，保留已计算输出；没有提前返回绕过计算。
- `cost_model.py` 的数值、质量、批准文件哈希和收益检查保留；没有为测试修改生产批准文件。
- M3 的历史准入只影响插入文本；缺失或拒绝时保持原始问题 ID 和完整当前视觉载体。KV 未命中或拒绝时 `reuse_length=0`，仍由 `PrefillExecutor` 运行完整 Prefill。`scheduler.py` 的两个独立准入检查及精确匹配约束均保留。
- 已有答案文件只会跳过关闭 M3 的原生参考生成路径；启用 M3 时原本就会重执行。本次验证并保留该行为。

实际发现并修复的缺口：

- 模型包装曾根据 `m3_answers is None` 决定是否调用原生 `lm_head.generate()`。正常 M3 不会返回 None，但异常返回契约可能静默绕过 M3。现在只允许在 M3 未启用时调用原生参考生成；启用 M3 却返回 None 或答案数不匹配时显式报错。
- 正式配置没有对模块缺失、视觉调用被意外绕过、未完成 Prefill/Decode 等情况进行请求级一致性检查。现在启用 `worldstream_require_m123=True` 并检查实际调用标记和运行时完成诊断。
- 空语言问题列表可能使 M3 返回空结果而没有任何 Prefill。正式 VLM 入口现在将其明确报告为无有效语言请求的输入错误，不伪造问题或 Prefill 成功。B0 原生参考行为不受该约束影响。
- M1 历史快照发布原来位于请求异常清理边界外；发布异常可能遗留待提交事务。现在从输入适配至生成均处于同一异常清理边界内。
- 原有 M1 日志在 Commit 时写出，早于 M3，缺少完整请求结果；原有 M2 顶层原因在成本拒绝后可能仍为 None。现在补齐真实执行诊断，并在 M3 完成或异常之后向既有日志写一次完整请求审计。

## 2. 修改文件及目的

| 文件 | 修改目的 |
|---|---|
| `projects/configs/OmniDrive/worldstream_m123_auto.py` | 唯一正式配置启用三模块执行一致性约束。 |
| `projects/mmdet3d_plugin/models/detectors/petr3d.py` | 请求级真实执行标记、正式入口完整性检查、M3 答案契约校验、异常事务清理、延后既有审计写出。 |
| `worldstream/m2/runtime.py` | 记录运行时进入/完成、真实窗口复用、全量重算及实际原因；计算图和准入逻辑保持。 |
| `worldstream/m3/runtime.py` | 记录运行时进入/完成、问题序号、实际历史文本消费、实际 KV 复用后的模式与完整 Prefill 原因。 |
| `worldstream/m3/prefill_executor.py` | 在真实 first-forward/生成完成位置记录 Prefill 进入/完成、完整 Prefill、Decode 完成；禁止未执行 Prefill 却报告成功。 |
| `tools/worldstream_m123/benchmark/worker.py` | 既有逐帧记录增加紧凑 `module_execution`；既有 B1/B2 离线消融显式不启用正式三模块约束，B3 保持约束，B0 保持原生参考。未新增配置或执行器。 |
| `tests/worldstream_m123/test_always_execute.py` | 新增贯穿实际推理方法、真实 M1、微型 EVA 和微型 Llama 的集成测试，以及拒绝、异常、跨请求诊断测试。 |
| `tests/worldstream_m123/test_benchmark.py` | 检查正式约束与现有 B0/B1/B2/B3 条件的隔离。 |
| `tests/worldstream_m123/test_final_route.py` | 检查正式配置启用约束、原生基线未启用约束。 |
| `tests/worldstream_m2/test_window_completion.py` | 验证真实增量诊断；成本、缓存、来源、数值、缓存发布拒绝均通过统一执行器完整计算。 |
| `tests/worldstream_m3/test_trusted_history.py` | 验证获准时历史文本确实消费、拒绝时仍完整 Prefill 并 Decode。 |
| `tests/worldstream_m3/test_execution.py` | 验证获准 KV 复用时真实 Prefill/Decode 完成标记。 |

没有修改 `state_manager.py`、`cost_model.py`、`scheduler.py`：真实控制流已满足其职责。本次没有修改模型权重、任务头、原生 Decode 算法、预处理或任何生产准入文件。基线配置、全部 dense head 源文件、M1 正式配置和四份生产证据文件均已与 Git HEAD 逐字节核对，记录于 `source_verification.json`。

## 3. 优化获准和未获准时的实际行为

| 模块 | 优化获准 | 优化未获准或历史缺失 |
|---|---|---|
| M1 | 真实 Precheck 校验和提交后的状态维护；按来源及有效性导出可信计算状态。 | 同样运行 Precheck 和当前感知后的 Commit，记录失效/刷新原因，不放宽有效性。 |
| M2 | 使用通过全部准入的依赖闭合更新集，真实复用未更新窗口。 | 同一个 `_execute` / `WindowExecutor` 执行全部窗口和全局依赖后缀，继续 Detection/Map；缓存是否发布只影响后续请求。 |
| M3 历史语义 | 可信且任务适用、独立质量准入通过时插入历史文本，保留全部当前视觉载体。 | 不插入历史文本，原始问题和当前视觉载体原样进入扩展输入及 Prefill。 |
| M3 KV/Prefill | 精确匹配且成本/数值/质量均获准时复用实际可用 KV 前缀，执行剩余 Prefill 和原生 Decode。 | `reuse_length=0`，同一 PrefillExecutor 执行完整 Prefill 和原生 Decode。 |

`historical_text_consumed` 来自实际插入及已完成的 Prefill；`reused_tokens` 来自执行器实际使用长度，包括执行器为末位 logits 调整后的长度，不能用匹配长度或配置开关代替。`actual_window_reuse` 来自实际 stage 的复用窗口计数。

## 4. 诊断读取及测试

每次主推理后可读取 `model.worldstream_execution`。它绑定 scene/frame/timestamp，区分 M1 Precheck/Commit 进入、完成、事务 committed/aborted/rejected，M2 进入/完成/复用/全量原因，以及 M3 按问题序号的实际历史消费、KV token 数、完整 Prefill 原因和 Prefill/Decode 完成。请求开始会清理上一请求的诊断，不清理有效计算缓存。

既有 benchmark 的逐帧 JSONL 直接保存 `module_execution`；开启既有 M1 logging/log_path 时，同一审计 JSONL 的 `experiment.module_execution` 保存完整请求诊断。默认主配置仍保留原有 `logging:false`，不会另建默认磁盘日志系统。运行时错误保留 `completed:false` 和错误信息；当前感知已经 Commit 后的 M3 失败不会错误报告为 M1 已回滚。

测试命令：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q -rs tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123
```

结果：**354 passed, 2 skipped, 23 warnings in 6.37s**。日志：`tests.log`。两项跳过的现有 CUDA 专用测试为 `tests/worldstream_m2/test_execution.py:118` 和 `tests/worldstream_m2/test_wsgrr.py:240`，原因为当前测试进程 `CUDA unavailable`。警告来自既有第三方弃用提示和微型生成模型的配置提示，无失败用例。

覆盖要求：

1. 有效窗口缓存及隔离测试批准证据 → 真实增量窗口执行、真实复用诊断。
2. 无成本、缓存未命中、来源失效、数值证据缺失、任务质量未批准 → 同一个执行器全量计算；结果与原生全量微型 EVA 一致；缓存发布失败不会跳过已完成视觉输出。
3. 隔离 state quality mock 获准 → 历史文本进入实际 Prefill，全部当前视觉载体保持。
4. 历史缺失或独立质量未获准 → 原始 token ID、全部当前视觉输入保持，完整 Prefill 和 Decode。
5. 首次前缀未命中、再次精确命中但批准缺失 → 均 `reuse_length=0`；对原生 multimodal generate 设置禁止调用检查，仍完成实际 PrefillExecutor 路径。
6. 正式模型包装 → 原始 Detection/Map 任务方法保持连通；微型 Llama forward 实测序列为完整 Prefill 长度 7，随后原生 Decode 长度 1、1。
7. 正式配置和未批准文件/空成本记录 → 连续两帧完整三模块执行；已有答案文件不跳过 M3；缺失模块、视觉被绕过、空问题、异常答案会明确报错。
8. 原生参考基线数值、原有答案文件行为、任务头和配置保持独立；现有离线消融不会误套正式配置断言。
9. 视觉、Commit、历史快照发布、生成异常，以及成功后下一帧异常 → 事务状态与真实阶段一致，不继承上一请求的成功标志。

测试批准只存在于 pytest 临时夹具或 monkeypatch；没有写入正式批准文件。

附加检查：修改的 12 个 Python 文件语法编译、12 个受保护文件逐字节比对、`git diff --check` 均通过，证据见 `source_verification.json`。现有 `tools.verify_final_route` 辅助脚本在读取缺失的历史删除清单 `outputs/single_route_convergence_20261022/deleted_files.json` 时退出，故本次不将其报告为通过，也没有伪造该历史清单。

## 5. 剩余边界与后续测量

在当前唯一正式配置、审计过的正常推理入口内，没有发现仍可静默跳过 M1/M2/M3 并成功返回的有效 VLM 请求路径。模块缺失、运行时未完成、空问题、不兼容模型/生成契约等会明确报错。独立 B0、既有 B1/B2 离线消融、训练及单独的底层特征/算子调用不属于正式三模块完整 VLM 请求入口。

本次验证的是调用契约、控制流、轻量模型结果及诊断真实性；不声称生产模型速度提升或端到端质量批准。后续仍需真实 GPU/完整模型测量：窗口复用的数值误差及任务质量、匹配部署形状的成本收益、历史文本独立质量、KV 复用 token/logits/RNG 一致性、GPU 内核/访存行为、诊断开销、命中率、显存及端到端吞吐/尾延迟。全验证集 Detection/Map 和语义任务质量也仍需单独执行；本次未运行。
