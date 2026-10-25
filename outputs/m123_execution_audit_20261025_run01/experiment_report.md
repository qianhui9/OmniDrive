# M1 + M2 + M3 正式执行与 GPU 短序列审计

日期：2026-10-25（Asia/Shanghai）。本轮实际完成正式模型 **32 帧 / 32 个真实问题** GPU 推理，1 帧预热不计入测量；逐帧执行审计、状态账本及最终资源对账均通过。没有运行基线、消融或完整 6019 帧推理，没有修改生产批准、模型权重、图像预处理和采样参数。

## 修改与入口

唯一正式配置：`projects/configs/OmniDrive/worldstream_m123_auto.py`。沿用现有 runner、worker、数据集、严格 checkpoint loader、Observer、scorer 和报告机制，增加 `--execution-audit` 短测模式，不建立另一套模型运行框架。

| 文件 | 目的 |
|---|---|
| `README.md`、`docs/worldstream_final_route.md`、`docs/worldstream_benchmark.md` | 修正原生回退描述、当前 M2 v2 策略与旧数值证据的适用边界、报告链接，说明批准为空与短测计时限制。 |
| `projects/mmdet3d_plugin/models/detectors/petr3d.py` | 补充每请求错误阶段、状态 epoch/有效/过期数量、M1 时延和运行身份；向 M3 传递默认关闭的 profile 标记。 |
| `tools/test.py` | 正式模型加载时绑定 config、checkpoint 和检测器源码 SHA256；不在每帧重新哈希权重。 |
| `worldstream/m3/prefill_executor.py` | 仅 profile 模式新增生成/Decode CUDA Event 跨度，保留 Prefill 含缓存写入计时；公开缓存容量与失败前的执行诊断。 |
| `worldstream/m3/runtime.py` | 保留执行器失败时的问题序号、前缀与不完整 Prefill/Decode 诊断，并继续抛出异常。 |
| `tools/worldstream_m123/benchmark/observe.py` | 分别观察 WindowExecutor 与各全局后缀 block，通过 hook 保留受验证的模型拓扑；普通模型推理不安装这些分析探针。 |
| `tools/worldstream_m123/benchmark/worker.py` | 单场景连续帧、运行身份与三模块源码哈希、逐帧执行核验、模型请求失败 JSONL；保留既有严格权重加载和评分。 |
| `tools/worldstream_m123/benchmark/run.py` | 仅调度一个正式 B3 短测，先回归再资源检查；修复通用 runner 对不存在报告索引的读取。 |
| `tools/worldstream_m123/benchmark/audit.py` | 复核运行时/执行器/窗口计数/实际复用/身份，生成专项 JSON 和 Markdown；只统计真实持久化记录。 |
| `tests/worldstream_m123/test_execution_audit.py`、`test_always_execute.py` | 覆盖错误成功标记、陈旧身份、计数不一致、错误阶段、跨请求时延清理、hook 拓扑及汇总分母。 |

M1/M2/M3 核心算法和准入器未重写。`state_manager.py`、`cost_model.py`、`scheduler.py`、WindowExecutor 算法、正式配置及四份生产证据文件保持不变。

正式 `tools/test.py` 和 benchmark 的逐请求 `worldstream_execution` 均绑定运行身份；每次重新初始化模块进入/完成标记、阶段、时延与错误。自定义直接调用模型的 launcher 若未提供 `worldstream_audit_identity`，会显示 `unbound_by_launcher`，不能冒充已绑定 checkpoint 的正式审计记录。默认 M1 `logging:false` 保持，诊断从模型属性读取；benchmark 持久化到原 JSONL。

## 本轮实际命令与测试

全部代码执行使用 `taskset -c 0-3,6-31`，避开故障 CPU 4/5；解释器为 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u \
  -m tools.worldstream_m123.benchmark.run --execution-audit \
  --run-dir outputs/m123_execution_audit_20261025_run01 \
  --checkpoint ckpts/iter_10548.pth --timing-frames 32 --warmup 1

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m pytest -q -rs tests/worldstream_m1 tests/worldstream_m2 \
  tests/worldstream_m3 tests/worldstream_m123
```

第二条为最终回归命令，启动时通过父进程设置 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`，完整输出保存在 `tests_final_gpu.log`。runner 自动运行的同一组测试保存在 `raw_logs/regression_tests.log`。实际 worker argv 见 `run_manifest.json` 和 `experiments/B3_execution_audit/invocation.json`。

| 本轮运行 | 通过 | 失败 | 跳过 | 证据 |
|---|---:|---:|---:|---|
| 沙箱内回归 | 367 | 0 | 2 | `tests_sandbox.log`；沙箱无法访问 CUDA，两项专用测试跳过。 |
| 沙箱外、推理前回归 | 369 | 0 | 0 | `raw_logs/regression_tests.log`，6.18 s。 |
| 错误阶段断言补充后的局部回归 | 27 | 0 | 0 | 本轮终端输出，4.62 s；不计为额外独立覆盖。 |
| 最终沙箱外回归 | **372** | **0** | **0** | `tests_final_gpu.log`，8.26 s；包含两项 CUDA 专用测试及新增离线对账反例。 |
| 配置/资源/CUDA kernel witness | PASS | 0 | 0 | `preflight.json`；kernel 结果 510720.0，与期望一致。 |
| 正式 checkpoint 严格兼容 | PASS | 0 | 0 | `experiments/B3_execution_audit/checkpoint_validation.json`；无缺失/意外学习参数或 shape 不匹配，不接受随机初始化。 |
| 32 帧正式模型与执行账本 | PASS | 0 | 0 | `driver_status.json`、`audit.json`、逐帧 JSONL、`m1_scene_integrity.json`。 |

预训练构建器保留其既有日志提示，最终模型在任何推理前由完整正式 checkpoint 严格加载；仅忽略 loader 已识别的非持久 RoPE 派生缓存。未使用模拟数据、随机权重或 CPU 推理替代 GPU 短测。GPU 专用测试已在沙箱外实际运行，沙箱的 CUDA 不可见不代表服务器 GPU 不可用。

## 硬件、数据与计时

RTX 3090，24576 MiB；驱动 580.173.02；Python 3.9.25；PyTorch 1.13.1+cu117 / CUDA build 11.7；Transformers 4.31.0；FlashAttention 2.5.6；MMCV-full 1.6.2。详见 `preflight.json`。未修改或重建环境。

checkpoint：`ckpts/iter_10548.pth`，SHA256 `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`，最终再次哈希核对未改变。scene `01452fbfbf4543af8acdfd3e8a1ee806`，同场景按 timestamp 排序连续前 32 帧，无缩减；每帧六相机和一个既有 language workload 选择的真实 Conv/VQA 问题。输入、输出与源标识见 `sequence.json`、逐帧/逐问题 JSONL。

预热 1 帧后重置三模块、Detection/Map 状态和峰值显存，实测答案目录独立，避免文件存在触发跳过。图像预处理和生成参数不变：sampling，temperature=0.1，top_p=0.75，num_beams=1，max_new_tokens=320，use_cache=True。language workload 将 planning 问题来源切换为既有真实 Conv/VQA，每帧选择一个问题；不伪造正式配置原 planning 可能不存在的语言问题。

端到端 wall-clock 在请求前后同步 CUDA，包括数据读取/预处理、collate/H2D、模型、模型答案写入及 CPU 输出回传；JSONL 审计序列化、离线 scorer 和反事实检查在计时外。模块 CUDA Event 是 stream span，含主机提交间隙，不等于活跃 kernel 时间；M1 的 CUDA 空隙也不能解释为其 GPU 计算。M2 原有 host 分段未经独立同步，不能当 GPU 时长。Prefill span 包括 KV 写入，Decode loop span 包括 HF 采样循环；嵌套跨度不可相加。

分析同步只由短测 profile 启用，普通请求不默认添加。该计时带分析开销，未测关闭分析的开销差异。

| 完整系统指标 | 短序列参考值 |
|---|---:|
| 有效帧 / 问题 / 失败请求 | 32 / 32 / 0 |
| 端到端均值 / P50 / P95 | 3378.61 / 2262.82 / 9419.80 ms |
| 端到端最大 / 总测量时间 | 9635.98 ms / 108.115 s |
| 顺序请求吞吐 | 0.29598 frame/s |
| 模型 wall-clock 均值 | 3041.45 ms |
| 峰值 allocated / reserved | 15.635 / 16.295 GiB |
| 请求失败率 | 0/32 = 0% |

## M1 真实行为

Precheck 32/32、Commit 32/32，全部 committed，aborted/rejected/异常均为 0；状态 epoch 均为 6。7 帧有可用的历史观测支持状态，累计有效状态帧次 45，均值 1.406；过期状态条目累计 844。刷新原因分布为 scene_change=1、periodic=4、no_history=1（同帧可有多个原因，不强行相加成帧数）。

CPU wall-clock 均值：Precheck 200.52 ms，观测验证 40.51 ms（已包含在 Precheck 中），Commit 93.11 ms；含输入/输出适配及状态维护总量 315.88 ms。普通关联、投影和观测支持不构成物理身份或几何真值认证。逐帧状态、过期、刷新与完整 timings 均保留在 JSONL / `execution_summary.json`。

## M2 真实行为

Runtime 和 WindowExecutor 完成 32/32；实际复用帧 0/32，全量重算 32/32。每帧可缓存前缀总窗口 108（两个 block 合计）、实际执行 108、复用 0；全序列实际执行 3456 个窗口，跳过 Attention 窗口 0。全量原因 m1_full_refresh=4 帧、no_valid_cache=28 帧。

视觉完整执行的 CUDA stream span 均值 520.76 ms、P50 507.16 ms、P95 522.02 ms。Patch Embedding、Block 0/1 的 WindowExecutor、Block 2–23 的全局后缀均单独记录，完整阶段分布见 `execution_report.md` 和 JSON。Detection head / Map head forward 的平均 stream span 分别为 14.41 / 28.82 ms；这些不是官方任务精度。

缓存发布成功 0 次，发布失败 0 次；自适应捕获没有获准的有益成本记录，因此不发布生产窗口缓存。`cost_table_status=ready` 只表示空 JSON 表成功解析，**不表示已有成本或质量批准**。本序列没有可复用缓存，未走到部分窗口计划的成本拒绝分支；该分支拒绝计数为 0 不能解释为准入通过。不能把已有空表、模块执行成功或全量计算标记为增量收益。

没有实际复用，因而本轮没有复用对原生特征的数值差异样本；对应列表为空，不填虚构的零误差。原始 Detection/Map 输出已保存为 `predictions.pkl`，未额外运行原生基线或反事实全视觉比较。

## M3 真实行为

Runtime 32/32、Prefill 32/32、Decode 32/32 均完成，全部通过 PrefillExecutor；没有原生 `lm_head.generate()` 回退。历史文本消费 0/32：26 个 full_scene_task_requires_visual，6 个 no_task_relevant_verified_history；没有可用于独立历史文本质量评估的消费样本。

31 个请求匹配到已有前缀，但全部记录 prefix_evidence_unavailable，1 个 short_or_missing_prefix；实际复用 KV 请求数、token 总量、每请求均值均为 0。完整 Prefill 共 32 次。既有 scheduler 的 prefix_evidence_unavailable 合并成本和质量证据缺失，不能从该原因分别推断两种独立失败次数。

KV 写入成功 32 次、失败 0 次；容量 536870912 bytes（512 MiB），峰值实际占用 348918775 bytes。缓存写入成功与批准复用是不同事件。

| CUDA stream span | 均值 ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| Prefill（含 KV 写入） | 185.97 | 180.17 | 221.18 |
| Decode 完整生成循环 | 1899.06 | 818.27 | 8040.03 |
| PrefillExecutor 完整生成 | 2085.59 | 997.15 | 8220.38 |
| M3 完整 runtime | 2101.12 | 1012.42 | 8236.17 |

长尾受问题与生成长度影响，本轮没有把这些值解释为稳定线上 P95。历史文本成功注入、KV 复用和答案生成是独立事件，本轮只观察到后者。

## 任务指标、批准与剩余证据

32 个问题均有真实参考答案。沿用既有粗粒度评分：answer exact match=0/32，token F1=0.36329；8 个含数字参考问题的数字数量匹配为 0/8，numeric MAE 无有效分母，记录为 null。该评分不是官方语义任务完整集精度，也不是历史文本或 KV 的质量批准。Detection/Map 仅完成模型推理并保存预测，官方完整集指标明确 NOT_RUN / null。

`audit.json` 的受保护文件检查与 checkpoint 哈希复核全部通过。`window_costs.json` 中 records / approved_counts 为空，其余三份 M3 证据的 approved 均为空，本轮没有写入或晋升任何批准。来源/参数/阈值、正式预处理和生成控制保持。推理完成后仅增强离线 `audit.py` 的身份/标记对账和汇总，并从相同 JSONL 重生成报告；未重写测量快照。`report_regeneration.json` 记录这一报告代码版本差异，模型及计时实现的测量源码均未改变。

后续完整验证之前仍需分别补齐：当前 M2 v2 的模型/部署签名匹配的数值误差与独立下游质量，视觉完整管线的净成本收益；历史文本的任务适用性和独立质量；KV 复用的 token/logits/RNG 与成本证据；完整验证集 Detection/Map/语义精度；更多场景与重复运行的命中率、显存、关闭分析后的延迟/吞吐和尾延迟。当前短测证明执行链和审计可用，未证明缓存复用质量、生产加速或完整集精度。

## 原始证据

- [结构化专项统计](experiments/B3_execution_audit/execution_summary.json) 与 [自动阶段报告](experiments/B3_execution_audit/execution_report.md)
- [逐帧记录](experiments/B3_execution_audit/per_frame_metrics.jsonl)、[逐问题记录](experiments/B3_execution_audit/per_question_metrics.jsonl)、[M1 状态账本](experiments/B3_execution_audit/m1_state.jsonl)
- [最终对账](audit.json)、[资源与环境](preflight.json)、[实际运行命令及源码哈希](run_manifest.json)
- [最终测试日志](tests_final_gpu.log)、[GPU worker 日志](raw_logs/B3_execution_audit.log)

最终静态复核：10 个相关 Python 文件语法通过，22 个文档链接均存在，`git diff --check` 通过，7 个受保护配置/批准文件 SHA256 保持；见 [source_verification.json](source_verification.json)。本轮实际运行无失败测试；完整集精度、生产复用认证、基线加速与 profiling 开销对照均明确未执行。
