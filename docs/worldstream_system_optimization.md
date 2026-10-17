# 跨模块 Stateful VLM Runtime

Python 环境为 `OmniDrive`。本次实现复用原有 M1/M2/M3，不新增研究模块，实验保存到 `outputs/system_optimization/`。

## 使用

`Petr3D` 增加可选的 `stream_config`。默认不配置，原入口保持兼容。也可以在权重加载完成后设置：

```python
from worldstream.runtime import UnifiedRuntime
model.stream_runtime = UnifiedRuntime(dict(
    service='persistent_state',
    model_version='已核对的 checkpoint 内容版本',
    lora_version='已核对的 LoRA 内容版本',
    memory_budget_mb=1024,
    cost_aware=True,
    profile_stages=False,
    trace_path='outputs/system_optimization/live_trace.jsonl',
))
```

`persistent_state` 要求 M1 manager，且 M2 的 `m1_service='maintain'`。即使不存在视觉复用机会，M1 仍 precheck/commit。缓存预算不足时先淘汰 M2/M3，仍不足则明确失效 M1，并在记录中说明；不承诺在预算不足时保存全部历史。

`current_output` 每帧显式清空 M1 历史并跳过状态维护，始终运行当前 Head 和 LLM。M3 不会得到可消费的持久语义视图。这是另一种服务契约，不能把其省去的维护成本算作同服务算法加速。

场景、相机顺序、时间倒退/重复、长间隔、模型参数版本或执行设置变化使三个模块共同失效。执行异常后清空缓存与两个 StreamPETR Head 的时序 memory；错误向调用者返回，下次从干净状态执行，不在已推进的时序状态上重试。

统一入口启用时即使语言输出路径已有文件，也执行真实生成。不会缓存完整答案。

## 成本与质量准入

`cost_aware=True` 默认没有准入证书，因此 M2/M3 的优化被提前绕过；Persistent 服务的 M1 仍维护。不会用旧实验的全局平均速度批准新输入。独立模块消融设为 `cost_aware=False`，该设置本身不改变模块各自的 Exact/Auto 质量门限。

成本文件和质量文件都使用 `records` 数组。每行完整 `key` 必须匹配模型/LoRA 版本、执行后端、生成契约、图像形状、问题长度列表、服务、策略、冷暖缓存、内存预算、刷新间隔及实际帧间隔。质量行还要求互斥 calibration/heldout 场景、完整 Exact 目标、协议和证据哈希。成本行必须明确包含全部增量工作和峰值缓存；总优化耗时加必要准入开销小于完整执行耗时才可能通过。实际缓存仍由 M2/M3 独立验证，系统证书不证明缓存输入等价。

当前原生生成契约固定为 Sampling、temperature=0.1、top_p=0.75、num_beams=1、max_new_tokens=320、use_cache=True。其他调用契约需要独立记录/校准，不能复用本入口的证书。本次不生成生产批准证书。

## 低成本改动

Cheap Gate 只读取每个 Window 的九个固定采样点，保存这些点即可精确复现旧的拒绝决策。高级索引生成独立存储，之后调用者修改图像不会改变历史。未采样像素从未参与该 Gate；Gate 始终只拒绝，完整图像/特征依赖仍由原 M2 精确验证。源图像快照不再由 Cheap Gate 持有；真实特征缓存仍持有其合法计算来源。

M1 在真实 commit 时写入 source_frame_id、source_state_epoch、source_execution_signature、evidence_origin；FeatureRecord 同样记录来源帧和 epoch。旧数据缺失来源时保持 Unknown，precheck 不会把当前预测写成历史。未知计算来源或 annotation/oracle 来源不进入 M3 语义视图；历史帧 ID 本身缺失仍如实保留 Unknown。没有放宽任何物理认证条件。

整个相机的 EVA 输出被 M2 复用时，M1 的 backbone 记录保留实际计算来源；相同来源不因本帧验证而增加计算版本，混合相机分别记录源帧/时间，缺失来源为 Unknown。Detection/Map carrier 仍是本帧实际计算，保留各自的新版本；M2 证据不会提升 M1 的物理有效性。缓存保存 OOM/淘汰不抹掉本次已返回输出的来源。执行键绑定 FP16 reduction、确定性设置及原数值选项；旧成本行不匹配新键时按未校准回退，不自动继承批准。

M3 新增 `adaptive_batch=True, max_batch_size=3` 离线选项，需要已有 `batched_decode=True, offline_experiment=True`。按实际输入长度稳定分组，恢复原问题顺序，单问题或无收益组走独立完整 Prefill。Sampling 在重排前直接原生顺序回退。成本表行必须精确绑定执行/生成环境和输入长度，包含共享 Prefix seed 与 padding。新分组形状没有继承旧 Batch 的质量证书。

## 时间与来源

`FrameExecutionRecord` 包含真实帧标识、版本、问题数、M1/M2/M3 诊断、源计算版本、内存、回退及质量。StageRecorder 记录嵌套父子关系、墙钟、进程 CPU 时间与 CUDA interval，独占墙钟只扣一次直接子阶段；GPU interval 不能当 kernel active 时间。完整 E2E 按每帧实测，包含输入准备与同步，不能相加组件 P50。缓存预算约束请求后保留的数据，瞬时算子工作区及 allocator 由实测 GPU peak / RSS 另报；M1 的历史图像、几何、掩码和证据数组计入保留量，共享存储去重。

`profile_stages=True` 仅用于诊断/同插桩的公平对照。请求结束一定恢复包装器，保留函数签名供 Transformers 检查。真实 kernel、H2D/D2H、同步与分配来自独立完整 `torch.profiler` trace。Profiler 控制运行不混入普通计时表。KV 分支保留 prefix_lineage，视觉缓存来源与物理状态有效性不互相推导。

后缀即使已有 Past KV，仍属于 Prefill；新记录携带实际 Forward Shape/Token 数。较早语言候选记录的原计时标签保留，汇总器只依据其显式 KV 分支事件重新归类相应后缀，不估计或修改实际时间。单/多问题配对回归使用每帧固定的 CPU/CUDA seed（准备输入用 index，生成用 index+10000），重复计时不增加独立随机种子或独立数据数。

长序列默认另跑真实缓存故障控制：受控重复图像仅验证非零相机复用的来源版本，随后在原生 Head 已推进后注入异常，检查三模块/Head 清空，并与干净原生请求比较。此控制单独归档，不计入真实移动场景的覆盖率、长流独立帧或速度表。

## 复现

```bash
conda activate OmniDrive
export PYTHONPATH=.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 0-3,6-31 python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q
```

完整运行（需要 GPU 和全量本地权重/nuScenes）：

```bash
taskset -c 0-3,6-31 bash tools/system_optimization/run.sh
```

脚本只观察其他 GPU 进程，等待至少 19500 MiB 空闲显存。每个推理实验目录必须不存在；若重跑，设置 `SYSTEM_ROOT=outputs/system_optimization_repeat`。中断/失败实验保留，使用新的目录名重试，不能混入正式统计。显存空闲不代表 GPU 独占，环境快照保留并发进程；有竞争的时间只能按其实际运行条件解读。

图和报告：

```bash
taskset -c 0-3,6-31 python -m tools.system_optimization.report --root outputs/system_optimization
```

报告区分历史证据、新实验、受控重复、离线候选与正式准入。未测的官方检测/地图和闭环指标保持未测。图清单 `figures/manifest.json` 明确说明缺少数据的图；不画虚构点。

本轮长序列曾因诊断脚本将配置 `memory_len` 当成时序状态而在后置控制失败；原目录保持 failed，正式结果采用新目录 `long_stream_retry`。修正后的独立 `lifecycle_validation/lifecycle_probe/results.json` 已通过真实六相机复用来源、故障清空、恢复及场景切换对照；该控制不进入性能表。`outputs/system_optimization/resume_after_probe_fix.sh` 记录补跑命令。

最终源码/历史文件完整性检查可独立重建：

```bash
taskset -c 0-3,6-31 python -m tools.system_optimization.delivery
```

它只更新本轮交付补丁、清单和源版本差异审计，不改写逐运行源码快照或旧模块结果。

本轮本地实验已完成：正式计时 760 次方法帧执行，覆盖 76 个唯一真实帧、4 个场景；最终 GPU 可见环境回归 246 项通过。报告与 S1–S10（PNG/PDF）见 [experiment_report.md](../outputs/system_optimization/experiment_report.md)。严格质量目标下尚无生产批准的端到端净加速，Prefix/Batch 保持 Auto Full 回退；物理 FAR/关键变化召回的新独立 Oracle、官方检测/地图和闭环安全仍未测。长序列使用完成的 `long_stream_retry`，旧失败目录不进入统计。

## 第二轮优化（system_optimization_v2）

第二轮修复了覆盖报告的 Dataset/raw-pickle 下标混用：Dataset 会按 timestamp 排序，所有身份连接现在使用 sample_token，并检查 scene_token、timestamp、重复帧、时间间隔和位姿跳变。历史实际执行连续，原时延记录保留；修正对照见 [sequence_identity_audit.md](../outputs/system_optimization_v2/sequence_identity_audit.md)。

`RuntimeConfig.m1_overlap` 默认 `serial`；`experimental` 仅用于显式研究，`auto` 必须通过成本与质量双证书。只有完整 EVA、不依赖本帧 M1 结果的路径可并行。输入 Adapter 创建独立 CPU Frame，一个有界 Worker 独占 M1 manager/store；主线程只执行 EVA，在 heads、M1 commit 和 M3 读取前 join。超时或异常先等待 Worker 退出再 reset，不承诺硬截止，也不发布上一帧状态。`reuse_m1_signature=True` 利用运行时已有完整参数版本 witness，避免重复扫描，保留原 M1 签名内容。

单问题 Decode 候选在 `worldstream/m3/native_decode.py`，由可恢复 context 显式启用，仅跳过已经证明为全零的单 Token additive mask。完整 all-ones 检查和主机同步计入候选成本；不更改 Q/K/V 形状、LoRA、权重、KV、采样循环或停止条件。默认不启用。Batch 继续复用原执行器，只增加不可能容纳公共前缀时的预算预拒绝；它不是复用认证。

成本键增加执行实现和 CPU/OpenCV 线程资源。仅性能 frame gap 可以在独立测量证实的闭区间内匹配，状态有效性、服务、形状、问题长度和生成合同仍严格检查。校准采用保守的逐帧配对增量上界，并要求独立场景验证；无收益或证据不足就保持空批准表。原有正准入和负回退测试保留。

在 `OmniDrive` 环境完整执行：

```bash
taskset -c 0-3,6-31 bash tools/system_optimization/run_v2.sh
taskset -c 0-3,6-31 bash tools/system_optimization/finalize_v2.sh
```

已有 smoke 自动启动器时，用 `taskset -c 0-3,6-31 bash tools/system_optimization/run_v2.sh after-smoke` 接续，避免同时运行两个模型。推理输出目录不可覆盖；失败重跑需选择新目录并保留旧记录。第二轮固定输出根目录，重跑前应在新分支副本调整 `v2.ROOT`，不能删除旧实验来重跑。

`v2.py` 复用原 experiment runner，冻结提交 `b4782ab` 的 `simple_test` 作 Original Persistent Serial 对照；统一协调/测量层共同使用本轮版本，新增记录开销计入各方法 E2E，不用历史旧时延代替本轮基线。所有方法使用同一实际模型实例、内容哈希、问题和 RNG。正式场景 102–105，独立长序列 106–109，stride/refresh 110–111，校准后 Auto 112–113；100–101 仅为探索性冒烟，不混入正式统计。Profiler、逐层 Logits 和独立物理 Oracle 单独运行，不混入时延样本。

`finalize_v2.sh` 是本次任务的第二阶段：等初始正式 GPU 流程完成后，以源 SHA 守卫补全成本键的 profiling/trace/timeout 策略字段，保留初始版本，再在固定 102–105 场景复测；新 Auto 场景固定 114–115。模型计算保持一致。脚本在新模型加载后还会等原流水线全部分析/回归退出，才开始新的计时。初始证书归档，新键必须重新取得实测证书。已有完成目录不能覆盖重跑。

本轮成本校准候选的部署键为 M1-only 的 Persistent-State：完整 EVA、两个感知头和原生 LLM 都真实执行，M2/M3 缓存策略关闭。含 M2/M3 实例的组合独立消融，不继承这个不同策略键的证书。默认仍为 serial；实际批准范围由结果和证书决定。

`long_state_supply` 在四个完整 32 帧历史的第 8/16/32 帧调用真实五问题原生/状态候选，记录实际冻结的 commit 前历史、当前真实 carriers 和进入 LLM 的状态；附加请求属于混合服务，不加入普通单问题 E2E。物理 Oracle 在 GPU 工作结束后使用独立标注评估这些历史记录。资源控制另以 20ms 采样记录进程 resident 峰值、线程数和 GPU allocator 峰值，范围明确且不当作整段增量内存。

结果、失败状态、原始帧记录、源快照和图表均保存在 `outputs/system_optimization_v2/`。可分别运行 `taskset -c 0-3,6-31 python -m tools.system_optimization.physical_oracle` 与 `taskset -c 0-3,6-31 python -m tools.system_optimization.v2_report` 重建离线分析；它们不向在线推理提供 GT。资源不足或未完成的项目必须在报告中保留未测原因。

## 服务器 CPU 核约束与故障恢复

用户确认当前服务器物理核 4/5 存在故障。后续 Python、测试及实验启动命令统一加 `taskset -c 0-3,6-31`，子进程继承亲和性。此处使用用户指定的 Linux CPU ID 范围；不另外推断 SMT 对应关系。`run_v2.sh`、`finalize_v2.sh` 和第二轮 `commands.sh` 已内置此约束。

初始正式程序完成所有 GPU 工作并写出完整结果后以 139 退出，原始日志与 `formal_process_failure.json` 保留。逐帧完整性检查通过不等于该进程正常退出；初始未绑定 CPU 的性能结果保留为历史测量，硬件故障的影响不能排除。后续在排除故障核的配置下重跑 GPU 回归、独立成本校准和新场景 Auto 验证，不覆盖初始逐帧结果。`cpu_affinity_revision.json` 记录资源与源版本变化；CPU 亲和性现在属于成本键，不允许旧环境的证书继承准入。

成本校准首次尝试在回归完成门检查失败，尚未创建任何测量目录；只有这种无测量的失败允许恢复，并保存原失败标记。其他失败或已存在测量目录仍拒绝覆盖。

## 第三轮：物理证据、双阶段任务验证与执行优化

新结果与冻结风险预算见 `outputs/system_optimization_v3/protocol.json`，旧第二轮数据不覆盖。所有执行使用健康 CPU 核和 `OmniDrive`。完整入口：

```bash
taskset -c 0-3,6-31 bash tools/system_optimization/run_v3.sh
```

`M1Config.vectorized_occlusion` 是原逐对遮挡公式的向量化实现，默认关闭；保持相同投影、遮挡门和独占 Worker，不增加并发 Store 写者。`bidirectional_geometry` 为单独的拒绝式物理候选：要求真实来源时间、双向观测、可观视差和源/当前三维范围内的对应支持。动态目标不使用静态场景三角化授权运动，地图深度保持未认证。它不改变 Transformer 依赖，也不是独立物理真值。两类候选分别验证状态等价或风险/覆盖，不混为一种保证。

`M3Config.post_perception=True` 仅允许显式离线状态实验。完整 Heads 执行后、Commit/LLM 前，使用冻结历史和当前预测进行唯一、双向的同类几何核对；当前结果不回填 Precheck，不追溯改变 M2 准入。保留历史源帧/时间/版本及单独的当前预测核对记录。只支持输入需求明确的局部任务，原始全局计数、不存在性、语义和 Planning 保持 Full Carrier。Direct Query 没有独立任务证书不执行。实际任务激活与真正减少 Prefill 另报，不能以记录数量替代消费证据。

第三轮成本校准同时保存最坏观测增量、预先固定分位数和场景均值单侧上界。仅在 calibration 实测帧间隔范围、精确部署/线程/Shape/问题长度/生成合同内评估新场景。统计均值性能不是确定性截止保证。准入匹配缓存绑定实际键和证书内容，仍对每次实际开销复核；模型或环境变化不能继承许可。若没有足够支持或风险未过，则空批准和 Full 回退。三种证据始终分开：物理可信、计算可复用、任务质量认证。

性能分析使用本轮真实 GPU kernel traces；M2 的掩码依赖上界是乐观可行性分析，不是 Exact 视觉复用批准。通用 Decode 候选同样作用于 Original 对照，Batch 仍遵守完整 Token 等价。实现与测量文件清单将随最终报告保存，不将尚未完成的实验记作成功。
