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
python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q
```

完整运行（需要 GPU 和全量本地权重/nuScenes）：

```bash
bash tools/system_optimization/run.sh
```

脚本只观察其他 GPU 进程，等待至少 19500 MiB 空闲显存。每个推理实验目录必须不存在；若重跑，设置 `SYSTEM_ROOT=outputs/system_optimization_repeat`。中断/失败实验保留，使用新的目录名重试，不能混入正式统计。显存空闲不代表 GPU 独占，环境快照保留并发进程；有竞争的时间只能按其实际运行条件解读。

图和报告：

```bash
python -m tools.system_optimization.report --root outputs/system_optimization
```

报告区分历史证据、新实验、受控重复、离线候选与正式准入。未测的官方检测/地图和闭环指标保持未测。图清单 `figures/manifest.json` 明确说明缺少数据的图；不画虚构点。

本轮长序列曾因诊断脚本将配置 `memory_len` 当成时序状态而在后置控制失败；原目录保持 failed，正式结果采用新目录 `long_stream_retry`。修正后的独立 `lifecycle_validation/lifecycle_probe/results.json` 已通过真实六相机复用来源、故障清空、恢复及场景切换对照；该控制不进入性能表。`outputs/system_optimization/resume_after_probe_fix.sh` 记录补跑命令。

最终源码/历史文件完整性检查可独立重建：

```bash
python -m tools.system_optimization.delivery
```

它只更新本轮交付补丁、清单和源版本差异审计，不改写逐运行源码快照或旧模块结果。

本轮本地实验已完成：正式计时 760 次方法帧执行，覆盖 76 个唯一真实帧、4 个场景；最终 GPU 可见环境回归 246 项通过。报告与 S1–S10（PNG/PDF）见 [experiment_report.md](../outputs/system_optimization/experiment_report.md)。严格质量目标下尚无生产批准的端到端净加速，Prefix/Batch 保持 Auto Full 回退；物理 FAR/关键变化召回的新独立 Oracle、官方检测/地图和闭环安全仍未测。长序列使用完成的 `long_stream_retry`，旧失败目录不进入统计。
