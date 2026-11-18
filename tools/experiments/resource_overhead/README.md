# WorldStream resource overhead

独立评测 Native VLM 与完整 frozen WorldStream R0；不修改生产模型、模块算法及主实验默认行为。

## 运行

从项目根目录运行；脚本自动使用 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，所有进程和采样线程继承 `taskset -c 0-3,6-31`。默认物理 GPU 0，可用 `--gpu` 修改。不会停止任何其他任务，GPU 忙时每 15 秒检查一次。

```bash
# CPU Smoke：真实 CPU storage/RSS 采集和真实模块 API 的小型测试；无模型实验结果
bash tools/experiments/resource_overhead/run.sh --cpu-smoke

# 只校验文件、权重 SHA256、冻结场景与负载并保存可复现实验清单
bash tools/experiments/resource_overhead/run.sh --prepare-only --smoke \
  --output outputs/experiments/resource_overhead/preflight_smoke

# GPU Smoke：两种方法各一个独立进程，单场景前三帧，默认 S4、1 次重复
# GPU 忙时立即退出；不指定 --check-gpu-once 时等待
bash tools/experiments/resource_overhead/run.sh --smoke --check-gpu-once

# 正式实验（耗时较长，确认后运行）：18 个主实验完整场景、S1–S5、3 次重复
bash tools/experiments/resource_overhead/run.sh

# 配置重复次数、负载与较短数据范围
bash tools/experiments/resource_overhead/run.sh --repeats 3 --workloads S1 S4 S5 \
  --max-scenes 1 --max-frames-per-scene 10 --warmup-frames 4

# 重新生成全部统计和图表（拒绝缺失运行或不成对的数据）
bash tools/experiments/resource_overhead/run.sh --summarize outputs/experiments/resource_overhead/<run>
```

可直接修改 `config.json`，或传 `--config`。默认每个 method × workload × repeat 独立加载权重并退出，共 30 个新进程；同一进程按时间顺序执行所有选择场景。交替执行方法顺序；同一次重复双方使用相同请求 RNG seed，跨重复 seed 独立。支持 `--methods native` 等诊断子集；子集结果不代表完整对比。已有输出目录拒绝覆盖。

GPU 驱动检查复用已有 `driver_compat` 的已校验进程级库目录，只有检测到版本不匹配且已有匹配包时才重新执行本实验，不下载或修改系统驱动。CPU Smoke 不需要 GPU。GPU 不可访问/忙时不生成虚构 GPU 结果。

当前 PyTorch 1.13.1 不支持通过 UUID 正确计算可见设备数。因此脚本使用 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和数字 `CUDA_VISIBLE_DEVICES` 绑定 GPU，UUID 只用于记录；`--gpu` 仍指 nvidia-smi/NVML 的物理编号，多 GPU 时按 PCI 地址转换为 CUDA 序号。每个 worker 在加载权重前验证可见设备数和设备名称。worker 失败时，终端会显示日志末尾的具体异常，完整日志保留在输出目录。

## 与主实验的一致性

- 复用 `tools.worldstream_m123.benchmark.worker.effective_config/load_model/reset`，严格校验同一 `ckpts/iter_10548.pth`。
- Native 使用 B0；WorldStream 使用 B3，M1/M2/M3 配置逐项与 `outputs/m123_r0_only_reproduction_run01/frozen_main_config.json` 校验。仅策略文件路径可不同，但两份文件 SHA256 必须完全相同；校验凭据保留在 `runtime_validation.json`，两份策略文件均进入输入配置 hash 清单。
- 场景直接复用该目录 `scene_selection.json` 中 selected：18 个日志各一个完整、可读取场景。
- 使用主实验 `WORKLOADS` 和 `continuous_stream.question_schedule`：S1 每四帧一问、S2 每两帧一问、S3 每帧一问、S4 每帧三个核心问题、S5 每帧核心问题加两条自然问题。
- 六相机、640×640、batch=1、相同模型精度、语言提示编码、generation 参数（包括 320 最大生成 Token）均保持原主实验配置。推理直接调用原模型，不重写 backbone/LLM 流程。
- 汇总验证每对运行的实际图像摘要、Prompt ID 摘要、问题、相机、shape/dtype、请求 seed 和运行配置。失配即报错。

## 采集与边界

1. CUDA 初始化后启动监控，严格权重加载完成后记录模型 baseline；数据集尚未加载。
2. 用首场景前缀进行配置数量的 warmup；Smoke 最多三帧。预热输出使用独立目录，避免 Native 的已有回答文件跳过逻辑。
3. 按主实验正常场景生命周期 reset 后正式测量。**不调用 CUDA empty_cache，不调整状态预算，不额外清空模块内部缓存。** 峰值统计在正式测量前 reset 一次，保留 warmup 结束峰值供全进程统计。
4. 每帧模型完成后 CUDA synchronize，读取 allocated/reserved 和累计 peak；后台 psutil RSS 与 NVML 默认 50 ms，USS 默认 1 s；另有逐帧 RSS、最近 USS。保留 Linux ru_maxrss 生命周期峰值，覆盖模型加载阶段。
5. GPU incremental peak = 正式 CUDA peak allocated − 模型加载后 baseline allocated；reserved 同理。CPU incremental = 正式采样 RSS peak − baseline RSS。基础模型/增量/全进程峰值分开保存。GB=10^9 bytes，MB=10^6 bytes。
6. NVML 通过安装的 libnvidia-ml C ABI 查询**当前 PID**（无 pynvml 也能使用），不是整卡占用。不可用/进程无记录写 null 和错误原因；NVML 峰值是采样值，不保证捕捉瞬时尖峰。
7. RSS 平均为定时样本加帧边界样本的算术均值；USS 可能权限不可用。采样误差和监控本身开销如实保留；`accounting_ms` 记录逐帧模块统计成本。本工具的 forward_ms 不是正式时延基准。

## 模块内存

`memory.module_snapshot` 为只读外部 instrumentation，无需修改模块。递归读取真实 retained runtime graph，使用全局 ledger（OGSM→DAVP→QCSE）去重对象及底层 storage。nn.Module 不递归，模型 parameters/buffers 的 storage 被排除（包含它们的 view）。CUDA Tensor 不搬到 CPU：直接读取真实 storage 字节。NumPy 追踪 base owner；Python 容器及对象通过 `sys.getsizeof` 递归估算并单列 `estimated_python_bytes`。含不透明 C 扩展内存时可能低估，`opaque_objects` 可审查。模块数值无法替代整个进程 RSS/NVML。

| Component | 实际状态根 | Entries / Elements |
|---|---|---|
| OGSM | m1_manager：store.anchors/features/provenance，历史 cameras/masks/evidence、last_result、事务/验证状态；模型上的 committed/verified snapshot | 锚点+特征记录 / 锚点数量；另列 object/map anchors |
| DAVP | m2_runtime.cache：images、block0_input、outputs、sources、ages、descriptors、global_anchor；planner/executor 内实际 geometry/selection/footprint 缓存 | 跨缓存层窗口槽位+全局锚点 / 跨层窗口槽位；另列 window_features 和 global_residual storage |
| QCSE | m3_runtime.answer_memory.records/acceptance/performance、历史世界快照、绑定/tokenizer 元数据、实际诊断 | 历史回答条目 / 历史候选回答 Token 数；Prompt Token 单列 |

QCSE 当前不保留 persistent KV Cache，`persistent_kv_bytes=0` 来源于真实实现；临时生成 KV 仍包含在 CUDA peak 中。字段 `candidate_cache_bytes` 指实际 AnswerMemory 递归占用。`window_features/global_anchor/global_residual/world_state/candidate_cache` 是独立测量的子集，可能彼此共享，**不得再次加到模块 Total**。

模块 CPU = 精确 Tensor/NumPy storage + Estimated Python Memory，GPU = 精确 CUDA storage。模块总量也包括保留的诊断/元数据；进程整体还包含模型、检测/地图头公共时序状态、框架和分配器等。存储在多个模块共享时归属第一个模块，不能解释为独占内存。Entries/Elements 的 Total 是异构计数之和，仅用于完整性核验。

## 输出与统计

输出在 `outputs/experiments/resource_overhead/run_<time>/`：

- `manifest.json`：选定场景、负载、参数、运行时间、CPU 亲和性、权重/源码/输入配置 SHA256、源码版本。此项目为无 `.git` 的导出目录时，live Git Commit 如实记 null，同时保留导出前 Commit 和 SHA256 inventory。
- `runs/<workload>_r<repeat>_<method>/`：独立进程 PID、effective_config、strict checkpoint 校验、runtime、baseline、warmup、逐帧原始 JSONL、result.json 和回答原文。
- `summary.json`：完整运行资源/模块结果及采集健康状态。
- `summary.csv`：两方法各 S1–S5 和 ALL，全部基础/峰值/增量指标的 mean/std/peak/n。
- `paired_overhead.csv`：同一负载、重复编号下 WorldStream − Native 的配对资源差值，另有 ALL 汇总；可与相对模型加载 baseline 的 incremental 指标区分。
- `per_frame_memory.csv`：逐帧资源和模块字节、计数、子集、摘要、accounting_ms。
- `module_memory_breakdown.csv`：模块 temporal_mean/temporal_max/final、基础以上增量峰值，再统计独立重复间 mean/std/peak。
- `report.md`：两张主表、各负载结果和口径说明。
- `resource_comparison.{pdf,png}`、`module_cache_breakdown.{pdf,png}`、`memory_over_frames.{pdf,png}`：只由已完成真实模型运行产生，不从测试夹具绘制。曲线展示第一次独立重复的所有负载，场景边界保留。

单负载统计的独立样本是每次新进程重复，SD 使用 ddof=1，不把帧当独立重复。ALL 先按 repeat 聚合：峰值取各负载最大值，baseline/平均指标按负载等权平均，再统计重复间均值/SD/最大值。模块 table 使用 temporal_mean；temporal_max 的 Peak 可在 CSV 查看。n=1 的 SD 无法估计，数值占位 0 且 std_defined=false；正式默认 n=3。不同模块的 Peak 未必发生于同一帧，Total 先逐帧相加后再取 Peak。

`--cpu-smoke` 的夹具测试仅验证 storage 去重、权重排除、循环引用、模块 API、RSS 监控和统计公式；结果写独立 cpu_smoke.json，不进入 GPU 实验汇总。
