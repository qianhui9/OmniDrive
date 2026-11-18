# WorldStream resource overhead

独立评测 Native VLM 与完整 frozen WorldStream R0；不修改生产模型、模块算法及主实验默认行为。

## 扩样重跑（协议 v2，推荐）

```bash
taskset -c 0-3,6-31 bash tools/experiments/resource_overhead/run_expanded.sh
```

该命令使用 `config_54_s4_s5.json`：54 个完整场景、2,166 帧/负载、S4/S5、Native/WorldStream 各 3 次独立重复，共 12 个独立进程、25,992 帧正式测量。自动使用 OmniDrive Python 和指定 CPU 亲和性，GPU 忙时等待。结果新建于 `outputs/experiments/resource_overhead_expanded/run_<time>/`，不覆盖旧18场景结果。脚本支持将额外参数传给原 runner，例如 `--workloads S1 S2 S3 S4 S5` 可扩至全部负载。

本次代码交付不启动正式 GPU 实验。可先用 `run_expanded.sh --prepare-only` 进行文件/权重/选择验证，或使用 `run_expanded.sh --smoke --check-gpu-once` 自行进行三帧 GPU 验证。`--cpu-smoke` 只执行 CPU/统计回归测试。

### 扩样选择与可追溯性

- 保留原18场景，候选限定在相同18个原始日志的完整验证场景。核对官方首尾帧、前后链、时间顺序、六相机文件与自然问题文件完整性。
- 按 GT 对象数量、已有 `map_geoms` 几何 primitive 数量及官方场景描述中的 night/rain 标志，在每个日志内选择特征差异较大的场景并适度偏向稠密尾部。GT 仅用于推理前的采样，绝不作为模型输入；`map_geoms` 计数只是地图复杂度代理，不是模型地图预测质量。
- 原18日志有1个仅1场景，无法每个日志都选3个。该日志取全部，其余名额按最少已入选数量、日志ID排序确定性重分配；当前为1个日志×1场景、15个日志×3场景、2个日志×4场景，总计54。不得重复场景凑数；可用场景不足时直接失败。
- `scene_selection.json` 保存原场景、全部候选特征、筛除理由、每个新场景的入选理由和日志分布。`manifest.json` 冻结实际选定帧，并保留选择文件、配置、实际相机图像与问题文件的 SHA256。worker 启动后复核；按结果挑场景或运行中修改源码/输入会失败。
- 这是有意覆盖稠密/复杂情况的采样，不是完整验证集的等概率抽样。按日志的 bootstrap 是所选覆盖下的稳健性描述，不能无条件视为整个 nuScenes 总体的置信区间。

### CPU 基线与同步采集修正

- 每个新 worker 启动前统一 `MALLOC_ARENA_MAX=2`、`MALLOC_TRIM_THRESHOLD_=131072`，在其导入 Python/Torch 前生效；只影响本次实验子进程，不修改系统环境或其他任务。
- 权重加载后保存 raw baseline，`gc.collect()` 回收无引用的加载对象，再用 fresh RSS/USS 读取观察平台。默认至少5次、间隔0.2秒、窗口相对极差≤0.5%、最长8秒；取最后一次真实同读快照作为模型基线。未达到平台时停止并保留 trace，不挑最低值、不强制得到“好看”的基线。
- 不调用 `malloc_trim`、CUDA `empty_cache`，不裁剪或额外清空 live 模块状态；场景 reset 完全复用主实验生命周期。
- 帧结束和模型/预热后基线均强制调用 `psutil.memory_full_info()`，RSS/USS 取自同次结果；不再把加载阶段的旧 USS 配到新 RSS。后台低频 full read 之外只读 RSS，未新读 USS 时该样本 USS 为 null。
- `memory_full_info` 涉及多个内核读取，并非原子快照；保存读取耗时。后台及主线程串行采样，避免互相覆盖；整个进程加载峰值继续由 ru_maxrss 保留。
- 标准 Incremental 仍相对模型加载后的基线；另记录 warmup 后基线及其增量，供 `baseline_quality.csv` 敏感性比较。达到短时平台不保证不同进程的 allocator RSS 相同；跨进程实际基线及差异仍需报告。
- 平均 RSS 改为按采样间隔积分的时间加权值。CPU/USS/模块统计本身有开销，采集读取时长及 accounting_ms 原样保存，仍不能将本工具作为正式时延基准。

### 场景峰值、尾部与样本单位

- 每场景只重置 CUDA **计数器**，不重置 CUDA 分配器；整次运行 GPU peak 为各场景 peak 的最大值。后台 RSS/NVML 同时按运行和场景归集。`gpu_peak_scope=scene`，原协议的累计运行 peak 不会误标成场景 peak。
- `scene_memory.csv`：每方法/负载/重复/场景的实际 CUDA peak、RSS/USS/NVML sampled peak、相对模型与相对场景起点增量。
- `scene_module_memory.csv`：各场景模块 retained state 的 Mean/P95/P99/Max；Total 先逐帧全局去重，再做任何统计。
- `tail_memory.csv`：每次独立运行内先计算逐帧快照/模块状态的 Mean/P95/P99/Max，再统计重复间 Mean/SD/Peak。帧结束显存快照的分位数不能称为 CUDA 瞬时峰值的分位数。
- `scene_mean_over_repeats.csv`、`scene_distribution.csv`：先平均同一场景的三次重复，再对54个场景峰值计算 Mean/P95/P99/Max；避免把同场景重复当作新增场景。
- `paired_scene_ci.csv`：同场景/重复配对 WorldStream − Native，先平均重复，再按原日志聚类 bootstrap（默认2000次，固定 seed），给出“场景峰值差的平均值”的95%区间。不是对全局最大值的保证，也不把54个场景简单当成54个独立日志。
- 保留原要求的 JSON/CSV/报告/三组 PDF/PNG，新增 `extended_statistics.json`、`extended_report.md`、自动中文 `结果.md`。全部新旧协议结果分开，不能把旧18场景与新54场景当作独立样本混合计数。

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
