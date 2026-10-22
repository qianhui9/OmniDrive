# M2 精确特征比较与唯一窗口执行器

正式配置 `projects/configs/OmniDrive/worldstream_m123_auto.py` 固定 `mode=auto`、`reuse_boundary=window`、`state_interface=source_linked`、`m1_service=maintain`、`cache_capture=adaptive`。缓存缺失、来源失效、签名不匹配、成本未批准等情况将更新全部窗口，仍使用同一个 `WindowExecutor`。没有相机缓存、Token 输出缓存、近似阈值、图像补偿或第二个执行器。

模型图像为 640×640，16×16 Patch 产生 40×40 Token；16×16 Token 窗口补齐到 48×48。六相机每 Block 54 个窗口，仅 Block 0、1 属于经过结构检查的独立前缀，共 108 个输出窗口。Patch Embedding 和 Block 2 起的后缀始终完整执行。

`dependency_planner.block0_input()` 完整执行原生 Patch Embedding 和位置嵌入，每个生产请求仅执行一次，并直接传给后续计算。`exact_token_changes()` 对 BHWC 张量逐元素精确比较，任一通道不同或当前 Token 非有限即更新。历史输入是上次完成请求的实际 Block 0 输入，既不是图像采样，也不是 M1 的复用许可。`exact_pixel_changes()` 只保留在离线对照中，生产代码不调用。

M1 的 dirty、局部来源过期区域仍是预处理图像 NHW 掩码，经卷积真实 kernel/stride/padding/dilation 的感受野传播后与特征变化 Token 取并集。unknown 同样经过 M2 自己的精确比较。依赖规划沿真实窗口拓扑闭包；跨窗口残差或全局 Attention 结束可缓存前缀。非零 padding mode 等无法证明的结构直接拒绝初始化。

原生 FP32 GEMM 的归约随形状变化，Q/K 转 FP16 后可能放大误差。因此局部更新保留原始 QKV、投影和 MLP 形状，仅 gather/scatter 选定窗口的 Attention。只有整 Block 零更新时才能跳过其全部点积。`reused_windows` 是复用的输出窗口数；`attention_windows` 是实际 Attention 窗口；`pointwise_window_work` 与 `pointwise_tokens` 单列完整点积处理规模，`actual_skipped_block_computations` 按整 Block 计数，`skipped_pointwise_window_work` 单列零更新时跳过的窗口等价点积工作。六相机有更新时 QKV／投影处理全部 54 个补齐窗口（13,824 个 Token），MLP 处理原网格 9,600 个 Token；不能把输出复用数当作整 Block 计算节省。

`StateCache.commit(..., block0_input=...)` 在发布前完成预算、有限值检查、所有张量和上下文／来源元数据的独立复制及版本记录。新特征、原有输入和各 Block 输出全部计入预算。失败不发布半个请求；运行时对预算、非有限值、CUDA 分配失败清除缓存并返回已完成的输出，接口或编程错误直接抛出。缓存版本、张量 `_version`、地址／布局及上下文／来源元数据变化使缓存失效，且先于 M1 解引用检查。

成本键包含完整 EVA 权重和缓冲区指纹、Block 指纹、拓扑／Patch 几何、Torch/CUDA/GPU、输入形状与布局、数值策略和精度上下文。旧键不能用于新路线。生产准入需要精确键／规模记录、数值证据、独立下游质量、双时间收益、明确批准及仍可读取且哈希相同的证据文件。自适应缓存仅在当前键有获批且划算的规模时捕获：第一个完整请求建立缓存，后续请求才能复用；候选记录本身不触发捕获。

离线测量不依赖生产批准，复用现有工具，输出目录必须不存在：

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m2.window_validate \
  --output outputs/window_validation_new --frames 4 --warmup 3 --repeats 10
```

自然帧按 scene/timestamp 排序，仅比较同场景严格递增时间的相邻帧。覆盖相同输入、单窗口、填充边缘、所有相机同时变化，以及默认 2/4/8/27/53/54 个窗口规模；可用 `--counts` 指定其他规模。实际规模始终来自依赖闭包，每个可缓存 Block 都测量。每个样本用 CUDA 同步、预热、交替顺序的配对计时，保留中位数及全部原始样本。

`results.json` 单列像素变化率、旧像素感受野 Token 率、新特征 Token 率、窗口更新率、视觉管线 wall/GPU 时间以及缓存发布时间和字节数。视觉管线包括完整 embedding、比较／闭包、窗口执行、完整后缀；纯 Block 测量排除加载、embedding、比较、发布和任务头。缓存发布单独同步测量，不将两个中位数相加当作实际完整请求延迟。GPU event 为流跨度，包含提交间隙，不是 kernel 活跃时间总和。M1、Detection、Map、完整 VLM 服务的配对端到端时间没有测量时明确标记 `NOT_MEASURED`。

`window_costs_provisional.json` 按键／规模汇总，保留每个场景的块级成本、数值差异和样本，取保守成本案例且要求所有覆盖案例位级相等。始终 `quality_certified=false`、`approved_counts=[]`、`production_ready=false`，不会改写正式表。

晋升分三步：

1. 完成上述数值／成本验证，保留候选和 `results.json`。
2. 独立完成对应键、规模与场景的 Detection、Map、完整 VLM 评估，由研究人员出具 `m2_window_task_quality_v1` 评估清单。历史无复用的 smoke、一部分预测哈希一致、EVA 位级相同均不能替代它。
3. 明确选定规模并执行晋升命令。工具重新加载正式权重、按当前正式六相机几何构造实际键，逐条检查证据、所有场景的 wall/GPU 收益和质量清单，再原子写出批准表。仅允许初始化空表；已有批准使用新路径，防止覆盖。

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m2.promote_costs \
  --candidate outputs/window_validation_new/window_costs_provisional.json \
  --quality-report /path/to/independent_task_assessment.json \
  --checkpoint ckpts/iter_10548.pth \
  --approved-counts 0 --approved-by researcher_name --confirm-approval \
  --output outputs/final_route/window_costs.json
```

上述命令只是满足条件后的操作示例；当前没有质量评估清单，不能实际晋升。可只批准有证据的规模；未列出的规模即使更小也不能推断批准。生产进程需要重新加载新表（重建 runtime／重启服务）。

独立质量清单的契约（字段说明，不是已通过的报告）：顶层 `schema="m2_window_task_quality_v1"`、`independent=true`、`status="passed"`、`assessor` 和候选文件 `candidate_sha256`；`records` 每项含原始字符串 `key`、整数 `updated`、覆盖候选场景名称的 `covered_cases` 和 `tasks`。`tasks` 必须包含 `detection`、`map`、`vlm`，每项包含 `status="passed"`、`evaluator`、非空实际 `metrics`、实际评测产物 `path` 和 `sha256`。相对路径相对于质量清单；数值证据路径相对于候选。清单是独立评估者对实际任务评测的判断，不由视觉校准程序自动生成。批准表保存候选／数值／质量清单的绝对路径和哈希，部署时需要保留这些文件。

2026-10-22 本次实测见 `outputs/m2_feature_boundary_20261022_run02/experiment_report.md`。自然帧没有降低窗口更新比例；只有完全相同输入在本次视觉管线测得收益。正式批准依然为空，所有规模待独立质量认证；非零局部规模还未满足成本条件。历史记录保留不覆盖。
