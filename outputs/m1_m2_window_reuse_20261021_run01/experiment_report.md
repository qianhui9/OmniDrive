# M1→M2 局部窗口复用：代码与实测交付

已完成原生窗口路径、局部失效、计算来源、安全回退与诊断代码，并完成真实模型验证。**尚未达到自然场景安全加速验收**：自然正式测量实际复用为 0；严格窗口 P50 比同 M1 完整执行慢 1.30%，Auto 慢 1.01%。真实局部变化能实际跳过 106/108 个 Block-Window，但未通过既定检测/地图质量门槛，不能作为生产精确复用能力。

## 实际代码

| 文件 | 作用 |
|---|---|
| `projects/configs/OmniDrive/worldstream_m2_source_linked_window.py` | 新增 opt-in 原生 `source_linked + window` 配置；Auto 保留原成本/质量门，M1 服务 maintain。原相机配置保留。 |
| `worldstream/m2/runtime.py` | 窗口级 source-linked 准入、状态读取/适配耗时、实际 Block 执行 hook、逐相机统计、故障发布后返回当前输出一次、研究质量标签与模块图签名。 |
| `worldstream/m2/m1_adapter.py` | 每相机独立 lease；过期窗口按真实输入依赖足迹局部失效；逐窗口层号/位置/版本/时间及 M1 producer 记录匹配；非法时间/epoch 回退。 |
| `worldstream/m2/dependency_planner.py` | 缓存前缀增加原始 EVA 模块与逐通道 LayerNorm 检查，拒绝保留参数对象但引入全局依赖的自定义 norm/attention。原依赖闭包保持。 |
| `worldstream/m2/state_cache.py` | 发布前独立检查输入/前缀特征有限性；原子发布、原计算来源与数值资格沿窗口保留；历史坐标空间与 epoch 必须匹配。 |
| `worldstream/m1/mask_codec.py` | 坏压缩/base64 payload 统一为验证异常，避免完整回退后诊断再次解码而报错。正常编解码不变。 |
| `tools/worldstream_m2/state_interface_integration.py` | 增加 window study、单像素 local 控制、短预热选项；只保留当前序列参考以限制主存；源码归档不再因绝对路径写回原文件。 |
| `tools/worldstream_m2/window_study_summary.py` / `window_source_audit.py` | 排除预热、同帧配对汇总；审计窗口来源、期限、版本及执行/复用计数守恒。 |
| `tests/worldstream_m2/test_window_local_admission.py` / `test_source_archive.py` | 新增针对性回归，未删除旧断言或放宽阈值。 |

原 PETR3D 已调用 `forward_from_m1`，无需修改 detector。真实链路为 `Petr3D.simple_test → M1.precheck → extract_img_feat → M2.forward_from_m1 → M1.export_m2_inputs → 窗口准入/原 EVA 前缀执行 → 原检测/地图/语言/规划 → M1.commit`。新配置与实验开关真正选择窗口分支，未把相机复用数当窗口数。

实际 EVA 输入 6×3×640×640，Patch Conv kernel=stride=16、padding=0、dilation=1，Window=16 tokens。每层每相机 9 个窗口，独立前缀只有 Block 0/1，共 108 个 Block-Window；第 2 个 Block（索引 2）的全局注意力及其后 22 层完整执行。使用现有 `patch_updates` 的完整卷积感受野及 DependencyPlanner 闭包；不支持的投影几何/拓扑完整回退。未跨全局或残差依赖增加缓存边界。

## 测量范围与口径

- `natural_delivery/`：3 个真实连续场景，各 20 个不同帧；每组预热 4 帧，正式重复 2 轮。4 组各 120 次正式请求，共 480 次；另有 48 次预热。
- `repeat_delivery/`：1 场景、4 帧受控重复图像；每组 4 帧预热、2 轮测量，各 8 次正式请求。
- `local_delivery/`：当前最终数值执行/拓扑检查代码，1 场景 4 帧、1 轮、无预热，仅作机制与质量验证，不用其冷启动时延宣称加速。
- 原始模型及 `ckpts/iter_10548.pth`，真实 greedy 语言生成与规划；各组独立新保存路径，无跳过下游任务。三组 M1 使用相同 source-linked 配置，世界 epoch/anchor ID 序列在自然和重复实验中均与 full_maintain 相同；M2 的缓存来源记录及其额外成本保留。
- 全组统一 TF32 matmul/cudnn=False，未改模型权重。原始 original 不维护 M1，仅作额外参考；主要比较基线是 full_maintain。M1 各组顺序随轮次轮换。
- E2E 在 CUDA 同步前后计时，包括 M1、M2、缓存、全部任务及 extract 中诊断 CPU 特征副本；数据 scatter 和结果压缩写盘在计时外。峰值为各序列实际 CUDA max_memory_allocated，包含诊断开销。
- M2 各组件是 host wall/enqueue，包含 `.nonzero` 与有限性检查 `.item()` 的必要隐式同步。保存/不保存缓存的请求同步位置不同，**不能把这些组件的差值当作 GPU 加速**，也不把组件 P50 相加成 E2E。主要性能结论只使用同步后的 E2E。
- candidate/verified/recomputed/actual_skipped_windows 均按可缓存前缀 Block-Window 对计数；actual_skipped_block_computations 是实际执行分支完成后记录的跳过次数。另记首层空间窗口数。candidate/dirty/unknown 在窗口层面可重叠；verified 是精确比较并通过 M1 失效与依赖闭包后的资格数，不是对象身份或单独候选数。

## 自然连续图像：正式结果

| 路径 | 请求 | E2E P50/P95 ms | M1 P50/P95 ms | 实际跳过 Block-Window | 峰值 GiB |
|---|---:|---:|---:|---:|---:|
| original | 120 | 2826.14 / 2900.18 | 0.00 / 0.00 | 0 | 15.295 |
| source_linked_window_exact | 120 | 3585.24 / 6128.42 | 680.13 / 3240.63 | 0 | 15.430 |
| source_linked_window_auto | 120 | 3574.88 / 6191.80 | 673.40 / 3350.02 | 0 | 15.430 |
| full_maintain | 120 | 3539.06 / 6216.72 | 685.24 / 3360.02 | 0 | 15.295 |

严格窗口与 Auto 分别有 2300 个 M1 候选 Block-Window；通过完整严格资格验证的为 0，实际各重算 12960 个、复用率均为 0%。不能把候选或缓存发布解释为收益。两组 120/120 请求均成功读取当前 M1 快照。

相对同 M1 完整执行，严格窗口 P50 改善为 **-1.30%**，Auto 为 **-1.01%**；逐帧配对中位差分别为 -49.11/-39.56 ms（负数为更慢）。未达到 +5% 目标。

360 次正式质量对照全部通过，最终视觉特征、输出与任务 memory 的 max_abs 均为 0；检测、地图、文本及规划与完整参考一致。另 36 次预热对照也保留，不计入正式统计。所有方法物理认证为 0。此为指定 3 个可用场景的探索性重放，不是独立保留场景或全量 nuScenes mAP/漏检率认证。

严格窗口 M2 validation/execution/wall P50 为 7.36/480.49/501.26 ms，状态读取 32.75 ms、M1→M2 适配 3.03 ms（适配属于 validation，不重复相加）。Auto 对应 2.30/255.03/271.37 ms，状态读取 32.66 ms；其执行时间分布混合了有/无缓存发布同步的请求，因此较小值不代表跳过了视觉计算。M1 本身 P95 达 3.24–3.36 秒，已完整计入 E2E。缓存张量最大 136.125 MiB，预算 512 MiB。

严格组有 12 次全局 M1 refresh；其余 108 次请求的每个前缀 Block 都因无稳定窗口全量执行。Auto 回退为：m1_full_refresh 12、no_valid_cache 60、no_measured_beneficial_count 22、validation_cost_exceeds_window_saving 26。顶层 reason=none 不表示发生复用，必须查看实际计数与各 Block 的 reason。

## 受控重复与局部变化

| 重复帧路径 | 请求 | E2E P50/P95 ms | 跳过/重算 Block-Window | 质量 |
|---|---:|---:|---:|---|
| original | 8 | 2486.42 / 2528.25 | 0 / 未记 M2 计数 | 参考 |
| source_linked_window_exact | 8 | 5348.32 / 5818.61 | 432 / 432 | 通过 |
| source_linked_window_auto | 8 | 5392.64 / 5481.27 | 0 / 864 | 通过 |
| full_maintain | 8 | 5292.98 / 5436.51 | 0 / 未记 M2 计数 | 通过 |

重复帧严格组跳过 432/864 个 Block-Window（50%），24 次正式任务质量对照全部通过，特征/输出/memory max_abs=0，文本和规划相同。第二帧可复用；第三帧原始计算年龄约 1.000767 秒超过 1 秒 lease，必须刷新，不能靠验证时间延长计算寿命。该控制仍比同 M1 完整执行慢约 1.05%，不可代替自然加速证据。

局部测试只改变 CAM_FRONT 的一个预处理像素（三通道），每帧先恢复首帧输入，再加入 ordinal×0.125；其余 5 路图像不变，实际位姿/时间及全部下游任务继续执行。

| 局部帧 | 执行 Block-Window | 实际跳过 | 完整任务门槛 | 最终视觉 max_abs |
|---|---:|---:|---|---:|
| 0 | 108 | 0 | 通过 | 0.000000000 |
| 1 | 2 | 106 | 失败 | 0.403118134 |
| 2 | 108 | 0 | 通过 | 0.000000000 |
| 3 | 2 | 106 | 失败 | 0.414880753 |

发生复用的两帧中，CAM_FRONT 重算 2、复用 16；另外 5 相机各重算 0、复用 18。确实按窗口执行，没有折叠成整张相机。到期帧全部 108 个重算，符合真实来源期限。

但局部强制执行仅 2/4 帧通过冻结门槛。第 1 个部分更新帧有 2 个未匹配检测、匹配中心差最大 0.0118468 m、地图坐标差约 0.0158768 m，超过原 0.01 m 条件。文本与规划始终相同也不足以通过。第 2 帧完整视觉已恢复 0 误差，但检测仍有约 3.97 mm 差异，说明未认证步骤会通过任务历史传播。全部强制 Exact 请求保持 task_quality_certified=False；Auto 完整执行的 4 个局部请求均通过。

局部原始逐索引 output/memory max_abs 为 90.55/87.87，包含检测排序和 memory 槽位重排，不能当作 90 m 物理框位移；判定使用既有匹配检测/地图门槛，原始与匹配诊断均保留。

真实权重的逐步探针显示：首窗口 norm1 完全一致，Q/K/V FP32 投影因 GEMM 批量形状变化出现数个 1e-6 差异，原 FlashAttention 的 FP16 转换使数百个元素改变，并经完整后缀放大。关闭 TF32 和固定 cuBLAS 工作区未消除这些差异。完整 54 窗口批量的前缀探针 valid_output=0，部分批量不为 0。未改权重、删除断言或放宽任务阈值。详见 numeric_probe*.log 与 local_delivery/quality.json。

这证明了局部执行机制与来源生命周期，**没有证明真实部分更新已经安全**。只在已有认证、匹配签名且有收益的数量上允许 Auto；现有窗口表 formal_quality_policy.approved_counts=[0]，未把本轮失败的部分更新加入认证。跨位置/近似 conditional 路径默认关闭，其 task_quality_certified=False。

## M1 与来源审计

Geometry 原协议重放 4 个开发场景、161 帧：正确 804、错误 3、未知 2417；已评分错误率 3/807=0.37175%，覆盖率 69.3930%，条件有效召回 82.9721%；IDS 258、碎片 141、连续性断裂 165；物理认证 0。P50/P95 88.86/272.31 ms。未知不计正确，GT 仅在 commit 后评分，属于既有部分 GT 协议。离线候选重放时延不能当作原生完整 M1 时延。

原 Geometry/association_observation_guard=False 保留，未修改关联阈值、物理认证器或 M3。新代码仅改变来源准入/失效和异常处理，不将 ID、几何一致性或 candidate 当作特征相等证明。

| 审计 | 窗口回执 | 检查的窗口来源 | 历史来源 | 失败 |
|---|---:|---:|---:|---:|
| natural_delivery | 264 | 21384 | 0 | 0 |
| repeat_delivery | 24 | 1944 | 648 | 0 |
| local_delivery | 8 | 648 | 212 | 0 |

审计包括预热及正式记录，核对相机/位置/producer layer、当前/历史帧、时间及 1 秒计算期限、版本、新计算数量与 executed_windows、历史来源数量与 reused_windows 的守恒。没有把该元数据审计当作完整任务数值认证。Scene/model/epoch/事务/全局版本异常全局回退；已定位的窗口过期、dirty 和相机 lease 异常局部重算。来源记录结构不一致仍保守全局回退。

## 测试、版本与可复现证据

- 最终完整 M1/M2 回归：`complete_regression.log/xml`，**771 passed、2 skipped、33 warnings，107.85 s**。2 个普通沙箱不可见 CUDA 项在 GPU 环境单独补跑：`delivery_cuda.log/xml`，**2 passed，2.71 s**。
- 覆盖相同/局部变化/多相机/dirty/unknown、每相机 horizon、单窗口期限、场景/帧/签名/epoch/版本/相机顺序/坐标、pending/aborted/提交后快照、缺失读取、字符串/None/NaN/Inf 时间、坏 mask、全局/残差/非局部替换、非有限缓存、OOM/发布失败及原子性。发布失败用例是注入测试，不冒充真实 GPU OOM。
- 用参数指针完全相同的非局部 norm/attention 替换验证不支持拓扑会完整执行并清缓存。归档测试验证绝对源路径只读、mtime 不变，目标别名原文件时拒绝。
- `result_summary.csv` 和各主目录 `summary.json`、`runs.json`、`quality.json`、`source_audit.json`、`protocol.json` 是实际结果。`memory/` 与语言目录保留任务输出与历史快照。
- 三个主目录各 64 个归档源码 SHA256 均通过：`source_integrity.json`。旧绝对 __file__ 归档曾写回同字节源文件而未产生独立副本；已修复后续归档，并以记录 SHA 完全匹配的旧字节补齐三个缺失快照，说明在 `source_snapshot_repair.json`。没有丢失源内容或覆盖历史实验。
- 正式自然/重复实验包含有限性验证成本。其后只补充 CPU 拓扑/签名检查、保守研究标签、异常/计时标签与启动归档修复；EVA 数值执行器、依赖闭包、权重和下游未改。当前代码原生 local 已覆盖拓扑/签名守卫；相对其快照仅计时文本及归档辅助不同。全部差异保存在 `source_revisions.json` 和 `*_to_current.patch`，不将旧版本时延冒充逐字节最终源码重测。
- 在同一真实 24-block EVA 上 100 次交错微测量，新增单次前缀检查 ΔP50=0.0521 ms、签名检查 ΔP50=0.1950 ms，见 `metadata_guard_overhead.json`。这是元数据检查测量，不替代 E2E，也未用组件相加制造“调整后”时延。
- 最终代码快照与 SHA 在 `delivery_source/`、`implementation_manifest.json`；检查点校验在 `weights_manifest.json`。硬件 RTX 3090 24 GiB，PyTorch 1.13.1+cu117、CUDA build 11.7；GPU 进程检查仅本任务。所有运行均使用 OmniDrive 解释器及 CPU 0-3,6-31，未停止其他用户进程。
- `local/`、`local_fp32/`、`natural/`、`natural_final/` 是保留的迭代/停止记录，不混入正式表。最终配置默认 Auto；`enabled=False` 关闭，旧 legacy/camera 保留；强制 Exact 仅机制研究，不自动认证。

复现需选择新的输出目录；不要覆盖以上证据：

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests/worldstream_m1 tests/worldstream_m2
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests/worldstream_m2/test_execution.py::test_cuda_real_eva tests/worldstream_m2/test_window_local_admission.py::test_cuda_tf32_partial_windows_fail_closed
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.state_interface_integration --window-study --case natural --frames 20 --scenes 3 --warmup 1 --warmup-frames 4 --repeats 2 --output outputs/window_natural_repro_new
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.state_interface_integration --window-study --case repeat --frames 4 --scenes 1 --warmup 1 --warmup-frames 4 --repeats 2 --output outputs/window_repeat_repro_new
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.state_interface_integration --window-study --case local --frames 4 --scenes 1 --warmup 0 --repeats 1 --output outputs/window_local_repro_new
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.window_study_summary outputs/window_natural_repro_new
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.window_source_audit outputs/window_natural_repro_new
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.dqtrack_policy_eval --protocol outputs/m1_dqtrack_policy_20261021/protocol.json --candidates outputs/m1_dqtrack_policy_20261021/omnidrive/model_observations.pkl --dqtrack-root outputs/m1_dqtrack_policy_20261021/dqtrack --methods geometry --output outputs/window_geometry_repro_new
```

## 验收边界

执行链、空间映射/局部失效、来源期限/版本/事务回退、原生真实模型调用、指定范围的自然时延/显存/质量记录及原基线复现已交付。**真实部分更新的完整任务质量未通过，且自然场景实际复用为零、P50 未改善**。因此本轮不能宣称获得真实场景安全加速，也不能把条件/跨位置近似路径升级为已认证能力。
