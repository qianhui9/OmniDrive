# 实现交付验证

日期：2026-11-16（Asia/Shanghai）。项目当前没有 `.git` 元数据；未初始化仓库。写入范围仅为 `tools/environment_stability_eval/`，现有模型、R0 配置、主实验脚本和历史实验结果未编辑。

## 已完成

- 29 项测试通过（包含真实 Native/完整 R0 配置检查以及临时目录内的报告/五组 PDF+PNG 导出契约测试）。记录：`outputs/validation_run01/unit_tests.json`。
- 补充运行原项目两组场景重置/缓存失效回归，参数化展开共 **8 项通过**（5.42 秒）。命令：`taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m pytest -q tests/worldstream_m2/test_global_residual.py::test_scene_reset_and_failed_scene_initialization_are_atomic tests/worldstream_m2/test_reuse_diagnostics.py::test_forced_refresh_scene_reset_and_invalid_state`。
- 全部 18 主实验场景/723 帧六相机图像、标定、位姿、连续性、查询文件和冻结权重依赖预检查通过。
- 用 132 个其他验证场景/5,296 帧的官方标注校准四级阈值。真实覆盖率和分组：`outputs/coverage_dev_run01/`，无模型时延或准确率结果。
- 3,615 条真实查询的 GT 可评分状态与主实验评分器核对：**0 个不一致**；无重复查询；1969 条可评分、1123 次 Stable、72 次 Changed。验证 GT adapter/classifier 的源码与生成时一致，未使用模型预测作为真值。记录：`outputs/validation_run01/gt_audit.json`。
- 指标/绘图测试的构造数字仅写入 `SYNTHETIC_TEST` 临时目录并清理；没有进入真实 `outputs`。

## GPU Smoke

`outputs/gpu_smoke_run01/completion.json` 为 **COMPLETED**，进程退出码 0。真实 RTX 3090 上 Native 和完整 R0 均完成 4 帧独立预热 + 3 帧正式测量，即各 15 条正式查询；全部原始收据均来自此次新执行。现有 GPU 实验释放项目锁后才开始，没有停止任何原有任务。

- 两种模型均通过 `checkpoint_validation.json` 的严格权重检查；R0 的 M1/M2/M3 生效配置与完整冻结 JSON 一致。
- 两个配置在独立新解释器中加载，均使用同一物理 GPU、相同模型精度、batch size、图像/标定/位姿、prompt token IDs、查询顺序、请求 seeds；逐帧配对检查 PASS。
- R0 的第一正式帧 historical_candidates=0，执行完整 OGSM commit 和冷视觉刷新；预热缓存没有进入正式序列。
- 正式帧/QA、分布、Latency/Accuracy/Compute/Response Delay 等 10 张 CSV 均已生成，`experiment_summary.json` 为 COMPLETED；5 组 PDF + PNG 共 10 个图像导出已生成，并检查了延迟图和无数据 Stale 图的渲染。
- 这 3 帧只覆盖 S3 的一个窗口前缀，不代表完整窗口或四等级实验。15 问中 8 问可评分、7 问 UNKNOWN；两方法均 6/8 完全正确、4/4 稳定查询正确。**没有可评分 Changed 查询**，故 Changed Accuracy、Stale Answer Rate 和单序列置信区间均为 N/A，不能解释为 0% 过期率。
- 计算接口产生了实际执行 MACs、DAVP 刷新/复用、QCSE 直答和 Prefill/Decode 计数。没有 candidate token 时 acceptance rate 为 N/A。
- 模型/资产读取校验和加载在本机较慢，全部位于测量范围以外；没有用启动耗时影响 E2E/FPS。

机器可读的真实 Smoke 核验见 `outputs/validation_run01/gpu_smoke_audit.json`；Smoke 图有明确执行验证标记，不作为论文泛化结论。

上述为 2026-11-16 的 Smoke 交付状态；完整实验的后续启动状态见下。

## 2026-11-17 驱动兼容修复与完整实验启动

- 原失败目录 `outputs/run_20261117_113054_463092/` 保留。失败发生在 Native 的空闲 GPU 检查：系统 NVML 用户库为 580.178.04，而已加载的 NVIDIA 内核模块为 580.173.02，`nvidia-smi` 返回 18。
- 启动器在 GPU 运行前检测该错误，并复用项目已有、哈希校验通过的 580.173.02 驱动兼容包，在全新解释器中设置进程级库路径。没有修改系统驱动或重启服务；CPU 检查和报告命令仍不依赖 GPU。驱动来源记录进入运行清单和有效运行配置；worker 失败时总日志附带末尾错误。
- 环境稳定性测试与已有驱动兼容测试共 **41 项通过**。真实 CUDA 矩阵乘法通过，且进程实际加载项目兼容包内的 `libcuda.so.580.173.02`；记录：`outputs/driverfix_validation_20261117_113802/kernel_witness.json`。
- 新的独立会话按 README 原命令于北京时间 11:39:51 启动完整实验，主 PID `3431528`，Native PID `3432407`。输出目录：`outputs/run_20261117_113951_740642/`；总日志：`outputs/launch_full_20261117_113951_152056.log`；启动记录：`outputs/launch_full_20261117_113951_152056.json`。
- 独立文档验证确认原命令自动切换兼容库，Native 于 11:40:54 达到 `CUDA_READY`，11:41:04 达到 `LOAD_FROZEN_MODEL`，父子进程均排除 CPU 4/5，无 `failure.json`。这仅证明修复后成功启动并越过原故障点，**不代表完整实验已经完成**；最终状态以该输出目录的完成标记和报告为准。

## 2026-11-17 完整结果及 Change Response Delay 补充

- `outputs/run_20261117_113951_740642/completion.json` 为 COMPLETED，非 Smoke。Native 于 12:31:07 完成，WorldStream 于 13:10:21 完成，全部报告于 13:10:45 完成；各 723 帧 / 3,615 条查询，输入配对通过。
- 新增 `change_response.py`，在原结果外生成 `outputs/run_20261117_113951_740642/change_response_v1/`，保留原始报告；真实来源/收据/输入和运行配置配对均再次校验，并重新评分当前 GT 与变化前冻结 GT。
- 新增 15 项科学语义边界测试；环境稳定性和现有驱动兼容测试合计 **56 项通过**（18 项已有 matplotlib 弃用警告）。真实补充结果、来源和实现 SHA256 复核通过，PDF/PNG 均导出，并目视检查 PNG。
- 全部 72 个变化事件中观察到响应 Native 23 / WorldStream 56；最长连续历史事实错误 Native 10 / WorldStream 1；历史错误输出 47 / 10。紧邻前帧已知 GT 的变化事件为 25 个，进一步可确认同一持久目标为 9 个，均单列，避免扩大物体运动响应结论。
- 新报告遇未知 GT、查询缺失或观测断点停止追踪，区别于旧版跨未知帧搜索；未响应不填 0，连续错误跨度仅为采样下界，缺少在线同步墙钟和独立 TEST。结果解读见该次目录 `change_response_v1_analysis.md`。没有为本次指标请求追加 GPU 消融。

## 2026-11-17 纯前向时延口径补测（排队，尚无新性能结果）

- 用户要求从 WorldStream 时延中排除数据加载和预处理。旧 E2E 与 `execution.host_elapsed_ms` 均不满足严格同步、无导出的完整前向定义，因此没有把已有值直接改名或拼接组件时延。
- 新增 `forward_timing.py`、`forward_latency.py` 和 `tests/test_forward_timing.py`；文档及观察器清单见 `FORWARD_LATENCY.md`。全部修改在独立补测文件中，原冻结 evaluator、模型和算法文件不变。
- 前向测量在输入已准备并传入 GPU 后开始，CUDA 前后同步，覆盖一次完整模型根调用（每帧六相机及 5 个查询）；实际扣除最终答案解码和文件写入，保留模型内部 OGSM/DAVP/QCSE、必要状态适配/传输及只读观察器开销。正常答案仍完整导出。
- 8 项新增时序/输出保持/异常清理测试通过；与现有测试合计 **64 项通过**（18 项已有弃用警告）。这属于 CPU 验证，不替代真实 CUDA 短测。
- 完整配对补测于北京时间 15:04:28 独立后台启动，父 PID `17526`。新输出 `outputs/run_20261117_113951_740642/forward_20261117_run01/`；日志 `outputs/launch_forward_20261117_run01.log`；启动记录 `outputs/launch_forward_20261117_run01.json`。
- 当前系统驱动已为 580.178.04，NVML 正常；GPU 被其他任务占用约 17,351 MiB，父进程等待项目 GPU 锁，没有中断其他任务。资源可用后自动做两方法的真实 4 帧预热+3 帧短测，验证每帧排除 5 次最终答案解码和 1 次文件写入，再开始两方法各 723 帧的正式配对测量。
- 原结果页已增加纯前向主口径状态块；完整实测报告成功后自动回填新均值/P95及分组区间链接。原 E2E 为历史口径，**当前没有可报告的新纯前向性能值**。既有准确率及 Change Response Delay 不重命名为墙钟时延。
- 原环境账本引用的历史 spec 文件当前缺失；现有解释器包版本与原运行一致，未重建或安装。真实 seeded CUDA witness 放在空闲资源获取后的 worker 中，GPU 未空闲前不宣称 CUDA 测量验证完成。相关状态记录见新输出的 `launch_validation.json`。
- 独立新代理原样执行文档的 CPU `--check-only` 命令，退出 0，输出 `FORWARD_CHECK_PASS 723 frames/method; CPU-only; no inference`；无文档调用偏差，状态 PASS_WITH_LEDGER_SPEC_NOTE。证据：`outputs/forward_doc_validation_20261117.json`。

## 2026-11-18 FQ-3 Stable/Changed 新主协议

- 用户明确移除原始生成QA。新增 `fq3.yaml`（`other_questions:0`）、`fq3.py`、`tests/test_fq3.py` 和 `FQ3.md`，原五问源文件、算法与已完成输出保留；README 和旧结果页入口指向新协议。FQ-3是原固定 q0/q1/q2，不是另一个 q0/q1/q3 时延估算子集。
- 新论文主表仅Stable/Changed，包含Factual QA Accuracy、Stale Answer Rate及重点Change Response Delay；另报告响应率、删失/替代原因、共同响应事件、连续历史错误次数/跨度下界和95%区间。GT首帧、UNKNOWN、schema不一致及时间断点不强行分类，不能跨未知帧连接变化起点。
- 真GT预检查：18场景、723帧、每方法2169查询；已知1214、UNKNOWN955。严格相邻已知口径Stable1063、Changed25、Unclassified1081（其中已知但无可比邻帧126）。每个任务每帧执行，类别/存在性/行人位置各723条；无`other`。
- Changed为12个类别回答状态变化、13个相对方向变化；存在性任务没有可评分Changed事件，显示N/A。Appearance/Disappearance/物理Motion/Occlusion的可靠原因标签不足，不用UNKNOWN伪造负例或因果类型。
- 新增14项测试（含自动清理临时合成收据的完整报告/绘图测试），与已有测试合计**78项通过**；18项已有matplotlib弃用警告。合成测试不写入真实outputs。
- 运行技能要求的新代理文档验证已通过：原样执行FQ3.md的CPU命令，退出0、输出`FQ3_CHECK_PASS`；独立核验每帧exact q0/q1/q2与以上GT分母。记录 `outputs/fq3_doc_validation_20261118.json`。Summary/manifest为PREPARED_NO_INFERENCE、status.json为PREPARED，均不冒充模型推理完成。账本旧spec缺失限制仍记录，未重建或安装。
- 最新真实三问配对实验于北京时间2026-11-18 09:30:36后台启动，主PID `246385`。输出 `outputs/fq3_full_20261118_run01/`；启动记录 `outputs/launch_fq3_full_20261118_run01.json`；日志 `outputs/launch_fq3_full_20261118_run01.log`。先两方法真实Smoke后各723帧正式运行，沿用纯前向同步计时（每帧三问），不会用五问子集估算冒充三问结果。GPU忙时只等待，不中断任何任务。
- 本条为启动与协议验证记录，最终性能结果以新目录 `completion.json` 和 `结果.md` 为准；旧五问结果页顶部运行状态会在完整报告成功后自动更新。没有把旧72个变化事件的统计直接当作新25事件的三问结果。
- 真实两方法FQ-3 Smoke已通过：每方法4帧独立预热+3正式帧/9查询；两方法CUDA kernel witness通过，精确三任务、输入配对、严格权重和新报告全部通过。逐帧确认排除3次最终文本解码和1次写文件。Smoke无Changed机会，相关指标为N/A，不作为论文结果。09:35:17完成Smoke报告，随后自动启动正式Native PID `247577`，09:35:32达到CUDA_READY。父PID仍为246385，完整结果尚未完成；证据见新目录的 `launch_validation.json` 和 `smoke_validation/completion.json`。
- 后续正式运行已完成：Native于2026-11-18 10:06:31、WorldStream于10:25:26完成，全部评分/报告于10:26:01完成。新目录`completion.json`为COMPLETED、非Smoke；各723帧/2169查询、配对核验通过。原结果页已自动指向新正式报告。
- 完成后复核：原始收据SHA256、精确FQ-3问题组、完整汇总SHA256和完成清单16项产物摘要全部通过。Stable准确率58.33%→88.62%，Changed准确率32%→72%，Stale9/25→4/25；观察响应10/25→21/25，成功样本均值450.00→71.41ms。三问纯前向2204.32→1248.47ms/帧。完整主表在新目录`结果.md`，独立分析在`结果解读.md`，没有修改已完成原始产物或性能数据。

## 2026-11-18 FQ-3 四等级 × 四项核心指标补齐

- 四等级元数据一直保存在正式FQ-3收据/评分中；新增 `fq3_levels.py` 离线导出 S1–S4 × 每查询Stable/Changed交叉指标，没有重跑模型或改变GT阈值。新增 `tests/test_fq3_levels.py` 的11项测试，合计**89项通过**（18项已有matplotlib弃用警告）。
- 复算132个其他val场景/5296帧、469个完整GT校准窗的原分位数，阈值精确复现；逐窗等级与收据对应通过。重新评分4338条两方法查询、完整时序25个配对变化事件，均与原保存结果一致。按变化起点等级聚合，后续跨窗口响应不截断。
- 新结果目录 `outputs/fq3_full_20261118_run01/stability_levels_v1/` 包含四级四指标主表、分母/置信区间、响应及删失、配对统计、GT分布及事件数和真实PDF/PNG。新完成记录COMPLETED、`new_model_inference=false`。图像已目视检查；缺失响应显示N/A，不补0。区间按每项指标实际贡献Scene聚类，少于两个Scene为N/A。
- 论文汇总入口：`outputs/fq3_full_20261118_run01/四级稳定性评估.md`。旧环境结果页顶部追加四级主表/链接，修改前入口保存在 `four_level_documentation_history_20261118/旧环境报告入口.md`；原FQ-3已签署总体Markdown/CSV/JSONL和完成摘要未改动，原16项产物及新报告全部输出摘要复核通过。
- Changed事件在S1/S2/S3/S4为1/2/5/17。S4的WS/Native Changed准确率94.12%/35.29%，Stale5.88%/35.29%，成功响应均值29.40/562.50ms；S3则Changed20%/40%、Stale40%/20%，如实报告退化。不能声称各级均改善，低等级极小样本不支持稳健泛化结论。
- 65个四等级完整窗口661帧，另7个尾窗62帧仍为Unclassified；四级加尾窗所有计数/分母/分子与原723帧、Stable1063、Changed25及响应10/25对21/25严格对账。CSV/JSON和`four_level_report_link.json`保留来源/实现/文档摘要，README/FQ3.md给出独立离线复现命令。


## 2026-11-18 FQ-3 扩充评估实现与真实 CPU 验证

- 新增 fq3_expanded.py、run_expanded.sh、EXPANDED.md 和6项测试；总计95项测试通过，18项已有弃用警告。
- 固定哈希保留原18场景，30场景校准、120场景评估；真实4811帧、每方法14433问题。严格Stable7375、Changed143，覆盖64个变化场景。四级变化数5/38/39/39，尾窗22；各级仍未达到50事件规划目标，尤其S1仍为小样本。
- 重拟合阈值仅使用互斥校准集，明确新版本和已暴露val属性；无性能筛选或UNKNOWN伪负例。完整时间流响应追踪后按变化起点分级。
- 新代理逐字执行文档 --help 和真实 --check-only 命令，退出0，未用GPU。最终可恢复目录：outputs/fq3_expanded_20261118_142340_968242。源码、脚本、划分、GT及输入均已冻结，最终源码与GT摘要复核通过。旧冻结Python及原正式产物摘要无变化。
- 正式推理由用户执行，当前PREPARED_NO_INFERENCE，无新性能值。命令自动等待GPU，先真实短测，再完整配对推理并生成四级四指标、响应覆盖、置信区间、纯前向和逐事件证据；支持完整方法复用、部分方法归档重跑。
- 机器记录：outputs/fq3_expanded_doc_validation_20261118.json；环境未安装或重建，账本旧spec缺失限制保留。
