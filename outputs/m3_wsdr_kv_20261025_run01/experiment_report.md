# M3 WSDR-KV 聚焦实现与真实验证（2026-10-25）

本轮交付可运行的 **group 级候选规划器**，真实 OmniDrive 保留原 object→ego→map 布局。没有可证明的 M2 区域到 LLM token 映射，也没有载体重排契约，因此没有启用区域级 KV 或静默重排。正式配置仍为 exact_prefix；未修改模型权重、M1/M2、正式成本/质量批准文件。

服务验证中 exact_prefix 与 WSDR-KV 均匹配 11/12 个请求，候选前缀合计 4562 token；批准和实际复用均为 0，Prefill 节省为 0。本轮不能声称生产复用或加速已达标。

## 1. 接口审计与可靠映射

| 来源 | 实际数据流及本轮使用范围 |
|---|---|
| Detection → M3 | `vlm_memory` 中 256 个学习 latent query 输出 + 1 个 `can_bus_embed` ego token；不是逐检测对象槽位。 |
| Map → M3 | 256 个独立学习 latent query 输出；与 Detection 拼成 513 个 carrier。 |
| M2 → M3 | EVA 局部窗口经 Block 2–23 的全局后缀、稠密 Detection/Map 和时序记忆才成为 carrier。没有 window ID/version→carrier token 映射。只检查 M2 当前 scene/frame/timestamp 的执行上下文，报告 `m2_region_mapping_unavailable`。 |
| M1 → M3 | Commit 前冻结的 VerifiedWorldState，经摘要、scene/frame/time、epoch、状态版本、执行签名及来源/年龄验证；只影响候选和失效。不能据此证明 KV 相等或完整物体计数。 |
| carrier → ExpandedInput | 调用原 `prepare_inputs_labels_for_multimodal`；视觉段的 token ID 为 -200，embedding 由真实 native expansion 确定，位置和 mask 跟随实际展开结果；新增 spans 记录各组边界。 |

原模型训练与调用均使用固定 object/ego/map 顺序，没有顺序可交换性保证。实际 `layout_permutation` 全为原顺序，退化原因 `native_group_order_unverified`。候选顺序不是已执行排列，更不是质量批准。M1 anchor ID 也没有到 object latent query 的可靠对应；本轮不实现指定对象 token 的定点复用。

## 2. 实现与安全边界

规划复用既有任务分类并细化 object 属性、map 关系、object+ego 运动及 full_visual 四种 profile；计数、缺失、全局安全和未知任务保留完整视觉范围。布局候选依次按组精确稳定性、任务相关性、原始顺序确定。所有当前 carrier 均保留，历史动态文本仍须独立准入，并保持在视觉标记之后。

组稳定性通过当前实际展开 embedding 与不可变缓存分组视图精确比较，不以余弦相似度、M1 状态或 M2 漂移代替精确相等。规划只一次回传三个布尔结果；不另存大型特征，不在普通路径新增显式 CUDA synchronize。

只有显式声明 `CarrierLayout.order_contract=group_permutation_v1` 的生产者才可在 expansion 前重排完整组，再由 native expansion 生成对应元数据。当前 OmniDrive 没有此声明；测试中合成生产者的重排会改变完整 forward logits，已被检测，不能据此宣称真实模型重排无损。

最终始终使用原 PrefixKVCache、严格连续 LCP、PrefillExecutor 和 Decode。模型/部署、权重、scope、layout signature 以及 token IDs/embedding/position/mask 均必须兼容；不能跳过第一个变化 token 后复用后方 KV。短请求命中旧长缓存时，仍重算最后 token logits；候选、准入和实际复用计数分别记录。

WSDR 的质量和成本证据绑定 `wsdr_kv_group_v1`、模型/权重/部署、任务/profile/布局、输入/候选/实际复用长度、dtype、生成参数、证据与协议哈希、数值检查、独立任务质量、逐样本耗时。exact_prefill 的批准不能迁移。缺少证据安全完整 Prefill；没有新增批准。

场景/相机/epoch 变化、时间倒退、状态版本回退、来源无效、缓存存储或布局签名变化均有失效原因。相同帧相同状态的多问题调用可以共享候选缓存。

## 3. 变更文件

| 文件 | 改动 |
|---|---|
| `worldstream/m3/reuse_planner.py`（新增） | ReusePlan、真实分组稳定性、任务优先级、确定性布局和映射缺失诊断。 |
| `config.py`, `task_conditioner.py` | 默认 exact_prefix 不变；显式 wsdr_kv 和简洁任务 profile。 |
| `visual_state.py`, `prefix_matcher.py`, `prefix_kv_cache.py` | 分组边界/排列契约、布局签名和缓存完整性；保留精确前缀约束。 |
| `world_state.py`, `runtime.py`, `scheduler.py` | 来源有效性诊断、规划集成与失效、策略独立证据绑定及安全退化。 |
| `prefill_executor.py` | 暴露最后 token 调整后的实际长度，增加受 profile 控制的分支耗时；执行算法保持原路径。 |
| `projects/mmdet3d_plugin/models/detectors/petr3d.py` | 传递现有 M2 诊断与 checkpoint 身份，补充紧凑 M3 审计字段。 |
| `tools/worldstream_m123/benchmark/{worker,observe,prefill_validate}.py` | 现有工具的三组短比较、原生分阶段计时、独立受控数值探针、真实问题 offset 选择。 |
| `tests/worldstream_m3/test_wsdr_kv.py`（新增） | 38 项针对性回归，包括精确连续复用、重排输出变化、证据隔离、失效及缓存引用寿命。 |
| `tests/worldstream_m123/test_benchmark.py` | 验证 offset 保留原问题及对应输入，不改答案与预处理。 |
| `docs/worldstream_m3_wsdr_kv.md`（新增） | 输入审计、边界、证据约束与复现实验命令。 |

发现诊断局部变量持有旧 CacheEntry 会延长显存占用，已改为布尔存在标志并及时释放分组视图引用；两个策略均有 weakref 回归。保留最初实验文件，修复后的 exact_prefix_final / wsdr_kv_final 才用于下表。

## 4. 测试与实验身份

最终完整 GPU 回归：**425 passed，0 failed / skipped，7.78 秒**，见 [tests_final_gpu.log](tests_final_gpu.log)。覆盖 M1/M2/M3/M123；包含新增 38 项 WSDR 测试及 1 项问题选择回归。测试 fixture 的批准只在 pytest 临时目录，不触及正式文件。

测试覆盖确定性任务布局、完整 token 保留、重排元数据一致、没有虚假 M2 映射、M1 场景/时间/版本/epoch/来源失效、精确与高相似输入的区别、不连续 KV 禁止、策略/布局/长度/哈希绑定、缺质量或成本证据的拒绝，以及原 exact_prefix 行为。

环境：OmniDrive Python，RTX 3090 24 GiB，所有测试和实验以 `taskset -c 0-3,6-31` 启动，避开故障 CPU 4/5。真实权重 `ckpts/iter_10548.pth`，SHA256 `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`；各 worker 严格检查实际 state_dict，无缺失 learned key、unexpected key 或形状不符，仅忽略注册为 nonpersistent 的派生 RoPE cache。

主比较取同一场景连续四帧、每帧三个真实标注问题，1 帧 warmup 后重置状态并重新测量，12 个请求/6913 个输入 token。三组均使用原生成参数：sampling=True、temperature=0.1、top_p=0.75、beams=1、max_new_tokens=320。M1/M2、权重、预处理和问题相同。native_full 只关闭自定义 M3，仍保留 M1/M2。

## 5. 四帧主比较

| 配置 | 候选命中请求 | 候选 token | 批准/实际 token | 实际复用率 | Prefill P50/P95 ms | 整帧请求 P50/P95 ms | 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| native_full | N/A | N/A | 0 / 0 | 0% | 176.528 / 177.259 | 3352.844 / 4423.024 | 15.110 |
| exact_prefix | 11/12 (91.67%) | 4562 | 0 / 0 | 0% | 176.696 / 177.409 | 3637.925 / 4491.208 | 15.402 |
| wsdr_kv | 11/12 (91.67%) | 4562 | 0 / 0 | 0% | 177.417 / 178.384 | 3434.423 / 4317.314 | 15.402 |

候选匹配率按达到当前最小前缀 32 token 的请求计算；native_full 没有运行 matcher，记为 N/A。实际复用率为实际复用 token / 当前总输入 token。全部请求实际 Prefill token 等于输入 token，节省 0。整帧 request_ms 包含图片加载/预处理、M1/M2/任务头和三个问题的生成；GPU Prefill/Decode 是每问题 CUDA Event stream span，不能与整帧或 host wall 混作同一时延。P95 仅四个整帧样本，不构成稳定性能结论。

WSDR 的整帧 P50 并没有低于 native_full，Prefill 也没有变快；与 exact_prefix 的差异不能作为实际复用收益。

| 配置 | 准备平均 ms | 精确匹配平均 ms | 服务 KV 分支 ms | Decode P50/P95 ms | 最大 owned cache MiB | peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| native_full | N/A | N/A | N/A | 546.177 / 1206.246 | N/A | 15.740 |
| exact_prefix | 0.557 | 0.978 | 0 | 546.375 / 1206.074 | 296.191 | 16.166 |
| wsdr_kv | 1.613 | 0.881 | 0 | 547.456 / 1206.223 | 296.191 | 16.166 |

GPU peak allocated 相比原生多约 299.87 MiB；owned cache 最大 296.19 MiB，低于 512 MiB 限额。reserved 是 allocator 保留值，不等于有效缓存量。准备计时包括 native expansion/规划，不把全部准备开销都归因于额外规划。无实际复用，所以服务没有 KV clone 分支；探针的分支成本另列。

主比较中三组 12/12 回答文本和生成 token 序列完全一致，4/4 Detection/Map 输出摘要一致。真实 GT 有效 12/12，token F1 均为 0.35722494，字面 exact-match 均为 0；后者是自由文本精确字符串度量，不能称为通用任务准确率。主问题集数字参考数为 0，numeric_count_match/MAE 为 null，不以 0 或一致性冒充数字正确性。数字补测单独见下一节。

主工作负载中同帧后续问题有 8 次 object/ego/map 全部精确稳定；跨帧第一问共有 3 次任务相关组变化。**自然跨帧部分组稳定样本为 0**，不能声称本段覆盖了自然的部分稳定跨帧复用。任务 profile 为 object_attributes 3 / full_visual 9；M2 映射可用 0/12；保留原布局 12/12。

exact_prefix 拒绝原因：short_or_missing_prefix 1、prefix_evidence_unavailable 11；WSDR：short_or_missing_prefix 1、wsdr_quality_evidence_unavailable 11。当前成本批准也为空，质量未过时未执行成本准入，不将其记录为已获成本批准。

## 6. 真实数字问题补测

使用原始 Conv/VQA 标签与问题 token，offset=10，确定性选择两帧各两个问题（第一帧 VQA 第 1/2 问；第二帧第 3/4 问），没有基于答案过滤，也没有修改加载器或模型输入内容。三组均完成，同为 4 问、3 个数字参考；输入/源码/权重一致，见 [numeric_service/comparison_summary.json](numeric_service/comparison_summary.json)。

| 配置 | 候选请求 / token | 批准 / 实际复用 | Prefill P50/P95 ms | 整帧 P50/P95 ms | allocated GiB | 回答/token 一致 | 数字个数匹配 |
|---|---:|---:|---:|---:|---:|---:|---:|
| native_full | N/A | 0 / 0 | 197.165 / 217.506 | 7538.679 / 8115.138 | 15.221 | 4/4 / 4/4 | 0% (3问) |
| exact_prefix | 3/4 / 1159 | 0 / 0 | 199.740 / 220.664 | 7675.528 / 8186.589 | 15.545 | 4/4 / 4/4 | 0% (3问) |
| wsdr_kv | 3/4 / 1159 | 0 / 0 | 198.567 / 219.647 | 7592.722 / 8157.488 | 15.547 | 4/4 / 4/4 | 0% (3问) |

三组仍无实际复用，Prefill 节省和实际复用率均为 0；4/4 回答与 token 序列一致，2/2 Detection/Map 摘要一致。token F1 均为 0.39037340、字面 exact-match 为 0。3 个数字参考问题的回答数字个数均与参考不符，故现有 numeric_count_match 为 0%，numeric_mae=null、分母 0；这不是数字值误差为 0，也不能宣称数字正确。该指标只是现有数字抽取检查，不是独立语义质量批准。

该小样本补齐了数字参考问题的失败观察，不能证明完整 VQA 数字质量。本节只作为独立短工作负载，不与四帧主比较合并计算延迟或命中率。

## 7. 独立受控探针：实际局部 Prefill 与数值边界

探针在所有服务计时和峰值显存记录之后运行。使用最后一帧真实 carrier 和已拥有的真实缓存，明确修改 map 或 object 首 token，比较修改后同一输入的 native full 与现有 PrefillExecutor 直接执行。它们不走生产 scheduler，`production_admitted=False`、`approved_prefix_tokens=0`，不冒充自然帧，也没有自然 GT 的任务质量。

| 探针 | 稳定组 / 变化组 | 精确连续匹配 / 离线实际复用 | 实际 Prefill / 输入 | tokens/RNG | scores close | 最大 logits 误差 | native / candidate wall ms | clone GPU ms |
|---|---|---:|---:|---|---|---:|---:|---:|
| map carrier +1 | object,ego / map | 292 / 292 | 278 / 570 | 一致 / 一致 | **失败** | 0.15625 | 561.134 / 510.928 | 0.496 |
| object 首 token +1 | ego,map / object | 35 / 35 | 535 / 570 | 一致 / 一致 | 通过 | 0 | 562.062 / 563.855 | 0.665 |

数值阈值沿用既有校验 atol=1e-6、rtol=1e-5。map 探针实际跳过 51.23% 输入 token，虽回答相同并有单次 wall 降低，但 logits 超阈值，不能批准；object 探针只跳过 6.14% 的文本前缀，未出现净成本收益。没有把后方稳定 ego/map 的 KV 跨过已变化 object 直接复用。两次都是单次受控数值观察，不能拿来签发净成本或独立任务质量证明。

保留 [controlled_probes.json](service/experiments/wsdr_kv_final/controlled_probes.json) 逐 token、RNG、误差、分支和耗时记录，以及 [wsdr_candidates.json](service/experiments/wsdr_kv_final/wsdr_candidates.json)。候选始终 production_ready=False、approved=[]、independent_task_quality=False。

## 8. 诊断与证据入口

真实诊断样例见 [diagnostic_example.json](diagnostic_example.json)：同帧三组均稳定、matched=558，但 approved=0 / reused=0 / effective_prefill=572，拒绝原因明确为 wsdr_quality_evidence_unavailable，布局退化 native_group_order_unverified，M1 有效而 M2 映射缺失。稳定组或匹配前缀不等于跳过了计算。普通推理 GPU 时延字段未采集时为 null；此处开启 profiling，因此服务分支时延为测得的 0。

完整逐帧和逐问题 JSONL、生成 token、实际执行量、模型输出、checkpoint 严格加载结果、各阶段耗时、输入序列、命令和源码哈希都保留。最终主比较选择见 [selected_runs.json](selected_runs.json)，汇总见 [comparison_summary.json](comparison_summary.json)；初始结果完整保留，不用于修复后的内存比较。

主比较 native_full 保留首次结果：其关闭 M3，缓存引用修复不在其执行路径。final 两组共享同一模型实现快照；随后新增 worker 的默认 0 offset 选项仅影响显式补测选择，不改变主比较默认输入。各轮真实源码哈希独立保留，不声称不同轮源码字节完全一致。

39 个受保护文件均未变化，见 [protected_verification.json](protected_verification.json)。包括所有 M1/M2 Python、正式/基准配置、M1 yaml、四个正式成本/质量 JSON、Detection/Map/LLava 核心 head。

## 9. 尚未通过的准入项与后续

1. 当前模型没有可交换组顺序契约，真实路径不能执行稳定组优先排列；需要单独验证输入策略对模型任务质量的影响。
2. 缺少 M2 窗口/anchor 到 carrier 的可证明来源映射，不能宣传区域级/窗口级 KV 复用。
3. 自然跨帧部分组精确稳定样本尚缺；受控扰动已明确标记，不补造这种覆盖。
4. WSDR 没有独立 heldout 任务质量批准或逐样本净成本批准；map 受控探针数值失败，object 探针成本不优。
5. 本轮小段自由问答/数字指标只做可追溯参考，不替代独立场景任务质量评估；尚未建立 WSDR 对正式独立标准的收益结论。

后续应在不改变权重和正式入口的前提下，先获得真实映射或明确的布局输入契约，积累自然合格帧，再运行既有独立任务质量验证与同策略/同布局数值和净成本采样，生成可追溯 evidence/protocol，由独立流程审批。正式记录必须使用 WSDR 独立绑定，不能套用 exact_prefill 的批准。未完成这些步骤时继续完整 Prefill。

复现主比较命令和约束见 [实现说明](../../docs/worldstream_m3_wsdr_kv.md)，实际运行命令见 [service/invocations.json](service/invocations.json)、[final_invocations.json](final_invocations.json) 与 [numeric_service/invocations.json](numeric_service/invocations.json)。本轮没有提交 Git commit 或自动推广证据。
