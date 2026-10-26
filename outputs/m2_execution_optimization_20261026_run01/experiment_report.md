# M2 GPU 执行与缓存管理优化交付

日期：2026-10-26。本轮完成代码、最终 GPU 回归 **480 passed**、40 帧冻结 M1 配对、三轮交替顺序完整服务、nuScenes/OpenLane 子集评估。使用 OmniDrive 环境、RTX 3090 和 `taskset -c 0-3,6-31`。没有覆盖旧实验结果。

## 验收结论

| 问题 | 结论 |
|---|---|
| 正式 M2 是否只有一条主路径？ | 是，仍是 WSGRR → DependencyPlanner → WindowExecutor → StateCache。没有成本批准、耗时选择器或备用全量执行器。 |
| 执行器减少了什么？ | 静态窗口元数据、直接输入 gather、有效 Token 整数索引、一次按原生 stride 重建网格、规划 ID 版本证明，减少重复打包、动态 nonzero 和重复验证。 |
| Clone/Commit 是否降低？ | 每帧缓存 clone 少约 108 MiB；实际隔离 Commit 中位数 2.946→2.382 ms。公开 commit 仍克隆外部数据。 |
| 同步是否减少？ | 代表帧 stream synchronize 26→11、item 10→1、nonzero 6→0；GPU 工作的等待仍存在，不能将次数减少等同等待时间全部消失。 |
| 窗口行为是否不变？ | 是。两个 40 帧实验的旧/新计划、执行、年龄、版本、来源、packed outputs 和最终视觉特征逐帧完全一致。 |
| 完整 EVA 路径是否比旧 WSGRR 快？ | 同一已导出 M1 快照的受控配对为 502.41→498.83 ms，降低 0.71%。收益有限，未达到显著整体加速。 |
| 是否比原生视觉净加速？ | 否。同快照边界原生 492.53 ms，优化后 498.83 ms，加速比 0.9874×。 |
| 完整服务是否净加速？ | 否。三轮原生均值 2898.89 ms，正式 3238.24 ms，加速比 0.8952×。完整正式视觉 span 也未获得净收益。 |
| 是否新增精度退化？ | 本轮未观察到。三轮各 40 帧 Detection/Map 输出 digest 和 40 问回答均与旧 WSGRR 完全一致；原有 2 个关键数字退化案例保留。该结论只限本场景与这些问题。 |
| 根本限制是什么？ | 实际能省的前缀计算只有约 3–5 ms，而状态/规划/发布开销更大，后续 22 个 Block 约占 91% 的 kernel active 时间。两者共同限制收益。 |

## 修改文件与具体优化

- `window_executor.py`：保持 execute 接口和 EVA 数学语义。缓存相机/窗口/有效位置及重建索引；部分输入直接 gather；MLP 用已知有效位置的 index_select/index_copy_ 替换布尔索引；直接写原生 stride 的网格，避免两次大布局复制。全更新仍由此执行器调用原生 Block。Q/K/V 独立投影、bias、RoPE、FlashAttention、LayerNorm、残差与数值策略 v2 不变，没有融合新权重或近似注意力。
- `packing.py`、`region_refresh.py`：有界缓存静态几何、有效 token 数和卷积坐标。区域矩形使用与原二维 token mask 等价的分离 OR 减少 padding；GPU 漂移、年龄和不确定性评分保持原顺序。预算仍为每相机 3，最大复用年龄仍为 8。
- `dependency_planner.py`：真实依赖闭包保持不变。批量读取闭包 mask，并缓存从 CPU 合法范围生成的唯一窗口 ID。`_WindowIdProof` 绑定指针、Tensor 版本、窗口数和 CPU ID，重复 GPU 范围扫描移出热路径；修改或不匹配直接报错，独立 Tensor 输入仍验证。缓存拓扑验证与 footprints，并保留结构/方法/参数/Buffer 身份守卫及逐帧 Tensor 版本签名。
- `state_cache.py`：内部接管执行器私有 packed outputs，避免局部执行 clone 后又在 commit 重复 clone。输入 images、Block 0 输入、age、descriptor 仍保留私有拷贝；元数据保持所有权。外部 commit 默认仍 clone。检查与输入/返回特征的 storage alias，有限值、容量和全部可失败工作仍在原子发布之前完成；失败不覆盖旧缓存。
- `runtime.py`：合并输入/Block 0 有限值回传，最终视觉特征合并到缓存发布前检查；减少重复区域/来源拷贝，缓存诊断闭包与几何；参数/Buffer 指针与版本、M1 来源/时间/epoch、场景和缓存完整性检查保留。对外诊断 footprint 独立复制，不能修改内部静态几何。

没有修改 `worldstream/m1/`、`worldstream/m3/`、正式配置、权重或正式策略。所有新增开关只存在 benchmark；服务无法选择旧包。冻结旧代码保存在 baseline_m2/，用于同时期因果对照，不是第二条正式运行路线。

详细存储契约见 [实现说明](../../docs/worldstream_m2_execution_optimization.md)，完整补丁见 [implementation.patch](implementation.patch)。

## 所有权与数值验证

480 项最终 GPU 回归全部通过，见 [tests_gpu_final.log](tests_gpu_final.log)。新增 18 项覆盖按原生 stride 重建与 storage 独立、公开 commit 私有拷贝、私有输出接管、旧缓存不提前覆盖、失败原子性、最终特征 nonfinite、非法 alias、窗口 ID 篡改、静态 footprint 保护、Buffer 替换、描述子分母精确相等、Profiler CUPTI 归因。

Runtime 转移的 packed outputs 不与输入、Block 网格或最终返回特征共享 storage。旧窗口在部分更新时先写到新的 packed buffer，发布失败时历史不变。外部可写输入和返回结果仍独立。零更新共享的是已发布的只读历史，后续局部更新继续先 clone。没有通过无条件 detach 接管外部 Tensor。

真实 checkpoint `ckpts/iter_10548.pth`，六次完整模型的 SHA256 均为 `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`，严格 learned keys/shapes 校验全部通过。没有改动原有数值容差。

[provenance_verification.json](provenance_verification.json) 检查冻结旧包未变、M1/M3 未变、六次完整服务及配对实验的核心源码与交付源码一致。

## 真正的 GPU/CPU 瓶颈

采用 PyTorch Profiler、cProfile 和 CUDA Event。先对旧实现真实两帧测量，再在最终实现与冻结旧包上对同一帧重放。所有 profile 调用、同步隔离与质量评估均在正常性能样本之外。

代表帧更新 49/54 个窗口/Block，以下为 trace kernel 事件按 CUPTI correlation 关联 CPU launch 与嵌套作用域后的**单次计数**，单位 ms；不是重叠 operator CUDA 总时间，也不是无 profiler 条件的最终延迟估计。

| 阶段 kernel active | 旧 WSGRR | 优化后 |
|---|---:|---:|
| 输入 gather | 0.509 | 0.623 |
| 前缀 QKV | 7.482 | 7.386 |
| 前缀 Attention | 1.198 | 1.160 |
| 前缀有效 Token MLP | 16.198 | 15.520 |
| 前缀 Norm/残差/其他 | 7.840 | 7.477 |
| 窗口 scatter/clone/清零 kernel | 1.097 | 0.543 |
| 网格重建 | 0.801 | 0.324 |
| WSGRR GPU 评分 | 0.640 | 0.579 |
| 缓存有限值 kernel | 1.159 | 0.967 |
| 后续 22 个 Block | 434.935 | 434.028 |
| 全部 kernel active（含其他） | 478.063 | 473.491 |
| kernel 数量 | 1208 | 1135 |

输入 gather kernel 本身 **0.509→0.623 ms，略慢**，不能宣称所有阶段都下降。主要改进来自网格重建 0.801→0.324 ms、scatter/clone/清零 kernel 1.097→0.543 ms、去除动态 nonzero 与重复同步。QKV/MLP 等变化还含单次 profiler 的运行波动，最终总收益以交替配对为准。

| 代表帧开销 | 旧 | 优化后 |
|---|---:|---:|
| 动态 nonzero 次数 | 6 | 0 |
| item 次数 | 10 | 1 |
| stream synchronize 次数 | 26 | 11 |
| DtoH 次数 / 字节 | 19 / 1940 | 4 / 1137 |
| HtoD 次数 / 字节 | 7 / 12288340 | 7 / 12288340 |
| Commit 内 clone 次数 | 7 | 5 |
| Commit clone 数据量 | 173.837 MiB | 65.837 MiB |
| 全过程 DtoD 数据量 | 398.798 MiB | 290.798 MiB |

Cache clone 数据量减少 **108.000 MiB/帧**；不是显存总峰值减少同等数值。HtoD 总字节并没有明显减少，主要减少的是 DtoH 次数和重复复制。

独立实际 Commit：在真实 commit 调用之前 synchronize，再测原来参数和所有权路径，三次原始样本保留：

- 旧 wall 中位数 2.946 ms；优化后 2.382 ms，降低 19.14%。
- 优化后包含最终特征和缓存有限值检查，公开路径仍保留拷贝行为。
- 原始数据：[旧提交](isolated_profile/reference_synchronized_actual_publication.json)、[新提交](isolated_profile/synchronized_actual_publication.json)。

**约 431 ms 的发布段不是 clone 本身。** Trace 显示长等待出现在实际 commit 之前：一次 8 字节年龄标量 HtoD 后的流同步等待旧 422 ms、新 446 ms，等待的是排在此前的 GPU 计算，不是传输 8 字节耗时数百毫秒。隔离前置同步仅约 0.02 ms，说明先前等待已在此处发生。同步次数减少后等待集中到更后面，不能把等待总量当作可凭空消除的 CPU 开销。非有限值检查和原子发布完成边界均保留。

原始 trace：[旧](isolated_profile/reference_profile_output/trace.json)、[新](isolated_profile/profile_output/trace.json)；CPU 调用树和 operators.json 在同目录。统计方法有专门回归：Torch 1.13 的 External id 在后续 profiling session 与 CUPTI 编号不能直接连接，当前使用 correlation→CPU launch 时间→作用域归因。

## 40 帧严格控制变量的边际性能

同一 checkpoint、图像预处理、历史真实 Detection 回放维护的 M1 状态、源缓存和 WSGRR 参数。每帧 warmup=1、repeats=3；每次重复交替新/旧或新/原生顺序。重复调用恢复同一源缓存，只有一次实际 M1 Commit。所有帧保留全部原始样本，不选最快一轮。

“纯 M2 快照边界”预先导出同一 M1 snapshot，旧/新直接调用同一个 forward 核心；不包含 M1 Precheck/export。“正式入口边界”通过 forward_from_m1，包含 snapshot export。它们都是同一 GPU 执行实现，不能混用范围。

| 边界 / 全路径平均 | 原生 EVA | 旧 WSGRR | 优化 WSGRR |
|---|---:|---:|---:|
| 同 snapshot 同步 wall | 492.53 | 502.41 | 498.83 |
| 同 snapshot CUDA span | 492.52 | 502.39 | 498.81 |
| 正式入口同步 wall | 489.75 | 515.65 | 512.16 |
| 正式入口 CUDA span | 489.74 | 515.63 | 512.14 |

单位 ms。纯 M2 优化前后加速比 **1.0072×**，相对原生 **0.9874×**，没有净加速。与原生的 wall 差距从 9.88 缩至 6.30 ms，缩小约 36.3%。

两个边界各 40/40 帧的计划、执行、年龄、版本、来源、packed outputs 和最终特征均与旧包**逐元素完全相等**。实际复用为 516/4320（含首帧）=11.9444%，连续帧 516/4212=12.2507%；Block 0/1 各复用 258 个窗口。34/40 帧有复用（85%），与旧行为一致；不是通过删必须事件提高复用。

配对过程峰值 allocated 约 3.008 GiB、reserved 3.520 GiB，包含一个模型、前后对照历史和工作区，不能解释为单一缓存大小。

逐帧证据：[snapshot40/frames.jsonl](snapshot40/frames.jsonl)、[paired40/frames.jsonl](paired40/frames.jsonl)。每帧 reference_pair 保存前后原始样本、执行 ID/年龄/版本/来源一致性及 exact 特征结果。

## 完整实时 M1+M2+M3 三轮

固定 40 帧、全部 40 个原问题、checkpoint、采样参数、seed；顺序为 B0→B3、B3→B0、B0→B3。每条件预热 1 帧，完整正式入口开启 M1/M2/M3。下表端到端为 mean/P50/P95，单位 ms。

| 轮次 | 顺序 | 原生端到端 | 正式端到端 | 视觉 CUDA span 原生/正式均值 |
|---|---|---:|---:|---:|
| 0 | 原生→正式 | 2871.94 / 1749.91 / 9141.75 | 3215.13 / 2306.73 / 8439.83 | 480.03 / 544.70 |
| 1 | 正式→原生 | 2967.08 / 1849.49 / 9179.04 | 3188.77 / 2168.35 / 8408.96 | 480.18 / 543.36 |
| 2 | 原生→正式 | 2857.65 / 1741.79 / 9120.56 | 3310.83 / 2382.24 / 8477.64 | 480.17 / 543.76 |

| 三轮汇总（120 帧测量，40 个唯一帧） | 原生 | 正式 |
|---|---:|---:|
| 完整视觉 CUDA span 平均 | 480.12 ms | 543.94 ms |
| Prefill forward CUDA 平均 | 189.18 ms | 185.00 ms |
| Decode forward CUDA 平均 | 1886.94 ms | 1824.66 ms |
| 端到端平均 | 2898.89 ms | 3238.24 ms |
| 端到端 P50 | 1775.22 ms | 2306.73 ms |
| 端到端 P95 | 9150.46 ms | 8466.96 ms |
| 吞吐量 | 0.3450 frame/s | 0.3088 frame/s |
| 峰值 allocated | 15.353 GiB | 16.014 GiB |
| 峰值 reserved | 15.742 GiB | 16.658 GiB |

正式 M1（含适配/Commit）CPU wall 平均 314.61 ms，M2 内部 host/wait 493.00 ms，M3 host 2053.26 ms。这些含异步/嵌套等待，不能直接相加。CUDA span 含主机提交间隙；上文 profiler kernel active 是另一种指标。

每轮实际复用均为 **350/4320=8.1019%**，29/40 帧复用；连续帧 350/4212=8.3096%。三轮合计 1050/12960，不能将重复轮次增加当作更多独立场景。每轮窗口 ID、Detection/Map digest、全部回答均与上一版 40 帧完全一致，缓存版本连续 2→41（预热产生了版本 1），每帧发布一次，正式验收和 M1 场景完整性通过。

与上一轮**非同期**完整服务记录比较，当前正式视觉平均 543.94 ms 高于 519.59 ms，而端到端 3238.24 ms 低于 3342.97 ms，不能把这两项都归因于本轮执行器。在第一轮逐段核对中，M2 内部 494.42→493.60 ms，窗口执行 CUDA 36.19→35.27 ms，但 M1 snapshot export 24.00→49.89 ms，完整包装的 span 增大主要记录在导出阶段。M1 源码未改；本轮没有同轮旧版完整服务对照来确定导出时间变化的原因。受控的旧/新 M2 因果结论以上文同快照配对为准。当前正式服务相对原生仍慢约 11.71%，没有宣称端到端净加速。

M3 仍用原有 wsdr_kv 和 quality_budget=0.02，三轮实际复用 token=0。原生/正式每轮生成 3022/2926 tokens，Decode 的时间差不能冒充 KV 复用收益。

## 任务质量

未修改 nuScenes/OpenLane 评分函数、阈值、类别过滤。对第一轮实际 predictions.pkl 的完整声明 40 帧 GT 子集重评；其余轮次任务输出 digest 和回答完全一致。以下不是 6019 帧完整验证集分数。

| 指标（比例） | 原生 | 优化 WSGRR | 相对旧 WSGRR |
|---|---:|---:|---:|
| Detection mAP | 0.288546 | 0.288720 | 0 |
| Detection NDS | 0.341442 | 0.341355 | 0 |
| Lane DET_l | 0.232000 | 0.231822 | 0 |
| VLM GT Token F1 | 0.359270 | 0.369617 | 0 |

三轮各 40/40 回答与旧 WSGRR 相同，无新增不可解释数值或任务退化。相对原生仍有原先 **2 个关键数字代理退化问题、4 项退化检查**，没有删除：

- `88a6b5e35eb44deba6b1065d74b0a3c4`，道路布局问题，数字 exact=True→False、precision 下降。
- `fd612e8660e74196af7da7e6487e5d53`，交通元素问题，数字 precision/recall 下降。

[vlm_pairs.json](vlm_pairs.json) 保留所有问题的数字/计数、空间、运动、驾驶安全词法代理及逐问比较；retained_critical_cases 在机器汇总内单独列出。词法代理和单场景不能代替人工语义判断或独立多场景联合验证。

官方子集原始输出：[原生](service/r0_native_official_subset/summary.json)、[正式](service/r0_formal_official_subset/summary.json)。Map 只将真实 lane head 的 DET_l 用于结论，不把无预测头的拓扑/交通元素占位分数当作模型能力。

## 两个真实场景的缓存切换

优化后额外跑两个真实场景各两帧，M1/M2/M3 完整链完成。执行/复用窗口依次为 108/0、106/2、108/0、104/4，缓存版本 2→3→4→5；M1 epoch 6→7。新场景首帧来源全部为本帧，没有旧场景 buffer 泄漏或提前覆盖，第二帧恢复增量。正式验收 PASS，详见 [scene_switch_summary.json](scene_switch_summary.json)。

## 为什么仍未净加速

- 代表帧后续 22 个 Block 约 434.03 ms，占总 kernel active 约 91.7%。本轮没有关闭依赖或强行复用它们。
- 单独原生前两个 Block 的 CUDA span 合计约 40.56 ms。即使前缀 100% 免费、忽略全部管理开销，按串行 CUDA span 的乐观上限，完整视觉最多约 1.09×。按当前 8.10%/12.25% 窗口复用、假设窗口计算均匀，理想节约只有约 3.29/4.97 ms，对完整视觉约 1%；该均匀窗口估算不是非均匀 padding 窗口的严格性能界。
- 纯 M2 边界仍比原生多约 6.30 ms；实际 commit 约 2.38 ms，仍有来源检查、选择、几何/布局和缓存管理开销。正式入口还承担 M1 snapshot 导出，完整服务另有 M1 Precheck/Commit。当前可省计算不足以抵消这些固定开销。
- 不同快照边界下冻结旧/新小幅变快是已测结果；完整服务并没有因此获得净加速。一次 profiler 不足以证明所有 kernel 阶段统计显著下降，三轮也只有一个独立场景。
- 后续若追求明显端到端收益，需要评估扩大**可证明正确**的可复用计算范围，以及降低 M1 状态导出成本；这属于后续算法/状态服务范围。本轮没有擅自修改后 22 个 Block、M1/M3、预算、必须刷新或成本准入来制造指标。

## 复现与 instrumentation 变更表

完整命令在 [commands.json](commands.json)，六次服务的实际顺序/开始结束/退出码在 [service_commands.json](service/service_commands.json)。新目录复现：

```bash
taskset -c 0-3,6-31 bash outputs/m2_execution_optimization_20261026_run01/reproduce.sh outputs/m2_execution_optimization_reproduction
```

脚本需要新的输出路径，按顺序执行 GPU 回归、两个 40 帧边界、隔离 profiler、三轮服务和子集质量/汇总；旧包只作为工具参数。GPU 不可见的沙箱需使用本轮相同的设备访问权限。所有子进程继承避开 CPU 4/5 的亲和性。

| 性能分析相关文件 | 改动 | 位置 / 用途 |
|---|---|---|
| tools/worldstream_m2/profiling.py | 修改 | instrumentation:16、profile_call:66、measure_publication:87；临时包装、Profiler/cProfile、真实提交隔离 |
| tools/worldstream_m2/window_validate.py | 修改 | run_wsgrr:181；冻结旧包同源缓存配对、严格相等检查、快照边界 |
| tools/worldstream_m2/execution_benchmark.py | 新建 | run:11；三轮顺序启动现有正式 worker，不改变模型或运行时 |
| tools/worldstream_m2/execution_analysis.py | 新建 | trace_summary:31、run:96；解析 kernel/correlation、原始时延和质量 |

Profiler 使用上下文退出后恢复所有方法和 hooks，正式代码没有持续 profiler 注入。各包装和原始 trace 可独立审阅/删除，不影响正式 WSGRR 算法。

最终机器数据：[comparison_summary.json](comparison_summary.json)。代码哈希：[source_hashes.json](source_hashes.json)。环境：[environment.json](environment.json)。新增测试覆盖：[test_execution_optimization.py](../../tests/worldstream_m2/test_execution_optimization.py)。

初始 PyTorch 1.13 storage API 兼容性、benchmark 旧包缓存类型转换及可选文件等待脚本问题已修复，初始日志保留，未计入最终性能样本。没有隐去失败后只选择最快的正常样本。
