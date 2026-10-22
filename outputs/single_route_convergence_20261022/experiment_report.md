# OmniDrive M1/M2/M3 单一主线收敛报告

研究目标：**World-Referenced Stateful VLM Serving with Dependency-Aligned GPU Execution and Efficient Semantic Consumption**。

代码已收敛为普通 A + source-linked EVA 窗口执行 + 可信历史查询/精确 Prefill。只有一个正式入口。M2 数值差异已修复；M3 完整 Prefill 等价性通过。当前真实数据流未复用，历史文本与局部前缀资格仍未通过，不宣称整体提速或完整验证集质量通过。

## 实际代码改动

- 唯一入口：`projects/configs/OmniDrive/worldstream_m123_auto.py`，直接继承原始模型；M1 开启 temporal_visibility/local_support/source_linked_state，M2 固定 window/source_linked/maintain，M3 固定完整视觉补充与精确 Prefill。
- M1 删除替代身份、Shadow/Hint、DQTrack、深度/身份搜索与实验支持分支；保留世界变换、普通几何门控与一次分配、实际时序观测、来源摘要/epoch/过期及原子 Commit。普通关联从不计入物理身份认证。
- M2 只使用 WindowExecutor。缓存未命中、刷新及未获准入都扩大同一图的更新集，没有相机/条件缓存或 Exact 绕过。Patch Embedding 和 block 2 起的后缀完整计算。
- M3 只使用 PrefillExecutor；未命中为 reuse_length=0。删除载体裁剪、近似压缩、批解码、Prefix Seed、借用 KV 和原生生成回退。原生 generate 仅保留在独立原始模型基线与验证参考中。
- M3 冻结并读取 Precheck 的 pending 历史 Store，修复此前读到尚未验证 committed Store 的问题；快照绑定内容摘要、版本、场景/帧/时间、来源签名。历史只补充完整当前视觉，未适用/未取得独立质量资格时不添加文本。
- 检测/地图模型、原始 EVA 类、LoRA、Tokenizer、Decode 和实际权重未替换；M3 保持原始 FP16 LLM 与 FP32 LoRA 精度设置。

按逐文件调用方/继承/测试清单删除 **187 个已跟踪源码、配置、驱动与专用测试文件**。分类：`{'configs': 28, 'projects': 7, 'tests': 56, 'tools': 57, 'worldstream': 39}`。未删除已有 outputs、数据、权重或受保护未跟踪目录。
完整修改/删除清单：`change_manifest.json`、`deleted_files.json`；保留原因：`retained_files.json`。删除前计划为 `deletion_plan*.json`。共享投影、账本、序列与离线评分函数迁入主线通用模块，保留对应正确性测试；旧策略专用测试随策略删除。

## M2 数值差异定位与修复

真实首个 Q 投影因打包改变 GEMM 形状出现约 9.54e-6 的 FP32 差异，经过 FP16 FlashAttention 后放大；简单增加部分 padding 仍有误差。证据为 `window_numeric_diagnosis.json`。修复保留原生 QKV、输出投影与网格 MLP 形状，只 gather/scatter 需要更新的独立 Attention 窗口，同时保持原生特征步长。未放宽 1e-5 atol/rtol，未设置 Exact。

| 真实图像上的受控变化 | 复用输出窗口 | 最大绝对误差 | 位级相同 | 原生视觉算子 wall ms | 增量 wall ms |
| --- | ---: | ---: | --- | ---: | ---: |
| identical | 108 | 0 | True | 477.168 | 437.517 |
| one_window | 106 | 0 | True | 479.115 | 479.163 |
| padded_edge_window | 106 | 0 | True | 478.521 | 478.531 |
| one_window_per_camera | 96 | 0 | True | 479.693 | 479.790 |

上表每例预热后 3 次，属于受控视觉算子验证，包含依赖规划，**排除缓存发布及整个 M1/M3 服务**。自然连续 4 帧的实际复用为 0，输出均位级等于原始 EVA。
局部更新时完整点积仍执行：每个前缀 block 的 QKV/投影覆盖 54 个 padded 窗口（13,824 token rows），MLP 使用 6×40×40=9,600 个有效网格 token；不能将复用 106 个输出窗口解释为跳过 106 个完整 block-window 的全部计算。只有零更新 block 才省去全部前缀工作。局部变化微基准接近原生耗时，未证明净收益。

## M3 真实模型生成等价性

移除原生回退前已验证，收敛后又复验；使用同一真实模型、真实视觉载体、同一生成配置和 seed=1729。局部前缀案例明确是末尾一个 token 变化的受控测试，不作为自然命中率。

| 测试 | 生成 token | RNG | 严格 scores/logits | 最大差异 |
| --- | --- | --- | --- | ---: |
| full_prefill / greedy | True | True | True | 0 |
| identical_prefix / greedy | True | True | True | 0 |
| partial_prefix / greedy | True | True | False | 0.0390625 |
| full_prefill / sampling | True | True | True | 0 |
| identical_prefix / sampling | True | True | True | 0 |
| partial_prefix / sampling | True | True | False | 0.15625 |

完整 Prefill 和相同全前缀位级通过；局部前缀在该测试中生成 token 一致，但 logits 超出原严格容差（atol=1e-6、rtol=1e-5），不能据此授予正式局部前缀优化。前后原始证据分别在 `prefill_before_removal/` 和 `prefill_after_convergence/`。历史文本有独立任务质量与 M1 协议门控，不能以 KV 匹配代替认证。当前批准表全部为空。

## 完整服务的真实配对短测

原 checkpoint `ckpts/iter_10548.pth` 严格加载 PASS，无缺失/意外权重或形状不匹配；只忽略非持久 rotary caches。RTX 3090，12 个连续 keyframes、1 个场景、每帧 1 个真实问题；每个条件预热 1 帧后测量 1 次。相同帧、问题、预处理和逐帧 seed。检测/地图输出摘要与生成 token **12/12 完全一致**。成功配对验证见 `paired_service_validation.json`。

| 指标 | 原始基线 B0 | 单一主线 B3 |
| --- | ---: | ---: |
| 完整请求 mean ms | 3077.541 | 3313.562 |
| 完整请求 P50 ms | 1757.564 | 2130.595 |
| 完整请求 P95 ms | 9235.810 | 9442.529 |
| 峰值 GPU allocated GiB | 15.345 | 15.635 |
| 峰值 GPU reserved GiB | 15.740 | 16.131 |
| 真实 GT token-F1 | 0.359925 | 0.359925 |
| 真实 GT exact match | 0.000000 | 0.000000 |

当前主线延迟增加 **7.67%**，没有净加速。仅一个场景、一次测量，不推断总体性能或置信区间；完整验证集官方 Detection/Map 指标未运行。文本 token-F1 是有限参考匹配，不代表全面语义质量。

- M1 每帧有效历史观测数：`[0, 7, 7, 6, 6, 6, 7, 6, 0, 0, 0, 0]`；累计 **45 个观测事件**、11 个不同 anchor ID、峰值 7；不是 45 个不同实体，物理认证数为 0。冻结历史数与其逐帧一致，账本来源/版本检查 PASS。
- M2 每帧实际复用窗口：`[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`；实际执行前缀 Attention 1296 个窗口，与 kernel 输入 batch hooks 完全一致。
- M3 每个问题实际复用前缀长度：`[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`；12 次均由 PrefillExecutor 执行完整 Prefill，未调用原生生成回退；缓存峰值 296.19 MiB。
- 历史查询未适用原因：`{'no_task_relevant_verified_history': 2, 'full_scene_task_requires_visual': 10}`。当前问题子集没有可用的历史增强候选，未添加历史文本；不能声称历史文本已获独立任务质量验证。

## 回归、失败记录与未解决事项

- 全量当前回归：`231 passed, 1 skipped, 19 warnings in 5.71s`。这套测试对应已收敛策略，旧退役策略的专用测试不再计入。
- CUDA 专项：`1 passed, 19 warnings in 2.96s`；另有上述真实尺寸窗口与真实 LLM 验证。
- 静态审计 `route_verification/verification.json`：唯一正式入口、单一执行器、配置、导入、原任务 Head 和逐文件删除范围检查 PASS。
- 初次开发短测 `service/experiments/B3_language` 在首帧审计导出时引用了已退役的 `_depth_history`，记录失败。已修复并增加连续 Commit 的账本回归；成功主线为 `B3_language_final`。service/audit.json 如实保留开发失败，因此聚合状态仍是 FAIL；成功配对的所有审计项另列 PASS，不能混称全体实验均成功。
- 未解决/未获准：M3 局部前缀严格 logits 等价性；历史文本增强独立任务质量；真实连续流可复用率与净加速；完整验证集官方质量。M2 为保持数值一致仍保留完整点积工作，这限制了局部更新收益。
- 没有伪造/迁移相机级成本或签发优化证书，没有放宽数值门限，没有恢复此前停止的长时评测。全部执行使用 OmniDrive 环境和 CPU 0-3,6-31；验证结束 GPU 空闲。

运行与重现说明见 `docs/worldstream_benchmark.md`、`docs/worldstream_m2_window.md`；实际修改与删除清单、保留原因及机器可读指标均在本报告同目录。
