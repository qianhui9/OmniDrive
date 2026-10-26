# Quality-Bounded WSDR-KV

M3 主线为 `M3Runtime → ReusePlanner → ReuseDecision → PrefillExecutor → 原始 HF Decode → 缓存更新`。完整 Prefill 使用同一执行器的 `reuse_length=0`。`exact_prefix` 只限制同一规划器的近似候选，保留为测试/benchmark 参考；不存在另一套运行时回退执行器。

## 输入与复用含义

保留原始 object 256 → ego 1 → map 256 顺序、全部当前视觉输入、原权重与预处理。object/map 是学习 latent query，不能当作逐物体槽位。M2 EVA 窗口经过全局后缀和稠密任务头，现有接口没有到 carrier 的可靠空间映射，因此始终报告 `m2_region_mapping_unavailable`。

32-token 块只表示当前 LLM 序列中的固定连续计算块，组边界也保留。文本 embedding、token IDs、position IDs、attention/visual masks、模型签名与场景作用域必须兼容；不能跨过变化或不合法的前缀使用非连续 KV。

同帧匹配保持严格精确；跨帧对每个视觉块计算对称归一化 Frobenius/RMS 差异及最大 token 变化。风险使用 task profile 的 object/ego/map 权重、可信 M1 来源连续性和 M2 实际刷新/年龄信息。M1 普通关联不等于物理身份认证，M2 只提供风险线索。未知、计数、安全类使用统一较高权重。所有前缀候选都包含之前块的最大风险，遇到首个超界块便不能复用其后 KV。

## 一份策略

`outputs/final_route/m3_reuse_policy.json` 替代 M3 的旧成本/质量批准分支。旧 JSON 和研究结果保留，当前 M3 不再读取它们；M2 的独立窗口成本机制保持不变。

策略由真实输入的配对实验生成，不包含人工 `approved=True`。记录模型/权重/部署环境/生成参数/布局签名、块大小、任务阈值、质量预算、校准及验证场景、逐请求测量和数据哈希。每个 profile 在校准场景中按质量预算选最低运行时延阈值；独立场景只验证该选择，失败时不改用另一个在验证集表现好的阈值。

统一决策在实测范围内最小化预测运行时延。成本拟合配对的完整 M3 wall，包含输入比较、KV 分支复制、剩余 Prefill、缓存更新及变化后的 Decode；拟合固定开销并用当前决策实耗替换校准决策耗时。没有匹配策略、样本不足、质量超预算或预测无净收益时，输出 `reuse_length=0`。

配对单问题回放后还需执行真实整帧服务验证，覆盖自然采样 RNG 链和缓存演变。`prefill_validate --qualify-service` 会将实际服务中的质量退化写回同一策略，使失败规则失效，不重选阈值。规则撤销后的同样本复跑是回归证据，不能冒充另一份全新独立测试。

默认质量预算为 GT Token F1 绝对下降 0.02 的初始研究预算，`quality_budget` 可配置，不能解释成通用任务标准。数字检查比较实际数字值（含英文数词）、精确序列、值覆盖及对齐误差；空间/运动/安全同时记录显式词法标签与新增退化。它们是可审计代理指标，不能替代完整语义标注或人工安全评估。策略中的小样本经验上界不是理论保证。

## 近似来源和执行

`ReuseDecision` 区分 exact / bounded_approx / none，并记录长度、风险、预测节省、缓存帧、年龄、近似深度及理由。执行器校验当前张量版本、缓存版本、连续前缀证明和策略界限；不能简单传一个近似长度跳过精确检查。精确 benchmark 的整数长度接口仍必须通过严格 LCP。

缓存保留 prefix lineage、根帧、近似深度和风险所用 profile。当前问题改变时重新加权继承风险；近似缓存即使当前 embedding 相同也不能标成 exact。当前校准限制年龄最多 2 帧、跨帧近似深度最多 1；达到限制后仍用同一执行器完整计算。近似全输入命中也重算末 token logits，不使用旧帧末 token logits。

历史文本仍经现有来源和任务筛选，作为视觉标记之后的补充输入；是否启用属于同一校准策略。本轮保持 `history_text=false`，避免把未经本轮校准的输入文本变化混入 KV 近似收益。没有额外历史文本批准路线。

标准 `tools/test.py` 在实际加载 checkpoint 后只计算一次文件身份，供 M3 绑定权重；不会每帧复制或哈希模型参数。只读 benchmark 时间包装不改变可移植的后端键，其余模型/LoRA/数值包装仍纳入签名；live cache 签名仍会检测包装发生变化。

## 复现

所有命令使用 OmniDrive 解释器，并以 `taskset -c 0-3,6-31` 启动。最终实验使用 24 帧、3 个场景、每帧 3 问：校准 12 帧、开发 6 帧、全新验证 6 帧。长前缀失败后仅追加一个最多 128 视觉 token 的短前缀候选；原验证帧不再计入最终独立验证。场景和问题仅按真实元数据、问题文本选择；标签只在离线评分中使用。

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.benchmark.worker \
 --output outputs/new_native --run-root outputs/new_run --condition B3 --workload language \
 --limit 24 --warmup 1 --questions-per-frame 3 --question-selection balanced \
 --sequence-file outputs/m3_quality_bounded_20261026_run01/sequence_extended.json \
 --m3-strategy native_full --capture-m3-inputs

taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.benchmark.prefill_validate \
 --calibrate-wsdr --capture outputs/new_native \
 --sequence-file outputs/m3_quality_bounded_20261026_run01/sequence_extended.json \
 --quality-budget 0.02 --block-size 32 --max-visual-tokens 128 --checkpoint ckpts/iter_10548.pth --output outputs/new_calibration
```

再以 worker 的 `--m3-strategy wsdr_kv --m3-policy <生成策略>` 对同一真实序列执行完整 M1/M2/M3 验证。校准工具以逐问题共同 RNG 进行配对；实际服务运行保留原生整帧采样顺序，须单独报告实际质量和时延。不能把校准回放的 M3 wall 当作端到端时延。

本轮结果、Pareto 图、逐请求记录和可复现命令见 [实验报告](../outputs/m3_quality_bounded_20261026_run01/experiment_report.md)。上一版严格候选规划的结果保留于 [2026-10-25 报告](../outputs/m3_wsdr_kv_20261025_run01/experiment_report.md)，不作为新策略的质量批准。
