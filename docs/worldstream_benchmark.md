# 单一主线运行与验证

唯一正式入口是 `projects/configs/OmniDrive/worldstream_m123_auto.py`。Python 使用 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，所有代码执行以 `taskset -c 0-3,6-31` 启动，子进程继承亲和性，避开故障 CPU 4/5。GPU 被其他任务使用时等待，不终止其他进程。无需安装依赖、替换权重或改写数据。

正式顺序为 M1 Precheck → M2 窗口 → 原 Detection/Map → M1 Commit → M3 历史质量门控与精确 Prefill → 原 Decode。M2 的 miss 是同一窗口执行器的全量更新；M3 的 miss 是同一 PrefillExecutor 的 `reuse_length=0`。不支持的契约明确报错。

## 回归与静态审计

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.verify_final_route
```

静态审计默认另存至 `outputs/single_route_convergence_20261022/route_verification/`，引用本次 tests.log，不覆盖旧清理审计。逐文件删除计划由 `tools/audit_single_route.py` 生成；该工具只写计划，不删除文件。实际删除记录含逐文件 SHA 和原因，且不包含 outputs/data/ckpts。

## 真实服务短测

输出目录必须是新目录，现有 checkpoint 为 `ckpts/iter_10548.pth`。以下使用实际 Conv/VQA 问题、完整当前视觉载体和原采样配置，按场景/时间顺序运行：

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m123.benchmark.worker \
  --output outputs/service_new/experiments/B3_language \
  --run-root outputs/service_new --condition B3 --workload language \
  --limit 12 --warmup 1 --measurement-set smoke
```

`B0` 是原始模型的独立参考，只用于相同样本、相同预处理和种子的质量/延迟对照，不是第二条研究路线。模块归因仅使用同一配置的临时开关，不增加正式配置。

`tools/test.py --eval bbox` 不能自动证明语言流程已执行；实际问题、原生/PrefillExecutor 调用及生成 token 必须记录。当前原配置 planning 请求可能没有语言问题，完整服务验证使用上面的 language workload。

每帧记录 M1 有效历史观测数、冻结历史数、M2 更新/复用 Attention 窗口与完整点积工作、M3 真实复用前缀长度、完整请求延迟、CUDA stream span、显存及真实 GT 评分。CUDA event span 含主机提交空隙，不等于所有 kernel 活跃时间之和；嵌套跨度不相加推断端到端。

## 数值与生成等价性

窗口验证命令见 `docs/worldstream_m2_window.md`。完整 Prefill 与原生生成参考的独立验证如下：

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m123.benchmark.prefill_validate \
  --output outputs/prefill_validation_new
```

该验证工具保留原生 generate 作为对照，仅调用同一个 PrefillExecutor 检查完整 Prefill、相同全前缀和受控改变一个末尾 token 的局部前缀。比较 greedy/sampling token、logits 与 RNG，不写生产批准，也不构成历史文本增强质量认证。

当前 M2 全部受控情形位级一致。M3 完整 Prefill/相同全前缀通过严格检查；局部前缀生成 token 一致但 logits 未通过严格检查，因此不授予对应正式复用。历史增强需要独立的任务质量、M1 协议和部署模型证据；KV 精确匹配不能替代此证据。当前成本/质量批准为空。

## 本次结果边界

`outputs/single_route_convergence_20261022/experiment_report.md` 汇总 12 帧成功配对、4 帧自然视觉验证、受控窗口数值验证与前缀生成对照。成功配对的 Detection/Map 输出和生成 token 逐帧一致，主线当前没有真实复用收益。历史增强没有适用问题，不声称其已通过任务质量验证。完整验证集官方精度未运行。

开发短测中出现过一次审计导出引用退役字段的失败，已修复并新增账本回归；失败记录保留在 service/experiments/B3_language，成功主线为 B3_language_final。service/audit.json 会如实保留此失败，成功配对检查单独见 paired_service_validation.json。以前按用户要求停止的长时评测没有恢复。
