# 单一主线运行与验证

唯一正式入口是 `projects/configs/OmniDrive/worldstream_m123_auto.py`。Python 使用 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，所有代码执行以 `taskset -c 0-3,6-31` 启动，子进程继承亲和性，避开故障 CPU 4/5。GPU 被其他任务使用时等待，不终止其他进程。无需安装依赖、替换权重或改写数据。

正式顺序为 M1 Precheck → M2 窗口 → 原 Detection/Map → M1 Commit → M3 历史质量门控与精确 Prefill → 原 Decode。M2 的 miss 是同一窗口执行器的全量更新；M3 的 miss 是同一 PrefillExecutor 的 `reuse_length=0`。不支持的契约明确报错。

## 回归与静态审计

本阶段只检查正式执行链，使用现有 runner 的短序列模式：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
  -m tools.worldstream_m123.benchmark.run --execution-audit \
  --run-dir outputs/m123_execution_audit_new \
  --checkpoint ckpts/iter_10548.pth --timing-frames 32 --warmup 1
```

该模式仅调用一个 B3 worker，按场景/时间选择同一场景连续帧；不足 32 帧会记录实际帧数与原因。既有 language workload 每帧选择一个真实 Conv/VQA 问题，保持原图像预处理、完整视觉输入、权重和采样参数；planning 可能无问题，不能用于证明 Prefill/Decode 完成。预热后清空模块/任务头状态，并将预热答案目录与实测目录分开。先运行本轮回归、配置/资源检查和 CUDA kernel witness，再严格校验 checkpoint 并执行实际推理。不会调度基线、消融或完整验证集。

`experiments/B3_execution_audit/` 保留 `per_frame_metrics.jsonl`、`per_question_metrics.jsonl`、`m1_state.jsonl`、`predictions.pkl`、`execution_summary.json` 和 `execution_report.md`。异常写入 `request_failures.jsonl` / `failure.log`，worker 非零退出，不返回伪成功。`audit.json` 对账实际 Attention 批次数、执行器与受保护文件哈希。

计时分析仅在该 benchmark 启用：请求前后 CUDA 同步得到含预处理、传输、模型及输出回传的 wall latency；CUDA Event 记录各窗口前缀/全局后缀、Prefill（含 KV 写入）、Decode 生成循环与完整生成的 stream span。span 含主机提交间隙，嵌套跨度不可相加；原 M2 `M2_*_ms` 为未经独立同步的 host 诊断。峰值显存在预热后重置。普通正式推理不启用这些分析同步；`worldstream_execution.host_elapsed_ms` 是 host 时延，不能宣称纯 GPU 时间。短序列均值/P50/P95/吞吐仅作参考，不推断加速或完整集质量。

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

历史全形状 M2 策略的位级一致不能迁移到当前 v2 选定窗口 GEMM 策略，需按当前签名重新验证。历史 M3 完整 Prefill/相同全前缀通过严格检查；局部前缀生成 token 一致但 logits 未通过严格检查，因此不授予对应正式复用。历史增强需要独立的任务质量、M1 协议和部署模型证据；KV 精确匹配不能替代此证据。当前成本/质量批准为空。

## 本次结果边界

历史配对记录不作为本轮新测量。当前可访问的报告入口为 [执行契约测试](../outputs/m123_execution_contract_20261025/experiment_report.md)、[WSGRR 历史修复](../outputs/wsgrr_mainfix_20261023/experiment_report.md) 和 [本轮短序列审计](../outputs/m123_execution_audit_20261025_run01/experiment_report.md)。完整集官方精度与历史文本独立质量需要单独评估。

历史开发记录曾包含审计导出引用退役字段的失败与修复，这些记录不作为本轮测试结果。以前按用户要求停止的长时评测没有恢复；本轮只执行上面的单场景短序列。
