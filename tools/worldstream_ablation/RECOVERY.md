# 缺失实验恢复运行

本次恢复目标为原目录 `outputs/worldstream_ablation_full_run01` 中的两组：

1. `qcse_semantic_cache`：Semantic Answer Cache。
2. `qcse_prompt_kv`：Prompt/KV Cache。

每组仍按原 manifest 执行 18 个 Scene / 723 帧 / 3615 个查询，种子、权重、问题、算法及评分口径不变。两组共 1446 帧 / 7230 个查询，以独立进程依次执行。原来的 12 组通过校验后跳过。

## 当前驱动问题

2026-11-17 实机只读检查发现：

```text
nvidia-smi: Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.178
/proc/driver/nvidia/version: loaded kernel module 580.173.02
```

这属于服务器驱动/用户态库版本不一致。**后续已验证本项目可通过隔离匹配用户态库免重启继续运行，当前优先使用 [DRIVER_COMPAT.md](DRIVER_COMPAT.md) 中的 `recover_compat.sh`。** 系统驱动环境的统一仍由服务器维护人员处理。实验代码不会更新系统驱动、重置 GPU 或停止其他任务。普通恢复入口在驱动不可用时每 30 秒记录 `WAIT_GPU_DRIVER` 并等待；GPU 被其他任务占用时记录 `WAIT_GPU_IDLE` 并等待。默认不设等待上限，不在等待时加载模型。

## 启动命令

前台执行：

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/worldstream_ablation/recover.sh \
  --output /data/qh/phdCode/work4/OmniDrive/outputs/worldstream_ablation_full_run01
```

推荐后台执行，日志追加写入以保留历次启动信息：

```bash
taskset -c 0-3,6-31 nohup bash /data/qh/phdCode/work4/OmniDrive/tools/worldstream_ablation/recover.sh \
  --output /data/qh/phdCode/work4/OmniDrive/outputs/worldstream_ablation_full_run01 \
  >> /data/qh/phdCode/work4/OmniDrive/outputs/worldstream_ablation_full_run01.recovery.log 2>&1 &
```

脚本自动进入项目并使用 `/home/qh/miniconda3/envs/OmniDrive/bin/python`；所有子进程继承 CPU 0–3、6–31，默认使用 GPU 0。源文件和资产哈希校验需要数分钟，期间不代表卡住。补跑本身预计约 1–2 小时，GPU/驱动等待时间另计。

只校验完整原始输入、已完成收据和续跑计划，不启动 GPU、不修改输出目录：

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/worldstream_ablation/recover.sh \
  --dry-run \
  --output /data/qh/phdCode/work4/OmniDrive/outputs/worldstream_ablation_full_run01
```

可选 `--gpu-wait-timeout 3600`：每个子进程等待驱动/GPU 超过约一小时后退出，保留失败记录；默认 `0` 持续等待。单次驱动查询另有 15 秒超时。

## 恢复与汇总规则

- 原 `manifest.json`、原推理 worker、模块实现及 12 组已完成收据不修改；完整组哈希或帧序有异常就拒绝续跑。
- 已标记失败的目录先整体移动到 `recovery_history/<时间戳>/<配置>/`，再建立新目录，旧日志和部分收据均保留，不拼接历史缓存或跨进程续接半个 Scene。
- 对没有失败标记的部分目录拒绝自动覆盖，避免误移动仍运行的任务；同一输出目录有专用恢复锁，防止重复启动恢复程序。
- 驱动恢复后按原队列顺序补跑缺失配置。某组推理失败会停止队列并记录原因，再执行同一恢复命令即可跳过已完成组、重试失败组。
- 两组成功后，自动重新进行所有 14 组的离线 GT 评分，更新 `结果.md`、`ABLATION_REPORT.md`、四个 CSV 和 `metrics.json`；报告按实际状态生成，不再固定显示 12/14。
- 只有全部配置完成并评分核验通过，才写入整套 `completion.json`。恢复代码、Shell、报告代码和原 manifest 的 SHA256 保存在 `recovery_history/<时间戳>/recovery.json`。

## 验证

本次代码验证包含驱动 exit 18 后重试、GPU 被占用时等待、查询超时不误放行、失败证据归档、完整结果拒绝覆盖、损坏收据拒绝复用、恢复进程互斥，以及 14/14 报告不残留缺失状态。与既有模块机制回归合计 29 项通过；另通过 Bash 语法检查。

当前驱动故障下没有重新执行真实 GPU 推理；Semantic 和 Prompt/KV 的原始机制测试及上一轮真实三帧启动验证仍可参考 `VALIDATION.md`，不能把本次 CPU 检查当作新的正式推理结果。
