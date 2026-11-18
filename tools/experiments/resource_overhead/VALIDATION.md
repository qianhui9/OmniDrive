# 实现验证记录（2026-11-17，Asia/Shanghai）

- CPU Smoke：10 项通过，0 failure，0 error。检查 storage/view/NumPy 共享底层去重、模型参数排除、递归循环、真实 StateCache.commit 和 AnswerMemory.remember API、RSS/USS 采集、S1–S5 调度、流式输入摘要、独立重复统计以及不完整结果拒绝汇总。
- Python AST、Shell 语法和 CLI 帮助检查通过。
- 三帧 Smoke preflight：模型权重 SHA256 和输入配置文件校验通过，独立清单已保存。
- NVML：初始化成功；CPU 探测进程没有 CUDA 上下文，进程显存值如实为 null。没有将整卡占用作为当前进程占用。
- GPU Smoke：`gpu_smoke_final_20261117/status.json` 为 `GPU_BUSY`，退出码 75；检查时已有其他任务，整卡显存 16837 MiB、利用率 99%。没有加载模型、分配实验 GPU 状态或停止其他任务。
- 首次诊断发现协作 GPU 锁先于忙时检查阻塞，已修正：check-only 先检查 GPU，再尝试非阻塞锁；只终止了本评测自己创建的等待 worker。默认正式运行仍使用同一协作锁并等待资源空闲。
- 未运行完整 GPU 实验，未获得真实视频推理时的 GPU/模块缓存数值，未生成实验对比表或图像。缓存接口的 CPU 测试夹具不会进入实验汇总。

验证产物（相对项目根目录）：

```text
outputs/experiments/resource_overhead/cpu_smoke/cpu_smoke.json
outputs/experiments/resource_overhead/cpu_smoke/tests.log
outputs/experiments/resource_overhead/cpu_smoke/gpu_probe.json
outputs/experiments/resource_overhead/preflight_smoke_20261117/manifest.json
outputs/experiments/resource_overhead/gpu_smoke_final_20261117/status.json
outputs/experiments/resource_overhead/gpu_smoke_final_20261117/S4_r00_native.log
```

GPU 空闲后应先运行 `run.sh --smoke`，再在确认后运行默认 S1–S5、3 次独立重复实验。报告与图形代码目前尚未经过完整 GPU 结果的端到端验证。

## 21:36 修复后的验证（更新以上早期状态）

- 用户运行 `run_20261117_212401_982577` 的实际异常为 `AssertionError: Invalid device id`。当前 PyTorch 1.13.1 的 `_parse_visible_devices` 仅识别数字；原 UUID 绑定使 `device_count()` 返回 0，而 CUDA 驱动仍能加载模型。
- 修复：按 PCI 地址顺序设置数字 `CUDA_VISIBLE_DEVICES` 和 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，UUID 仅记录；加载权重前检查设备数量和名称，保留 nvidia-smi/NVML 物理 GPU 编号。
- 完整 Smoke 又发现 M3 策略配置仅路径不同，内容 SHA256 完全相同（`33d7dc927b5771a78f3e8181f8876a09cb1727ef11490f3b24035b91c31c6c52`）。校验改为仅允许 byte-identical 的路径变化，其他参数和策略内容变化仍报错；凭据保存到 `runtime_validation.json`。
- worker 失败时现在在终端直接显示最后一个 traceback，避免用户只看到外层 `Worker failed`，且不打印超长 checkpoint 初始化警告。
- CPU 回归：13 项全部通过，包含数字设备绑定、不同 PCI 排序、策略文件内容校验和错误日志显示测试。
- GPU Smoke：`outputs/experiments/resource_overhead/device_fix_gpu_smoke_v2_20261117/status.json` 为 `COMPLETED`，退出码 0，完成时间 2026-11-17 21:36:16 +08:00。Native 和 WorldStream 各 3 帧预热、3 帧正式测量、独立进程、S4、单次重复；无完整长实验。
- 图像/Prompt/问题/相机/shape/dtype/请求 seed 配对检查通过。两个进程的 NVML 和 RSS/USS 采样均无错误，NVML 进程峰值为非空。
- `summary.json`、`summary.csv`、`per_frame_memory.csv`、`module_memory_breakdown.csv`、`report.md`、配对差值表及三组 PDF/PNG 已成功生成；已检查输出存在、非空及模块图可读。
- 正式测量末帧实际模块状态：OGSM CPU 15415814 bytes / GPU 0；DAVP CPU 773909 bytes / GPU 261482488 bytes；QCSE CPU 320637 bytes / GPU 0。CPU 含标记的 Python Estimated Memory；这些数值仅是三帧 Smoke 的末帧快照，不是完整实验结论。

修改文件：`run.py`、`smoke.py`、`README.md`、`VALIDATION.md`。主实验和三个模块实现未修改。
