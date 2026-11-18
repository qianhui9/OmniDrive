# 项目进程内的 NVIDIA 驱动库兼容方案

本机已加载内核驱动 `580.173.02`，系统用户态驱动库为 `580.178.04`。原环境既不能初始化 NVML，PyTorch CUDA 初始化也返回错误 804，不能仅跳过 `nvidia-smi` 检查。

本次已验证可用的处理方式是：从 NVIDIA 官方 Ubuntu 22.04 仓库下载与已加载内核模块精确匹配的 `580.173.02` 用户态库，只解包到项目输出目录，然后在实验新进程中通过 `LD_LIBRARY_PATH` 优先加载它们。没有安装系统软件包、修改系统符号链接、重新加载内核模块或重启服务器。

## 已验证结果

- 隔离 NVML 能正常查询 RTX 3090，报告驱动 `580.173.02`。
- OmniDrive 环境的 `torch 1.13.1+cu117` 能初始化 CUDA。
- 在获取项目 GPU 锁并确认 GPU 空闲后，真实执行并同步了 GPU 矩阵乘法，结果正确。
- `/proc/self/maps` 确认 PyTorch 加载的是项目隔离目录中的 `libcuda.so.580.173.02`。
- 兼容库与恢复逻辑的 13 项 CPU 测试通过。

证据：[probe_result.json](../../outputs/worldstream_driver_compat/580.173.02/probe_result.json)。库文件的来源、下载包 SHA256、各解包文件 SHA256 和符号链接记录见 [manifest.json](../../outputs/worldstream_driver_compat/580.173.02/manifest.json)。

## 实验恢复入口

在当前服务器上，匹配库已经准备好。后台启动命令为：

```bash
taskset -c 0-3,6-31 nohup bash /data/qh/phdCode/work4/OmniDrive/tools/worldstream_ablation/recover_compat.sh \
  --output /data/qh/phdCode/work4/OmniDrive/outputs/worldstream_ablation_full_run01 \
  >> /data/qh/phdCode/work4/OmniDrive/outputs/worldstream_ablation_full_run01.recovery.log 2>&1 &
```

本次会话已使用该入口替换原先处于 `WAIT_GPU_DRIVER` 的恢复子进程，并重新启动两组补跑，启动记录在 `outputs/worldstream_ablation_full_run01/recovery_compat_launch.json`。**已有兼容恢复进程运行时不需要重复执行。** 原等待任务的日志及失败记录保留，12 组已完成推理保持不变。

`recover_compat.sh` 为新进程选择匹配库后调用既有 `recover.sh`，仍执行全部输入/权重/源码核验，只补跑缺失配置，GPU 忙时继续等待。恢复过程将兼容库清单路径、SHA256 和 `LD_LIBRARY_PATH` 记录在本次 `recovery_history/<时间戳>/recovery.json` 中。

查看日志：

```bash
taskset -c 0-3,6-31 tail -n 30 -f /data/qh/phdCode/work4/OmniDrive/outputs/worldstream_ablation_full_run01.recovery.log
```

## 在同类环境中重新准备

仅在本地匹配库不存在时执行，需要网络访问 NVIDIA 官方仓库：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u \
  -m tools.worldstream_ablation.driver_compat prepare
```

下载地址固定为 NVIDIA 官方仓库，包版本必须与当前内核驱动匹配，所有包均按仓库元数据核验 SHA256 和大小后才解包。缓存存在时重新核验各库文件及符号链接。该实现目前限定 Ubuntu 22.04 / x86_64 / 580 驱动分支。

这是本项目的进程级兼容环境，系统默认 `nvidia-smi` 仍可能报版本不一致。内核驱动版本变化后，旧隔离库会被拒绝使用；服务器驱动环境统一后可改回普通 `recover.sh`。最终汇总仍需等两组正式推理和离线评分全部完成。
