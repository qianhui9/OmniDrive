# 当前服务器环境与 R0 运行入口

本机继续使用现有预热的 `OmniDrive` 环境，解释器固定为 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。本次清理无需重装依赖、训练或修改权重。数据、模型文件与校验方式见 [运行指令](../tools/运行文件.md)。

先进入项目根目录：

```bash
cd /data/qh/phdCode/work4/OmniDrive
```

CPU 4/5 存在硬件故障。执行命令必须以 `taskset -c 0-3,6-31` 启动，子进程继承该亲和性。

检查解释器和依赖版本：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -c 'import os,sys,torch,transformers; print(sys.executable); print("torch",torch.__version__,"transformers",transformers.__version__); print("CPU",sorted(os.sched_getaffinity(0))); assert not {4,5}&os.sched_getaffinity(0)'
```

检查正式主入口帮助：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.frozen_main_benchmark --help
```

检查本次清理的完整参数、源码、数值逻辑及已删模块引用：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.r0_cleanup_validation audit
```

唯一改进研究配置为 [完整 R0 参数](../configs/worldstream_r0.json)，执行边界见 [主路线](worldstream_final_route.md)。Native PyTorch 保留作必要基线。旧 EVA-base/TinyLlama 的 TensorRT/EdgeLLM 部署代码已删除；它没有接入本轮 R0，不属于当前运行入口。

CPU/CUDA 回归、完整五负载主实验、单独任务和官方评价的命令集中在 [运行指令](../tools/运行文件.md)。GPU 阶段使用能够访问真实 CUDA 的主机终端；其他任务占用 GPU 时等待，不停止其他用户进程。
