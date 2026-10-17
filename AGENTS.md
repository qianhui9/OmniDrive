# 本服务器执行约束

- 用户确认服务器 CPU 4/5 存在硬件故障。执行代码、测试和实验时，命令必须以 `taskset -c 0-3,6-31` 启动；子进程继承该 CPU 亲和性。不要使用 4/5 号 CPU。
- Python 环境为 `OmniDrive`，解释器路径为 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。
- GPU 被其他任务占用时只监测并等待资源空闲，不停止其他用户的进程。
