# GPU 补测入口

服务器 GPU 与当前执行沙箱的设备可见性分别记录。当前沙箱没有 /dev/nvidia*，
不能将沙箱内 PyTorch 的 CUDA unavailable 判作主机无 GPU。主机 nvitop 可用性按用户反馈记录，
实际 CUDA 验收必须由可访问设备的进程完成。已通过审批在主机 RTX 3090 上完成 gpu_validation_v2；该目录包含本次实际 CUDA 结果。

从项目根目录执行以下一条命令，无需安装或修改 OmniDrive 环境：

    taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m2.state_interface_cuda_batch --output outputs/module1_state_interface_rearchitecture/gpu_validation_reproduction

该入口会：

1. 保存 nvitop 输出，以 nvidia-smi 核对空闲卡及计算进程。GPU 占用时每 30 秒监测，不终止其他进程。
2. 在 OmniDrive 环境运行固定种子的 CUDA 矩阵核见证。
3. 执行 tests/worldstream_m1、tests/worldstream_m2、tests/worldstream_m3 全部回归；CUDA 测试不再以 CPU 替代。
4. 使用原 Checkpoint 跑 8 帧 M1 开/关感知一致性对照。
5. 按 CPU 回放协议的相同两场景起点，执行 12 帧自然序列的五臂完整模型比较。
6. 另跑 16 帧受控重复 RGB 序列，保留真实时间、自车位姿、Head、Memory 和 LLM 的当前计算。
7. 各序列 1 次预热、2 次测量；延迟统计排除预热，质量核验包含预热。
8. 保存原生 dtype Memory、实际执行的 EVA 中间输出误差、来源、回退、显存、完整模型 Wall/CUDA 区间。
9. 自动合并新 GPU 原始结果到报告；CPU 阶段文档另存快照。任何未完成或失败项均不标作通过。

每个语言输出目录是全新的。自动成本/质量门保持原样；没有放宽局部更新或近似复用准入。
受控重复图像结果与自然驾驶结果分别报告。GPU 存在或小核见证通过都不等于完整模型质量通过。

输出目录拒绝覆盖。若该目录已经开始过执行，请先保留它，然后为新的批处理使用一个全新目录，
并将其状态和原始结果明确作为另一轮记录，不混入旧轮次。

当前签名的零更新成本与完整任务补测：

    taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m2.state_interface_cuda_batch --zero-cost-followup --repeats 1 --output outputs/module1_state_interface_rearchitecture/gpu_zero_reproduction

该补测在新目录中实测成本；初始记录无任务资格，通过完整模型零更新探测后才供 Auto 使用，不修改历史表。旧表签名不匹配而全部回退的结果仍保留。最终采用的自然和补测目录由 gpu_validation_current.json 记录。
