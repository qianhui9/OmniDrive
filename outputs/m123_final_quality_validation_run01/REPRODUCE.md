# 复现

基准提交40c8cdce74ef2e5cbdb8c2bb6eaa5ef42e2ad071；当前审计提交包含本目录与两个外部脚本。环境warm reuse，无安装/训练/下载；权重ckpts/iter_10548.pth，SHA256 6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d。GPU RTX3090。所有命令必须使用上述taskset；GPU被占用只等待，锁/tmp/omnidrive_m123_latency_gpu.lock。

以下命令须在 /data/qh/phdCode/work4/OmniDrive 项目根目录执行。select仅允许第一次选择，已有final_research_policy.json时拒绝覆盖；之后只能report重新核验，或使用新的复现目录。

最终参数只有R0一套。policy_archive/pre_confirmation_policy.json保留确认前的原始冻结文件及SHA；final_research_policy.json随后只补充已观察到的残余严重错误风险元数据。policy_qualification_revision.json逐项证明M1/M2/M3、质量预算、候选选择均未修改。档案不构成第二套推荐运行路径。

无GPU历史完整性校验：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_analysis verify
```

本次三个真实GPU阶段的执行状态、源文件哈希、严格checkpoint验证、seeded CUDA witness与原始请求均在对应子目录。完整请求JSONL已无损gzip，receipt_archive.json包含原文与压缩SHA256；读取器可直接读取.gz，不需要大型Tensor。

在现有收据上重新组装离线报告（不会训练或运行推理，保留冻结策略）：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_analysis report
```

重新推理必须使用全新输出目录；复用相同帧只是回归/性能重放，不是新的独立TEST。顺序如下：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_analysis prepare --root /tmp/omnidrive-final-quality-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_capture causal --root /tmp/omnidrive-final-quality-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_capture dev_quality --root /tmp/omnidrive-final-quality-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_analysis select --root /tmp/omnidrive-final-quality-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_capture selected_confirm --root /tmp/omnidrive-final-quality-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_analysis prepare_scene_start --root /tmp/omnidrive-final-quality-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_capture scene_start_causal --root /tmp/omnidrive-final-quality-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_analysis map_quality --root /tmp/omnidrive-final-quality-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.final_quality_analysis report --root /tmp/omnidrive-final-quality-replay
```

Full必须先于对应Incremental，以在/tmp/omnidrive-final-quality-features生成同序列当前返回NCHW参照。不能同时运行两个重复实验写同一临时特征命名空间；GPU锁串行整个阶段。临时Tensor不提交Git，不参与在线准入。

新阶段统计从实际请求逐行计算，历史复用不计新增；causal40帧、DEV128帧、冻结后五负载确认208帧、官方Scene首帧起点causal80帧。确认中的32个三问B/C收据来自先前DEV，其余208实际执行。既有正式R1五负载实验保持独立，绝不混入本轮均值。

CPU/CUDA回归：

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --output /tmp/omnidrive-final-quality-cpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/omnidrive-final-quality-cuda
```

GPU命令需要真实主机CUDA访问；沙箱可能隐藏CUDA/NVML，应在已授权主机执行，不能把沙箱不可见误记为GPU实验失败。生产默认关闭；零DEV新严重错误并非严格风险认证。独立TEST不足与历史NO_GO均保留。
