# 主实验冻结与空间清理

主路线以 `c8ddb66bf90ee2c4d4627751d5f9d3961133f810` 的最新研究结果为准，配置与风险结论逐字节保留。唯一研究参数：M1=.3、M2预算3、M3 current_only/C-basic/.45/.75；残余严重错误与独立TEST不足不因冻结而改变，生产关闭。

已删除 20801 个无版本旧输出、重复原文和可重建临时/安装缓存。清单占用合计 27.811 GiB；整个文件系统实测可用空间净增加 27.807 GiB。逐磁盘测量在 cleanup_result.json / disk_before.json / disk_after.json。项目 outputs 从约27GB降至3.7GB。空间单位GiB为1024³字节，du的四舍五入显示可能不同。

保留全部版本化源码/配置/历史审计文件、主实验gzip收据、约33GB权重与初始化文件、原Conda环境、原始nuScenes数据链接和Git历史。旧参数仅作为主审计的历史依赖保留。未运行GPU推理、未停止进程、未清空RAM/GPU显存/系统页缓存。

## 文件

- freeze_manifest.json：唯一主路线与冻结绑定。
- protected_tracked_files.json / protected_checkpoints.json：删除前固定的文件SHA。
- cleanup_plan.json / deletion_receipts.jsonl：逐文件清单与实际删除记录。
- archive_validation.json：15份主线/前轮请求gzip副本及原文SHA校验。
- preservation_validation.json：源码/全部权重/主配置/数据链接保持不变。
- reanalysis_validation.json：从仅gzip检出副本成功重建报告，原性能与风险结论一致，原文件未改写。
- frozen_entry_validation.json：只准备R0的15格A/B/C五负载入口，240次未来调用计划，本次GPU调用为0。
- cpu_validation/summary.json：清理后CPU {'passed': 889, 'skipped': 8}。

主入口说明：[MAIN_EXPERIMENT.md](../../MAIN_EXPERIMENT.md)。归档的主实验：[最新结果](../m123_final_quality_validation_run01/result_answers.md)。

## 检查保留完整性

在项目根目录执行；只读取模型权重和收据，可能产生较多磁盘读取，不运行GPU。

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.freeze_cleanup verify
```

原始JSONL可按receipt_archive.json从同目录.jsonl.gz解压恢复；旧实验中无版本、无压缩副本的大型临时产物已按授权删除。清理过程不重写Git历史，也不强制移动已有标签。
