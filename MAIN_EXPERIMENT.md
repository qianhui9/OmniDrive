# 唯一冻结主实验：完整 R0

本地删除旧 `.git` 后，必要 Native 审计源码由独立快照提供，已有执行时源码哈希保留不变；迁移方式见 [源码溯源说明](docs/provenance/README.md)。

当前唯一改进配置为 [configs/worldstream_r0.json](configs/worldstream_r0.json)，正式模型入口为 [worldstream_m123_auto.py](projects/configs/OmniDrive/worldstream_m123_auto.py)。参数来自此前最终选择：M1 检测阈值0.3；M2 普通刷新预算3、全局间隔2；M3 current_only / C-basic / basic / 0.45 / 0.75，保留精确 prompt 历史 K8 目标模型验证和请求 RNG 隔离。完整参数以 JSON 为准。

Native PyTorch 保留作必要的 A/C 配对基线。WSDR-KV、持久 Prefill KV、旧视觉前缀与答案复用候选、替代执行器、参数搜索及未接入 R0 的 TensorRT/EdgeLLM 部署入口已退役。固定空值或关闭字段仅用于和原始完整配置清单对照。

## 当前运行与验证

- [运行指令](tools/运行文件.md)：统一环境、正式五负载、单独任务、官方评价、CPU/CUDA 回归和监测命令。
- [主实验说明](R0_MAIN_BENCHMARK.md) 与 [主路线执行边界](docs/worldstream_final_route.md)。
- [本次清理状态](outputs/r0_route_cleanup_validation_run01/WORK_STATE.md)：删除前后真实运行、源审计、回归和完整复现的进度。
- [清理后完整复现输出](outputs/m123_r0_only_reproduction_run01/)：18 Log / 18 完整 Scene / 723 不同帧，五种负载、7,230 次正式调用及8次 pilot。最终是否完成以该目录 completion.json 和清理目录 full_reproduction_pipeline.json 为准，不能按计划数推定成功。

主实验由保留的 `tools.worldstream_m123.frozen_main_benchmark` 准备与运行，报告由 `frozen_main_report`、`frozen_main_finalize` 和 `frozen_main_synthesis` 从真实收据生成。新实验必须使用新的不存在目录；参数不再搜索。

## 历史证据与研究边界

[最终策略与原始审计](outputs/m123_final_quality_validation_run01/) 保持原字节，用于参数来源、模型资产校验和历史追溯。该目录的原始记录属于历史实验，其中旧工具命令不是当前运行入口。历史全18场景主实验可从 Git 提交 `b0b234af5cac169bdb620c7da4f4c3db8be12f29` 追溯，本次新调用不复用其收据冒充新实验。

冻结的是 R0 研究配置。已暴露数据、UNKNOWN、残余错误和新增严重错误均按原口径报告；用户允许一定错误比例，仍应明确报告分母和不可评分比例。本次不训练、不修改权重，不据此宣称独立 TEST 或生产风险认证通过。

所有执行命令以 `taskset -c 0-3,6-31` 启动，使用 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。GPU 被其他任务占用时等待，不停止其他用户进程。
