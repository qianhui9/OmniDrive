# 本轮最终实验产物

- [中文结果报告](../refinement_report.md)：指标、失败候选、适用范围和结论。
- [算法与运行说明](../../../docs/worldstream_m1_coverage.md)：配置、在线/离线边界和复现命令。
- `protocol.json`、`selection.json`：预先记录的场景/token 与确认前冻结的配置选择。
- `holdout_summary.json`、`holdout/`：16 场景、192 个唯一连续 keyframe 的完整结果；逐帧掩码保存在 state JSONL，部分帧另存 NPZ。
- `verification.json`、`per_scene_metrics.json`：经验检查与按场景 bootstrap 区间；不代表统计非劣已获证明。
- `physical_ablation.json`、`critical_recall_by_category.json`：独立物理状态 Oracle、Geometry Only / Appearance Only 和分事件召回。
- `scope_fix_verification.json`：960 帧次的像素掩码一致性及锚点作用范围修正检查。
- `roi_verification.json`、`roi_timings.json`：最终代码 768 帧次实际重放，精确输出一致性和最终 CPU 耗时。
- `warmed_latency.json`、`warmed_latency_before_roi.json`：交替次序预加载微基准，保留计算范围优化前后记录。
- `gpu_roi/`：最终代码 RTX 3090 完整 Forward 对照与独立特征采样，10 轮共 80 帧次（8 个唯一帧）。`results.json` 汇总计时及输出一致性；`gpu/` 保留优化前记录。
- `feature_summary.json`、`m2_candidates.json`：实际特征误差与图像候选统计；精确深层 token 复用认证数仍为 0。
- `figures/`：五组 PNG/PDF，包括 Risk–Coverage、Invalidation–Recall、Runtime Breakdown、真实图像与特征误差。
- `tests.log`：89 项测试通过。
- `final_manifest.json`、`source_final/`：65 个代码/配置/测试文件快照及 SHA256、环境版本、数据索引/历史预测/模型 checkpoint 校验和。

历史阶段原样保留于 `../coverage_round3/`、`../coverage_confirmation/`、`holdout_before_scope_fix/` 和 `source_before_roi/`。不要把开发/校准数据上的结果当作独立确认，也不要把 CPU 原图回放的面积统计直接当作模型预处理后的 token 覆盖率。
