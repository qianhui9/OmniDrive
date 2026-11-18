# 18 条日志扩样实现与验证（2026-11-18）

本次交付准备扩大样本后的正式重跑，由用户执行最终命令启动；没有自动占用 GPU 运行完整实验，也没有把预检覆盖统计当作新推理结果。

## 新增与修改

- 新增 expanded_config.yaml、run_expanded.sh：18 条完整连续日志、每帧严格五问、每方法 3 次独立重复、两方法独立进程。
- 配置要求完整日志数为 18，并检查官方 scene.log_token：这 18 条 scene 对应 18 个不同 driving log，不静默缩减或通过重复样本补齐。
- data.py：模型运行前生成逐日志/逐阶段 GT 样本覆盖、唯一变化查询、known/UNKNOWN 数；日志时长不相加冒充连续时长。冻结 shell 入口、源代码、配置和输入哈希。
- statistics.py：2000 次配对场景聚类 bootstrap，保留场景内时间依赖和所有重复；区分唯一查询/变化转变与重复观测，不能由增加 repeat 人为缩窄区间。
- repeat_statistics 单列重复运行间 mean ± sample SD，包括每次 P95 的 mean ± SD，不混同主表 pooled P95。
- report.py / report_statistics.py：自动生成结果.md、report.md、表1.md、paper_table.csv、overall/scene/repeat summary、paired_comparison。保留原始七列和逐帧/模块缓存结果。
- worker.py 仅扩充进度记录的日志与总帧数；原 OGSM、DAVP、QCSE 算法、阈值、刷新和淘汰机制未修改。
- Protocol v3 使用新输出目录；既有 v1/v2 收据和报告未覆盖。

## CPU 与真实数据预检

完整验证目录：`outputs/experiments/long_run_state_aging_expanded/preflight_20261118_final/`。

**68 项测试通过**，包括本目录 27 项与既有语义/变化响应回归。覆盖：场景聚类不因重复观测扩大、配对抽样相同、零错误边界区间处理、无变化分母处理、重复 P95 统计、跨日志状态边界、完整两方法多日志报告生成与中断统计。测试夹具仅存在于 pytest 临时目录，不是实验数据。

完整 18 条真实日志预检通过：六相机原文件与标定/位姿、连续 next/prev 链、官方 scene 首末样本及总数、每帧三个固定事实问题和两个原 QA、冻结权重和 GT/input/source 哈希均核验。

| Stage | 不同帧 | 不同查询 | Known GT | UNKNOWN GT | 不同采样事实变化查询 | 包含变化的场景数 |
|---|---:|---:|---:|---:|---:|---:|
| Early | 249 | 1245 | 713 | 532 | 22 | 12 |
| Middle | 240 | 1200 | 660 | 540 | 24 | 13 |
| Late | 234 | 1170 | 596 | 574 | 26 | 9 |
| ALL | 723 | 3615 | 1969 | 1646 | 72 | 17 |

这些数字不乘重复次数。变化单位是 scene/frame/question 的采样 GT 事实转变，不宣称 72 个独立物理事件。正式推理评分还需满足评价器的可评分条件；预检数量不保证与最终所有分母完全一致。

正式执行规模：每方法每次 723 帧 / 3615 查询；两方法各 3 次，共 4338 帧次 / 21690 查询执行。最长连续视频仍是 19.949757 秒；18 条日志时长之和约 351.85 秒，不能称为一段 351.85 秒连续视频。更多场景改善覆盖，不足以替代真实长日志验证缓存饱和稳态。

预检报告状态为 PREPARED_NO_INFERENCE；准确率/时延/实际缓存等均为 N/A，没有生成虚构推理图表。CPU 报告验证保存为 cpu_validation.xml，冻结输入与模板核验保存为 validation.json。

## GPU 与正式启动

实现期间 GPU 被其他任务占用，观测显存约 16.8 GiB、利用率 99%。未停止、干预其他进程，没有启动新版 GPU Smoke 或完整重跑。此前完整五问模型的 v2 Smoke/完整运行是历史验证，不能当作本次扩样结果。

用户正式运行：

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/experiments/long_run_state_aging/run_expanded.sh --run-full
```

会自动使用 OmniDrive 解释器、安全 CPU 亲和性与线程限制；GPU 忙时只等待。可选先执行同入口 `--smoke`（两条不同日志、各首 9 个相邻帧），以验证实际模型跨日志边界。

结果默认写入新目录 `outputs/experiments/long_run_state_aging_expanded/run_<时间>/`。完成后在该目录查看结果.md、表1.md；场景 CI 与重复 SD 分开报告。仍可能出现真实 0 或无有效分母的 N/A，不通过伪计数或选择性删样消除它们。
