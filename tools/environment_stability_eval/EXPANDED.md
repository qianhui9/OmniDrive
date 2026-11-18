# FQ-3 扩充环境稳定性评估

这是新的扩充入口，保留既有 `fq3.py` 和旧结果。原18场景保留在评估集；从其他场景中按 `sha256(20261118:scene_id)` 排序选30个校准场景，其余全部评估。默认共120个评估场景、30个校准场景。划分不读取 GT 事件数量或模型性能，不为凑齐每级数量改变种子。

四级阈值仅用新校准集的完整5秒窗口拟合，在推理前冻结。因此属于新版本，不能把旧四级表的百分比与新四级表直接混合。已有 val 全部曾用于相关开发/GT分析，不声明独立未见 TEST。场景互斥也不等于数据从未被模型或开发者见过。

## 一条命令启动正式实验

从任意目录执行：

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/environment_stability_eval/run_expanded.sh
```

脚本默认 `--run-full`，固定使用 `/home/qh/miniconda3/envs/OmniDrive/bin/python`。所有子进程继承 CPU 亲和性；不使用4/5号CPU。GPU忙时等待，绝不停止其他进程。环境沿用现有 OmniDrive，不安装依赖、不修改权重。启动阶段打印 `EXPANDED_OUTPUT`，其后是本次新目录。

执行步骤：真实 GT 与输入校验 → 冻结场景划分与四级阈值 → 统计覆盖 → 等待项目 GPU 锁和空闲 GPU → 每方法4帧预热及3帧 smoke → 完整 Native 推理 → 完整 WorldStream 推理 → 生成全部指标和图表。每个方法使用独立解释器，完整场景按时间顺序执行，场景边界清空时序状态。

日志在 `<本次目录>/evaluation/native/runtime.log` 和 `worldstream/runtime.log`，进度在各方法 `progress.json`，总体阶段在 `<本次目录>/status.json`。无参数脚本为前台执行；长任务请使用已有终端会话保持运行。

## CPU 验证

只检查命令入口，不准备数据、不使用 GPU：

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/environment_stability_eval/run_expanded.sh --help
```

真实 GT 准备和覆盖统计（不加载模型、不调用 CUDA）：

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/environment_stability_eval/run_expanded.sh --check-only
```

正式运行和 `--check-only` 自动创建独立时间戳目录，不覆盖旧输出；`--help` 仅显示帮助，不创建目录。也可以用 `--output /绝对路径/新目录` 指定位置。`--check-only` 输出的 `结果.md` 明确标为尚无新模型性能，不把 GT 事件数当作已正确响应数。

## 恢复与重新汇总

使用已经完成 GT 准备的目录启动 GPU 推理，或恢复中断任务：

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/environment_stability_eval/run_expanded.sh --run-full --resume --output /绝对路径/本次目录
```

所有源码、GT、配置和输入哈希必须仍一致。已完整完成且收据哈希通过的方法可复用；未完成的方法默认报错，避免误读旧答案。确需重试时增加 `--retry-incomplete`：把未完成的方法目录移至本次目录的 `interrupted/` 下，然后从第一帧重跑该方法；不删除历史，不从任意中间帧恢复缓存。此操作也适用于未完成的 smoke 方法。

全部推理完成后单独重建汇总，每次创建新的 report 子目录：

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/environment_stability_eval/run_expanded.sh --report-only --output /绝对路径/本次目录
```

## 输出与解释

- `<本次目录>/结果.md`：完成前是 GT 盘点，完成后是四级核心表及完整报告链接。
- `gt_audit.json/csv`：每级 Stable、Changed、变化场景数、连续 GT 跟踪机会、任务组成。
- `report_*/结果.md`：四级和总体四指标、响应率/停止原因、配对响应、场景聚类置信区间、任务分项、真实三问纯前向时延。
- `report_*/core_metrics_by_stability.csv`、`scored_queries.jsonl`、`change_response_events.jsonl`：完整分母及原始评分证据。
- `plan/split.json`、`evaluation/manifest.json`：冻结划分、配置、源码和输入哈希。
- `completion.json`：仅在完整推理及报告验证完成后生成。Smoke 报告单独标识，不能用于正式论文结果。

只保留三个事实任务 q0/q1/q2。相邻、同问题、双方 GT 已知、间隔不超过1秒才能定义 Stable/Changed；UNKNOWN 不跨越。响应先在完整时间流中追踪，再按变化起点的四级标签汇总，窗口边界不截断事件。响应时延是视频采样时间差，纯前向是实际计算墙钟，两者不可混用。

每级50个事件、10个变化场景是覆盖规划目标，不是统计充分性保证，也不是启动门槛。未达到时仍如实运行全部评估数据，并在汇总标注不足；不改变种子、评分器或 GT 来制造足够事件。N/A 保留。成功响应均值必须结合响应率和删失原因解释；全0/全1的稀疏 bootstrap 区间可能退化，不能声称确定性。

当前评分器仅认证可靠正例，部分存在性负例仍为 UNKNOWN。扩充样本不能自动补齐物理出现/消失/遮挡的标签，也不能证明三个模块的独立因果贡献。
