# FQ-3 环境事实变化响应协议

本协议替换五问环境稳定性实验的论文主评估。Stable 与 Changed 均执行全部 FQ-3：

| 索引 | 任务 | 问题 |
|---|---|---|
| q0 | vehicle_category | What type of vehicle is directly ahead? |
| q1 | vehicle_existence | Is there a car in front? |
| q2 | pedestrian_position | Where is the nearest pedestrian relative to the vehicle? |

以项目 `selective_answer_reuse.QUESTIONS` 的原始固定文本为准；`fq3.yaml` 设置 `other_questions: 0`。不包含任何原始生成 QA。另一个 component-latency 报告的 q0/q1/q3 包含原始 QA，不属于本协议；也不能把其三问离线估算当作本次真实结果。

## 状态与核心指标

- 每个 Scene 的每一帧都推理三个任务，不根据离线 GT 筛选模型输入。Stable / Changed 是同一问题所需 GT 回答事实的状态，不表示整幅环境绝对静止/变化。
- 严格相邻帧、两端 GT 已知、事实 schema 相同且时间差 `(0,1]` 秒时，状态相同为 Stable，不同为 Changed。GT 未知、首帧、缺问或时间断点为 Unclassified；不跨未知帧追溯最近已知 GT。UNKNOWN 不当作模型答错或答对。GT 仅用于离线评分。
- 主表只有 Stable / Changed 两类，列 Factual QA Accuracy、Stale Answer Rate、Change Response Delay；Stable 对后两项为 N/A。Accuracy 为既有主评分器的已知 GT 平均事实得分（另存完全正确比例）。
- Changed 当前帧回答错且符合变化前 GT 才算 Stale。新增响应追踪对后续 Stable 帧也反复与本次变化之前冻结 GT 评分，记录连续历史错误次数和采样持续时间下界。
- `T_response = t_first_correct - t_change`，以视频时间戳计；当前变化帧即正确时为0。0不是在线推理零耗时。遇下一 GT 变化、UNKNOWN、缺问、观测断点、Scene 结束或5秒上限停止；未响应为null。均值只对已响应样本，必须同时报告响应率、删失和同事件配对结果。
- 模型相邻帧继续复用真实历史计算与缓存，仅在 Scene 边界重置。完整WorldStream与Native分别在新解释器运行；统一冻结权重、帧/问题、请求seed、精度和GPU。

## 可选变化细分

现有 GT 能直接给“回答事实类别变化”和“回答方向变化”，但不是可靠的物理原因标签。最近目标可能切换；另存能唯一确定相同 instance ID 的子集。没有可见可靠正例时主评分器为UNKNOWN，不构造“对象不存在”的新负例，因此不假称Appearance/Disappearance覆盖完整。Occlusion、物理Motion与自车运动原因不能仅由相邻答案状态推出。本次细分仅作有据可查的附录，不进入论文核心表，不补造上述五种原因标签。

## CPU准备与文档验证

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.fq3 --check-only --output tools/environment_stability_eval/outputs/fq3_cpu_validation_20261118
```

只读取已有资产、生成真实GT和清单，不构造模型或启动CUDA。输出目录必须新建，拒绝覆盖。预期最后出现 `FQ3_CHECK_PASS`；每方法723帧、2169查询，并且无 `other` 问题组。该目录是PREPARED_NO_INFERENCE，没有性能结果。

## 完整执行

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/environment-stability-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.fq3 --run-full --output tools/environment_stability_eval/outputs/fq3_full_20261118_run01
```

先完成真实GT预检查与权重/输入冻结，再取得项目GPU锁并等待GPU空闲；不停止任何其他任务。先自动执行每方法4帧独立预热+3帧正式Smoke，只有输入/计时/评分/报告全部成功后才继续两方法各723帧正式运行。每个worker含 seeded CUDA kernel witness，排除故障CPU4/5，不安装/重建软件环境。

只用现有 OmniDrive 解释器、冻结模型和数据；当前正常系统驱动直接使用，已知 NVML 库冲突时只复用验证过的进程兼容包，不下载/修改系统驱动。环境账本 `.aris/compute/local.md` 引用的旧spec目前缺失，记录该限制，不据此声称重建了环境。真实kernel与Smoke验证在资源可用后进行。

本次需要真实三问运行：删掉五问中的后两问会改变跨帧候选/答案记忆和实际执行，不能保证五问核心输出子集等同真实三问负载。旧五问完成状态、答案与新增CRD补充保留，不把它们冒充新FQ-3结果。

## 输出与文档

- `结果.md`：Stable/Changed核心表、逐任务结果、响应率/均值/median/P95、观察停止原因、连续历史错误和95%区间。
- `stable_changed_summary.csv`、`accuracy_by_task_and_state.csv`、`change_response_summary.csv`、`change_response_events.jsonl/csv`、`scored_queries.jsonl`。
- `forward_latency_summary.csv`、`raw_forward_latency.csv`：真正三问负载的纯前向，输入加载/预处理/外部H2D/prompt编码与最终导出排除；完整模型内部成本计入。共享帧可能同时有Stable与Changed任务，不将整帧时延双算为两个环境分组。
- `experiment_summary.json`、`manifest.json`、完成标记、严格权重检查、真实收据/答案和来源摘要。
- 新结果页自动写入后，旧 `run_20261117_113951_740642/结果.md` 顶部指向新主协议并标记运行/完成状态，后文五问内容只留作历史。

这是既有暴露val数据上的系统评估，不是独立TEST认证。Native/完整R0对比不独立证明OGSM×QCSE因果协同；若要归因需同一协议下的匹配模块消融。
