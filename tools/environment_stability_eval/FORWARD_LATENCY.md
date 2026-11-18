# 纯模型前向时延补充测量

新主口径是一帧六相机及本帧 5 个查询的一次 `model(...)` 调用，不是单个 Transformer token 的 forward。输入读取、预处理、collate/scatter、外部 CPU→GPU 传输、prompt 编码全部在计时前完成；计时前后均调用 CUDA synchronize。完整视觉/检测/地图头、OGSM/DAVP/QCSE、Prefill/自回归 Decode，以及模型内部必要状态适配和传输计入。模型加载和独立预热不计入。

模型当前在 forward 内导出答案。补充观察器对最终 `batch_decode` 和答案文件写入建立独立、非重叠、CUDA 同步区间，从完整模型调用时间扣除；QCSE 内部语义解析/必要 Token 解码保留，不能借“预处理”名义删去算法成本。真实答案仍正常生成/写入，收据记录原始调用时长、扣除项及最终 `forward_ms`。两方法均保留原有只读 MACs/性能诊断观察器，其开销仍存在；这不是 GPU kernel-only 时延。

旧实验收据的 E2E 含数据预处理；`execution.host_elapsed_ms` 包含模型内部导出且没有独立 GPU 完成同步边界。它们不能直接改名为纯前向，也不能通过相加/相减局部视觉和语言计时精确推断新指标。需补充真实测量，原权重、算法和已完成的原始准确率/变化响应结果保留。

服务器 CPU 4/5 有故障，全部命令必须使用指定 taskset。Python 环境沿用 OmniDrive，无安装/重建。GPU 被其他任务占用时，仅监测等待；不停止任何进程。当前系统驱动若健康直接使用系统库；若检测已知 NVML 版本冲突才验证并选择已有的匹配兼容包，不自动下载驱动。

## 文档验证命令（CPU，只读）

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.forward_latency --check-only
```

预期：校验旧 manifest 的源码/GT/输入摘要后输出 `FORWARD_CHECK_PASS 723 frames/method; CPU-only; no inference`。不启动 CUDA、不创建测量目录。

## 真实 GPU Smoke

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/environment-stability-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.forward_latency --smoke
```

各方法沿用 4 帧独立预热 + 3 帧正式测试；不是完整论文结果。GPU 资源等待规则与完整测量相同。

## 完整测量

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/environment-stability-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -B -u -m tools.environment_stability_eval.forward_latency --run-full
```

默认来源为 `tools/environment_stability_eval/outputs/run_20261117_113951_740642`。自动创建全新 `forward_<timestamp>/` 子目录，或通过 `--output <不存在的目录>` 指定。Native 和完整 WorldStream 均重新测量 723 帧，沿用相同 GT 分组、冻结权重、问题和请求随机数。每方法新解释器和缓存，独立预热后正式场景重置，整个配对流程继承同一项目 GPU 锁。第一方法的 GPU 空闲检查通过前不会加载模型。

`--run-full` 会先在其目录内的 `smoke_validation/` 自动执行两方法的真实短测，核验同步时间有效且每帧 5 次最终答案解码和 1 次答案文件写入已完整排除；只有短测报告成功后才启动完整配对测量。若任何阶段失败，会保留明确错误，不把短测结果填入正式报告。

完整执行完毕后生成 `raw_forward_latency.csv`、`forward_latency_summary.csv`、`forward_latency_report.json`、`结果.md` 和 `forward_completion.json`；按 S1–S4/Unclassified/ALL 给平均、P95 和 Scene cluster bootstrap 95% 区间，并比较与原实验的逐字答案一致性。新延迟不用除以 5 伪造单问题前向时延，不用平均时延倒数冒充持续 FPS。

排队时原结果页新增主口径说明，并明确原 E2E 为历史口径；完整测量完成后自动在原结果页的主时延段写入新均值/P95及链接。原 E2E 表、持续 FPS 和旧窗口加速比保留为历史测量，禁止混用分母。Smoke 不改正式结果页。

## 观察器变更清单

| 文件 | 类型 | 内容 |
|---|---|---|
| `forward_timing.py` | 新增 | 同步根前向计时；最终答案导出独立扣除；作用域结束还原所有 hook/方法绑定 |
| `forward_latency.py` | 新增 | 冻结来源校验、进程内安装观察器、新目录配对执行、统计与报告更新 |
| `tests/test_forward_timing.py` | 新增 | 前后同步、排除项不重叠、内部解码保留、输出保持、异常清理 |
| `FORWARD_LATENCY.md` | 新增 | 测量边界、CPU/Smoke/完整调用和局限 |

原 `evaluator.py`、主模型 `petr3d.py`、仪器 `instrumentation.py`、算法和冻结配置均不修改。观察器只在补充 worker 内临时安装，退出作用域时还原。
