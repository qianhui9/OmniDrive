# WorldStream 消融实验

**最新统计口径：** 正式结果已按主实验评分器、UNKNOWN、平均事实得分及 FPS/QPS 分母统一，见 [MAIN_PROTOCOL.md](MAIN_PROTOCOL.md) 和 [最新结果](../../outputs/worldstream_ablation_full_run01/结果.md)。本页原 `--report-only` / `summarize_results` 属于历史扩展评分入口，不用于覆盖当前 `main_protocol_v1` 主表。

正式一键顺序运行（14 个独立 GPU 子进程，完成后自动评分和生成四张表）：

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/worldstream_ablation/run.sh --output outputs/worldstream_ablation_full_run01
```

脚本自动进入项目目录，使用 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，无需手动激活 conda。默认 GPU 0，等待空闲并与项目其他实验共享锁；不停止任何已有任务。默认使用既有主实验固定的 18 个完整 Scene / 723 帧，每帧三个固定事实问题，加最多两个当前帧原始 QA 问题。按 Scene 时间顺序运行，Scene 与配置之间重置全部状态。实际数量以新输出的 `manifest.json` 为准。

输出目录必须不存在。完整运行会执行 10,122 次视觉帧推理；这是较长的正式任务，本次交付只执行启动验证。不要用 smoke 指标写论文。首次检查会读取数十 GB 权重和标注以计算 SHA256，加载阶段不代表卡住。

后台运行并保留总日志：

```bash
taskset -c 0-3,6-31 nohup bash /data/qh/phdCode/work4/OmniDrive/tools/worldstream_ablation/run.sh --output outputs/worldstream_ablation_full_run01 > /data/qh/phdCode/work4/OmniDrive/outputs/worldstream_ablation_full_run01.launch.log 2>&1 &
```

其他入口：

```bash
# 仅检查依赖、权重哈希、图像/QA和完整时间流，不加载模型
taskset -c 0-3,6-31 bash tools/worldstream_ablation/run.sh --check-only

# 所有配置的真实推理启动测试：首个Scene连续3帧，共42次调用
# 只在smoke模式共用一次完整权重加载；各配置独立清空状态
taskset -c 0-3,6-31 bash tools/worldstream_ablation/run.sh --smoke --output outputs/worldstream_ablation_smoke_new

# 单独选择一张表；Native自动加入以提供实际计算量分母
taskset -c 0-3,6-31 bash tools/worldstream_ablation/run.sh --group ogsm --output outputs/worldstream_ablation_ogsm_run01
# --group 可选 module / ogsm / davp / qcse / all

# 全部当前可读val场景；缺图/缺QA立即失败，不静默跳帧
taskset -c 0-3,6-31 bash tools/worldstream_ablation/run.sh --scope full-val --output outputs/worldstream_ablation_full_val_run01

# 只运行前三个固定事实问题，可减少语言执行负载
taskset -c 0-3,6-31 bash tools/worldstream_ablation/run.sh --other-questions 0 --output outputs/worldstream_ablation_facts_only_run01

# 从完成的原始收据重算全部表格
taskset -c 0-3,6-31 bash tools/worldstream_ablation/run.sh --report-only --output outputs/worldstream_ablation_full_run01

# 完整运行被中断后，跳过哈希验证通过的已完成配置
# 仅支持完整配置之间续跑；失败配置的部分帧不自动覆盖，需换新输出根目录
taskset -c 0-3,6-31 bash tools/worldstream_ablation/run.sh --resume --output outputs/worldstream_ablation_full_run01

# 小网络机制回归，不占用GPU
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m pytest tools/worldstream_ablation/test_mechanisms.py -q -p no:cacheprovider
```

主要文件：

| 输出 | 用途 |
|---|---|
| `ABLATION_REPORT.md` | Module / OGSM / DAVP / QCSE 四张表及场景覆盖 |
| `module.csv`, `ogsm.csv`, `davp.csv`, `qcse.csv` | 可导入论文表格的实际指标和分母 |
| `metrics.json` | 核心及辅助指标、场景分组、准确率/Stale分母 |
| `manifest.json` | 固定场景、配置矩阵、问题负载、随机种子、代码/数据/权重哈希 |
| `planned_commands.json` | 各独立进程的可审计启动命令 |
| `<variant>/receipts.jsonl` | 真实回答、MACs、CUDA时延、状态验证及路由收据 |
| `<variant>/per_question.jsonl` | 当前GT评分、前次GT、变化机会、过期回答判定 |
| `<variant>/effective_runtime.json` | 实际挂接的模块和替代策略 |
| `<variant>/runtime.log`, `stage.json`, `exception.json` | 完整运行的日志、阶段与故障；smoke共用根目录`smoke_runtime.log` |
| `completion.json` | 只有推理完整且评分成功后才生成 |

实现细节、指标边界和可支持的结论见 [EXPERIMENT_TASK.md](EXPERIMENT_TASK.md)。启动验证的实际证据见 [VALIDATION.md](VALIDATION.md)。
