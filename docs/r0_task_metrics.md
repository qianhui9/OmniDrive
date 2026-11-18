# R0 事实问答与规划任务评估

一条命令依次执行完整 R0 的事实问答、规划两个连续流任务，并从原始回答生成指标。不训练、不修改冻结 R0 参数。脚本自动进入项目目录、设置 CPU 亲和性与离线模型环境，并在 GPU 0 被占用时等待。

```bash
taskset -c 0-3,6-31 bash /data/qh/phdCode/work4/OmniDrive/tools/run_r0_task_metrics.sh
```

默认使用 `outputs/m123_r0_only_reproduction_run01/scene_selection.json` 固定的完整主实验 Scene。当前依赖检查得到18个Scene、723帧；每帧分别运行事实问答和规划，合计1446次完整模型执行。两种任务各按Scene时间连续运行，在任务和Scene边界重置M1/M2/M3与任务头状态。

默认每个事实问答帧包含三个固定问题（前方车辆类别、前方车辆存在性、最近行人相对位置），再按conv、vqa文件原始顺序选取最多两个不重复问题。只把问题送入模型，参考答案和未来轨迹只用于离线评分。因此该任务默认最多5问/帧，其FPS不能直接与旧S1～S5不同查询负载的FPS比较。

输出目录默认为 `outputs/r0_task_metrics_YYYYMMDD_HHMMSS/`，必须尚不存在。固定目录或选择完整验证集：

```bash
taskset -c 0-3,6-31 bash tools/run_r0_task_metrics.sh --output outputs/r0_task_metrics_main_run01
taskset -c 0-3,6-31 bash tools/run_r0_task_metrics.sh --scope full-val --output outputs/r0_task_metrics_full_val_run01
```

`full-val` 使用当前info pickle中的所有Scene和帧，逐一检查6相机文件存在及帧间链接；不跳过缺失或失败帧。缺失依赖会报错，不能把不完整运行当作全验证集结果。`--max-scenes N` 只用于预先确定的小规模运行，保留前N个完整Scene，报告中仍保留实际场景清单。`--other-questions 0` 可仅运行三个固定事实问题。不同负载、范围的指标分别报告。

只检查依赖与数据、不构建模型或执行推理：

```bash
taskset -c 0-3,6-31 bash tools/run_r0_task_metrics.sh --check-only
```

只检查完整启动路径（CPU预检查、新Python进程、CUDA、模型插件与配置），不加载大模型权重或执行推理：

```bash
taskset -c 0-3,6-31 bash tools/run_r0_task_metrics.sh --runtime-check-only
```

该模式会等待GPU 0空闲并短暂初始化CUDA。成功时输出 `RUNTIME_CHECK_PASS_NO_MODEL_INFERENCE`，写入 `runtime_check.json`，不生成实验指标。

从已完成的原始输出重算，不重新推理：

```bash
taskset -c 0-3,6-31 bash tools/run_r0_task_metrics.sh --report-only --output outputs/r0_task_metrics_main_run01
```

没有断点续跑或覆盖已有结果的隐式行为。失败时保存 `failure.json`，更换输出目录后重新运行；`--report-only` 只接受已完成的两项推理，并核验当时的代码、配置、info、标注、地图和问答参考文件哈希。

## 事实问答指标

- `mean_e2e_ms`、`p95_e2e_ms`：每个视觉帧及其全部问题的完整端到端时延。CUDA显式同步，包含图像读取/预处理、提示编码、H2D、全部模块和答案导出；包含各Scene的冷启动帧。模型加载、排队、离线评分及收据写入不计入。
- `FPS = 1000 × 帧数 / 全部帧E2E毫秒之和`，不是逐帧FPS的算术平均；另给出QPS。
- `scorable_fact_accuracy`：复用主实验事实评分，按可评分问题求平均；同时输出百分数、完全正确比例、可评分覆盖率和UNKNOWN原因。UNKNOWN从分母排除并计数。
- 四个组：`vehicle_category`、`vehicle_existence`、`pedestrian_position`、`other`。前三组用当前帧可见GT对象与现有目标判定规则；其他问题复用当前帧原始QA的有限事实评分，可能为小数，不能解释为通用自然语言答案准确率。

## 规划指标

模型用原始规划提示生成 `[PT, (x1,y1), ..., (x6,y6)]`，六个点分别对应0.5、1、1.5、2、2.5、3秒的ego坐标系绝对XY位置。解析器支持整数、小数、正负号和科学计数法，拒绝点数错误、非有限坐标和多个候选轨迹。

| 指标 | 1s / 2s / 3s 的定义 |
|---|---|
| `L2_ADE_m` / `ADE_m` | 前2/4/6个点相对于GT的欧氏距离均值，单位米 |
| `endpoint_L2_FDE_m` | 第2/4/6个点的端点欧氏距离，避免将端点L2与ADE混用 |
| `collision_rate_percent` | 时域内任意一步自车包围框与车辆GT占用相交的轨迹比例，单位%；沿用原ST-P3式碰撞内核，掩码GT已发生碰撞的步骤 |
| `raw_collision_rate_percent` | 未掩码GT碰撞步骤的同类碰撞率，供解释主值 |
| `road_boundary_rate_percent` | 时域内任意一步旋转自车矩形离开地图drivable_area的比例，单位%；初始自车已越界的帧单独计数并排除 |

碰撞继承原项目的车辆类别集合与0.1米、100×100米BEV栅格，不表示包含所有行人、交通锥等障碍物的碰撞认证。道路检查使用与碰撞一致的自车尺寸/参考偏移，对旋转后的整个矩形做栅格相交；预测驶出BEV范围计为道路越界，而碰撞标为不可评估并另计分母。

每个时域独立要求该时域内GT的X/Y mask全部有效且GT有限；当前18个主实验Scene对应GT有效帧数为1s 687、2s 651、3s 615。无效预测不填零、不算安全，单独给出解析失败数、有效预测覆盖率、碰撞/道路分母。零可评分样本时指标为JSON null，报告显示N/A。

## 输出

| 文件 | 内容 |
|---|---|
| `TASK_METRICS.md` | 可直接查看的事实问答与1s/2s/3s规划表 |
| `metrics.json` | 两项任务全部指标与定义 |
| `fact_qa_metrics.json` | E2E、p95、FPS、事实准确率与四类问题分组 |
| `planning_metrics.json`、`planning_metrics.csv` | 各时域误差、碰撞/越界百分比、GT/预测/几何分母 |
| `facts_receipts.jsonl`、`planning_receipts.jsonl` | 每帧真实模型回答、同步时延、实际LLM调用统计 |
| `per_question_metrics.jsonl` | 每个问题的答案和事实评分依据 |
| `per_frame_planning_metrics.jsonl` | 解析结果、各时域是否可评分及逐帧指标 |
| `run_manifest.json`、`effective_runtime.json` | 数据、代码与配置哈希、固定场景、模型SHA、生效完整R0参数 |
| `progress.json`、`inference_status.json`、`completion.json` | 进度、真实推理完成状态和评分完成状态 |
| `runtime.log`、`runtime_stage.json`、`runtime_stages.jsonl` | GPU子进程标准输出、最近阶段与阶段流水，便于定位初始化或逐帧故障 |
| `native_crash.log`、`worker_failure.json` | 原生段错误栈、子进程退出码与故障阶段；Python异常另见failure.json |

实现与测试阶段没有启动模型推理；示例命令由用户后续执行后才产生实际模型性能结果。

## 启动阶段的段错误修复

2026-11-15复现了原启动顺序在完成数据预检查后，于 `torch.cuda.set_device(0)` / `torch.cuda._lazy_init` 内触发SIGSEGV（退出码139），此时模型尚未加载。独立干净进程中的CUDA初始化正常，故GPU工作阶段改为 `subprocess.Popen` 启动全新解释器，不继承预检查进程的原生库状态；子进程先初始化CUDA再导入模型插件和读取数据。

CPU父进程负责预检查、清单与离线评分；GPU子进程负责原有完整R0加载和推理。CPU亲和性、离线环境、随机种子、模型参数和GPU锁照常继承。原命令无需改变；失败子进程的退出码与原生栈会保留，不再只有终端的segmentation fault。实机已通过完整 `--runtime-check-only`，55项相关CPU测试通过；未运行完整模型推理。
