# OmniDrive 项目分析与模块一综合优化

当前推荐配置是 `configs/worldstream_m1_balanced.yaml`，完整模型入口是 `projects/configs/OmniDrive/worldstream_m1_balanced_observe.py`。本轮按照风险、有效覆盖和成本共同选择，实测结果见 [优化报告](../outputs/module1_optimization/optimization_report.md)。原有两轮配置和结果保留。

## 项目主链路与限制

| 环节 | 本地实现与分析 | 本轮处理 |
|---|---|---|
| 数据和坐标 | `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py` 提供自车到世界位姿和各相机标定；图像预处理会改变投影坐标。模型解码框使用 gravity center，离线 JSON 框在世界系。不能直接拿 bottom center 与 GT center 比较。 | 保留逐相机时间、位姿与图像仿射变换；历史输出只进入 commit，GT 只用于离线评估。 |
| EVA 视觉骨干 | `models/backbones/eva_vit.py`，配置为 24 层、1024 维、16 像素 patch。每两层窗口注意力后有全局注意力。 | 局部像素稳定不能证明全局特征不变；本轮继续执行完整 backbone。 |
| 检测和地图 | `StreamPETRHead` / `PETRHeadM` 包含时序 memory、top-k 更新和 ego motion 变换；配置 memory_len=600、topk_proposals=300。 | M1 是独立状态库，不修改 Head memory。关联优化针对 M1 的历史世界锚点。 |
| 视觉语言接口 | `Petr3D.simple_test_pts` 将 object/map carrier 拼接后交给 LLM；共享 query 与检测框缺少逐对象一一对应。语言输出目录已有文件时原代码会跳过生成。 | GPU 对照每轮使用新的语言目录，保证实际执行 LLM；没有用跳过生成伪造速度提升。 |
| 状态认证 | `worldstream/m1/state_manager.py` 在前向前检查历史状态，真实前向后 commit。双向光流、几何与局部 patch 是同源 RGB 证据。 | 保持分层证据和保守失效规则；新增历史质量元数据及可配置门限。 |
| 局部失效 | 精确像素掩码来自实际通过验证的 patch，未知区域失效；导出的 tile 仅为保守包围。 | 六相机并行处理，保持精确像素掩码一致。没有扩大稳定区域来提高覆盖率。 |
| 评估 | 原两轮已经查看过的场景不能再作为本轮独立验证。重复帧/同一实例也不是独立样本。 | 新增冻结校准/验证场景清单、源码哈希、修改前快照、逐帧等价核验和完整 GPU 对照。 |

用户所说的“结果最优”在本轮具体为：在现有模型、数据和 observe-only 边界下，避免降低风险/覆盖表现并减少 M1 成本。检测 mAP/NDS、规划误差或碰撞率需要另外的任务评估与训练实验；本轮没有把状态候选率当作检测精度。

## 代码变化

`association.py` 先按锚点类型/类别分组，对检测框距离与尺寸门限向量化；地图折线在每次关联调用中每个输入只重采样一次。尺寸、方向、距离门限和合法 Hungarian 分配保持一致。随机混合对象/折线测试与修改前真实回放确认配对一致。

`state_manager.py` 使用受限 `ThreadPoolExecutor` 并行生成相机证据，线程全部完成后再按原相机顺序做遮挡排除、状态判断和掩码输出。线程数默认 1，推荐配置根据本机预热基准选为 6；库不修改进程级 OpenCV 线程数。实验程序显式设 OpenCV 单线程，避免嵌套并行干扰。其他 CPU/资源共享场景应重新测量线程数，不能将 6 视作所有机器的最优值。

`state_types.py` 和适配器增加历史检测 `source_score`；commit 同时记录 `history_observations` 与 `last_innovation_m`。位置创新量是新计算中心与历史预测中心的距离，不是与 GT 的误差；历史计数是同一关联 ID 的累计匹配观测次数，不等于真实身份认证或每帧连续可见。precheck 不使用当前帧检测分数，不递增历史计数，也不更新计算版本。

新增配置均向后兼容：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `temporal_workers` | 1 | 相机证据计算线程数，支持 1..6 |
| `certification_min_source_score` | 0 | 历史预测分数下限；0 关闭，启用时缺少分数也拒绝 |
| `certification_min_history` | 1 | 至少多少次匹配的实际 commit 观测 |
| `certification_max_innovation_m` | 0 | 历史位置创新量上限；0 关闭，启用时缺少历史创新量也拒绝 |

推荐配置保留这些质量门限的默认值。校准中只有 2 个被接受的独立实例，无法证明开启新门限的风险收益；较严格的分数/历史门限甚至会保留错误对象并删除正确对象。地图或其他没有分数的输入仍按未知处理，启用分数门限时失败关闭。

## 证据与实验设计

`optimize_balanced.py` 在查看校准指标之前写出 32 个场景的精确 sample token 列表。校准评估 1600/800 个光流点与 5 种历史质量门限，质量门限只过滤候选，不改变状态提交和局部像素证据，因此可在校准日志上做精确后筛选。独立实例/场景数量不足时，必须保留原判断策略。

有足够证据时，候选方案须满足错误接受率不升高、候选量至少原来的 75%、有效状态恢复至少 90%、新区域召回下降不超过 0.5 个百分点、失效面积增加不超过 2 个百分点；再在 Pareto 集合中选中位成本最低者。此规则明确表达本轮偏好，并不构成通用最优目标。

`selection.json` 在独立验证前冻结。384 个唯一数据帧中，192 帧用于校准，另外 192 帧用于验证。验证比较的是本轮开始前的源码快照与最终配置，审计包含全部精确失效掩码、候选和无效 ID、版本、provenance 及关联配对。真实 GPU 实验另用已知场景 8 帧验证运行等价性，不能把它说成新的独立泛化数据。

CPU 回放计时不含模型、磁盘读取和离线 oracle；GPU 的 M1 计时包含适配和数据传输。多相机阶段耗时之和在并行执行时是累计工作量，应使用 `precheck_total`、`commit_total_ms` 和 `m1_total_including_adapters_ms` 衡量墙钟。各实验的基线必须成对比较，不能跨机器/跨轮直接相减。

## 使用与复现

在仓库根目录执行：

```bash
conda activate OmniDrive
export PYTHONPATH="$PWD"
export MPLCONFIGDIR=/tmp/omnidrive-mpl
python -m pytest tests/worldstream_m1 -q
```

真实 GPU 等价/成本实验（选择新的输出目录，避免覆盖）：

```bash
python -m tools.worldstream_m1.gpu_refinement \
  --output outputs/module1_optimization_repeat/gpu \
  --config configs/worldstream_m1_balanced.yaml \
  --reference-root outputs/module1_optimization \
  --frames 8 --warmup 1 --repeats 3 --skip-feature-oracle
```

完整 32 场景回放复现需要保留修改前快照，不能用当前优化代码冒充基线：

```bash
mkdir -p outputs/module1_optimization_repeat/baseline_source
cp -a outputs/module1_optimization/baseline_source/worldstream outputs/module1_optimization_repeat/baseline_source/
cp outputs/module1_optimization/baseline_manifest.json outputs/module1_optimization_repeat/
python -m tools.worldstream_m1.optimize_balanced \
  --output outputs/module1_optimization_repeat \
  --selected-config outputs/module1_optimization_repeat/balanced.yaml
python -m tools.worldstream_m1.summarize_optimization --root outputs/module1_optimization_repeat
```

该程序拒绝复用已经存在的 protocol，防止覆盖或悄悄混用部分实验；baseline 加载会核对源码与原配置 SHA256。若要合并完整报告，GPU 对照应使用同一个输出根目录下的 `gpu/`。

全量完整模型可以使用 `projects/configs/OmniDrive/worldstream_m1_balanced_observe.py` 替换原测试配置。若要重新测量 LLM 推理时间，务必指定尚无已有回答的 `model.save_path`；原代码会根据已有回答文件跳过 LLM。全量验证耗时较长，本轮已完成上述成对实验，没有额外重跑全部 6019 帧。

## 后续最值得解决的问题

当前跨场景锚点接受稀少，且部分错误由持续定位偏差造成。更有价值的后续方向是建立可独立验证的定位不确定性与动态运动证据、使用更高频实际输入验证状态持续性，以及为具体 backbone 层/视觉 carrier 建立依赖和身份映射。单纯降低门限、扩大稳定掩码或信任未经校准的速度，都会让候选率看起来更高但缺乏风险依据。
