# M2 世界状态引导的区域刷新（WSGRR）

WSGRR 是唯一生产候选策略。正式配置保持 source-linked window Auto、维护 M1、自适应缓存和原成本批准表；没有批准时完整更新。`evaluate_offline()` / `evaluate_offline_from_m1()` 是明确的离线入口，使用相同 DependencyPlanner、StateCache 和 WindowExecutor。普通 M2Config 不提供绕过准入的开关。离线与生产缓存的执行签名不同。

## 实际 M1 数据与时间边界

`precheck → computation_state.build_bundle → export_m2_inputs → runtime` 传递 `ComputationStateBundle.region_states` 和 `region_events`。区域来自已提交 WorldAnchor 在当前相机的真实投影，使用当前预处理 intrinsic 和 camera-time pose。只导出通过已有投影有效性检查的区域，不把图像差分转换成对象。

字段包括 camera_id、anchor_id 对应的 track_id、bbox_xyxy、semantic_class 对应的 class_id、(anchor_type, semantic_class) 状态摘要、source_state_version、source_score、观测支持/遮挡假设、世界中心、仅在 velocity_reliable 时导出的速度、最后实际观测时间、出生时间、关联依据与身份可靠性。visible=True 只表示 M1 RGB 观测支持；occluded=True 是既有深度重叠遮挡假设，其余使用 None。几何版本是来源信息，不能把其每次变化当成语义突变。

当前帧 Detection/Map 结果在视觉计算后 commit 才产生，因此区域状态是历史检测在当前相机的预测投影。M1 几何关联 ID 没有独立物理身份认证；检测速度目前也没有可靠运动认证，通常不会导出。当前没有信号灯状态、确认消失、可靠语义突变生产者。本实现不声称覆盖这些语义。状态寿命到期来自真实 expired_anchor_ids，导出 state_expired；单帧缺席不产生 disappeared。首次进入某相机的已知历史投影标记 region_entered；出生时间支持时才标记 appeared。

## 选择与依赖

每路 640×640 经真实 16×16 Conv2d 产生 40×40 Token，16×16 Token 窗口补齐到 48×48；六相机每 Block 54 个窗口。只有经过结构检查的 Block 0、1 独立窗口前缀缓存，Block 2 起全部执行。不同窗口大小也通过真实拓扑传播闭包。

每帧只执行一次 block0_input。区域到 Token 的映射逐轴检查真实 kernel、stride、padding、dilation 的采样位置是否与图像区域相交。事件影响区域为旧、新 bbox 各自膨胀 margin 后的并集。固定图像位置缓存不支持对象输出搬移。

必须刷新来源：显式 M1 事件、状态摘要变化、已知身份关联资格失效、有效观测在遮挡假设后重现、明显 bbox 位移或中心跨窗口、局部计算/观测租约失效、年龄达到上限、当前非有限 Token。bbox 各坐标变化最大值达到 16 像素视为明显位移；中心根据真实 Conv2d 几何定位窗口。普通单像素 bbox 边缘变化不会直接成为必须刷新事件。全局上下文、模型/策略签名、缓存来源/版本失效以及 M1 full_refresh_hint 仍完整更新。

普通评分：

```
V = clamp(1 - cosine(mean_valid_tokens(x0_window), cached_descriptor), 0, 2)
A = min((frames_since_actual_refresh + 1) / max_reuse_frames, 1)
U = mean_valid_tokens(mean_receptive_field(dirty_hint OR unknown))
S = drift_weight * V + age_weight * A + uncertainty_weight * U
```

窗口 padding 不参与描述平均。描述只在对应 Block 0 窗口实际重算时更新，反映相对计算来源的累积漂移；Block 0 输入本身保存每次成功请求的当前值，用于离线精确对照。每个 Block 单独记录实际刷新年龄。相等零向量漂移为 0，零/非零向量漂移为 1。

默认参数：ordinary_budget=3 / 相机，max_reuse_frames=8，region_margin_px=8，movement_threshold_px=16，权重 (1, 0.25, 0.25)，score_threshold=0.05。普通候选满足评分阈值后按分数降序、年龄降序、camera-major 窗口索引升序选取。必须集合不占普通预算，可以超过预算。这些参数只是未校准研究起点，真实来源租约可早于八帧失效，不会为提高复用而放宽。

选定窗口经 plan_windows 转为 Token，再走同一 plan 依赖闭包。成本拒绝导致某 Block 扩为完整执行时，后续前缀也相应扩展。保持一个 WindowExecutor，QKV、投影、MLP 的原生全形状与精度不变，只减少独立窗口 Attention；全局依赖后缀全部执行。输出窗口复用率不能当作总计算减少率。

## 原子缓存和成本准入

StateCache 同时提交图像、当前 Block 0 输入、各层窗口输出及逐窗口实际计算来源、区域快照、各层窗口年龄、Block 0 描述、上下文/时间戳/签名。分配、复制、预算和有限性检查全部成功后一次发布。失败不发布混合版本；运行时已知资源失败清空缓存，编程错误传播。区域元数据和描述/年龄张量也纳入修改检测。普通列表缺席保留历史区域，直到 M1 明确过期/消失事件或缓存重置。

缓存签名和正式成本键绑定全部关键策略参数、模型和数值上下文；旧精确路线成本不能授权 WSGRR。批准需要对应模型/策略/规模的成本、独立 Detection/Map/VLM 质量报告及可验证证据哈希。离线调用不更新批准表，cache_capture=always 本身不绕过生产成本门槛。

## 离线复现

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.window_validate \
  --wsgrr --frames 40 --repeats 3 --warmup 1 --budgets 2 3 4 6 9 \
  --m1-predictions /path/to/actual_full_model/results_nusc.json \
  --output outputs/wsgrr_new
```

真实连续关键帧按 scene/time 排序；可用 --scene 指定场景。真实全模型预测只在每帧 M1 commit 时回放，不能以标注对象 ID 代替跟踪；标注类别只用于离线覆盖率描述。各实验独立维护 M1 与 M2 提交链。

- A：旧精确 Block 0 比较与 dirty/source 约束，离线基线。
- B：区域漂移、年龄和局部优先级，不使用对象事件、无普通数量预算。
- C：加入 M1 真实区域关联和事件，仍无普通数量预算。
- D：完整 WSGRR，测试各预算。

frames.jsonl 保存各层执行/复用窗口、必须/普通集合、事件区域及漏刷数、漂移/年龄/评分、精确 Token 变化率和实际选中 Token 比例、原生特征误差、墙钟/CUDA 流跨度样本、M1 precheck/read 与评分/执行/发布的 host 时间。视觉管线含 embedding、校验、选择、前缀、完整后缀、发布和诊断；M1 precheck/read 单列。CUDA event span 包含 host 提交间隙，不等于活跃 kernel 时间；host 分段未经独立同步，不是 GPU kernel 时间。

--profile-block-costs 额外测量对应实际输入的纯 Block 成本，生成 window_costs_provisional.json；始终 production_ready=false / approved_counts=[]。近似候选记录定量误差，不要求位级相同。独立清单 m2_window_task_quality_v1 的每条记录除 tasks、covered_cases 等既有字段外，还必须给出独立评估的 feature_error_limits={max_abs,mean_abs}。promote_costs 同时要求所有覆盖案例块级与净视觉管线 wall/GPU 收益，不会自动选择容差或批准。晋升工具只匹配当前正式配置的完整策略键，研究预算不能偷偷覆盖默认预算。

该 profiling 开关同时在前序 CUDA 完成后重新测量同一缓存负载的 StateCache.commit，记录 synchronized_cache_publication。运行时未同步的 publication host 时间会包含等待前序算子完成的开销，不能视为纯复制成本。

没有 --wsgrr 时保留旧精确校准工具，用于原生对照，不能作为 WSGRR 正式批准证据。

实际下游任务 smoke 复用全模型 benchmark worker：

```bash
taskset -c 0-3,6-31 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.benchmark.worker \
  --output outputs/wsgrr_tasks_new --run-root outputs/wsgrr_task_run \
  --condition B2 --workload language --limit 8 --warmup 1 \
  --offline-wsgrr D --window-budget 3
```

B2 保留真实 M1 和原生语言生成，避免混入 M3。使用同输入 B0 作对照。短测预测差异/问答指标不能替代官方完整验证集或独立质量评估。完整集评估遵循原 benchmark 标注完整性检查和真实 Detection/Map evaluator。

两个短测完成后，可用 window_validate 的 `--compare-task-runs BASELINE目录 CANDIDATE目录 --output 比较.json` 对齐 sample token，输出实际 Detection/Map 预测距离和回答差异；这些一致性指标不会标记为官方质量认证。

本次实测见 outputs/wsgrrr_20261022/experiment_report.md，历史实验目录不覆盖。
