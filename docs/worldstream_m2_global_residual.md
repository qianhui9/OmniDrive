# WSGRR-AR 全局锚帧残差

正式路径为 `M1 State → WSGRR → DependencyPlanner → WindowExecutor → Global-State Update → StateCache Commit → Detection/Map → M3`。运行时只有一个 `WindowExecutor` 和一个 `update_global`；锚帧与普通帧属于同一模块的时间更新。模型权重、M1 Precheck/Commit、M3 推理算法不变。

`ordinary_budget=3`、`max_reuse_frames=8` 和原有区域必须事件保持原值。每帧都真实执行 Patch Embedding、来源验证、窗口选择、依赖闭包和前缀计算。没有新增 Attention KV 拼接、在线成本预测、质量认证文件或备用 EVA Runtime。

## 更新规则

`global_residual.update_global()` 接收当前 BHWC `H2`。首帧、新场景、M1 整帧刷新、锚帧到期或 H2 漂移超过固定阈值时，计算后续完整 Suffix，保存 `R = Hfinal - H2`。其余帧输出 `H2_current + R_anchor`，显式标记 `approx_global_residual_reuse`。这不是全局 Attention 的精确复用，也不会保证旧 Token 的输出不变。

年龄按当前序列的调用帧数计数。间隔 2 表示第 0、2、4 帧更新锚帧，中间最多复用一帧；间隔 3/4 类推。漂移为每相机 `sqrt(mean((H2-H2_anchor)^2) / max(mean(H2_anchor^2),1e-12))` 的最大值，使用 float32 归约。判定在执行 Suffix 之前完成，只读取一次漂移标量。阈值属于离线校准参数。

后续 Suffix 必须保持 H2 的 shape/stride/dtype/device，否则明确报错。复用时通过 `empty_strided` 与 `torch.add(...,out=...)` 保持原生布局。相机顺序、模型/输入签名、M1 epoch 和发布版本均在复用前检查。

## 缓存事务

`StateCache.global_anchor` 只保存一个锚帧，包括 scene/frame/timestamp/epoch/signature/camera 顺序、前缀边界、shape/stride、H2、Residual、年龄、生成版本和发布版本。H2 实际用于漂移检测，因此需要保留；不保存逐层 KV 或 22 层 Hidden State。

锚帧 Tensor 与窗口 Tensor 在同一次 `StateCache.commit()` 发布。公开 commit 拷贝外部 Tensor；内部更新可转移私有 H2/Residual，复用帧继续借用已发布的只读锚帧 Tensor。输入、packed outputs 和最终返回特征都不能与锚帧非法共享 storage。Tensor 版本/指针/布局与锚帧元数据修改均会被检测。

提交容量包含 H2 和 Residual。所有分配、有限值检查、来源和完整性记录在发布前完成。Runtime 为请求创建浅层事务视图；失败时恢复上一缓存对象，包含新场景初始化失败的情况。正常新场景在该事务视图清空旧锚帧并初始化，成功后整体生效。这里的 `finally` 是事务回滚，不执行任何替代推理算法。

## M1 Snapshot

`export_m2_inputs(...,computation_only=True)` 保留原有 scene/frame/timestamp/epoch/signature/pending/source digest 检查。通过 `readonly_snapshot.freeze()` 一次分离可变容器，分享不可变标量和压缩 mask 字符串。只读字典/列表保留原有消费接口与 JSON 可序列化性，禁止普通修改操作；其 `deepcopy` 返回自身，因此 M2 到 Cache Commit 不再复制同一只读区域树。

Snapshot 的 wrapper 和 transaction diagnostics 仍私有复制。M1 Commit 修改原始 bundle 的事务标志不会影响已导出的快照；稠密 RGB correspondence 数组从不进入该接口。完整审计导出仍保持独立副本。正式接口没有 Snapshot 缓存池或简化状态模式。

## 测量与结果

放宽预算后的主实验保存在 `outputs/m2_global_residual_20261026_run03/`；`run02` 保存较激进阈值的失败开发结果，原 `run01` 保留当时预算与结论。新目录下 `previous_budget_reassessment/` 只重评旧预测，不把修改预算当成模型精度或推理速度改善。新参数的校准、独立验证和三轮完整服务使用新的真实推理输出，详见主实验目录的 `experiment_report.md` 和 `refinement_history.md`。

离线允许 Detection mAP、NDS、Lane DET_l、VLM Token F1 各项绝对下降 **≤0.05**，即 `native_score - candidate_score <= 0.05`，不是相对下降 5%。四项分别判断，不允许某项提升抵消另一项超标；数字、空间、运动、安全词法代理退化仍单列。默认值在 `tools/worldstream_m2/quality_budget.py`，实验 CLI 的 `--quality-budget` 默认 0.05，每个输出目录冻结 `quality_budget.json`；同一目录不能切换标准。旧标准可用 `--quality-budget 0.005 --vlm-quality-budget 0.02` 显式回放；复现原阈值时还需传入 `--drift-quantile 0.25`。旧输出中的脚本记录的是当时默认值，跨版本回放需显式传参。

先固定开发场景 age-1 H2 漂移的 75% 分位数比较间隔 2/3/4，三项候选均因精度失败；再按已记录的单次保守校准计划改用 50% 分位数（0.912088305），重测原生和三个间隔。在满足四项预算的候选中按开发集完整请求均值选择，选定间隔 4。单轮均值不能证明统计显著最优。独立场景按预先声明的规则选择，排除开发与旧验证场景；选定参数后只验证一次。任何候选超标都会被保留在曲线中；全部超标时不生成可用的 `selection.json`。三个间隔只用于离线校准，正式运行仍为同一条 M2 路径。

预算不加入在线 Runtime，也不修改 M3 内部 `quality_budget=0.02` 的 KV 策略。最终结果分别记录独立精度预算是否通过、完整请求是否平均加速、两者是否同时满足；精度通过不能代替端到端加速证据。三轮计时来自相同 40 帧的重复测量，其置信区间不作为跨场景保证。

本轮实验间隔 4 在新独立 4 场景上的 mAP 下降 0.070974，超过 0.05；NDS/DET_l 下降为 0.049255/0.014071，Token F1 提高 0.019196。该实验参数不替换正式默认值。`worldstream_m123_auto.py` 和 `M2Config` 均保留间隔 2、阈值 0.772102788；原独立子集的回顾性重评 mAP/NDS 下降为 0.016355/0.014083，按新预算通过。新实验的失败没有通过再次调整独立集来消除。

`global_suffix_skipped_blocks` 和 `global_suffix_reuse_frames` 在完成对应计算后生成。完整 worker 还通过 Suffix Block 的只读 forward hooks 记录实际调用次数，并逐帧核对计划诊断。Profiler 仅位于实验工具中，正式模块不创建 profiler 或 CUDA Event。

全局更新 CUDA Span 包含漂移判定、锚帧/残差更新与主机提交间隙；Kernel Active 从 trace 的 kernel 事件单次计数。Host enqueue、同步 Wall、CUDA Span、Kernel Active 不混用。端到端结果必须来自完整请求，不能直接用跳过 Block 的比例换算。
