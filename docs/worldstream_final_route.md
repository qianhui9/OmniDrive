# OmniDrive 唯一研究主线

唯一推荐集成入口是 `projects/configs/OmniDrive/worldstream_m123_auto.py`。结果与本次清理审计统一放在 `outputs/final_route/`。其他保留配置只作为继承基础或现有回归测试的兼容夹具，不是并列推荐入口。

M1 使用当前单主线的普通 A 关联：有限状态/历史、统一运动预测、合法几何代价矩阵、一次全局一对一分配、实际观测更新及原子提交。最终 YAML 使用当前默认算法参数，只关闭日志/可视化。`dqtrack_mode="off"`；B/C 离线包装、外部身份元数据决策、Shadow、Hint 及其工具已经退役，非 off 模式显式拒绝。普通关联不签发物理认证，源绑定、场景/epoch/时间检查、回滚和物理证据验证均保留。

模型先构造当前帧并进行 M1 precheck，M2 使用该历史快照、当前精确图像输入和独立计算来源决定 EVA 复用或完整执行。Detection/Map 继续执行，实际输出再交给 M1 commit；语言生成通过 M3 Auto 或原生 generate 执行。最终 M3 `state_context=none`，不会把 M1 身份或未认证事实注入语言上下文。提交后的新状态不能偷偷替换已交给 M2 的历史快照。

2026-10-22 起，按用户要求，正式 M2 改为 **window Auto**，继承 `worldstream_m2_window_auto.py`。实际图像像素差异与 M1 失效区域经 patch 感受野和窗口依赖闭包传播，只执行需要更新的窗口，将结果与未变化窗口缓存拼回。当前 EVA 的独立前缀为 block 0、1；block 2 的全局注意力是边界，其后所有 block 仍原生执行。patch embedding 仍完整执行。窗口来源、原子发布、预算、修改检测、场景/epoch/模型/布局验证均保留。

窗口 Auto 使用独立 `outputs/final_route/window_costs.json`，初始批准计数为空；相机成本证据不转移到窗口路径，未校准或未通过下游质量验证时回退。当前真实 GPU 实验中，完全相同输入的窗口复用保持位级一致；局部 packed 重算存在超过严格容差的 FP32/FlashAttention 差异，未授予生产批准。Exact 仅作为显式实验模式，不代表获得数值/任务质量认证。复现命令及结果见 `docs/worldstream_m2_window.md`。

历史记录：此前正式 M2 为 camera Auto。原 camera 成本表迁移与 SHA 保留记录仍有效，`camera_costs.json` 原字节保留，camera 配置继续用于兼容回归；本次没有改写该表及既有清理审计。它不再是当前推荐入口的 M2 执行边界。

M3 使用原 Auto 默认生成语义，原成本表与质量表分别按字节迁移至 `m3_cost_table.json`、`m3_quality_certificate.json`。两表 approved 均为空，因此不授予新增优化，执行原生生成回退。缺失、损坏或结构非法的证据表按空证据处理；请求/模型/生成模式/执行键不匹配也不能获得许可。未修改原模型、LoRA、Tokenizer、生成长度、检测或地图配置。离线策略实现中仍被运行时和正确性测试依赖的辅助模块保留，不代表获得在线许可。

运行前需要现有部署权重、原数据、现有环境及可用 GPU。窗口实现已完成真实权重的 2 帧集成短测（另有 1 帧预热），M1 提交、窗口 Attention 计数及原生语言生成通过检查，Auto 未发生复用。这只是执行路径验证，不是全量精度或最终组合加速结果。记录见 `outputs/m2_window_incremental_20261022_run01/experiment_report.md`。以下全量评估命令本次未执行：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
 tools/test.py projects/configs/OmniDrive/worldstream_m123_auto.py \
 /absolute/path/to/deployed_checkpoint.pth --launcher none --eval bbox
```

配置、接口和原生回退验证可在 CPU 执行：

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest \
 tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
 -m tools.verify_final_route \
 --output-dir outputs/m2_window_incremental_20261022_run01/route_verification \
 --tests-log outputs/m2_window_incremental_20261022_run01/tests.log
```

`outputs/final_route/verification.json` 保留清理当时的审计，其中 NOT_RUN 不覆盖后续短测；窗口阶段的验证单独保存在上述新目录。历史 A/B/C 仅支持不采用 DQTrack 的选择，不作为当前集成路线的性能值。先前可评分开发数据中 B/C 没有改善正确延续、错误延续或身份切换，C 没有实际满足完整映射条件的消歧边。

删除前已经生成逐文件 cleanup_plan；保留完整物理/事务/M1→M2/缓存/质量门控/原生回退测试，仅移除专用于退役 DQTrack 的测试。无法确定可删、仍有测试依赖的工具保留为兼容辅助函数，并移除其旧 main 驱动。历史结果的已跟踪文件已清理，必要证据原字节保留。任务开始前已有的未跟踪目录 `outputs/m3_state_guided_verified_decode/` 和 Git 忽略的旧文件按用户保护要求保留，记录在审计清单；这些不作为正式结果或推荐入口。没有执行 git clean、删除数据/权重、提交或推送。
