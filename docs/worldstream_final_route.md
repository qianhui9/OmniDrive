# OmniDrive 唯一研究主线

唯一推荐集成入口是 `projects/configs/OmniDrive/worldstream_m123_auto.py`。结果与本次清理审计统一放在 `outputs/final_route/`。其他保留配置只作为继承基础或现有回归测试的兼容夹具，不是并列推荐入口。

M1 使用当前单主线的普通 A 关联：有限状态/历史、统一运动预测、合法几何代价矩阵、一次全局一对一分配、实际观测更新及原子提交。最终 YAML 使用当前默认算法参数，只关闭日志/可视化。`dqtrack_mode="off"`；B/C 离线包装、外部身份元数据决策、Shadow、Hint 及其工具已经退役，非 off 模式显式拒绝。普通关联不签发物理认证，源绑定、场景/epoch/时间检查、回滚和物理证据验证均保留。

模型先构造当前帧并进行 M1 precheck，M2 使用该历史快照、当前精确图像输入和独立计算来源决定 EVA 复用或完整执行。Detection/Map 继续执行，实际输出再交给 M1 commit；语言生成通过 M3 Auto 或原生 generate 执行。最终 M3 `state_context=none`，不会把 M1 身份或未认证事实注入语言上下文。提交后的新状态不能偷偷替换已交给 M2 的历史快照。

M2 保留 camera Auto 的精确输入、模型版本、相机顺序、场景、缓存来源、成本与下游质量准入。原 camera 成本表按字节迁移至 `outputs/final_route/camera_costs.json`，SHA 不变。目前表中通过质量标记的是特定执行键下所有相机精确相同的请求；部分相机组合的 quality_certified 为 false。即使输入相同，执行键、成本或验证预算不满足条件仍执行原生 EVA；不能宣称任意跨帧图像可复用。window 依赖闭包与必要验证模块因运行时及回归测试依赖而保留，没有将 Exact 或条件近似设为默认。

M3 使用原 Auto 默认生成语义，原成本表与质量表分别按字节迁移至 `m3_cost_table.json`、`m3_quality_certificate.json`。两表 approved 均为空，因此不授予新增优化，执行原生生成回退。缺失、损坏或结构非法的证据表按空证据处理；请求/模型/生成模式/执行键不匹配也不能获得许可。未修改原模型、LoRA、Tokenizer、生成长度、检测或地图配置。离线策略实现中仍被运行时和正确性测试依赖的辅助模块保留，不代表获得在线许可。

运行前需要现有部署权重、原数据、现有环境及可用 GPU。本次没有执行以下昂贵推理命令，也没有测量最终组合的速度或质量：

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
 -m tools.verify_final_route
```

以 `outputs/final_route/verification.json` 中的实际日志为准。完整模型最终集成结果标为 NOT_RUN；历史 A/B/C 仅支持不采用 DQTrack 的选择，不作为当前集成路线的性能值。先前可评分开发数据中 B/C 没有改善正确延续、错误延续或身份切换，C 没有实际满足完整映射条件的消歧边。

删除前已经生成逐文件 cleanup_plan；保留完整物理/事务/M1→M2/缓存/质量门控/原生回退测试，仅移除专用于退役 DQTrack 的测试。无法确定可删、仍有测试依赖的工具保留为兼容辅助函数，并移除其旧 main 驱动。历史结果的已跟踪文件已清理，必要证据原字节保留。任务开始前已有的未跟踪目录 `outputs/m3_state_guided_verified_decode/` 和 Git 忽略的旧文件按用户保护要求保留，记录在审计清单；这些不作为正式结果或推荐入口。没有执行 git clean、删除数据/权重、提交或推送。
