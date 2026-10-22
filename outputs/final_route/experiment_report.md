# OmniDrive 单一研究主线最终交付

已完成代码收敛和已跟踪历史产物清理。唯一推荐入口为 `projects/configs/OmniDrive/worldstream_m123_auto.py`。完整相关 CPU 回归与配置/接口验证通过；最终配置的完整模型推理 **NOT_RUN**，速度与质量 **NOT_MEASURED**。本次不以旧实验数字冒充最终集成结果。

## 最终路线与准入

M1 使用普通 A 关联，配置 `configs/worldstream_m1_final.yaml` 与当前 M1 默认算法参数一致，仅关闭日志和可视化。保留统一预测、唯一合法代价矩阵、一次全局一对一分配、有限状态/历史、来源校验、物理认证隔离及原子事务。删除 DQTrack B/C 离线适配、Shadow/Hint 调用、专用导出/对比工具和测试，非 off DQTrack 模式显式拒绝。移除研究子类委托钩子，恢复管理器直接调用原 associate；没有回退算法版本。

M2 使用当前 camera Auto / maintain 状态服务。精确图像、相机顺序、场景、执行键、模型/缓存来源、质量标记、实测成本和验证预算仍共同控制复用，不能用 M1 身份代替特征相等。原 camera_costs.json 原字节迁移，实际保留 2 个 quality_certified profile，仅针对特定执行键的所有相机相同请求；其他情况继续原生 EVA。没有强制 Exact、部分窗口或近似策略。

M3 保留 Auto 与原生 generate，成本和质量表原字节迁移，两表 approved 都为空。没有将 exact/compact/combined/batch/state-context 强制启用。审计发现原证据加载器在文件缺失/损坏时直接抛异常，现已改为空证据，继续拒绝优化并回退原生；新增测试覆盖缺失、损坏和非法表结构。模型、LoRA、Tokenizer、生成长度、Detection/Map 配置没有改动。

实际顺序：M1 precheck → M2 利用历史快照和当前精确输入决定复用/完整 EVA → 原检测/地图 → M1 commit → M3 Auto/原生生成。最终 M3 state_context=none，不注入 M1 未认证事实。M2 已取得的旧快照不会被新 commit 偷换。

配置继承：projects/configs/OmniDrive/worldstream_m123_auto.py → projects/configs/OmniDrive/worldstream_m2_camera_auto.py → projects/configs/OmniDrive/worldstream_m1_coverage_observe.py → projects/configs/OmniDrive/worldstream_m1_observe.py → projects/configs/OmniDrive/mask_eva_lane_det_vlm.py → projects/configs/OmniDrive/../../../mmdetection3d/configs/_base_/datasets/nus-3d.py → projects/configs/OmniDrive/../../../mmdetection3d/configs/_base_/default_runtime.py。其中旧基础和兼容配置仅服务继承及既有测试，不再推荐为并行主线。

## 审计与清理

任务开始时仓库为 main，HEAD `cf37f207fd8c238309e7f2e9836b94ce10f2f8c9`，没有已跟踪未提交修改；已有未跟踪目录 `outputs/m3_state_guided_verified_decode/` 完整保护。先建立 initial_workspace、artifact_manifest 和逐文件 cleanup_plan，再删除逐个已跟踪文件；没有 git clean、泛化 rm、提交或推送。

实际删除 **46644** 个已跟踪文件，其中旧 outputs 文件 **46399** 个。其余为不再被运行时/必要测试引用的实验工具、DQTrack 模块、17个配置入口/配置、18份旧文档和4份 DQTrack 专属测试。没有删除任何原数据、部署权重、检查点或第三方依赖。outputs 中 .pt 文件经路径审计属于旧 carriers/logits/前缀输入等实验缓存，实际检查点路径不在删除范围。

早期相对导入闭包检查发现13个辅助模块仍服务物理、质量和缓存回归，已恢复这些依赖；随后移除其旧 main 及无消费者的实验驱动函数。最终相关测试均保留原有效断言。运行时中的条件路径/批解码等辅助模块因仍有接口或正确性测试依赖而保留，不代表默认允许或已经获得质量许可。无法确认可删的配置/模块见 retained_files.json，保留理由为真实运行时、基础继承或回归依赖。

三个必要表位置和 SHA256 见 artifact_manifest.json：M2 camera_costs、M3 cost_table、M3 quality_certificate，内容与旧文件完全一致；配置已同步指向 final_route。历史 A/B/C 只压缩保留 A 的选择背景与原摘要 SHA，不保留多组逐帧表或完整比较产物，不作为最终集成测量。

**保护例外：outputs 并非物理上只剩一个目录。** 按“不得删除任务前用户未跟踪文件”的明确要求，保留前述用户目录及 **2515** 个原 Git 忽略/未跟踪文件（旧日志、缓存和部分结果）。这些文件没有被重写或当作正式交付结果；逐文件见 protected_ignored_files.json。唯一正式结果是 final_route。若要求连这些已有用户文件也删除，将与本次保护约束冲突，本次没有擅自删除。

## 真实验证

- 最终完整测试：**925 passed, 2 skipped, 19 warnings in 112.57s (0:01:52)**。
- 清理辅助驱动后的聚焦复核：285 passed，19 warnings，2.49秒；见 tests_post_cleanup.log。
- 两个 CUDA 单元测试因当前检查进程 CUDA 不可用而跳过，未计作通过；19条为现有库弃用提示。
- 新集成测试覆盖实际配置继承、M1 与普通默认算法一致、保留模型头、M2 原生回退、M3 空/无效证据拒绝及退役外部模式拒绝。
- 导入与编译验证、唯一关联入口、所有必要证据表存在/哈希保持、原 detector/base config/association/StateStore/world_motion 哈希未变、用户保护文件未变均通过；见 verification.json。
- 删除模块导入检查与删除路径引用检查均无最终遗留；README、旧运行说明、setup 与主线文档已同步。保留工具不再提供旧 main 驱动。
- 早期测试收集曾发现相对导入与残余驱动依赖，均已修复；最终不是通过删除安全断言或放宽门限取得绿色结果。

执行命令与完整日志在 tests.log、config_verification.log。关键源文件差异为 manager 退役外部调用、M3 fail-closed 证据读取、集成配置及兼容工具清理。原 EVA/检测/地图/语言模型实现与权重不变。代码级回归通过不能证明完整集成输出或服务性能；没有 GPU 全模型运行、训练或调参。

## 唯一运行命令与重现

已部署环境中使用原真实权重，不在本次任务中运行：

```bash
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python \
 tools/test.py projects/configs/OmniDrive/worldstream_m123_auto.py \
 /absolute/path/to/deployed_checkpoint.pth --launcher none --eval bbox
```

CPU 验证：

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest \
 tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q

taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.verify_final_route
```

## 交付与审阅

完整增删改清单 file_changes.csv；删除前清单 cleanup_plan.json；保留清单 retained_files.json；必要表与来源 artifact_manifest.json；最终配置链 config_manifest.json；代码版本及 SHA source_manifest.json；真实结果边界 summary.json / verification.json。关键文档 docs/worldstream_final_route.md，README 只推荐该入口。

工程主线收敛与 CPU 正确性验证通过。完整模型性能与质量未运行，不能发布新的加速/质量结论；M3 无审批记录仍回退，M2 仅满足原严格 profile 才可能实际复用。所有改动保留供用户审阅，未提交或推送。


## 实际评测运行 run_20261021_233721

[评测报告与当前状态](benchmark/run_20261021_233721/report.md)。真实评测由持久控制器继续执行；完整结果以该目录 status.json 为准。该链接不代表全量评估已完成，原清理审计记录保留。
