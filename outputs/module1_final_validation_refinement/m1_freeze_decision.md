# M1 最终冻结判断

- engineering_interface_freeze: **Go**
- candidate_quality_freeze: **No-Go**
- conditional_reuse_authorization: **No-Go**
- formal_physical_certification: **No-Go**
- handoff_to_M2: **Go**

工程接口可冻结的条件是当前源码的 CUDA、完整原模型、来源和持久状态回归通过；它不授予候选跳算或物理认证。候选仍有 14 个旧失败事件，另有 0 个新失败事件。新出现或微小目标缺少历史 Anchor 时，不能由局部自洽光流推导静止或输入相同。

保持严格刷新、Exact 输入及依赖闭包、现有成本/任务门。M2 可继续研究真实计算依赖与条件复用；不得把普通 Observation Link 当作物理真值或安全跳算证明。停止继续叠加 M1 运动模型或认证生产器。

推荐工程配置 configs/worldstream_m1_engineering_frozen.yaml 仅启用实测获益的 measurement_reuse；motion_risk_revocation=False。projects/configs/OmniDrive/worldstream_m2_final_validation.py 指向该配置。启用风险的完整方法只保留为独立实验对照，不列为质量改善或生产加速。
