# 实现与插桩变更

新增 TaskEvidenceContract、严格拒绝门、默认影子模式及显式实验直接执行分支；直接拒绝后可复用现有 Exact Prefix。未改变原始网络、权重、解码算法、M1 配置、M2 准入/执行算法。旧 post_perception.direct_query 入口统一调用同一严格门，不能只靠一个布尔值批准。

| 文件 | 类型 | 用途 | 行数 |
|---|---|---|---|
| worldstream/m2/conditional_prefix.py | 修改 | 运行时接口/测试 | 1–198 |
| worldstream/m2/runtime.py | 修改 | 运行时接口/测试 | 1–361 |
| worldstream/m3/config.py | 修改 | 运行时接口/测试 | 1–56 |
| worldstream/m3/runtime.py | 修改 | 运行时接口/测试 | 1–308 |
| worldstream/m3/post_perception.py | 修改 | 运行时接口/测试 | 1–73 |
| worldstream/m3/task_evidence.py | 新增 | 运行时接口/测试 | 1–136 |
| worldstream/m3/__init__.py | 修改 | 运行时接口/测试 | 1–8 |
| tests/worldstream_m2/test_source_aligned_conditional.py | 修改 | 运行时接口/测试 | 1–161 |
| tests/worldstream_m3/test_task_evidence_serving.py | 新增 | 运行时接口/测试 | 1–159 |
| projects/configs/OmniDrive/worldstream_m3_task_evidence_research.py | 新增 | 运行时接口/测试 | 1–5 |
| tools/worldstream_m3/task_serving_bootstrap.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–22 |
| tools/worldstream_m3/task_serving_supplement.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–62 |
| tools/worldstream_m3/task_serving_final_validation.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–28 |
| tools/worldstream_m3/task_serving_profile.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–198 |
| tools/worldstream_m3/task_serving_finalize.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–221 |
| tools/worldstream_m3/task_serving_checks.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–79 |
| tools/worldstream_m3/task_serving_recovery.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–40 |
| tools/worldstream_m3/task_serving_identity_probe.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–58 |
| tools/worldstream_m3/task_serving_confirm.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–31 |
| tools/worldstream_m3/task_serving_batch.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–26 |
| tools/worldstream_m3/m2_freeze_oracle.py | 新增 | 独立实验、计时或报告脚本；无模型内联插桩 | 1–107 |

计时位于独立 runner；各 prefill/decode 调用分别同步计时，所有臂均包含 observer 成本。模型服务结束恢复原方法。阶段嵌套时间不可相加。GPU trace 属诊断实验，不作为完整请求性能证明。
