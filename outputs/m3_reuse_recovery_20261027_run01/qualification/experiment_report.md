# 收紧 Map Approx 规则后的新独立场景验收

上一阶段出现自动 F1/关键词代理遗漏的新增道路规则断言，因此仅增加失败证据并禁用 Map Relations Approx；没有重新拟合阈值、成本或开放新规则。该失败场景不再视为未见验证数据。本阶段冻结策略后另选4个未观察场景，每场景6帧、3个真实标注问题，按原生标注顺序执行所选问题。

新场景：`{'a499ee875da34e2b9655afb999edb8a9': 'fresh_independent_validation', 'ddb615d9bb22484cabc6545b632a1025': 'fresh_independent_validation', '848ac962547c4508b8f3b0fcc8d53270': 'fresh_independent_validation', 'f24b4682fbb9482aba149534deed1cc9': 'fresh_independent_validation'}`。自然复用分解：`{'groups': {'same_frame_exact': {'requests': 7, 'tokens': 3908}}, 'input_tokens': 41841, 'effective_prefill_tokens': 37933, 'skipped_tokens': 3908, 'skipped_fraction': 0.09340120934012093, 'profiles': {'full_visual': 38, 'object_attributes': 17, 'map_relations': 15, 'object_ego_motion': 2}}`。自然答案相同 `72/72`；平均 F1 下降 `0.0`；新增自动关键代理错误 `0`；新增答案差异 `0`。不再使用失败场景的 F1 上升宣称语义通过。

| 条件（三轮） | Prefill CUDA mean/P50/P95 ms/帧 | 完整 M3 Wall | 模型 Wall | 请求 Wall | 生成 Tokens | 实际 Decode Forward |
| --- | --- | --- | --- | --- | --- | --- |
| native | 552.260 / 544.140 / 587.785 | 3174.656 / 3015.815 / 4942.698 | 3704.052 / 3542.462 / 5472.423 | 3937.427 / 3827.744 / 5658.150 | 7566 | 7350 |
| zero | 547.765 / 539.560 / 582.769 | 3188.543 / 3029.032 / 4951.130 | 3874.464 / 3677.268 / 5661.969 | 4057.380 / 3847.816 / 5827.836 | 7566 | 7350 |
| reuse | 504.036 / 537.942 / 581.735 | 3145.536 / 2954.889 / 4829.727 | 3831.283 / 3644.495 / 5536.893 | 4068.701 / 3888.142 / 5801.891 | 7566 | 7350 |

| 对照/阶段 | 加速比 | 平均节省 ms | 场景 CI95 ms | 所有场景/轮次均正收益 |
| --- | --- | --- | --- | --- |
| reuse_vs_zero/prefill_ms | 1.086758 | 43.729139 | [12.353, 67.899] | False |
| reuse_vs_zero/m3_ms | 1.013673 | 43.007556 | [11.407, 66.902] | False |
| reuse_vs_zero/model_ms | 1.011271 | 43.180700 | [22.359, 60.795] | True |
| reuse_vs_zero/request_ms | 0.997218 | -11.321042 | [-43.545, 14.425] | False |
| reuse_vs_zero/preprocessing_ms | 0.766177 | -54.424977 | [-64.325, -46.360] | False |
| reuse_vs_zero/non_model_ms | 0.770439 | -54.501741 | [-64.254, -46.481] | False |
| reuse_vs_native/prefill_ms | 1.095676 | 48.224232 | [17.058, 72.187] | True |
| reuse_vs_native/m3_ms | 1.009258 | 29.120517 | [-3.146, 53.927] | False |
| reuse_vs_native/model_ms | 0.966792 | -127.230620 | [-222.415, 4.895] | False |
| reuse_vs_native/request_ms | 0.967736 | -131.273291 | [-231.000, 9.984] | False |
| reuse_vs_native/preprocessing_ms | 0.982552 | -4.061107 | [-10.674, 4.359] | False |
| reuse_vs_native/non_model_ms | 0.982972 | -4.042671 | [-10.547, 4.419] | False |

场景 Bootstrap 为4个独立场景，三轮重复先在场景内平均，不当作12个独立场景；10,000次、seed=20261027。请求改善不能在 M3/模型区间跨零时全部归因于 KV，预处理/模型之外时间单独保留。

| 系统指标 | Native | 新策略 M123 | 绝对下降 | 相对下降 |
| --- | --- | --- | --- | --- |
| mAP | 0.215788 | 0.214526 | 0.001263 | 0.005851 |
| NDS | 0.264042 | 0.263404 | 0.000638 | 0.002415 |
| DET_l | 0.167848 | 0.169862 | -0.002014 | -0.011997 |
| Token_F1 | 0.456529 | 0.454117 | 0.002413 | 0.005285 |

| 场景 | mAP loss | NDS loss | Lane DET_l loss | Token F1 loss |
| --- | --- | --- | --- | --- |
| 848ac962547c4508b8f3b0fcc8d53270 | -0.001222 | -0.000550 | 0.000391 | 0.000000 |
| a499ee875da34e2b9655afb999edb8a9 | 0.030035 | 0.019898 | -0.000418 | 0.008298 |
| ddb615d9bb22484cabc6545b632a1025 | 0.000033 | 0.000068 | -0.000731 | -0.000137 |
| f24b4682fbb9482aba149534deed1cc9 | 0.006944 | 0.003544 | 0.003654 | 0.001489 |

四项 .10 预算通过：`True`。M3 内部 .02 不变。同初始实际 KV/同输入/相同强制 Token 的真实 Forward 与 logits 控制：`{'requests': 72, 'all_actual_fixed_histories_equal': True, 'initial_cache_storage_immutable': True, 'planner_equivalence_all_checks_pass': True, 'headline_performance': False, 'reused_requests': 7}`；详情在 same_input_controls.jsonl。强制答案不计质量。

正式安装状态：`True`。仅 Map Approx 的既有 critical_regressions 因新增图像审核证据上调；旧 Full Visual 错误记录和 Object Exact 质量拒绝保留，其他规则参数逐项相同。仍只有一个 ReusePlanner/PrefillExecutor/PrefixKVCache，M1/M2默认不变。

历史原策略5607 Tokens和本轮原策略1606 Tokens属于冻结候选复现，不应当作最终收紧策略的收益。最终收紧策略的实际复用与耗时以本文件为准。历史输出完整保留。
