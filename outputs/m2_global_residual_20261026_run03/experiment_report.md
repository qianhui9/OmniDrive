# OmniDrive WSGRR-AR：放宽精度预算后的实验

预算单位为原生分数减去正式分数的**绝对下降**，逐项独立判断，边界包含等号；负值表示改善。允许下降 0.05 相当于 5 个百分点，并非相对下降 5%。预算仅用于离线选择与验收，不参与在线 M2 判定。

四项预算：`{"mAP": 0.05, "NDS": 0.05, "DET_l": 0.05, "Token_F1": 0.05}`。旧结果未被改写；[原配置预算重评](previous_budget_reassessment/reassessment.md) 与本轮新参数实测分别记录。

本轮实测固定参数：间隔 4，最大相机 H2 相对 RMS 漂移阈值 0.912088305；前缀预算 3、最大复用年龄 8、checkpoint iter_10548.pth 不变。M1/M3 算法不变；M3 内部 quality_budget=0.02 是原有 KV 策略参数，不作为本次四项任务验收预算。

## 开发集校准

40 帧自然连续开发场景；漂移阈值取既有开发数据 age-1 最大相机 H2 漂移的 50% 分位数。漂移来源及 SHA256 见 calibration_parameters.json。只比较间隔 2/3/4；不使用独立场景调参。完整闭环真实 Detection/Map/语言输出由官方子集评分器评估。

| 间隔 | 后缀复用帧 | 请求 mean ms | mAP 下降 | NDS 下降 | DET_l 下降 | Token F1 下降 | 预算 |
|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 10 | 3000.17 | 0.028755 | 0.046661 | 0.007445 | -0.012356 | 通过 |
| 3 | 12 | 3128.56 | 0.040154 | 0.054144 | 0.010737 | -0.001142 | 未通过 |
| 4 | 14 | 2985.34 | 0.042929 | 0.047351 | 0.026407 | -0.006404 | 通过 |

Lowest measured complete service mean among candidates within all four frozen absolute quality budgets; numeric/spatial/motion/safety regressions are reported separately and are not certified by aggregate budgets

仅按单轮开发完整请求均值选择，未宣称统计显著最优；独立验证和三轮性能测量均不回调参数。逐问数字/空间/运动/安全词法代理退化保留在 *_quality_pairs.json，不被平均分数预算覆盖。

## 同快照视觉与 Snapshot

同一 M1 快照、当前帧和窗口选择，旧 WSGRR / 新 WSGRR-AR / 原生 EVA 同步 Wall 均值为 496.13 / 343.20 / 489.22 ms。旧→新节省 152.93 ms；相对原生纯视觉 1.4255×。此边界排除 Snapshot、M1 Precheck/Commit 和任务头。

对应 CUDA Span 均值 496.11/343.18/489.20 ms。Span 包含主机提交空隙；代表帧 kernel active 旧/新 474.557/40.697 ms。两种边界不能混用，也不能把代表帧推为所有帧。

前缀计划、packed outputs、年龄、版本、来源全部相同：True。最终 embedding 是近似输出，非精确等价。

同一 pending 请求交替导出 Snapshot：旧/新 mean 24.950/26.610 ms，P50 17.660/16.606，P95 30.176/28.659，max 320.517/334.823。所有正常长尾样本保留，不从 deepcopy 次数直接推断加速。

完整导出计算契约逐帧一致，来源检查未删。cProfile、实际缓存发布、kernel trace 和样本见 paired40/；无 GC 采样，不推断长尾来源。

## 三轮完整服务

顺序为原生→正式、正式→原生、原生→正式。同 40 帧/问题/seed/生成配置，预热后重置状态。120 次测量是 40 个唯一驾驶帧的重复，非 120 个独立样本。官方任务评分取第 0 轮真实 predictions.pkl。

| 轮 | 原生 mean/P50/P95 ms | 正式 mean/P50/P95 ms | 加速比 |
|---|---|---|---:|
| 0 | 2993.29/1882.22/9257.05 | 2977.93/2009.36/8431.49 | 1.0052× |
| 1 | 2866.11/1766.82/9120.26 | 3017.29/2069.20/8444.79 | 0.9499× |
| 2 | 2980.54/1888.70/9194.31 | 2978.06/2038.60/8439.69 | 1.0008× |

| 三轮合计 | 原生 | 正式 |
|---|---:|---:|
| 完整请求 mean ms | 2946.65 | 2991.09 |
| 视觉 wrapper CUDA Span mean ms | 480.03 | 365.73 |
| Prefill CUDA mean ms | 189.19 | 184.81 |
| Decode CUDA mean ms | 1887.53 | 1782.84 |
| 吞吐 frame/s | 0.3394 | 0.3343 |
| 峰值 allocated GiB | 15.3531 | 16.0842 |
| 峰值 reserved GiB | 15.7422 | 16.7598 |
| 生成 token | 9066.0000 | 8580.0000 |
| M3 实际复用 token | 0.0000 | 0.0000 |

完整服务加速比 0.9851×；平均时延降低：False。三轮配对节省时间 95% t 区间 [-274.61, 185.71] ms，正数表示节省；仅同场景三轮，不是跨场景保证。

重复输出一致性：`{"native": {"task_digests_equal": true, "answers_equal": true}, "formal": {"task_digests_equal": true, "answers_equal": true}}`。回答长度会影响 Decode 与总请求时延，不把全部变化归因于视觉或 KV 复用。视觉 wrapper 包含状态管理和主机间隙；嵌套 M1/M2/M3 host 时间不可直接求和。

轮 0：实际后缀复用 14/40 帧，独立 hooks 验证跳过 308 个 Block 调用，锚帧更新 26 次。缓存 248.837 MiB。M1/M2/M3 host 均值 292.37/340.43/2008.98 ms。
轮 1：实际后缀复用 14/40 帧，独立 hooks 验证跳过 308 个 Block 调用，锚帧更新 26 次。缓存 248.837 MiB。M1/M2/M3 host 均值 293.46/340.90/2010.10 ms。
轮 2：实际后缀复用 14/40 帧，独立 hooks 验证跳过 308 个 Block 调用，锚帧更新 26 次。缓存 248.837 MiB。M1/M2/M3 host 均值 290.56/340.74/2012.03 ms。

## 任务精度与独立验证

官方 Detection/Lane 在声明的 GT 子集上评分，不是全量 6019 帧成绩。Token F1 对照相同真实参考答案；词法代理检查不等于人工语义评判。

| 指标 | 开发原生 | 开发正式 | 独立原生 | 独立正式 | 独立下降 | 预算 | 通过 |
|---|---:|---:|---:|---:|---:|---:|---|
| mAP | 0.288546 | 0.245617 | 0.122287 | 0.051313 | 0.070974 | 0.050000 | False |
| NDS | 0.341442 | 0.294091 | 0.169732 | 0.120477 | 0.049255 | 0.050000 | True |
| DET_l | 0.232000 | 0.205593 | 0.077293 | 0.063222 | 0.014071 | 0.050000 | True |
| Token_F1 | 0.359270 | 0.365675 | 0.439463 | 0.458659 | -0.019196 | 0.050000 | True |

独立 4 场景、24 帧；四项预算通过：False。场景按固定规则预先选择，排除开发及旧验证场景，未按候选结果挑选；详见 validation_sequence24.json。

| 独立场景 | 帧 | 全局复用帧 | mAP 下降 | NDS 下降 | DET_l 下降 |
|---|---:|---:|---:|---:|---:|
| 01e4fcbe6e49483293ce45727152b36e | 6 | 3 | 0.156294 | 0.100401 | 0.021702 |
| 034dee1695304630b0692da8c1f153fc | 6 | 4 | 0.033509 | 0.016454 | -0.004707 |
| 04219bfdc9004ba2af16d3079ecc4353 | 6 | 2 | 0.006109 | 0.021059 | 0.003639 |
| 07aed9dae37340a997535ad99138e243 | 6 | 3 | 0.033233 | 0.021475 | -0.017835 |

独立集最大 mAP 下降位于 01e4fcbe6e49483293ce45727152b36e：6 帧中 3 帧全局复用，mAP/NDS 下降 0.156294/0.100401。开发 H2 漂移阈值没有保证跨场景任务误差稳定；任务头与 M1 后续状态也会传播历史差异。

开发原生 40/40 个问题的 Token F1 小于 1；逐问表同时保留原生答案、GT 和新答案，用于识别原有错误与新增退化。

开发集关键代理退化检查数：对原生 4，对历史 WSGRR 5；独立集对原生 1。一个问题可触发多项检查，不等于独立错误数。原 88a6b5... / fd612e... 案例及改善/退化均保留在逐问结果。

历史 WSGRR 闭环任务差：`{"development": {"mAP_loss": 0.04310298828736345, "NDS_loss": 0.04726367981569368, "DET_l_loss": 0.026229023933410645}, "scope": "Historical full-loop prediction comparison, including M1/task-head feedback; not an identical-state causal task-loss isolation. Latency causality comes only from paired_visual_summary."}`。任务头与 M1 后续状态会传播误差，不能当作同状态残差的因果精度损失。复用年龄、特征差和逐问损失见 visual_error_by_age.json / *_quality_pairs.json。

## 结论与复现

独立精度预算通过：False；完整请求平均加速：False；本子集同时满足精度和平均时延目标：False。三项结论分开记录，预算放宽不等于模型精度提高。

实际正式默认参数：`{"global_anchor_interval": 2, "global_drift_threshold": 0.7721027880907059}`；本轮实验参数覆盖默认值：False；当前默认参数在旧独立子集按本轮预算重评通过：True。独立预算失败时保留原默认参数，不依据独立集再搜索阈值。原参数按 0.05 的旧独立子集重评见 previous_budget_reassessment/。

只运行实验服务，未发布外部服务。平均指标达标不认证所有场景/关键问答；独立集不用于继续调参。单一路径执行与 hooks 计数核对见 formal_execution_audit.json；最新 GPU 回归见 tests_gpu_after_budget_guards.log。保留正常负收益样本。

本轮不继续使用独立数据调参。后续若研究更稳定的残差条件，需要更丰富的开发场景及另一份未参与调参的独立数据。

环境见 environment.json、各 worker manifest；命令/退出码见 commands.json。旧输出保留原预算含义，新结果记录本轮预算。

```bash
taskset -c 0-3,6-31 bash outputs/m2_global_residual_20261026_run03/reproduce.sh outputs/m2_global_residual_reproduction_budget_0p05
```

[质量—时延曲线 PNG](quality_latency_curve.png) / [SVG](quality_latency_curve.svg)；[最终机器结果](final_summary.json)；[同快照结果](paired_visual_summary.json)；[校准结果](calibration_curve.json)。

[先前开发阈值与本轮校准的完整记录](refinement_history.md)。
