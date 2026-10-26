# OmniDrive WSGRR-AR 实测报告

固定参数：锚帧间隔 2；最大相机 H2 相对 RMS 漂移阈值 0.772102788。窗口预算 3、窗口最大复用年龄 8、checkpoint iter_10548.pth 不变。

正式代码只保留一条 M2 主执行路径。锚帧与普通帧共用 Global-State Update 和原子 StateCache Commit；残差复用明确为近似推理。没有在线成本准入、质量批准文件、KV 拼接或备用 EVA Runtime。详见 [实现说明](../../docs/worldstream_m2_global_residual.md)、[静态核对](single_route_audit.json)。

## 开发集校准

40 帧自然连续开发场景；先测 2/3/4 帧漂移分布，再固定年龄 1 漂移的 25% 分位数作为阈值。仅比较三个间隔，不在独立场景调参。下面是完整闭环 Detection/Map/M3 的真实输出，不是 embedding 一致性评分。

| 间隔 | 实际后缀复用帧 | 完整请求均值 ms | mAP 下降 | NDS 下降 | Lane DET_l 下降 | Token F1 下降 | 四项初始预算 |
|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 5 | 3004.01 | 0.004890 | 0.003005 | -0.000172 | -0.010119 | 通过 |
| 3 | 7 | 3116.53 | 0.006171 | 0.004440 | -0.003120 | -0.006654 | 未通过 |
| 4 | 7 | 3006.79 | 0.004825 | 0.003249 | 0.001321 | -0.011579 | 通过 |

Lowest measured complete service mean among candidates within all four user-specified initial quality budgets; numeric/spatial/motion/safety regressions are reported separately and are not certified by aggregate budgets

间隔 2 与 4 的单轮平均请求仅相差约 3 ms，不能据此宣称统计显著的全局最优。选择较短年龄的 2，固定后不再修改。平均指标达标不认证数字、空间、运动或安全回答；完整逐问退化保留在 *_quality_pairs.json。

## 同快照视觉对照与 Snapshot

同一真实 M1 快照、同一当前帧、同一窗口选择下，旧 WSGRR / 新 WSGRR-AR / 原生 EVA 的同步 Wall 均值为 496.73 / 431.73 / 489.81 ms。额外节省 65.00 ms；对原生纯视觉加速 1.1345×。该边界排除 M1 Snapshot 导出、M1 Precheck/Commit 和任务头。

对应旧/新/原生 CUDA Span 均值 496.71/431.71/489.79 ms，旧→新实测减少 65.00 ms。Span 包含主机提交空隙，不能称为平均 kernel active 节省。

前缀计划、packed outputs、年龄、来源及版本全部相同：True。最终 embedding 允许差异，没有宣称精确等价。

同一 pending 请求的交替 Snapshot wall：旧均值 17.492 ms，新均值 21.792 ms；旧/新 P95 29.168/28.576 ms。原始样本和 cProfile 在 paired40/，完整计算契约逐帧一致。不可仅凭 deepcopy 减少推断导出加速。

Snapshot 旧/新中位数 17.118/16.391 ms，旧/新最大值 30.900/319.815 ms。新实现的均值负收益必须保留：尽管典型导出更快，长尾使本次均值变差。代表帧 cProfile 的 source_digest 累计 20.842 ms；未删减来源/状态检查。没有 GC 事件采样，不能断言长尾归因。

代表复用帧的旧/新完整 kernel active 为 477.329/41.441 ms。CUDA Span 与同步 Wall 是不同计时边界；trace 的 kernel 单次计数，未将重叠 operator 总时间相加。完整 trace、CPU profile 和真实 Commit 隔离样本位于 paired40/。

## 三轮完整服务

执行顺序：原生→正式、正式→原生、原生→正式。每轮同 40 帧、同原问题、同 seed 和生成配置，预热 1 帧后重置历史。三轮是 40 个唯一样本的重复测量，不是 120 个独立驾驶场景。

三轮 Detection/Map digest 与回答一致性：`{"native": {"task_digests_equal": true, "answers_equal": true}, "formal": {"task_digests_equal": true, "answers_equal": true}}`。官方任务评分来自第 0 轮真实 predictions.pkl。

| 轮次 | 原生 mean/P50/P95 ms | 正式 mean/P50/P95 ms | 端到端加速比 |
|---|---|---|---:|
| 0 | 2871.69/1741.04/9139.80 | 3003.82/2438.98/6588.94 | 0.9560× |
| 1 | 2874.99/1758.36/9158.39 | 3022.93/2456.75/6592.22 | 0.9511× |
| 2 | 2867.52/1727.36/9138.99 | 3143.08/2564.28/6719.53 | 0.9123× |

| 三轮合计 | 原生 | 正式 |
|---|---:|---:|
| 完整请求 mean ms | 2871.40 | 3056.61 |
| 视觉 CUDA Span mean ms | 480.75 | 489.00 |
| Prefill forward CUDA mean ms | 189.29 | 184.98 |
| Decode forward CUDA mean ms | 1888.11 | 1643.21 |
| 吞吐 frame/s | 0.3483 | 0.3272 |
| 峰值 allocated GiB | 15.353 | 16.085 |
| 峰值 reserved GiB | 15.742 | 16.955 |

完整服务加速比 0.9394×；没有完整服务净加速。视觉 wrapper CUDA Span 包含 Snapshot、主机提交间隙和状态管理，不能与同快照纯 M2 时延混用。M1/M2/M3 host 包含等待及嵌套范围，不可直接相加。

三轮原生/正式生成 token 总数 9066/7920；M3 实际复用 token 总数 0/0。回答长度改变会影响 Decode 与完整服务时间，不能将全部变化归因于视觉或 KV 复用。

按三轮均值配对的节省时间 95% t 区间：[-380.57, 10.15] ms（正值表示节省）。只有 3 轮同场景重复，不能当作跨场景保证。

轮 0 实际复用 5/40 帧，跳过 110 次后缀 Block；锚帧更新 35 次。缓存 248.837 MiB，512 MiB 预算足够。M1/M2/M3 host 均值 326.58/438.80/1868.43 ms；Snapshot 49.18 ms。M3 实际 KV 复用 token=0；未继承旧 M3 质量结论。
该轮 Global-State Update 的锚帧/复用 CUDA Span 均值 437.714/0.626 ms；包含固定漂移判定。前缀、Patch Embedding、Suffix 和 Commit 的组件样本见 final_summary.json 与逐帧原始日志。
轮 1 实际复用 5/40 帧，跳过 110 次后缀 Block；锚帧更新 35 次。缓存 248.837 MiB，512 MiB 预算足够。M1/M2/M3 host 均值 330.96/438.96/1870.38 ms；Snapshot 49.14 ms。M3 实际 KV 复用 token=0；未继承旧 M3 质量结论。
该轮 Global-State Update 的锚帧/复用 CUDA Span 均值 437.970/0.616 ms；包含固定漂移判定。前缀、Patch Embedding、Suffix 和 Commit 的组件样本见 final_summary.json 与逐帧原始日志。
轮 2 实际复用 5/40 帧，跳过 110 次后缀 Block；锚帧更新 35 次。缓存 248.837 MiB，512 MiB 预算足够。M1/M2/M3 host 均值 332.13/438.30/1869.24 ms；Snapshot 49.19 ms。M3 实际 KV 复用 token=0；未继承旧 M3 质量结论。
该轮 Global-State Update 的锚帧/复用 CUDA Span 均值 437.015/0.613 ms；包含固定漂移判定。前缀、Patch Embedding、Suffix 和 Commit 的组件样本见 final_summary.json 与逐帧原始日志。

## 任务精度与独立验证

以下均使用当前 nuScenes/OpenLane 评分函数在明确声明的 GT 子集上评估；不是 6019 帧完整验证集分数。数字/空间/运动/安全为词法代理，不能替代逐场景人工语义评价。

| 指标 | 开发原生 | 开发正式 | 独立原生 | 独立正式 |
|---|---:|---:|---:|---:|
| mAP | 0.288546 | 0.283656 | 0.365932 | 0.349578 |
| NDS | 0.341442 | 0.338437 | 0.371450 | 0.357366 |
| DET_l | 0.232000 | 0.232173 | 0.165651 | 0.168308 |
| Token_F1 | 0.359270 | 0.369389 | 0.394232 | 0.398143 |

独立误差主要集中在场景 905cfed4f0fc46679e8df8890cca4141：6 帧中 3 帧复用，mAP/NDS 绝对下降 0.030135/0.020607。另外三场景均未发生全局残差复用。固定间隔 2 下复用年龄均为 1，因此失败不能仅归因于 3–4 帧的长锚帧年龄；H2 漂移阈值不能保证全局非线性残差在不同场景中的稳定性，任务头/M1 的后续历史反馈也会传播误差。各场景原始差异在 final_summary.json。

相对历史仅复用窗口的 WSGRR，完整闭环任务精度差（包含 M1/任务头历史反馈，不能解释为固定同一 M1 状态的纯残差损失）：`{"development": {"mAP_loss": 0.005064044133448853, "NDS_loss": 0.002917888064714269, "DET_l_loss": -0.00035043060779571533}, "validation": {"mAP_loss": 0.016354926451520424, "NDS_loss": 0.014083422038940141, "DET_l_loss": -0.0026563555002212524}, "scope": "Historical full-loop prediction comparison, including M1/task-head feedback; not an identical-state causal task-loss isolation. Latency causality comes only from paired_visual_summary."}`。历史预测只做当前官方评分，不作为同期性能测量。

独立 4 场景、24 帧固定参数验证的四项初始预算：未全部通过。未用独立验证集重新调整阈值。开发集相对原生的关键代理退化检查数为 7，相对上一 WSGRR 为 6；一个问题可触发多个检查，不能称为同等数量的独立错误。独立集相对原生为 0 项。

此前 88a6b5... 道路布局和 fd612e... 交通元素两个数字案例保留完整回答和 GT；包括进一步退化和改善，都在 *_quality_pairs.json 的逐问对照中。即使平均 Token F1 提高，也不能掩盖遗漏车道数量、坐标或运动状态的问题。

visual_error_by_age.json 给出无严格漂移筛选探针中的特征误差与年龄；*_quality_pairs.json / final_summary.json 给出完整闭环任务误差与年龄。锚帧更新后的任务误差还包含之前 Detection/Map/M1 的历史反馈，不能全归于当前帧残差年龄。

## 结论、限制与复现

实际跳过后续 22 Block 已由独立 forward hooks 验证，局部窗口依赖和执行集合没有为提高复用而修改。简单 H2 漂移门限在驾驶流中常触发全局刷新；放宽年龄能增加可省计算，也会改变全局 Attention 上下文近似误差和任务历史。这限制了质量预算内的可用复用比例。

当前固定参数的独立 Detection 精度预算失败，未得到满足用户要求的质量—时延组合。代码保留研究实现，不推荐作为达标方案部署，也不根据验证集继续调参。

没有把 Block 次数下降直接当作端到端成功，没有自动切换备用全量模型，也没有建立在线批准机制。本轮只运行实验服务，没有发布或重启外部线上服务。若独立预算失败或关键回答不可接受，本策略不应被称为可部署质量方案。

真实 GPU 回归见 tests_gpu_final.log（早期 tests_gpu.log 也保留）；初始失败日志均保留。早期精确后缀断言已改为前缀精确和残差公式检查；一次临时 profiler 包装污染测试全局函数的问题已改为实例级只读包装。没有丢弃性能负收益的正常样本。

执行环境见 environment.json，源代码与 checkpoint 身份见各 worker manifest。命令及时间/退出码见 commands.json，纯视觉初始探针命令见 reproduce.sh。

```bash
taskset -c 0-3,6-31 bash outputs/m2_global_residual_20261026_run01/reproduce.sh outputs/m2_global_residual_reproduction
```

质量—时延曲线：[PNG](quality_latency_curve.png)、[SVG](quality_latency_curve.svg)。机器结果：[final_summary.json](final_summary.json)、[同快照结果](paired_visual_summary.json)、[校准曲线](calibration_curve.json)。
