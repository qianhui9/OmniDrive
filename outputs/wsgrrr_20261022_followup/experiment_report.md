# WSGRR 后续修改与真实验证报告

结果仅来自归档的真实执行。生产批准新增数为 0，既有实验、checkpoint、原生模型源码、官方评估源码和生产配置均未修改。完整验证状态以文末各组状态为准，运行中的结果不写成已完成。

## 可复现条件

- 环境：`/home/qh/miniconda3/envs/OmniDrive/bin/python`，RTX 3090；所有命令由 `taskset -c 0-3,6-31` 启动。没有停止其他进程。
- 固定权重：`ckpts/iter_10548.pth`，SHA256 `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`，严格键/形状校验通过，不接受随机参数。
- 同一帧序列、预处理、任务头和问题；生成参数来自原生源码：do_sample=True、temperature=0.1、top_p=0.75、num_beams=1、max_new_tokens=320、use_cache=True，按帧固定随机种子。独立进程，一帧内顺序执行。40 帧完整请求每组 2 次，每次先预热再重置历史，从首帧冷缓存开始。两次输出各自完全一致，重复不是新增独立场景。
- 连续视觉基准为同一场景 40 帧，每帧预热 2 次、重复 5 次，Native 每帧执行，候选顺序交替。报告首帧及 39 个后续帧；M1 输入来自相同的历史真实预测回放，只用于视觉剖析，不能代替候选任务质量。
- 6019 帧正式任务推理显式关闭语言问题，仅比较 Detection/Map；40 帧真实语言任务另列，不能把 perception-only 延迟当作含 VLM 的完整请求延迟。全量推理可能与前组 CPU-only 评估重叠，其单次延迟不用于跨组速度结论；性能比较采用独立的无评估并发配对重复实验。
- 资源预检：6019 帧、150 场景、36114 张图像齐全；SDK 重建 anns 后与原始 annotation 逆向链接一致，过滤前 187528 个 Detection GT boxes，Native 经官方范围/点数/bike-rack 过滤后为 121861 个。原始表中的 stale links 未用于推理或伪造标签。
- `run_manifest.json`、各组 `invocation.json`、`checkpoint_recheck.json`、source hashes、`repeat_consistency.json` 和 `protected_hash_check.json` 可复核。三组 fresh 完整验证前 40 帧与各自语言短测的全部感知输出一致。最早 Native/D3 repeat-0 未启用核心 hash 记录，不能补造；文件时间线显示核心在首测前固定，后续六组及全量运行核心 hash 均与最终一致，详见 `core_source_timeline_verification.json`。早期 M1 precheck 的 observer 覆盖后来修正；主请求计时和 GPU 同步边界未变，旧运行仍有源码内置 CPU 计时。

## 51.7928 m Map 异常

帧 `a7795dcadad44c1c85fa27eef18366f6`（ordinal 7）：旧匈牙利匹配将 Native raw 95 与候选 raw 87 配对。两者同为单类别地图线，各 11 个有限点，坐标均为 ego 米，pc_range、预处理和坐标输出路径一致。无坐标转换、索引偏移、NaN 或反向点序实现错误。

| 项目 | Native 95 | 旧配对候选 87 |
| --- | --- | --- |
| 置信度 | 0.37526482 | 0.31256533 |
| 起点 | [12.302318572998047, -4.747783660888672, 0.02976369857788086] | [11.58953857421875, 8.845451354980469, 0.05810880661010742] |
| 终点 | [48.5333137512207, -5.972682952880859, 0.008193016052246094] | [-50.04438781738281, 8.917938232421875, -0.0057926177978515625] |
| 范围 min | [12.302318572998047, -5.997524261474609, -0.021409988403320312] | [-50.04438781738281, 8.273963928222656, -0.0057926177978515625] |
| 范围 max | [48.5333137512207, -4.443183898925781, 0.040169715881347656] | [11.58953857421875, 9.161125183105469, 0.06862592697143555] |

旧配对正向距离 53.16554260 m，反向/旧成本 51.79280853 m。它是置信度过阈值导致候选集合改变后，**无门限强制一一配对造成的错误匹配**。Native 95 对应的候选 query 95 置信度降至 0.2776165，低于原有 0.3；候选 87 对应的 Native query 87 为 0.2861632，也低于阈值。两条各自近邻的几何差异约 0.1386 m、0.0192 m，并不是地图整体移动 51.79 m。仍存在真实的几何与置信度变化，不能删除该样本。

旧指标保留：原始 8 帧 mean 0.4066730 m、max 51.7928085 m。新增指标在运行前声明：类别兼容、匈牙利 dummy 匹配、101 个弧长均匀点的双向点到线段平均距离、3 m 门限。门限参考官方最大 Frechet 阈值，但几何定义不同，**不是官方 AP**。同一旧 8 帧新指标 mean 0.0640917 m、167 个匹配、Native 未匹配 6、候选未匹配 2。均值变化包含几何定义和匹配集合变化，不能解释为精度提升。

采样敏感性另存 `map_sampling_convergence.json`：将已报告的旧/新配对从 101 点加密至 401 点，D3 平均差异 0.000746 m、最大差异 0.037283 m；NoM1Region 最大差异 0.051099 m。三组这些配对均无 3 m 门限判定变化。该后验检查不修改已声明的 101 点指标，也不声称检查了所有未选候选的最优匹配稳定性。

所有旧配对仍在 `anomaly_traced/maps.json`；逐线原始索引、类别、分数、点数、起终点、范围、坐标、正反/对称距离、成本及拒配原因均保留。可视化：`anomaly_traced/map_extreme.svg`。最终三组 40 帧逐线/逐对象诊断分别在 `*_r0_diagnostic_v2/`（Matched 使用 retry01）。v2 修正了长序列 Map/Detection 极值不在同一帧时的归因坐标选择，原始 8 帧同帧归因不受影响；旧诊断目录仍保留，匹配指标未改。

## Detection 与缓存状态归因

同一帧 car，Native raw 4 ↔ candidate raw 4，互为同类最近预测，置信度 0.8864116/0.8790284。原始三维中心 delta = [-0.6036148, 0.1600876, -0.0322623] m，距离 0.6253157 m，属于真实预测变化。最近真实 car 标注 `07fb51af93cd429c91a87430c662a652` 的平面距离：Native 2.9602 m，最终 D3 3.5631 m；这只是逐对象几何归因，正式指标仍按官方分数顺序匹配计算。`anomaly_detection_index_binding.json` 另证实四组 raw 模型输出 4 对应官方格式化索引 4，世界中心距离均精确为 0，并非筛选重排造成归因错对象。

最近历史 car 假设 `01452fbfbf4543af8acdfd3e8a1ee806:g6:a4` 距 Native 世界中心约 0.4129 m，不能当作 GT 身份认证。CAM_FRONT 投影覆盖 local window 4，但 `geometry_valid=False`，原因为 `uncertainty_exceeds_search`，没有进入可信 region_states。该窗口 age=1、drift≈0.00054264、score≈0.281331，既非必须也未被普通预算选中。Map Native 95 的投影同时覆盖 window 4（复用）和 5（普通刷新）。这是现有世界区域引导的覆盖缺口，不应通过放松几何有效性来掩盖。详见 `anomaly_traced/state_window_trace.json`。

| 最终代码反事实（同一候选头历史） | D3 正常复用 | 仅该帧视觉全刷新 |
| --- | --- | --- |
| 检测 Native raw4 中心差异 m | 0.62597543 | 0.01078088 |
| Map query95 对称差异 m | 0.13896302 | 0.02505234 |
| Map query95 候选置信度 | 0.27780205 | 0.35627964 |

反事实支持当前帧局部窗口复用是实际偏移的重要来源；原始 v1 同样存在 0.6253 m 偏移。全刷新同时改变部分执行的算术形状，因此不能仅凭此反事实精确分离时间陈旧与舍入误差；残余差异仍包括历史任务头状态。它没有逐一干预各窗口，不能把单一窗口或历史假设认定为唯一原因。`anomaly_counterfactual.json`、`anomaly_actual_gt.json` 保存实际数据；`anomaly_gt_overlay.svg` 叠加 Native、D3、反事实与最近实际 Map/Detection 标注，近邻标注不等于身份认证。

## 公平离线消融：40 帧 × 2 次

D3 和 NoM1Region 每相机普通预算均为 3。后者只禁用 M1 世界区域/对象事件的额外优先级，来源失效、TTL、非有限值、基础不确定性、漂移和年龄约束保留。由于 D3 额外事件增加真实刷新数量，另加 NoM1RegionMatched，按 D3 每帧每相机总窗口数量选择自身评分最高的窗口，不读取 D3 位置；安全约束可超目标。全部为离线 API，未进入生产配置。

| 组 | 更新率 | 复用率 | 必须窗口累计¹ | 普通窗口累计¹ | 请求 P50 ms | 请求 P95 ms | VLM Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native | 1.0000 | 0.0000 | — | — | 1708.80 | 9108.50 | 0.3593 |
| D3 | 0.7370 | 0.2630 | 1898 | 1286 | 2106.76 | 8106.82 | 0.3637 |
| NoM1Region | 0.5926 | 0.4074 | 1264 | 1296 | 2088.40 | 9407.23 | 0.3588 |
| NoM1RegionMatched | 0.7370 | 0.2630 | 914 | 2270 | 2006.56 | 6850.06 | 0.3540 |

¹ 每帧每相机窗口按第一个可缓存 Block 计数，重复运行累计；更新率包含两个 Block 及冷启动。Native 的 100%/0% 来自完整 EVA 执行路径及逐帧 6 相机 forward 计数，必须/普通预算分类不适用于 Native。逐帧计数、普通预算使用量、特征误差和各阶段统计见 `measured_summary.json`。请求包括预处理、H2D、M1、M2、下游任务、真实生成、D2H 和完成同步；审计文件写入及独立反事实在计时外。

D3 已识别事件窗口覆盖 1006/1006。以 D3 事件足迹为共同参考，NoM1Region 覆盖 54.08%，Matched 覆盖 85.09%。各组自身必须安全刷新召回为 100%；**这不是现实变化零漏检**。禁用事件组的 event denominator 为 0，不能解释为覆盖良好。

Matched 的 80/80 帧逐相机总窗口数与 D3 完全一致，QKV token 数也一致；由于边缘窗口有效 token 数不同，Matched 的 MLP token 总数为 1131776，D3 为 1145344（约少 1.18%）。所以这是窗口数相同的强控制，仍有实际工作量混杂，不声称严格等 FLOPs。

40 帧 Prediction-to-prediction 差异（非 GT 精度，分数筛选始终 ≥0.3）：

| 候选 | 旧 Map mean/P95/max m | 新 Map mean/P95/max m | Map 未匹配 N/C | 检测旧 max m | 检测未匹配 N/C |
| --- | --- | --- | --- | --- | --- |
| D3_r0 | 0.3934/0.6138/51.7929 | 0.1070/0.3699/2.0654 | 34/22 | 63.7189 | 151/177 |
| NoM1Region_r0 | 1.5652/4.7717/55.3898 | 0.3505/1.4693/2.9253 | 73/72 | 64.7466 | 497/593 |
| NoM1RegionMatched_r0_retry01 | 0.6775/1.4567/52.5405 | 0.1589/0.4836/2.8349 | 40/33 | 58.4067 | 348/376 |

长序列旧 Detection 指标也出现超过 60 m 的强制配对。它们按预声明的 2 m 三维中心诊断门限报告为未匹配；原始距离仍保留，不与 8 帧里互为近邻的真实 0.6253 m 偏移混淆。

共同实际复用特征检查集合：29 个不同帧，含重复共 58 对。以下为与同输入 Native EVA 的真实特征差异；未独立检查的全刷新帧没有补测量零。它不是 GT 任务精度。

| 候选 | mean(&#124;feature delta&#124;) 的帧均值 | mean(&#124;delta&#124;) 帧 P95 | 所有帧最大 &#124;delta&#124; |
| --- | --- | --- | --- |
| D3 | 0.1810 | 0.3869 | 78.9718 |
| NoM1Region | 0.3042 | 0.4340 | 78.3441 |
| NoM1RegionMatched | 0.1857 | 0.4212 | 78.6530 |

真实生成长度分布（相同生成参数，不强制截断为同长度）：

| 组 | 输入 token P50 | 生成 token 均值 | 生成 token P50 | 生成 token P95 |
| --- | --- | --- | --- | --- |
| Native | 575.0 | 75.5 | 32.5 | 320.0 |
| D3 | 575.0 | 70.5 | 33.5 | 266.8 |
| NoM1Region | 575.0 | 75.7 | 33.5 | 320.0 |
| NoM1RegionMatched | 575.0 | 68.4 | 33.5 | 220.7 |

回答长度变化会改变完整请求延迟；某组 P95 请求更短不能单独作为 M2 加速证据。各帧原始 token、问题、真实回答和参考均保留。

## 真实任务精度

以下 40 帧分数来自官方程序及真实标签，但仅为一个场景的子集，不能外推完整验证集。无预测一致性代替精度。

| 40 帧子集 | mAP | NDS | mATE | mASE | mAOE | DET_l |
| --- | --- | --- | --- | --- | --- | --- |
| Native | 0.2885 | 0.3414 | 0.9337 | 0.4886 | 0.6802 | 0.2320 |
| D3 | 0.2651 | 0.3062 | 0.9557 | 0.4733 | 0.6919 | 0.2439 |
| NoM1Region | 0.2328 | 0.2733 | 0.9839 | 0.4878 | 0.7132 | 0.2221 |
| NoM1RegionMatched | 0.2560 | 0.3000 | 0.9347 | 0.4867 | 0.7204 | 0.2405 |

40 帧上 D3 比同普通预算 NoM1Region 提高检测/地图质量；相对窗口数匹配组仍有较小收益。但 D3 检测低于 Native，现有证据不能认证精度损失可接受。绝对值、差值、相对变化、逐类别 AP/TP errors、逐场景结果及官方曲线见 `measured_summary.json` 与各组 `subset_official/`。

完整 6019 帧、150 场景（fresh predictions）：

| 完整组 | 评估状态 | 帧数/进度 | mAP | NDS | mATE | mASE | mAOE | DET_l | Map F-score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native | COMPLETED | 6019 | 0.3774 | 0.4713 | 0.7723 | 0.2882 | 0.4240 | 0.1322 | 0.0811 |
| D3 | COMPLETED | 6019 | 0.3379 | 0.4406 | 0.8112 | 0.2893 | 0.4426 | 0.1357 | 0.0807 |
| NoM1Region | COMPLETED | 6019 | 0.2613 | 0.3877 | 0.8969 | 0.2910 | 0.4462 | 0.1176 | 0.0781 |

每组完整评估目录为 `Native_full/official_eval/`、`D3_full/official_eval/`、`NoM1Region_full/official_eval/`。预测进程完成后才由 CPU-only 子进程调用官方评估；`status.json`、`detection_metrics.json`、`map_metrics.json`、`scenes.json` 是状态和分数依据。全量 Detection 未使用诊断 0.3 分数筛选。Map 没有交通元素或拓扑预测头；DET_t/TOP_ll/TOP_lt 的空预测/零邻接结果仅按实际实现归档，不能作为学得拓扑能力的证据。部分空 GT 类别会产生看似非零的空预测分数，也不应作能力解释。

历史全量 Native 复算另存 `historical_native_full_reassessment/`：mAP=0.3773975，NDS=0.4713427，DET_l=0.1322065。它用于核对评估链路，与 fresh 完整组分开，不能冒充本轮新推理。新旧 Native 的 6019/6019 帧全部感知 digest 一致（`native_full_historical_consistency.json`）；同输入串行历史评估与 fresh 并行评估的全部 Map 指标精确一致（`official_map_parallel_full_parity.json`），是单元测试之外的全量官方评估等价证据。评估加速不计作推理加速。

VLM 真实标注验证范围为 40 个固定输入–问题对（每组重复两次，不同输入–问题对仍为 40，同场景时序样本不独立）。四组 EM 均为 0，Token F1 见前表；不将“相同回答”解释为正确。9 个含数字参考问题中，Native/D3/Matched 的数目匹配为 0/9，MAE 无定义（denominator=0）；NoM1Region 为 1/9，文本数字序列 MAE=5.6917（仅 1 个有效问题），不代表整体数值问答能力。空间、对象状态、关键驾驶问题的重叠文本分层及参考严格不匹配率见 `question_text_strata`；EM 不匹配不等于人工判定语义错误。**全量 48108 Conv + 24081 VQA 的正式质量及人工语义错误率未完成**，不声称完整 VLM 验收通过。

### 全量差值与场景分布

| 候选−Native | ΔmAP | mAP 相对变化 | ΔNDS | ΔDET_l |
| --- | --- | --- | --- | --- |
| D3 | -0.0395 | -10.48% | -0.0308 | 0.0035 |
| NoM1Region | -0.1161 | -30.75% | -0.0837 | -0.0146 |

| 类别 | Native AP | D3 AP | D3 ΔAP | NoM1Region AP | NoM1Region ΔAP |
| --- | --- | --- | --- | --- | --- |
| barrier | 0.4297 | 0.4250 | -0.0047 | 0.3650 | -0.0648 |
| bicycle | 0.4380 | 0.3514 | -0.0866 | 0.2689 | -0.1691 |
| bus | 0.3716 | 0.3395 | -0.0321 | 0.2617 | -0.1099 |
| car | 0.5448 | 0.4839 | -0.0609 | 0.3780 | -0.1669 |
| construction_vehicle | 0.1541 | 0.1273 | -0.0268 | 0.0718 | -0.0824 |
| motorcycle | 0.3810 | 0.3204 | -0.0605 | 0.2192 | -0.1617 |
| pedestrian | 0.4733 | 0.4146 | -0.0587 | 0.3297 | -0.1435 |
| traffic_cone | 0.5291 | 0.5201 | -0.0090 | 0.4112 | -0.1178 |
| trailer | 0.1036 | 0.0930 | -0.0105 | 0.0772 | -0.0264 |
| truck | 0.3488 | 0.3033 | -0.0455 | 0.2306 | -0.1182 |

D3 全量 mAP 相对 Native 变化 -10.48%，NDS 变化 -6.53%；自行车、行人、汽车的 AP 绝对差分别为 -0.0866/-0.0587/-0.0609。Detection 精度保持目标不通过，不能用 Map DET_l 的单项改善抵消该损失或批准配置。

D3：150 个真实配对场景，mAP 差值改善/相同/下降场景数为 13/0/137；场景差值 P5/P50/P95 为 -0.0763/-0.0179/0.0024。这是逐场景分数差的描述分布，不是另一个完整集官方 AP，也不把帧当作独立统计样本。

| 下降最大的场景 | 实际描述 | Native mAP | 候选 mAP | ΔmAP |
| --- | --- | --- | --- | --- |
| scene-0035 | Parking lots, trash can, bump, scooter, peds, potential jaywalkers, bus stop, oncoming bus | 0.3330 | 0.2044 | -0.1286 |
| scene-0110 | Parked car with open bonnet, parked cars, oncoming truck, oncoming cyclists | 0.3700 | 0.2516 | -0.1184 |
| scene-0522 | Parked bicycle, pedestrian, parked cars, cyclist, construction zone | 0.4247 | 0.3108 | -0.1139 |
| scene-0038 | Parked construction vehicles, peds, garbage | 0.3417 | 0.2366 | -0.1051 |
| scene-0523 | Oncoming bus, parked car on center divider, pedestrian, roundabout, parked trucks | 0.4176 | 0.3243 | -0.0933 |

NoM1Region：150 个真实配对场景，mAP 差值改善/相同/下降场景数为 2/0/148；场景差值 P5/P50/P95 为 -0.1950/-0.0813/-0.0080。这是逐场景分数差的描述分布，不是另一个完整集官方 AP，也不把帧当作独立统计样本。

| 下降最大的场景 | 实际描述 | Native mAP | 候选 mAP | ΔmAP |
| --- | --- | --- | --- | --- |
| scene-0110 | Parked car with open bonnet, parked cars, oncoming truck, oncoming cyclists | 0.3700 | 0.1336 | -0.2364 |
| scene-0039 | Busy street, parked cars, parking lot, ped sitting at bus stop, motorcycle driving | 0.3694 | 0.1365 | -0.2329 |
| scene-0035 | Parking lots, trash can, bump, scooter, peds, potential jaywalkers, bus stop, oncoming bus | 0.3330 | 0.1112 | -0.2218 |
| scene-0910 | Rain, parked cars, sign trailer, construction, construction vehicle, crane, cones, truck, ped with umbrella, cross intersection, bicycle, peds crossing crosswalk, bicyclist, potential jaywalkers | 0.3723 | 0.1567 | -0.2156 |
| scene-0107 | Cross busy intersection, bendy bus changing lane | 0.4543 | 0.2404 | -0.2139 |

完整集官方 Map 原始输出（交通元素与拓扑为缺少预测头的占位结果）：

| 组 | DET_l | DET_t | TOP_ll | TOP_lt | 组合 score | Map F-score |
| --- | --- | --- | --- | --- | --- | --- |
| Native | 0.132207 | 0.000000 | 0.003692 | 0.000000 | 0.048242 | 0.081051 |
| D3 | 0.135661 | 0.000000 | 0.003670 | 0.000000 | 0.049060 | 0.080703 |
| NoM1Region | 0.117560 | 0.000000 | 0.003352 | 0.000000 | 0.043865 | 0.078095 |

仅 DET_l 和 lane F-score 直接验证实际车道线头；组合值中的空预测/零邻接项不能解释为交通元素或拓扑能力。

M1 区域引导在全量同普通预算对照中的收益：

| 正式指标 | NoM1Region | D3 | D3−NoM1Region | 相对变化 |
| --- | --- | --- | --- | --- |
| mAP | 0.2613 | 0.3379 | 0.0765 | 29.28% |
| NDS | 0.3877 | 0.4406 | 0.0529 | 13.65% |
| DET_l | 0.1176 | 0.1357 | 0.0181 | 15.40% |
| Map F-score | 0.0781 | 0.0807 | 0.0026 | 3.34% |

两组每相机普通预算都为 3，D3 额外区域事件产生更多总刷新；该收益同时包含额外刷新数量与选择位置的影响。严格相同窗口数的额外控制仅完成前述 40 帧，不能把其结论外推到全量。

官方 Detection 2 m XY 匹配审计（全部保留分数，未经插值，**不是 AP/NDS**）：

| 组/范围 | GT | 预测 | TP | FP | FN | 原始 recall | 已评分类别原始 precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native_r0 | 1909 | 7460 | 1758 | 5672 | 151 | 0.9209 | 0.2366 |
| D3_r0 | 1909 | 7515 | 1738 | 5754 | 171 | 0.9104 | 0.2320 |
| NoM1Region_r0 | 1909 | 7682 | 1653 | 6018 | 256 | 0.8659 | 0.2155 |
| NoM1RegionMatched_r0_retry01 | 1909 | 7543 | 1711 | 5809 | 198 | 0.8963 | 0.2275 |
| Native_full | 121861 | 1283894 | 112054 | 1171840 | 9807 | 0.9195 | 0.0873 |
| D3_full | 121861 | 1294529 | 109598 | 1184931 | 12263 | 0.8994 | 0.0847 |
| NoM1Region_full | 121861 | 1306516 | 103435 | 1203081 | 18426 | 0.8488 | 0.0792 |

每组 `official_eval/detection_matches_2m/`（40 帧为 `subset_official/`）保存 `matched_objects.jsonl`、`unmatched_ground_truth.jsonl`、`prediction_status.jsonl` 及逐类 TP/FP/FN。此处 raw_index 指官方过滤前的格式化预测顺序，**不是原始头查询索引**；GT 索引指官方过滤后的该帧列表。原始模型输出索引另在逐对象预测比较诊断中保留。所有配对来自 SDK 自身执行，逐项校验其返回 TP。无 GT 类别标为未评分，不把其预测算成已验证正确。

真实 GT Map 逐线匹配审计：40 帧子集，官方 1/2/3 m 门限及全部分数（操作点计数，非 AP）。

| 组 | 门限 m | GT | 预测 | TP | FP | FN | 匹配官方距离均值/P95 m | 匹配原始 Frechet 均值 m | 分支 GT 匹配/总数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native_r0 | 1.0 | 725 | 12000 | 44 | 11956 | 681 | 0.7784/0.9615 | 0.9326 | 32/282 |
| Native_r0 | 2.0 | 725 | 12000 | 346 | 11654 | 379 | 1.4917/1.9375 | 1.7009 | 191/282 |
| Native_r0 | 3.0 | 725 | 12000 | 588 | 11412 | 137 | 2.0171/2.9212 | 2.2750 | 270/282 |
| D3_r0 | 1.0 | 725 | 12000 | 46 | 11954 | 679 | 0.8034/0.9895 | 0.9640 | 31/282 |
| D3_r0 | 2.0 | 725 | 12000 | 356 | 11644 | 369 | 1.4906/1.9412 | 1.6982 | 196/282 |
| D3_r0 | 3.0 | 725 | 12000 | 582 | 11418 | 143 | 2.0035/2.9024 | 2.2613 | 271/282 |
| NoM1Region_r0 | 1.0 | 725 | 12000 | 50 | 11950 | 675 | 0.7870/0.9853 | 0.9133 | 33/282 |
| NoM1Region_r0 | 2.0 | 725 | 12000 | 334 | 11666 | 391 | 1.4499/1.9285 | 1.6585 | 183/282 |
| NoM1Region_r0 | 3.0 | 725 | 12000 | 561 | 11439 | 164 | 1.9846/2.9133 | 2.2447 | 263/282 |
| NoM1RegionMatched_r0_retry01 | 1.0 | 725 | 12000 | 43 | 11957 | 682 | 0.8108/0.9859 | 0.9620 | 30/282 |
| NoM1RegionMatched_r0_retry01 | 2.0 | 725 | 12000 | 347 | 11653 | 378 | 1.4852/1.9586 | 1.6923 | 188/282 |
| NoM1RegionMatched_r0_retry01 | 3.0 | 725 | 12000 | 588 | 11412 | 137 | 2.0010/2.8983 | 2.2580 | 272/282 |

逐线原始预测索引、GT 索引/ID、距离、拒配原因、未匹配 GT 和逐场景分布保存在各组 `map_gt_audit40/` 或 `map_gt_audit_full/`。官方 ego relaxation 与 Chamfer 预筛选按原函数执行，没有另行放宽门限；预筛选占位值不算作真实距离。分支指 GT 图入度/出度大于 1，不是独立交叉口标签。模型没有拓扑头，所有预测连接为零，因此节点匹配覆盖不是连接预测正确率。全量逐线范围以已完成的行和各目录 status 为准，运行中不冒充完成。

真实 GT Map 逐线匹配审计：6019 帧完整验证集，官方 1/2/3 m 门限及全部分数（操作点计数，非 AP）。

| 组 | 门限 m | GT | 预测 | TP | FP | FN | 匹配官方距离均值/P95 m | 匹配原始 Frechet 均值 m | 分支 GT 匹配/总数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native_full | 1.0 | 150735 | 1805700 | 12958 | 1792742 | 137777 | 0.7554/0.9828 | 0.9012 | 3982/35225 |
| Native_full | 2.0 | 150735 | 1805700 | 54915 | 1750785 | 95820 | 1.3903/1.9471 | 1.6103 | 15497/35225 |
| Native_full | 3.0 | 150735 | 1805700 | 90406 | 1715294 | 60329 | 1.9415/2.9000 | 2.2269 | 24265/35225 |
| D3_full | 1.0 | 150735 | 1805700 | 12817 | 1792883 | 137918 | 0.7570/0.9818 | 0.9037 | 3946/35225 |
| D3_full | 2.0 | 150735 | 1805700 | 54755 | 1750945 | 95980 | 1.3934/1.9503 | 1.6144 | 15450/35225 |
| D3_full | 3.0 | 150735 | 1805700 | 90089 | 1715611 | 60646 | 1.9426/2.8998 | 2.2285 | 24185/35225 |
| NoM1Region_full | 1.0 | 150735 | 1805700 | 11771 | 1793929 | 138964 | 0.7541/0.9814 | 0.9021 | 3531/35225 |
| NoM1Region_full | 2.0 | 150735 | 1805700 | 50895 | 1754805 | 99840 | 1.3940/1.9493 | 1.6193 | 14247/35225 |
| NoM1Region_full | 3.0 | 150735 | 1805700 | 85354 | 1720346 | 65381 | 1.9588/2.9038 | 2.2516 | 22908/35225 |

逐线原始预测索引、GT 索引/ID、距离、拒配原因、未匹配 GT 和逐场景分布保存在各组 `map_gt_audit40/` 或 `map_gt_audit_full/`。官方 ego relaxation 与 Chamfer 预筛选按原函数执行，没有另行放宽门限；预筛选占位值不算作真实距离。分支指 GT 图入度/出度大于 1，不是独立交叉口标签。模型没有拓扑头，所有预测连接为零，因此节点匹配覆盖不是连接预测正确率。全量逐线范围以已完成的行和各目录 status 为准，运行中不冒充完成。

全量运行窗口计数和安全检查（两个缓存 Block 的总窗口计数；必须/普通按第一个 Block）：

| 组 | 帧数 | 更新率 | 必须窗口 | 普通窗口 | 已识别必须事件漏选 | 缓存提交错误 |
| --- | --- | --- | --- | --- | --- | --- |
| Native_full | 6019 | 1.0000 | 0 | 0 | 0 | 0 |
| D3_full | 6019 | 0.6911 | 128435 | 96186 | 0 | 0 |
| NoM1Region_full | 6019 | 0.5912 | 95264 | 96876 | 0 | 0 |

完整逐类 TP errors、几何误差/未匹配对象和逐场景 Map 值见 JSON。全量 inference 的单次延迟也保留，但由于可能与上一组 CPU 评估并发，性能结论仍只使用独立重复实验。

附加数字格式诊断：旧数字提取器无法正确处理 `-.2` 及科学计数法。旧结果完全保留；新增解析器重新执行在真实回答/参考上，结果如下。本批次整体数目匹配及 MAE 没有改变；等数字数量仍不保证实体对齐，不能视作数值回答正确。

| 组 | 不同数值问题数 | 数字数量匹配数 | 条件文本数字 MAE | 有效 MAE 问题数 |
| --- | --- | --- | --- | --- |
| Native | 9 | 0 | 未测/无定义 | 0 |
| D3 | 9 | 0 | 未测/无定义 | 0 |
| NoM1Region | 9 | 1 | 5.6917 | 1 |
| NoM1RegionMatched | 9 | 0 | 未测/无定义 | 0 |

## 瓶颈和真实计时

优化：选中窗口 QKV/投影，真实有效 token MLP；全刷新原生 Block；CPU 批量区域映射；复用 GPU 执行索引更新年龄；M1 只导出计算契约；来源摘要直接绑定 ndarray 字节/dtype/shape。保持事务原子发布、张量所有权、有限值检查、缓存来源/年龄和 Block 2 全局依赖屏障。数值政策版本 v2 使旧成本证据失效，正式批准仍为空。

单帧离线 profiler（同一真实 transition；插桩结果不纳入普通延迟样本）：

| 范围 | 旧 CPU 含等待 ms | 旧 GPU 归属 ms | 新 CPU 含等待 ms | 新 GPU 归属 ms |
| --- | --- | --- | --- | --- |
| m2/block0_input | 0.757 | 4.174 | 1.894 | 4.205 |
| m2/computation_admission | 7.962 | 0.000 | 9.067 | 0.000 |
| m2/select_regions | 23.789 | 51.775 | 5.585 | 2.885 |
| m2/dependency_plan | 0.153 | 0.017 | 0.165 | 0.255 |
| prefix/0/qkv | 0.130 | 4.001 | 0.165 | 2.906 |
| prefix/0/attention | 0.223 | 0.442 | 0.254 | 0.443 |
| prefix/0/output_projection | 0.039 | 1.339 | 0.048 | 0.930 |
| prefix/0/mlp | 0.352 | 9.368 | 0.141 | 5.742 |
| prefix/1/qkv | 0.094 | 4.014 | 0.125 | 4.503 |
| prefix/1/attention | 0.170 | 0.443 | 0.195 | 0.444 |
| prefix/1/output_projection | 0.054 | 1.346 | 0.046 | 0.927 |
| prefix/1/mlp | 0.218 | 9.594 | 0.292 | 6.202 |
| cache/commit | 2.907 | 1.726 | 2.956 | 1.721 |

前缀只占 2 个 Block，后续 22 个 Block 仍完整执行；不能忽略屏障获得更多可缓存层。QKV/MLP 有可测节省，但事件/来源处理、缓存复制以及后缀全量网络限制端到端收益。原 425–436 ms 的 publication 范围包含等待已排队后缀 GPU 工作，不能当作缓存拷贝成本。预先同步后单测 StateCache.commit：旧中位 2.7248 ms，新 2.6812 ms；profiler 中 cache/commit 的活跃 GPU 归属约 1.7 ms。CPU execution 范围约 40–54 ms 也只是提交/同步范围，不能与 425 ms 简单相加成计算分解。

数据复制取自 trace 实际 memcpy 事件：

| 单帧 trace | 类型 | 次数 | 字节 | GPU copy ms |
| --- | --- | --- | --- | --- |
| before_profile_run02 | Memcpy HtoD (Pageable -> Device) | 20 | 12289350 | 0.680 |
| before_profile_run02 | Memcpy DtoD (Device -> Device) | 81 | 315515766 | 0.841 |
| before_profile_run02 | Memcpy DtoH (Device -> Pageable) | 211 | 3663 | 0.019 |
| visual40_final | Memcpy HtoD (Pageable -> Device) | 8 | 12288716 | 0.680 |
| visual40_final | Memcpy DtoD (Device -> Device) | 16 | 393004736 | 0.959 |
| visual40_final | Memcpy DtoH (Device -> Pageable) | 18 | 1761 | 0.013 |

D2H 小拷贝次数由 211 降至 18，CPU StreamSynchronize 由 231 降至 26；但 D2D 总字节增加（compact scatter/stride/所有权副本），没有宣称消除全部复制。此表只含 memcpy 事件，不含以 kernel 执行的 stride/索引拷贝；`aten::copy_` 的嵌套 GPU 归属旧约 32.97 ms，新约 37 ms，不能当作相互独立的额外时间相加。必要 cache clone 保留，不能为速度破坏事务。kernel 总时长与 CPU/CUDA event 跨度分别保存，嵌套 operator 时间不叠加。

来源元数据 list comprehension 的 cProfile 累计 CPU 时间：旧约 0.326 ms，新约 0.348 ms（各 2 次）；并非 425 ms 的来源。缓存读取/合并、clone/stride 拷贝的成本包含在 executor 和 trace 的相应算子中，不能与其父范围重复计算。完整请求真实 allocator 峰值如下，缓存独立所有权副本没有被省略：

| 40 帧组 | peak allocated MiB | peak reserved MiB | cache MiB |
| --- | --- | --- | --- |
| Native_r0 | 15713.56 | 16120.00 | 0.00 |
| D3_r0 | 16106.84 | 16614.00 | 173.84 |
| NoM1Region_r0 | 16030.31 | 16510.00 | 173.84 |
| NoM1RegionMatched_r0_retry01 | 16099.51 | 16614.00 | 173.84 |

真实连续视觉延迟（排除首帧、每帧统计后汇总）：

| 版本 | 重复/帧 | D3 更新率 | Native P50 | Native P95 | D3 P50 | D3 P95 | 配对平均延迟变化 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before | 3 | 0.7075 | 489.54 | 491.40 | 505.24 | 517.58 | 3.490% |
| after | 5 | 0.7089 | 487.39 | 488.32 | 490.56 | 500.07 | -0.002% |
| baseline_final | 5 | 0.7089 | 485.38 | 487.19 | 500.47 | 517.63 | 3.590% |

表中历史 before 来自初版 WSGRR 长测，原报告已注明最终基线随后补充了区域过期处理；它与 after 的重复数也为 3/5，不能作为本轮纯优化的严格因果对照。两者虽使用相同回放文件和 token，事件/窗口轨迹略有差异。未修改历史记录；严格 before 使用 `baseline_final40_control/`：归档任务开始时 commit `4756dfac5c673b1a02abab65d0c485e37f3b5558` 的原始 M1/M2，统一 40 帧、预热 2、重复 5 的当前计时边界，详见 `baseline_control_manifest.json`。若该组尚未完成，不宣称有严格 40 帧优化前后结论。初次隔离快照缺少相对配置目录，在计时开始前失败；失败目录与日志保留，最终视觉结果通过 `baseline_control_alias.json` 明确指向 `baseline_final40_control_retry01/`。

当前 after 平均几乎持平（差异远低于噪声），中位/P95 仍慢，不能宣称稳定视觉净加速。含 M1 和 VLM 的完整请求 P50 也仍高于 Native。生成长度变化使请求尾延迟变化，不能据此归功于视觉。完整请求及视觉边界从未缩小来制造正收益。

M1 Precheck、状态读取、区域评分、窗口/后缀 CPU 范围和 publication CPU 范围的 P50/P95 见 `visual_before_after.json`；真实请求阶段见 `measured_summary.json`。M1 export cProfile 也归档于 `visual40_final/m1_export_cpu.txt`，主要剩余成本是来源元数据绑定，不能因其昂贵而跳过校验。

### 完整语言请求逐阶段实测

| 组 | 阶段 | P50 ms | P95 ms | 计时含义 |
| --- | --- | --- | --- | --- |
| Native | eva_native | 30.19 | 30.31 | CPU 墙钟，含等待 |
| Native | detection | 7.37 | 7.88 | CPU 墙钟，含等待 |
| Native | map | 7.75 | 8.11 | CPU 墙钟，含等待 |
| Native | prefill | 102.93 | 128.51 | CPU 墙钟，含等待 |
| Native | decode | 424.37 | 4314.11 | CPU 墙钟，含等待 |
| Native | native_generation | 990.68 | 8373.02 | CPU 墙钟，含等待 |
| Native | eva_native | 477.57 | 480.88 | CUDA event 跨度，非活跃 kernel 总和 |
| Native | detection | 14.35 | 14.59 | CUDA event 跨度，非活跃 kernel 总和 |
| Native | map | 28.69 | 29.18 | CUDA event 跨度，非活跃 kernel 总和 |
| Native | prefill | 182.57 | 223.24 | CUDA event 跨度，非活跃 kernel 总和 |
| Native | decode | 793.21 | 8061.52 | CUDA event 跨度，非活跃 kernel 总和 |
| Native | native_generation | 990.69 | 8373.02 | CUDA event 跨度，非活跃 kernel 总和 |
| D3 | m2_total | 506.59 | 564.81 | CPU 墙钟，含等待 |
| D3 | detection | 7.35 | 7.51 | CPU 墙钟，含等待 |
| D3 | map | 7.67 | 8.01 | CPU 墙钟，含等待 |
| D3 | m1_commit | 77.13 | 101.32 | CPU 墙钟，含等待 |
| D3 | prefill | 99.15 | 124.46 | CPU 墙钟，含等待 |
| D3 | decode | 436.91 | 3607.94 | CPU 墙钟，含等待 |
| D3 | native_generation | 1008.40 | 7034.43 | CPU 墙钟，含等待 |
| D3 | m2_total | 506.62 | 564.84 | CUDA event 跨度，非活跃 kernel 总和 |
| D3 | detection | 14.26 | 14.41 | CUDA event 跨度，非活跃 kernel 总和 |
| D3 | map | 28.29 | 28.61 | CUDA event 跨度，非活跃 kernel 总和 |
| D3 | m1_commit | 77.18 | 101.37 | CUDA event 跨度，非活跃 kernel 总和 |
| D3 | prefill | 178.88 | 219.28 | CUDA event 跨度，非活跃 kernel 总和 |
| D3 | decode | 816.73 | 6745.27 | CUDA event 跨度，非活跃 kernel 总和 |
| D3 | native_generation | 1008.41 | 7034.42 | CUDA event 跨度，非活跃 kernel 总和 |
| NoM1Region | m2_total | 498.31 | 517.81 | CPU 墙钟，含等待 |
| NoM1Region | detection | 7.32 | 7.61 | CPU 墙钟，含等待 |
| NoM1Region | map | 7.53 | 7.96 | CPU 墙钟，含等待 |
| NoM1Region | m1_commit | 75.84 | 112.01 | CPU 墙钟，含等待 |
| NoM1Region | prefill | 99.07 | 124.34 | CPU 墙钟，含等待 |
| NoM1Region | decode | 436.63 | 4315.21 | CPU 墙钟，含等待 |
| NoM1Region | native_generation | 1007.96 | 8374.70 | CPU 墙钟，含等待 |
| NoM1Region | m2_total | 498.33 | 517.84 | CUDA event 跨度，非活跃 kernel 总和 |
| NoM1Region | detection | 14.27 | 14.46 | CUDA event 跨度，非活跃 kernel 总和 |
| NoM1Region | map | 28.18 | 28.61 | CUDA event 跨度，非活跃 kernel 总和 |
| NoM1Region | m1_commit | 75.90 | 112.07 | CUDA event 跨度，非活跃 kernel 总和 |
| NoM1Region | prefill | 178.91 | 219.20 | CUDA event 跨度，非活跃 kernel 总和 |
| NoM1Region | decode | 816.32 | 8068.87 | CUDA event 跨度，非活跃 kernel 总和 |
| NoM1Region | native_generation | 1007.97 | 8374.72 | CUDA event 跨度，非活跃 kernel 总和 |

不同范围可能嵌套；CUDA event 跨度可能包含 CPU 提交间隙，不能将此表各行相加。早期 observer 的 precheck 覆盖差异见可复现条件；源码内置 M1 CPU 组件和逐帧计数仍归档。

### 任务开始时 Git 基线与最终代码：完整请求控制

| 版本/组 | 请求 P50 ms | 请求 P95 ms | 模型 P50 ms | M1-M2 读取 P50 ms | M2 内部墙钟 P50 ms | 生成 token 均值 | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BaselineNative | 1724.09 | 9126.99 | 1517.51 | 未测/无定义 | 未测/无定义 | 75.55 | 0.3593 |
| BaselineD3 | 2114.98 | 8597.04 | 1924.93 | 60.88 | 501.81 | 71.03 | 0.3687 |
| Native | 1708.80 | 9108.50 | 1516.11 | 未测/无定义 | 未测/无定义 | 75.55 | 0.3593 |
| D3 | 2106.76 | 8106.82 | 1857.78 | 24.25 | 482.80 | 70.55 | 0.3637 |

每组 40 帧 × 2 次；相同主请求计时边界、输入、问题、种子、预热和生成参数。早期短测尚未记录 observer precheck，补充基线采用修正后的只读阶段观测；主计时与同步边界不变，不声称插桩逐行完全相同。Baseline 仅将 M1/M2 还原到任务开始时的精确 Git 源码，M3 执行关闭；未修改当前生产代码。差值及逐帧配对记录见 `controlled_request_before_after.json`，生成长度差异不能视作视觉加速。

Native 优化前后 80/80 帧感知与回答完全一致。D3 相同种子的 80 个请求中，感知 digest 完全相同 2 个、回答相同 72 个；digest 不同不衡量差异幅度。D3 Token F1 0.36869085 → 0.36368929，差值 -0.00500156。这些实际输出变化不应表述为无损优化。

同一 40 帧优化前后真实 GT 回归：

| D3 版本 | 子集 mAP | 子集 NDS | 子集 DET_l |
| --- | --- | --- | --- |
| Git 基线 | 0.26511170 | 0.30629305 | 0.24259146 |
| 最终代码 | 0.26506062 | 0.30624701 | 0.24389519 |

此表是官方程序在相同 40 帧上的回归，不能外推全量集，也不自动批准 v2 的数值/任务误差。

优化后−Git 基线：mAP -0.00005108，NDS -0.00004605，DET_l 0.00130373。没有把小幅检测回归四舍五入为无损。

## 修改文件与验证

| 文件 | 实际目的 |
| --- | --- |
| worldstream/m2/window_executor.py | 选中窗口点式计算、有效 token MLP、padding/stride 保真，版本化数值政策 |
| worldstream/m2/runtime.py | 离线 NoM1 控制、同窗口数控制、GPU 索引复用、计算状态导出及真实工作量统计 |
| worldstream/m2/region_refresh.py | CPU 批量区域映射、一次读取选择数据；离线目标数选择仍保留安全约束 |
| worldstream/m1/state_manager.py | M2 最小独立计算快照，保留 pending/source 校验 |
| worldstream/m1/state_store.py | 完整元数据摘要改用数组内容哈希，避免列表化复制，仍检测原地修改 |
| tools/worldstream_m2/task_diagnostics.py | 旧/新匹配并列、逐线逐对象数据、状态归因及可视化 |
| tools/worldstream_m2/task_quality.py / map_eval.py / official_detection_audit.py / official_map_audit.py | 真实 GT 官方评估、子集/全量边界、逐对象官方匹配审计、逐场景质量和原函数内容绑定并行缓存 |
| tools/worldstream_m2/profiling.py / window_validate.py | CPU/GPU/复制/等待剖析及重复视觉计时 |
| tools/worldstream_m2/followup_summary.py / executor_regression.py | 从实际日志汇总、隔离真实 EVA 执行舍入与时间复用误差 |
| tools/worldstream_m123/benchmark/worker.py / observe.py | 公平离线任务组、反事实、感知-only正式推理、实际 M1 precheck 计时及独立官方评估 |
| tests/worldstream_m2/test_task_diagnostics.py / test_wsgrr.py | 坐标、点序/点数/无效点、类别、远距离配对、真实偏移、官方评估等价、预算/安全、compact 运算形状和快照所有权 |
| tests/worldstream_m1/test_state_store.py / tests/worldstream_m123/test_benchmark.py | 摘要内容/dtype/shape/object 修改检测；实际 precheck 计时观测 |
| tests/worldstream_m2/test_execution.py / test_feature_boundary.py / test_window_local_admission.py | 工作量断言对应真实选中执行；partial 的 bitwise 断言改用项目已有 3e-6 舍入容差，未改变任务评估阈值 |
| docs/worldstream_m2_window.md / worldstream_m2_followup.md | 数值政策、离线控制、评测边界、复现及插桩变更表 |

完整插桩清单/开关和代码定位见 `../../docs/worldstream_m2_followup.md`；trace、CPU profile 和 operator 表均保留。无需在生产代码中启用 profiler。

最终 GPU 回归 exit=0：`320 passed, 19 warnings in 8.09s`。原 297 项范围与新增项一起运行，日志 `tests_gpu_final_verification.log`；此前 320 项通过日志仍在 `tests_gpu_final.log`。

真实权重、同输入新鲜缓存的独立执行舍入回归：

| Block | 选择 | 局部 max abs | 最终视觉 mean abs | 最终视觉 max abs |
| --- | --- | --- | --- | --- |
| 0 | none | 0.00000000 | 0.00000000 | 0.00000000 |
| 0 | one_interior | 0.00051498 | 0.00004588 | 0.24181557 |
| 0 | padded_edges | 0.00065207 | 0.00009421 | 0.69119072 |
| 0 | half | 0.00000763 | 0.00027970 | 0.89028168 |
| 0 | all | 0.00000000 | 0.00000000 | 0.00000000 |
| 1 | none | 0.00000000 | 0.00000000 | 0.00000000 |
| 1 | one_interior | 0.00081027 | 0.00004418 | 0.28060150 |
| 1 | padded_edges | 0.00082037 | 0.00008821 | 0.36463928 |
| 1 | half | 0.00000429 | 0.00025654 | 0.41469765 |
| 1 | all | 0.00000000 | 0.00000000 | 0.00000000 |

Fresh exact native cache and unchanged current input: isolates execution rounding, excludes temporal reuse error. No task approval or new tolerance granted.

真实权重部分执行的局部最大差异约 8.20e-4，经过后续全局网络可放大到约 0.8903（特征值单位，不是米）。这里使用新鲜 Native 缓存，没有时间陈旧误差；不能将小型单元测试的 3e-6 舍入容差外推成真实整网的容差，也不把 exit=0 解释为部分执行与原生位级等价。真实任务影响另由 Native 对照及 Git 基线 D3 子集回归衡量，v2 不获得自动批准。

精确 Git 基线执行器与最终执行器的同输入数值对照（均使用新鲜 Native 缓存）：

| Block | 选择 | 旧局部 max | 新局部 max | 旧最终 mean/max | 新最终 mean/max |
| --- | --- | --- | --- | --- | --- |
| 0 | none | 0.00000000 | 0.00000000 | 0.00000000/0.00000000 | 0.00000000/0.00000000 |
| 0 | one_interior | 0.00000000 | 0.00051498 | 0.00000000/0.00000000 | 0.00004588/0.24181557 |
| 0 | padded_edges | 0.00000000 | 0.00065207 | 0.00000000/0.00000000 | 0.00009421/0.69119072 |
| 0 | half | 0.00000000 | 0.00000763 | 0.00000000/0.00000000 | 0.00027970/0.89028168 |
| 0 | all | 0.00000000 | 0.00000000 | 0.00000000/0.00000000 | 0.00000000/0.00000000 |
| 1 | none | 0.00000000 | 0.00000000 | 0.00000000/0.00000000 | 0.00000000/0.00000000 |
| 1 | one_interior | 0.00000000 | 0.00081027 | 0.00000000/0.00000000 | 0.00004418/0.28060150 |
| 1 | padded_edges | 0.00000000 | 0.00082037 | 0.00000000/0.00000000 | 0.00008821/0.36463928 |
| 1 | half | 0.00000000 | 0.00000429 | 0.00000000/0.00000000 | 0.00025654/0.41469765 |
| 1 | all | 0.00000000 | 0.00000000 | 0.00000000/0.00000000 | 0.00000000/0.00000000 |

此对照隔离时间复用误差：旧执行器全部 10 种真实输入 case 与 Native 的差异精确为 0，最终执行器的部分执行出现上述非零差异，确认本轮新算术形状引入了数值误差。这不是缓存陈旧误差，也没有通过原有位级等价要求。源码与结果见 `executor_regression_baseline/` 和对应执行器 hash。没有额外授予数值容差或生产批准。

原始失败（早期 profiler 关键字签名、Matched callback 捕获、测试 stub、隔离基线相对配置路径）和重试目录均保留，不覆盖失败记录。它们已修复，是否全部验收以最终日志和全量评估状态为准。

## 研究目标状态

- 异常归因、旧新指标并列、可复核逐对象/逐线状态、同预算与同窗口数消融、6019 帧正式 Detection/Map 及逐线审计、重复性能剖析和最小执行优化已落实。
- 当前连续视觉没有稳定净加速；含 M1 的完整请求更慢。不能以复用率或理论 FLOPs 替代吞吐/延迟结果。
- D3 在 40 帧上优于 NoM1 控制；完整 6019 帧 Detection mAP 相对 Native 下降约 10.48%，精度保持目标不通过，不批准为生产策略。M1 区域引导的全量收益应读取正式配对分数及逐场景分布。
- 新数值路径未保持旧执行器的位级等价，40 帧出现小幅 Detection/Token F1 回归；它是未认证研究实现，不通过数值与任务质量生产准入。完整 VLM 验证、独立世界事件漏检率和学得地图拓扑能力仍未达标；不自动批准任何配置。
