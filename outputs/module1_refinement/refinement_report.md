# M1 第二轮实现与实测报告

本轮完成在线证据模块、真实回放、消融、局部掩码、性能和特征 Oracle 验证。结果支持“存在可解释的局部图像连续性”，但不支持安全神经特征复用或已满足 5% 物理错误风险。默认原配置保持旧严格行为；新增配置仅用于 observe_only 研究。

## 第一轮根因

详见 [diagnosis.md](diagnosis.md)。首场景 468→363（运动）→362（投影）→327（外观）→0（可见性）。旧可见性日志覆盖了遮挡子原因，消融仍有 212 个 Anchor 被投影遮挡约束拒绝。第二场景 146 个可评分状态→140 同实例存在→125 有传感器支持→34 Visibility=4→1 中心误差≤1m；运动误差与低可见性同时存在。
旧支持掩码缺失使全部像素请求更新。旧 inf 残差还混入 changed_regions，因此不能从旧日志声称全部 Unknown 是观测变化。新实现只在实际局部对应里将显著残差记为 Changed，其余未支持像素保持 Unknown。

## 方法与代码

- `worldstream/m1/temporal_visibility.py`：上一实际 commit 的灰度图→Shi–Tomasi→双向金字塔 LK→FB≤0.75px、LK误差≤12、碰撞排除。每帧最多1600点/相机，保留一张历史灰度图，不链接未经计算的历史。
- `geometry_support.py`：实际相机时间位姿的双向极线距离≤1px、三角化正深度和0.5–200m范围；近纯旋转使用旋转射线投影。局部7×7像素支持需纹理、≥90%像素残差/梯度一致；只保留实际通过的像素，不填充Box或无支持Cell。
- Anchor 验证保留旧几何、外观、运动与投影遮挡约束，另需≥4支持点、≥80%选点通过、覆盖≥3/4象限，匹配受原投影搜索上限约束。来源是同一RGB传感器，不能称独立可见性证明；没有物体前景归属认证。
- 未知动态目标不因光流一致而被认定运动可靠。历史动态/潜在遮挡区域不参与静态背景解释；新视野、失败对应和未知遮挡继续Invalid。
- `novelty_detector.py`输出 Stable/Changed/Unknown 像素证据与精确失效掩码，最终 Anchor 仍为二元。常规失效=Changed∪Unknown；周期完整刷新可覆盖额外Stable像素。M2输出保留 `transformer_dependency_certified=False`。
- `state_manager.py`缓存有效相机标定检查，`geometry.py`优化标量裁剪；原模型结构、权重、Head Memory及LLM路径不变。适配器仅增加拷贝耗时测量。

## 数据与协议

主评价：6个独立nuScenes验证Scene，各12个真实连续关键帧，共72帧；额外同场景12帧stride=2压力回放单列，不能算新的独立Scene。4个保留Scene各12帧用于校准，共48帧。全部按Scene分离，覆盖停车/交通锥、转弯雨天、路口/行人、摩托车、施工及夜间。预测来自既有完整检测JSON；这些CPU实验是 Historical Full-Model Prediction Replay，不是本轮重新执行模型。另有6帧 Annotation-Assisted Map Replay 和下文独立 Newly Executed GPU Inference。
协议与所有阈值在回放前写入 protocol.json；主评价从未用于选择阈值。当前GT只由离线评价器在precheck后读取；commit仅获取已保存预测（Map演示明确用历史标注）。物理Oracle沿用原标准；未匹配身份保持Unknown。

## 风险、覆盖与消融

| 方法 | 候选率 | 物理错接受率 | 恢复率 | 可评分/全部 | 失效面积 | 稳定面积 | 新区域Recall | 新区域平均Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry_only | 38.19% | 30.27% | 50.42% | 827/1495 | 100.00% | 0.00% | 100.00% | 100.00% |
| appearance_only | 35.45% | 31.08% | 47.92% | 827/1495 | 100.00% | 0.00% | 100.00% | 100.00% |
| old_strict | 0.00% | N/A | 0.00% | 827/1495 | 100.00% | 0.00% | 100.00% | 100.00% |
| old_appearance_ablation | 8.90% | 15.73% | 20.78% | 827/1495 | 100.00% | 0.00% | 100.00% | 100.00% |
| without_temporal_visibility | 8.90% | 15.73% | 20.78% | 827/1495 | 95.71% | 4.51% | 95.93% | 98.20% |
| without_local_support | 3.08% | 28.00% | 4.99% | 827/1495 | 100.00% | 0.00% | 100.00% | 100.00% |
| new_full | 3.08% | 28.00% | 4.99% | 827/1495 | 95.71% | 4.51% | 95.93% | 98.20% |
| new_full_tight_patch | 2.81% | 27.27% | 4.43% | 827/1495 | 96.56% | 3.53% | 98.26% | 99.19% |
| new_full_loose_patch | 3.21% | 28.00% | 4.99% | 827/1495 | 95.33% | 4.94% | 94.77% | 97.46% |

w/o Temporal Visibility 仅移除Anchor认证中的可见性门控，其局部支持仍需时序对应；这是消融，不能部署。Geometry/Appearance基线没有局部支持，失效请求按全图计算。所有方法使用相同图像、历史预测、初始状态和Oracle。
新方法可评分接受 25，物理错误 7，试验级95%区间 [0.12071668850406661, 0.49387682180625553]。涉及 5 个唯一GT实例；连续帧相关，试验级区间不能作为独立风险保证。未知身份率 44.68%。
错误接受的Oracle因素（可重叠）：`{"same_instance_missing": 0, "sensor_support_missing": 0, "visibility_not_4": 0, "center_error_gt_1m": 7}`。外观一致可能保留历史定位偏差；时序匹配不等于完整物体可见性，尤其Box包含背景。

| 独立评价Scene | 候选率 | 物理错误率 | 失效面积 | 新区域Recall |
|---|---:|---:|---:|---:|
| 01452fbfbf4543af8acdfd3e8a1ee806 | 6.37% | 28.00% | 90.05% | 78.57% |
| 01e4fcbe6e49483293ce45727152b36e | 0.00% | N/A | 99.90% | 97.44% |
| 07aed9dae37340a997535ad99138e243 | 0.00% | N/A | 99.50% | 100.00% |
| 080a52cb8f59489b9cddc7b721808088 | 0.00% | N/A | 85.35% | 100.00% |
| 0ac05652a4c44374998be876ba5cd6fd | 0.00% | N/A | 99.45% | 92.86% |
| 0dae482684ce4cd69a7258f55bc98d73 | 0.00% | N/A | 99.99% | 100.00% |

静态/动态车辆/行人、速度、Visibility、实际帧间隔、自车位移/转向分组及几何误差见 validity_metrics.json。第二场景的低Oracle阳性主要由可见性与运动误差叠加，不能解释为M1只需取消可见性。Map演示不具备持久GT身份或前景标签，其物理错误率保持N/A。

## 局部失效：改善与失败

平均 Stable=4.51%，Changed=0.00131%，Unknown=95.49%。最终Invalid=95.71%，包含周期完整刷新。新区域数 172，Recall=95.93%，平均区域覆盖=98.20%。
这证明局部图像支持不再无条件全图失效，但没有证明必要更新的语义覆盖已得到保证。离线新目标评价是投影Box区域而非前景分割，新Box中可能有原先可见背景；该限制不能用来忽略召回下降。真实稳定像素负例不足，稳定区域误失效率和语义变化FPR保持null。
M1-R(c)显示首场景实测支持最多相机所在帧和场景reset全图刷新帧；精确NPZ与最终二值PNG保存，展示不会修改掩码。网格regions是保守包络，会掩盖细小支持；最终M1Result已添加无损压缩exact_invalidation_mask，逐像素解码等价。后续M2需依赖闭包，实际可跳过面积只能更小。

## 校准与风险曲线

状态：`insufficient_evidence`，threshold=null。4个独立校准Scene少于预先设定的10个；没有把测试集选择的阈值写回配置。图M1-R(b)为预设运行点在测试Scene上的描述性曲线，Geometry横扫age cap、Appearance/Joint横扫描述残差。实例去重与Bonferroni上界见calibration.json，不把相关帧当成独立保证。

## Feature Oracle

在完整模型中Hook patch embedding、首个窗口块、首个全局块和最终EVA。历史/当前特征按共同通过双向与相机几何检查的实际像素对应进行双线性采样；无法对齐的状态被排除，不用零误差填充。

| 层 | 接受记录数 | 接受Relative L2 | 拒绝记录数 | 拒绝Relative L2 |
|---|---:|---:|---:|---:|
| backbone_final | 45 | 0.122 | 525 | 0.141 |
| global_block_2 | 45 | 0.128 | 525 | 0.173 |
| patch_embed | 45 | 0.051 | 525 | 0.071 |
| window_block_0 | 45 | 0.065 | 525 | 0.088 |

计数为相关的Anchor×Camera×Frame记录，不是独立样本。patch embedding为局部16×16卷积；首个Window Block已有跨Token依赖与位置编码；第2号Block为全局注意力，最终表示积累24层依赖。位置对齐仅用于离线误差比较，不能证明输入支持域相同。Object/Map Carrier仅报告分支均值池化差异，不按Token索引假设对象对应。
特征误差与观测残差的Spearman统计：`{"backbone_final": {"n": 570, "spearman_residual_feature_error": 0.08934430165160437}, "global_block_2": {"n": 570, "spearman_residual_feature_error": 0.5555355835220183}, "patch_embed": {"n": 570, "spearman_residual_feature_error": 0.7778894166934335}, "window_block_0": {"n": 570, "spearman_residual_feature_error": 0.695364146149236}}`。不同层的误差和Cosine见feature_oracle_metrics.json；没有下游任务质量或精确复用结论。

## 实际开销与不干扰验证

| 配置 | 优化前P50/P95 ms | 优化后P50/P95 ms | 优化后traced峰值MiB |
|---|---:|---:|---:|
| old_strict | 114.78/128.81 | 44.36/50.42 | 4.11 |
| new_full | 209.98/228.83 | 144.49/152.43 | 14.56 |

CPU基准：预载真实4帧，warmup=2，repeats=10，每次略去reset帧计时，30个测量样本/方法；分配跟踪独立执行，tracemalloc不含OpenCV池。所有重复以及优化前后候选ID和精确掩码摘要完全一致。热点证据见*_hotspots.txt；runtime图使用组件均值，P50/P95不可逐项相加。IO/模型与离线Oracle不计入该CPU微基准。

GPU：NVIDIA GeForce RTX 3090，torch 1.13.1+cu117 / CUDA 11.7。完整模型warmup=1、repeats=3、每次8帧，另8帧Feature Oracle，后者不计入延迟样本。

| 模式 | 整帧P50/P95 ms | M1含适配/传输P50/P95 ms | 测量帧数 |
|---|---:|---:|---:|
| baseline | [2508.6797029944137, 2516.379317746032] | None | 24 |
| old_strict | [2687.4969385098666, 2701.9502728246152] | [137.0950504206121, 142.25113217253238] | 24 |
| new_full | [2860.8528280165046, 2884.7351755481213] | [296.23572586569935, 306.43757700454444] | 24 |

全部重复的Backbone、检测/Map结果与Temporal Memory摘要一致：True。LLM执行路径完整，采样文本不作为视觉摘要相等条件。GPU峰值allocated/reserved、逐帧输入适配与GPU→CPU同步拷贝耗时见gpu/results.json；拷贝是适配时间子项，不能重复相加。
上表优先使用gpu_optimized中的最终版本重复实验；优化前gpu目录保留独立对照和Feature Oracle。CPU优化及精确mask导出已核对判定与掩码不变。GPU时间按模式顺序执行，包含原LLM与数值摘要Hook；不能用于声称M2加速。

关闭新增功能与第一轮原始4组日志的Anchor判定复现一致；新增单元测试覆盖FB、遮挡、边界、几何失败、像素分区、场景隔离、动态不确定性、版本与原推理入口不干扰。测试数量见unit_tests.xml。


## 补充核对与可复现材料

最终版本在全部 132 帧回放上，候选ID、NPZ精确mask、M1Result压缩mask解码全部一致。检查见optimization_full_replay_checks.json，接口实例见m2_example.json。
更细的重叠约束、未可测条件和独立遮挡诊断见[diagnosis_independent.md](diagnosis_independent.md)。它在离线测量中打破短路以观测其余条件；实际在线首失败仍使用diagnosis.md原始日志口径。

Feature Oracle可比较覆盖：`{"all_historical_anchor_trials": 691, "comparable_anchor_trials": 552, "accepted_anchor_trials": 45, "comparable_accepted_trials": 45, "overall_fraction": 0.7988422575976846, "accepted_fraction": 1.0, "unknown_not_compared": 139}`。可比较子集选择会引入偏差；未可比状态没有被赋予零误差。feature_coverage图仅描述一个GPU场景的预设阈值运行点，没有用于校准。

最终GPU版本的组件P50/P95（毫秒，输入/输出适配包含其传输子项，不相加）：

| 部分 | 旧严格M1 | 新M1 |
|---|---:|---:|
| projection | 8.87/10.74 | 8.99/10.58 |
| temporal_correspondence | 0.00/0.00 | 106.78/113.61 |
| visibility_verification | 0.00/0.00 | 1.32/1.51 |
| geometry_support | 0.00/0.00 | 25.50/27.85 |
| observation_validation | 30.71/37.88 | 37.76/45.88 |
| novelty_detection | 3.61/3.69 | 7.43/7.99 |
| state_update | 15.78/19.18 | 22.47/25.85 |
| association | 53.61/56.34 | 53.82/57.50 |
| input_adaptation | 14.90/15.35 | 14.76/15.70 |
| output_adaptation | 7.49/9.50 | 7.59/9.53 |
| m1_overall_wall | 137.10/142.25 | 296.24/306.44 |
| gpu_to_cpu_transfer | 3.35/3.49 | 3.34/3.47 |

| 模式 | 优化前在线M1 P50/P95 ms | 最终在线M1 P50/P95 ms | 最终峰值allocated GiB |
|---|---:|---:|---:|
| old_strict | 226.91/247.70 | 137.10/142.25 | 15.103 |
| new_full | 385.73/407.72 | 296.24/306.44 | 15.103 |

## M2边界、未解决问题和结论

保留精确mask、坐标系、来源、Stable/Changed/Unknown比率及保守regions；这些是具有审计来源的更新种子。可见性证据同源RGB、前景归属未知、检测定位误差、动态运动未可靠估计、支持的Token依赖扩张和低覆盖Scene仍是限制。建议下一步先增加独立Scene和可见前景/遮挡负例验证，在风险校准通过后研究局部浅层特征的真实依赖闭包。

**Q1：有条件地识别到非零具有在线图像证据的候选，但尚未证明是低风险物理有效状态。** 当前错误率高于5%目标，不能把候选认证表述成部署级可靠性保证。
**Q2：能产生非平凡局部失效区域，但尚未保证必要更新覆盖。** 对应支持减少了更新面积，新增目标区域召回下降与缺少像素负例仍需解决；不能称已安全完成局部更新。
**Q3：有局部、描述性的对应关系，尚不足以证明深层复用。** 本次接受组在四层的平均误差均较低，patch embedding为0.051 vs 0.071，最终Backbone为0.122 vs 0.141；但最终层误差与图像残差相关性只有0.089。单场景相关记录不足以外推，Carrier没有对象一一对应，也没有下游质量验证。
**Q4：理论上只有跳过足够视觉计算才可能收益，当前尚未证明净收益。** 新增M1成本必须与低覆盖掩码经依赖扩张后可省计算比较；本轮没有跳过EVA/Head，也没有把LLM整帧时延当成可复用视觉预算。
**Q5：可以进入受控的M2依赖分析和离线增量误差实验，但尚不具备直接开启真实在线跳算的条件。** 保持observe_only、threshold=null和transformer_dependency_certified=False。

复现命令见commands.sh；完整参数、场景与来源见protocol.json。第一轮outputs/module1未被覆盖。
