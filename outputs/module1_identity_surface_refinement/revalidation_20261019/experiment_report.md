# M1 身份/局部表面需求：本次工程补齐与真实复现

需求主体已由基线 `0e78092` 实现。本次保留全部旧报告、数据划分、冻结方法和默认开关，复核上传需求后修复可复现的证据/来源验证漏洞。源码修改仅涉及 M1；没有改变模型、Checkpoint、LoRA、图像预处理、原 Detection/Map 推理、M2/M3 或实验阈值。

## 本次结果

- 新增18项受控边界测试：修复前13失败、5已通过；修复后全部通过。此前存在 NaN 来源时间绕过、来源裁剪/轨迹空间支持未重算、负误差/相机/共同视图验证不足、畸形资格计数、跨相机历史与查询执行签名问题。受控正例不计真实物理成功。
- 宿主 CUDA 环境完整回归 **421 passed、0 skipped、0 failed**。初次沙箱 CPU 回归为420 passed、1 CUDA-only skipped，两个原始XML均保留。
- 真实原始图像、原模型历史检测输出、独立离线 LiDAR/annotation，在20个已曝光 Development场景/240帧、4个已曝光Calibration场景/48帧上复跑五种方法和Full Refresh。身份测量/判定、区域评分、保守逐帧四类Mask与关联/证书结果全部与归档一致。没有把288帧当新的独立样本。
- 本次实际在空闲RTX 3090执行 **8帧原Checkpoint GPU对照**，outputs、backbone、head temporal memory均数值一致。初次nvidia-smi失败来自沙箱设备隔离，经只读宿主机核查后恢复设备可见性；未修改驱动或结束其他任务。
- Checkpoint SHA256：`6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`，本次重新计算，与原记录一致。

## 身份声明与覆盖

保留 A 严格RGB前景/静态射线路径。B 仅声明 **object_identity_continuity**，不推出静止、准确中心/速度、Box表面复用或地图稳定。证书绑定scene、anchor、source/current frame、geometry version、epoch、timestamp、图像digest、执行签名、生产器协议、资格artifact SHA256与claim scope。

| 复现分组 | 方法 | 几何候选 | 方法候选 | 可评分正确 | 错误候选 | 不可评分候选 | 正式认证 |
|---|---|---:|---:|---:|---:|---:|---:|
| development | strict_foreground | 5798 | 2 | 1 | 0 | 1 | 0 |
| development | selective_identity | 5798 | 175 | 144 | 0 | 31 | 0 |
| calibration | strict_foreground | 683 | 0 | 0 | 0 | 0 | 0 |
| calibration | selective_identity | 683 | 0 | 0 | 0 | 0 | 0 |

144个正确候选链接只涉及4个可评分支持场景，其条件场景事件CP95上界约52.713%；31个不可评分候选不能当正确。候选条件Precision不是物理认证准确率。Calibration仍无候选；不能用旧Heldout的2个正确链接替代失败的冻结选择，或重选方法。正式接受为0，正式FAR和上界均为null。

主要在线拒绝包括身份竞争、观测缺失/冲突及超出固定0.5m选择人口；计数可能重叠，不能相加充当互斥漏斗。完整关联、错误几何身份、2m截断、无GT人口、fragmentation、缺帧和候选测量保存在replay目录逐场景trace，摘要见identity_candidate_diagnostics.json、identity_recovery_results.json和identity_false_acceptance_cases.json。

资格仍要求场景/采集log为主要独立单位：任一错误接受使该单位失败。即使零错误，也至少需要59个完整可评分的独立接受单位，单侧95%上界才可能≤5%。历史选择后仅8个新log，理想全部可接受可评分且零错时仍至少缺51个；实际并非8个全合格单位，不能简单声称只补51个即可批准。本次全部场景已曝光，新独立资格单位为0。Train Population不能冒充模型独立泛化测试。

## 表面错误与区域结果

本次精确复现旧Local R12的169个错误稳定点：46个低视差，123个主要物理原因未知。相似纹理、表面切换、背景跟踪和参考误差不能单凭现有RGB闭合区分。低视差两射线可为错误对应给出自洽三维点；重投影通过不能当独立表面归属真值。

三帧表面原型保持原观测门槛和动态扫过保护。本次只新增历史相机/坐标空间/可靠定时的来源检查，正常输入数值不变。未进行Radius Grid、加权Score或新网络实验。缺表面证明继续Unknown，而非因相似光度接受。

| 复现分组 | 方法 | 有效稳定面积 | 事件召回 | 关键点召回 | 稀疏错误/接受 | FAR | 接受scene |
|---|---|---:|---:|---:|---:|---:|---:|
| development | conservative | 0.331562% | 99.969231% | 99.980976% | 9/306 | 2.9412% | 2 |
| development | dense_original | 4.406813% | 99.046154% | 99.228460% | 481/2939 | 16.3661% | 6 |
| development | local_r12 | 1.248860% | 99.938462% | 99.802993% | 169/1336 | 12.6497% | 3 |
| development | surface_observable_ablation | 0.091166% | 100.000000% | 99.978862% | 182/645 | 28.2171% | 2 |
| development | surface_visibility | 0.001666% | 100.000000% | 100.000000% | 0/0 | null | 0 |
| development | full_refresh | 0.000000% | 100.000000% | 100.000000% | 0/0 | null | 0 |
| calibration | conservative | 0.000220% | 100.000000% | 100.000000% | 0/0 | null | 0 |
| calibration | dense_original | 0.099596% | 100.000000% | 99.858444% | 6/6 | 100.0000% | 1 |
| calibration | local_r12 | 0.052948% | 100.000000% | 99.954959% | 0/0 | null | 0 |
| calibration | surface_observable_ablation | 0.080598% | 100.000000% | 99.926005% | 6/6 | 100.0000% | 1 |
| calibration | surface_visibility | 0.002660% | 100.000000% | 100.000000% | 0/0 | null | 0 |
| calibration | full_refresh | 0.000000% | 100.000000% | 100.000000% | 0/0 | null | 0 |

表面原型可评分稳定接受为0，FAR=null；虽关键召回为100%且有极少量未评分稳定面积，也不构成有用风险收益。Conservative仍只有2个开发接受scene，未达到正式完整条件；其他局部/稠密路径错误风险仍高。不得以全Unknown/Full Refresh或稀疏之外未标注像素证明成功。

原始逐点flow、FB、光度、梯度、纹理、Jacobian、pose、三角化、重投影、visibility和Oracle信息均保留。本次额外为全部169个Local R12错误点写入实际Source/Current RGB采样和PNG来源哈希，见 [逐点RGB证据](traces/local_r12_rgb_points.jsonl)；11张真实RGB输入PNG见visualizations/error_rgb/。八面板真实RGB/对应/Stable/Unknown/Refresh/错误位置/独立参考见replay_development/visualizations/。未提供完整实例像素Mask，没有把3D Box全部当作前景。

## Commit 时序闭环

`commit()` 仍返回原M1Result。当前证书通过只读 `get_current_certifications()` 查询，只在Store原子提交成功后可见。Precheck不消费当前Detection，也不反填后验认证；其字段保持Precheck时间语义。pending/异常只可能看到此前已提交状态；新几何、frame、scene、epoch、timestamp或已提交执行签名不匹配均不能读取旧证书。返回对象及防御性副本不能回写Store。正常/异常/替换/reset/声明不扩张均由受控完整证据与模拟资格验证。

## 单模块成本

固定8个已曝光开发驾驶帧预加载，对照HEAD完整源码快照和本次源码，交替运行；1次预热、3次计时，排除启动帧。每路径/版本21个相关单帧计时样本，完整原始值见runtime_cost_breakdown.json。

| 路径 | HEAD P50 ms | 当前 P50 ms | 当前 P95 ms |
|---|---:|---:|---:|
| conservative | 148.90 | 146.98 | 163.96 |
| identity_only | 214.45 | 222.90 | 248.09 |
| strict_only | 249.10 | 248.41 | 257.61 |
| both_producers | 317.66 | 317.79 | 338.85 |
| surface_identity | 467.73 | 469.04 | 496.19 |

本表包括Precheck+Commit，排除图像IO、模型推理、输出adapter、Oracle与日志；与旧报告含adapter的指标不能直接混比。960组逐相机Mask完全相等，身份测量/判定亦相等。P50小幅差异不能当显著加速；身份/表面原型仍有成本。本次没有因无质量收益而扩大P1优化，也没有用降低必要证据条件换取速度。

## 20项验收回答

1. 原严格前景候选少：来源可用、视差、双OBB、空间支持、背景环深度分离和跨视图静态一致性共同限制。
2. 有真实独立身份标签支持部分同实例链接，见开发144个正确候选；其余未评分人口不自动填正确。
3. 有条件候选覆盖，未取得非零正式可信身份认证；Calibration失败保留。
4. B只证明经资格批准后的身份连续性；A保留有限静态射线声明；都不能扩大为完整三维状态。
5. 正式身份FAR/上界null，不满足冻结5%资格；候选条件风险单独报告。
6. 独立接受场景不足，重复帧/实例不计独立样本。
7. 仍缺完整且可靠的所选人口身份证据及足量新log；表面/精确几何另缺独立观测参考。
8. Post-Commit查询闭环有效，受控证书可读，真实无资格时为空。
9. Precheck仍无当前完整Detection输入且不被后验回填。
10. 46点可确认低视差，123点根因Unknown，未主观标成前景/背景或表面切换。
11. 表面原型减少错误数，却没有可评分覆盖，不能据此判优化成功。
12. 有极少非零未评分面积，未形成满足风险约束的有用覆盖收益。
13. 本次各方法事件召回见表，保持原90% ROI事件定义及99.8%门槛。
14. 关键点召回与事件分别保留，未改分母。
15. 稀疏可评分风险与描述性CP上界均保存；零接受为null，未认证全图。
16. 表面路径无独立可评分接受scene，正式区域资格不足。
17. 实际同机CPU成本如表，GPU感知数值不变；原始逐对象/帧耗时均保存。
18. 本次仅提前拒绝违反必要来源条件的输入；288帧候选测量/判定一致，没有丢失原本正常候选，不能扩张到未验证人口保证。
19. 本次实际8帧GPU感知、backbone、head memory一致，完整421项回归通过，checkpoint哈希一致。
20. 可继续作为observe-only工程基础，不能正式批准身份/表面结果供后续模块消费。

## 六项独立结论

- **Engineering Go**：observe-only工程边界、完整回归、实际GPU数值对照通过。
- **Identity Feasibility Negative for frozen independent confirmation**：仅开发候选可行性，独立冻结选择未通过。
- **Physical Certification No-Go**：Qualification Insufficient，正式认证0，FAR=null。
- **Region Feasibility Negative**：未获得满足风险约束的可评分覆盖收益。
- **Region Validation No-Go**：不可用全Unknown、未标注面积或有限条件Precision批准。
- **Runtime Cost No-Go for deployment**：成本已测，无资格收益/部署加速证据。

本次无新训练、新网络或阈值开发，不证明所有轻量方法都不可能。下一步最小数据需求仍是完整独立身份与表面参考及足量未曝光scene/log。原Heldout/Final结果和原六项结论均只读保留于上级目录；本目录不是新的模型泛化或独立质量测试。

可复现信息：commands.sh、source_manifest.json、source_changes.patch、effective_configuration.json、checkpoint.sha256、environment.json、两组execution_manifest.json、regression_gpu.xml、gpu_parity/、各replay目录与isolated_timing/。
