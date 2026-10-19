# M1 几何自适应观测与联合消歧实验

正式结论：**No-Go**。本轮实际修改 M1 并完成 nuScenes 同一已曝光人口的实验；独立资格与研究性点估计分开报告。

基线 HEAD 为 `d95011a`。所有命令使用 OmniDrive 环境及 `taskset -c 0-3,6-31`；M2/M3、原模型、Checkpoint、训练和输入预处理未修改。历史文件哈希全部一致。

## 先诊断，再实现

冻结人口复现 1314 正确/230 错误候选，源验证后 177/80。1137 个损失正确点中的测量不足标记为 {'measured_parallax_or_triangulation_insufficient': 851, 'observation_missing': 467, 'measured_texture_support_insufficient': 647, 'Unknown_Root_Cause': 4}。这些标记有重叠，反映观测条件，并非物理根因标签。未确认的遮挡、深度边界、曝光及真实斜面原因继续 Unknown。

旧检查确实将中心深度扩展到整个邻域。但不能据源码把所有损失归因于前平行假设。局部 LK 诊断没有获得可靠的正确中心及全邻域联合支持，不能据此确认或否认真实斜面。

仅新增一个局部逆深度平面：以中心实测深度锚定，四角真实对应拟合斜率，四个空间分离轴向点留出验证，随后按原光度/纹理/梯度阈值检查实际投影 RGB。旧前平行路径作为明确回退和消融保留；无证据不扩张 Patch、ROI 或表面面积。

联合消歧逐个保留实际支持的竞争源射线。只有 H0 的第三图像正支持、直接双向 LK 与局部留出、源回投闭环、候选投影落在实测局部范围并与实测射线冲突时，才排除 Hi。任一实际第三视角仍支持 Hi 时保留竞争；观测缺失、离屏或 Patch 不匹配本身都不能排除。该证据仍有 RGB 相关性，不是物理所有权认证。

新局部模型源验证：{'correct': {'before': 1314, 'old_source': 177, 'new_source': 177, 'actual_geometry_applicable': 100, 'recovered_source': 0}, 'false': {'before': 230, 'old_source': 80, 'new_source': 80, 'actual_geometry_applicable': 23, 'recovered_source': 0}}。

## 同人口方法对照

| 方法 | 有效面积 % | 正确/错误点 | Sparse FAR % | 最差精确 ROI 刷新 % | 接受 Log |
|---|---:|---:|---:|---:|---:|
| full_refresh | 0.000000 | 0/0 | N/A | 100.000000 | 0 |
| conservative | 0.331562 | 297/9 | 2.941176 | 87.782805 | 2 |
| dense_original | 4.406813 | 2458/481 | 16.366111 | 23.529412 | 6 |
| local_r12 | 1.248860 | 1167/169 | 12.649701 | 23.529412 | 3 |
| surface_local_plane | 0.017963 | 155/39 | 20.103093 | 99.833224 | 1 |
| surface_consensus | 0.325636 | 494/196 | 28.405797 | 61.111111 | 2 |
| surface_constraints | 0.003654 | 5/0 | 0.000000 | 99.970238 | 1 |
| geometric_adaptive_patch | 0.003655 | 5/0 | 0.000000 | 99.970238 | 1 |
| joint_multi_observation | 0.003654 | 5/0 | 0.000000 | 99.970238 | 1 |
| correspondence_recovery | 0.003655 | 5/0 | 0.000000 | 99.970238 | 1 |
| recovery_no_extra_observation | 0.016662 | 84/35 | 29.411765 | 99.583333 | 2 |
| recovery_h0_only | 0.010241 | 51/14 | 21.538462 | 99.861111 | 1 |
| recovery_discriminating_third_only | 0.000000 | 0/0 | N/A | 100.000000 | 0 |
| recovery_no_depth_order | 0.003662 | 5/0 | 0.000000 | 99.970238 | 1 |
| recovery_conservative_neighbourhood | 0.000412 | 1/0 | 0.000000 | 100.000000 | 1 |
| recovery_no_motion_sweep | 0.005455 | 5/0 | 0.000000 | 99.970238 | 1 |

历史六种方法为 SHA256 固定的相同输入结果，不冒充新运行。旧 Surface Constraints 则在本轮逐相机重算并核对面积、错误、召回和最差 ROI，全部一致。新方法及所有逻辑消融均使用本轮实际在线测量；邻域基底和动态扫掠消融各用独立 manager 完整重放。

最终与旧 Constraints 配对：{'correct_retained': 5, 'correct_lost': 0, 'new_correct': 0, 'wrong_retained': 0, 'wrong_blocked': 0, 'new_wrong': 0}；旧5个正确点保留 5。与 Consensus 的494正确/196错误配对：{'correct_retained': 1, 'correct_lost': 493, 'new_correct': 4, 'wrong_retained': 0, 'wrong_blocked': 196, 'new_wrong': 0}。全像素精确接受交集为 {'total_pixels': 331776000, 'common_pixels': 12121, 'new_only_pixels': 5, 'baseline_only_pixels': 2}。详见 region_paired_support_analysis.json。本轮没有恢复新增可评分正确点；极少像素面积变化不能解释为已证实的正确覆盖提升。

第三观测排除条件性假设 0 个；稀疏合法恢复正确 0、错误恢复 0、继续 Unknown 138。本轮没有排除假设，真实错误排除数因此也为0。替代射线本身缺少逐个独立物理真值，不能把未来的条件排除数直接称作真实错误解释数。

所有面积分母包括20 Scene / 20 Log / 240 Frame / 1440 Camera 和启动、周期强制刷新。Correct/False Stable 是稀疏可评分参考点，正确/错误/不可评分稠密面积均无法由此推出，保持 null。每阶段像素与稀疏人口分别记录在 surface_acceptance_funnel.json。

最终 Event Recall=100.000000%，Point Recall=100.000000%，最差精确 ROI Refresh=99.970238%。四个历史失败事件在 critical_roi_replay.json 中逐项核对，GT ROI 在 Precheck 后评分，未控制刷新，也未用32px Tile包络替代精确掩码。

非零四项描述性点约束通过：True；Sparse FAR 的单侧95%描述上界为 0.45071972834694096。原冻结风险 Gate 通过：False。同一 Log 内像素、帧及对象不增加独立样本数；零错误不能推出认证。

## 身份、原模型与独立资格

Temporal Object、Shadow、Anchor 继承和认证隔离保持。13,314个身份候选边逐边与历史结果一致：Proposal 288，正确217，错误0，不可评分71，唯一正确实例45，正式认证0。StateStore、身份生产器及 Physical Certifier 未修改。Stable 不代表 EVA Token 或 LLM Feature Reuse。

本轮完整 CUDA 回归：538/538 通过，失败0、跳过0。本轮8帧原模型 Perception Parity：passed，输出/骨干/记忆摘要一致=True；未执行LLM，未用旧GPU结果替代。

复用已有 nuScenes 审计并重新核对数据与 Checkpoint 哈希。68个有标注采集Log均已曝光，Test缺少本地物理评分标注，没有新增独立数据。Calibration/Heldout/Final Test = Not Run；qualified_certifications=0，qualified_FAR=null，identity_selection=null，region_selection=null。按Log交叉验证仅是固定方法描述，不消除既有曝光。

## 科学问题与后续最小条件

前平行假设的局限存在，但这批损失主要伴随视差、纹理和投影可用性问题；本轮不能确认大量损失由斜面造成。自适应源验证恢复正确0、错误0，其最终收益必须结合完整配对和关键变化约束判断。第三观测的独立贡献由 joint_multi_observation 与旧 Constraints、完整路径与 geometric_adaptive_patch 直接比较，未通过时保留失败。

本轮不据失败继续堆叠评分器或多平面。Surface Consensus 作为对照保留，当前错误水平没有证明其作为正式接受路径的必要性。下一步需在同一nuScenes范围内获得具有可见局部二维纹理、足够视差且可直接可靠跟踪的第三采集；局部几何适用性与候选唯一性的观测缺口应分别解决。正式资格还需合法未曝光且完整可评分的采集Log，不能靠阈值或重分组补足。

复现命令见 commands.sh；实现细节见 implementation_changes.md；七项 Gate 及全部未通过结果见 formal_go_nogo.json。

## 七项 Gate 与问题回答

| Gate | 本轮结论 |
|---|---|
| Engineering Correctness | Pass：538项CUDA测试及8帧原模型一致性通过 |
| Geometric Patch Recovery | 未达目标：没有新增可评分正确支持 |
| Joint Hypothesis Disambiguation | 未达目标：实际排除0，无合法稀疏恢复 |
| Region Risk-Coverage Feasibility | No-Go：点估计门槛通过，冻结置信度及独立支持未通过 |
| Identity Preservation | Pass：候选逐边及完整指标保持 |
| nuScenes Independent Qualification | No-Go：新增独立可评分Log为0 |
| Scientific Feasibility | 已测量失败边界；当前机制没有证明正确覆盖收益 |

1. 前平行假设确有局限，但本轮没有确认大量正确损失由这一假设造成；可测视差、纹理及投影可用性不足必须分开统计。
2. 新局部模型恢复的可评分正确源对应为0；最终新增正确Stable为0。
3. 同时新增可评分错误源对应为0；最终新增错误Stable为0。
4. 实际第三观测排除的条件性竞争解释为0，尚未证实联合消歧收益。
5. H0获得附加观测支持但仍有错误的14个参考点，不能靠H0正支持证明唯一性；可靠第三几何约束缺失的竞争继续Unknown。
6. 新方法与旧Constraints的可评分支持及错误风险点估计相同；净3个未标注像素的增加不足以证明覆盖和风险同时改善。
7. 非零四项点估计工作点存在=True，但仅5个正确点、1个接受Log，绝非正式风险资格。
8. 当前证据没有证明继续依赖复杂Surface Consensus多平面接受机制的必要性；保留它作错误与正确损失对照。
9. 关键事件保护保持达标；本轮面积几乎未恢复，不能据此推断未来大幅恢复面积后仍达标。
10. 最小新增条件是同一区域实际可直接跟踪、局部可验证、具有新增视差的第三RGB采集；首先让真实几何适用与可见性条件可观测，再评价排除竞争的能力。独立认证另需合格未曝光Log。
