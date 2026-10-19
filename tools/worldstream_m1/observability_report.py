"""Chinese research report generated from actual saved nuScenes measurements."""
from pathlib import Path
import json
import numpy as np

ROOT=Path('outputs/module1_physical_observability_optimization')


def read(name):return json.loads((ROOT/name).read_text())
def percent(v):return 'N/A' if v is None else f'{100*v:.6f}%'


def main():
    region=read('region_method_comparison.json')['methods'];identity=read('identity_method_comparison.json')['methods']
    pairing=read('region_paired_error_correct_analysis.json');population=read('identity_population_analysis.json')
    diagnostic=read('surface_ambiguity_analysis.json');provenance=read('surface_evidence_provenance.json');funnel=read('surface_acceptance_funnel.json')
    audit=read('independent_data_adapter_audit.json');regression=read('regression_results.json');parity=read('original_model_parity.json')
    model=region['surface_constraints'];old=pairing['comparisons']['surface_consensus'];roi=read('roi_exact_mask_audit.json')
    head=read('baseline_source_manifest.json')['head']
    names=['full_refresh','conservative','dense_original','local_r12','surface_visibility','surface_local_plane','surface_consensus','surface_constraints']
    table='| 方法 | Effective Stable 面积 | 正确/错误参考点 | Sparse FAR | Event Recall | Point Recall | 最差 ROI Refresh | 接受 Scene/Log |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for n in names:
        v=region[n];table+=f"| {n} | {percent(v['effective_stable_area_ratio'])} | {v['independently_correct_stable_support']}/{v['false_stable_points']} | {percent(v['sparse_reference_far'])} | {percent(v['critical_change_recall'])} | {percent(v['critical_point_recall'])} | {percent(v['minimum_roi_event_refresh_ratio'])} | {v['accepted_scene_support']}/{v['accepted_log_support']} |\n"
    ablation='| 消融 | 面积 | 正确/错误 | FAR | 最差 ROI Refresh |\n|---|---:|---:|---:|---:|\n'
    for n in ['constraints_observation_base','constraints_two_view_only','constraints_temporal_only','constraints_camera_only','constraints_no_ambiguity','constraints_no_depth_order','surface_constraints']:
        v=region[n];ablation+=f"| {n} | {percent(v['effective_stable_area_ratio'])} | {v['independently_correct_stable_support']}/{v['false_stable_points']} | {percent(v['sparse_reference_far'])} | {percent(v['minimum_roi_event_refresh_ratio'])} |\n"
    itable='| 身份方法 | Proposal | 正确 | 错误 | 不可评分 | 唯一正确实例 | 完整可评分接受 Log | Log 风险上界 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for n,v in identity.items():
        itable+=f"| {n} | {v['identity_proposals']} | {v['independently_correct_proposals']} | {v['wrong_proposals']} | {v['unscorable_proposals']} | {v['unique_physical_identity_recovery']} | {v['log_risk']['fully_scorable_accepted_logs']} | {percent(v['log_risk']['complete_log_event_upper95'])} |\n"
    ptable='| 在线定义人口（非互斥） | 原始几何人口成员/5798 | 评价边 | Proposal | 正确/错误/不可评分 | 唯一正确实例 | 完整可评分接受 Log |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for n in ['all_measured_legal_edges','has_supported_crop','multiple_supported_temporal_camera_crops','measured_object_surface_and_background','background_depth_separable','unresolved_identity_competition','resolved_identity_competition','short_gap_source','predicted_displacement_over_05m']:
        v=population['populations'][n];ptable+=f"| {n} | {v['original_geometry_population_members']} | {v['evaluated_edges']} | {v['identity_proposals']} | {v['independently_correct_proposals']}/{v['wrong_proposals']}/{v['unscorable_proposals']} | {v['unique_physical_identity_recovery']} | {v['log_risk']['fully_scorable_accepted_logs']} |\n"
    roitable='| 帧/相机 | 事件 | 旧精确刷新 | 新精确刷新 | 旧 Tile 包络刷新 | 活跃预测动态扫掠覆盖 |\n|---|---|---:|---:|---:|---:|\n'
    for r in roi['events']:
        roitable+=f"| {r['frame'][:8]}/{r['camera']} | {r['event']['semantic_class']}: {','.join(r['event']['reasons'])} | {percent(r['old_exact_refresh'])} | {percent(r['new_exact_refresh'])} | {percent(r['old_tile_envelope_refresh'])} | {percent(r['predicted_dynamic_sweep_coverage'])} |\n"
    c=diagnostic['measured_comparison']['correct'];f=diagnostic['measured_comparison']['false']
    dist='| 测量分布 [Q25, 中位数, Q75] | 正确494 | 错误196 |\n|---|---|---|\n'
    for field in ['fb','photo','gradient','texture','jacobian','parallax_deg']:
        aa=[];bb=[]
        for r in diagnostic['records']:
            if 'surface_consensus' not in r['populations']:continue
            x=r['observability']['parallax_deg'] if field=='parallax_deg' else r['online'].get(field)
            if x is not None and np.isfinite(x):(bb if r['false'] else aa).append(x)
        dist+=f"| {field} | {np.round(np.percentile(aa,[25,50,75]),6).tolist()} | {np.round(np.percentile(bb,[25,50,75]),6).tolist()} |\n"
    minimum=1-.05**(1/20)
    summary=f'''# M1 基于可辨识物理证据的覆盖率优化

本轮完成 **nuScenes + OmniDrive 原模型 + M1** 的实际编码、20 个已曝光场景的 Development、消融、身份人口分析、全量 nuScenes 可用性审计、517 项 CUDA 回归及 8 帧原模型感知一致性验证。正式结论为 **No-Go**。新路径减少错误接受并恢复关键 ROI 刷新，但付出的正确覆盖损失很大，没有达到原有风险置信度和接受人口要求，也没有超过 Conservative。

## 实验边界和冻结

实际基线 HEAD：`{head}`。旧报告和历史结果未覆盖，9 项历史产物 SHA256 校验一致。仅修改 M1、M1 配置、M1 工具/测试及本轮输出；M2/M3、原模型、Checkpoint、训练与输入预处理未修改。环境为 `/home/qh/miniconda3/envs/OmniDrive/bin/python`，所有执行使用 `taskset -c 0-3,6-31`。GPU 空闲检查后运行，没有停止其他任务。

本轮所有主要结果来自同一组 20 Scene / 20 Log / 240 Frame / 1,440 Camera，含启动和周期强制刷新；区域参考分母 130,627 个可评分 Annotation-owned LiDAR 点、236,540 个关键点、3,250 个事件。身份原始几何分母固定 5,798，扩展合法边 13,314。全部标记 `exploratory_reused_population`。旧工具内部历史名称 calibration 仅指原开发样本加载入口，**本轮不是正式 Calibration**。

`observability_diagnostic_freeze.json` 固定方法修改前的 690 个 Consensus 和194 个 Local Plane 接受点，合并为780个实际参考位置；`development_source_freeze.json` 在新 Development 评分前固定算法、配置和消融。没有阈值网格搜索或事后按正确性挑选在线人口。完整命令在 [commands.sh](commands.sh)，源码/数据/模型/原始轨迹哈希分别在 source_manifest.json、data_manifest.json。正式方法选择始终 null。

## 可辨识性：同源一致为何不足

旧 Consensus 正确494/错误196中，同双帧 RGB 上重新运行 LK 且与 DIS 相差≤0.5px 的分别为 {c['same_RGB_LK_agreement_05']}/{f['same_RGB_LK_agreement_05']}；真实额外时间观测支持 {c['temporal_supported']}/{f['temporal_supported']}；实际测得互不相容双帧替代解释 {c['two_view_ambiguous']}/{f['two_view_ambiguous']}。额外时间支持本身仍包含23个错误，不能因多帧就称为独立物理认证。该冻结人口没有跨相机正支持，后续全候选实验确实测得少量跨相机像素支持，但没有可评分跨相机接受点。

{dist}

每个点保存实际 RGB 源像素、独立离线参考源像素、姿态/内参、FB/光度/纹理/Jacobian、实际视差、Consensus 单元/多平面、重新跟踪、有限替代源像素和不同采集观测。**196 个错误点的具体物理根因均保留 Unknown Root Cause**：独立参考确认对应错误，不据光流自洽或参考运动自动声称遮挡/前景混淆。Patch 不匹配是未获支持，不能当作已证明遮挡。示例在 visualizations/correspondence_cases、true_false_controls；所有原始测量在 surface_ambiguity_analysis.json。

## 新机制与接口

显式启用 `configs/worldstream_m1_physical_observability.yaml`，`dense_region_policy: surface_constraints`。原有 FB、纹理、光度、射线、完整邻域和动态/遮挡排除保留，半径2沿用旧 Consensus 的实验设置；另以 observation_base 消融披露其覆盖和风险。新机制不拟合更多平面，不作残差加权或最佳假设择优：

1. 对每个实际候选射线三角化，并直接在源 RGB 中验证局部深度的投影 Patch。
2. 要求至少一个**真实不同采集**支持：已 Commit 的更早图像，或同步误差≤10ms且真实投影共同可见的其他相机。时间顺序、标定、坐标、正深度与视差不足时 Unknown；不使用未来帧。
3. 在已有±4px搜索孔径内以0.5px采样极线对应；原解释与实际支持的>2px不相容源解释共存时拒绝为 Ambiguous。有限搜索没有发现替代解释不等于全局唯一。
4. 对具备附加支持的实测假设执行源像素深度排序冲突检查；这是拒绝冲突假设，不是确定真实遮挡者或物理表面所有权。

OnlineVisibilityEvidence 提供 observation_supported_mask、visibility_consistent_mask、ambiguity_rejected_mask、stable_candidate_mask、refresh_required_mask；最后一项包含周期/重置强制刷新。稳定候选不扩张原测量基底。不同采集仍有相关性；所有输出为条件性表面观测支持，**无 Transformer Feature Reuse / 物理前景所有权认证**。正常在线函数不接收 LiDAR/Annotation。局部前平行 Patch 是条件性探针，也可能拒绝真实斜面/非刚性表面；不能把该模型不适配的失败都归为“Camera-only 不可辨识”。

## 主要区域方法对照

{table}

历史基线是相同输入及未变评分协议的 SHA256 固定结果，明确为 archived same-input baseline，不冒充新独立实验。新方法实际接受5个正确、0个已知错误，只来自 **1 Scene / 1 Log**；Sparse IID 描述性上界为 {percent(model['far_upper95_iid_descriptive'])}，Log 事件上界 {percent(model['log_event_upper95'])}。Log 聚类 bootstrap 的零支持重采样353/1000，条件 FAR 区间 [0,0] 没有覆盖最坏未观测风险，不能据此认证。相对 Conservative 面积减少 {percent(1-model['effective_stable_area_ratio']/region['conservative']['effective_stable_area_ratio'])}，面积差 {100*(model['effective_stable_area_ratio']-region['conservative']['effective_stable_area_ratio']):.6f} 个百分点。

同一参考点配对：旧494个正确仅保留 {old['retained']['correct']}，损失 {old['rejected']['correct']}；旧196个错误全部阻止；另新增4个正确。旧 Local Plane 的39个错误全部阻止，155个正确只保留2个，另有3个新正确。**不能将这一巨大正确损失隐藏在“错误清零”中**。

## 消融与证据实际贡献

{ablation}

相同真实采集和 Commit 状态上使用预声明逻辑消融。独立的 two_view_only manager 在支持场景完整重放，连同 ROI 主路径重放核对共288个相机记录，面积/Stable 无差异，详见 independent_ablation_replay_check.json。主路径的区域掩码不会改变原模型完整输出或 Shadow 身份关联，所以共享观测消融没有跨方法状态反馈。

可评分漏斗：1544点（1314正确/230错误）→源投影支持257（177/80）→真实附加采集65（51/14）→歧义拒绝后5（5/0）。歧义检查单独排除14个错误，同时损失46个正确。源探针本身损失1137个正确，附加采集约束再损失126个；这些含未观测和模型假设不适配，不能全称为物理矛盾。深度排序在本人口仅减少少量像素，没有新增可评分错误剔除收益，不能包装成已证实的主要质量改进。

全候选、强制刷新前：时间支持35,001个像素；跨相机支持570个像素。实际跨相机几何可用探针19,794次；不是默认六摄像头共视。跨相机独立消融有效像素255，**可评分接受0，FAR=null**。探针计数可能在多个视角重复；独立风险单位始终是Log。帧/像素支持去重详见 surface_evidence_provenance.json。

离线 Oracle Upper Bound 在 observation_base 的1544个候选参考中知道1314个正确，主路径仅恢复5个，剩余1309个反映条件探针覆盖缺口。此为**稀疏候选上界**，没有稠密物理表面真值，dense area、完整 ROI Oracle 指标保持 null；不据 sparse 点外推全图真值。Oracle 从未进入在线决策或资格签发。

## Critical ROI 与精确掩码

{roitable}

四个历史失败事件均恢复到100%精确刷新，全体3,250事件的最差刷新 {percent(model['minimum_roi_event_refresh_ratio'])}。原最差交通锥 ROI 只有108个整数像素，其中42个被旧 Stable 占用，导致61.11%刷新。预测扫掠基于活跃私有 Precheck 状态重新核对，不能用已过期公开旧状态虚构保护范围。动态预测保护未覆盖一些真实事件；这并不足以证明具体物理漏检原因，仍保留 Unknown。

精确掩码与保守32px Tile包络是不同输出，旧四事件的 Tile 包络均为100%刷新；没有发现精确掩码到包络丢失覆盖的映射错误。所有主要统计坚持评分 exact mask，未将保守包络替换为更好看的面积/ROI数值。GT ROI只在在线 Precheck后评分与画图；没有 GT触发刷新。见 roi_exact_mask_audit.json 和 visualizations/roi_before_after。

## 身份：保留现有生产器，分析在线人口

{itable}

{ptable}

所有人口按实际在线输入定义，无 GT正确性筛选；非互斥人口不能相加。展开边的比例与原5,798人口成员比例分开报告，避免用13,314边代替原分母。完整预测类别、距离、位移和源年龄分层在 identity_population_analysis.json；短间隔人口只说明来源恢复，**不声称已确认真实遮挡**。多相机 crop 支持不等于同步可三角化的前景共视。

本轮逐边核对13,314边与现有 Temporal Object 的 candidate/correct/path/short-gap，差异0。现有217正确、71不可评分、45唯一实例保持；没有新生产器收益，保留 Temporal Object 默认 Shadow。Object Surface 人口仅4个正确候选、3个完整接受Log，Log上界约63.16%；完整Hybrid只有1个完整可评分接受Log，上界95%。保留全部71个不可评分接受，不用GT选出可评分人口发证。唯一 Claim 仍是 object_identity_continuity，不扩展为 stationarity/metric velocity/foreground ownership/box surface/feature或LLM复用。

## nuScenes 独立评价条件

逐帧核对Train 28,130帧/700 Scene/50 Log，Val 6,019帧/150 Scene/18 Log；实际六路 RGB 文件204,894个全部存在，LiDAR 34,149帧全部存在。非空Annotation对应Train 28,124帧、Val 6,016帧；空标注帧不是额外独立Log。实际内参/传感器与Ego Pose数值核对，使用帧的1,680个图像/LiDAR文件、元数据、预测缓存及Checkpoint均保存SHA256。

68个Trainval Log均有历史真实状态/评分轨迹的具体路径及SHA256见 historical_exposure_audit.json；没有把仅出现在可用性清单中的token算作曝光。两份真实原模型输出缓存共600个不同样本，缺少缓存不等于缺少原始数据或标注。Test有15个未曝光Log，但本地 sample_annotation.json 为0条，不能评价身份或物理对应；Mini的8个Log也已曝光。本轮未引入其他数据集、输入结构或跨数据集 Adapter。保留的文件名 independent_data_adapter_audit.json 内容仅为nuScenes同数据集可用性与曝光审计。

Calibration / Heldout / Final：**Not Run**。独立已标注新Log=0；qualified_certifications=0，qualified_FAR=null，Identity Selection=null，Region Selection=null。当前统一实验中的按Log探索性交叉验证仅作固定方法逐Log报告，19/20折无可评分接受，全部仍为 exploratory_reused_population，不能擦除开发曝光。采用95%单侧、每Log任意错误为失败的零错误风险上界，要≤5%至少需59个完整可评分、独立、实际接受Log；即使20个开发Log全是全新且零错误，上界仍有 {percent(minimum)}。重复场景、帧或对象不能补足Log单位。

## 六项 Gate

| Gate | 结论 | 实测依据 |
|---|---|---|
| 1 Engineering | Pass | CUDA 517/517；实际240次本轮Commit无重复历史ID，身份13,314边行为一致；8帧原感知/骨干/记忆摘要全部相等；旧产物哈希一致 |
| 2 Surface Physical Observability | Partial | 测得不同采集和冲突解释可排除错误；歧义边际14错误/46正确损失；没有物理所有权认证 |
| 3 Region Feasibility | No-Go | 非零5正确/0已知错，四项点估计门槛通过，但面积极低、仅1接受Log/Scene、点风险上界45.07%，原资格门槛未通过 |
| 4 Identity Feasibility | No new improvement | 保留217正确/45唯一实例及Shadow；人口分析未产生新增收益或认证 |
| 5 Independent Quality | No-Go | 新独立已标注Log=0；Calibration/Heldout/Final Not Run；正式选择null |
| 6 Scientific Feasibility | Boundary identified | 相关RGB证据可自洽；附加采集仍可歧义，条件Patch模型有额外覆盖损失；独立数据限制另行报告 |

## 下一轮最小必要证据

先在**同一nuScenes六摄像头输入与原模型**内验证：对实际共同可见、纹理可辨且具有足够非共线基线的第三次采集，直接测量可唯一排除竞争源像素的对应，而非继续增加同一DIS链上的平面或评分项。必须明确记录该观测真的可访问、时间可靠、未被遮挡与足够视差；本轮不同采集正支持仍有错误，尚未证明有限Patch搜索能提供这样的唯一性。前平行探针拒绝的真点应单独检查斜面/光照/遮挡等可实测模型条件，不能先给根因标签。

只靠当前弱重叠视角和两帧自洽不能支持已观测歧义区域的物理结论；此类区域保持Unknown/Refresh。物理资格方面，需要后续合法新增且未曝光的nuScenes acquisition Log及完整离线参考；本轮本地不能满足。算法继续可在复用人口研发，研究收益与独立认证应分开结论，不改变数据集、模型输入或目标来凑样本。

主要源码：[surface_visibility_constraints.py](../../worldstream/m1/surface_visibility_constraints.py)、[surface_observability.py](../../worldstream/m1/surface_observability.py)、state_manager.py/config.py/state_types.py/novelty_detector.py/dense_support.py；新配置 worldstream_m1_physical_observability.yaml。实现说明见 implementation_changes.md；可视化入口 region_risk_coverage.png、roi_and_paired_support.png、identity_population_recovery.png 及真实RGB病例目录。当前所有未通过指标和完整风险统计保存在 formal_go_nogo.json、region_method_comparison.json、identity_population_analysis.json。
'''
    (ROOT/'experiment_report.md').write_text(summary)
    (ROOT/'physical_observability_audit.md').write_text(f'''# Surface 物理可辨识性审计

冻结人口：690个新Consensus接受（494正确、196错误）与194个旧Local Plane接受（155正确、39错误），合并780个参考点；所有196错误具体根因为Unknown Root Cause。

{dist}

同源LK与DIS≤0.5px：正确385、错误150。双帧实际互斥解释：正确118、错误66。额外时间支持：正确62、错误23。附加时间观测本身的错误率23/85，不能当独立真值。冻结点跨相机正支持为0；新完整人口实际跨相机支持570像素但评分点为0。

逐点保留测得矛盾、未观测和同源一致的分离。不同源像素实际支持是有限孔径的几何歧义证据；Patch失败不是已证明遮挡，未发现替代不是全局唯一。LiDAR/Annotation仅离线确认错误，不进入源像素测量或线上掩码。

新机制逐级保留1314/230→177/80→51/14→5/0（正确/错误），大量正确损失尤其来自条件Patch探针，不能全归因物理不可辨识。歧义边际阻止14错、损失46真。深度排序没有可评分错误剔除收益。详情见 region_paired_error_correct_analysis.json、surface_acceptance_funnel.json、surface_evidence_provenance.json。

相同原双帧派生的FB/几何/拟合关键点并非独立传感器；跨时间/相机正支持也不能自动发物理认证。当前物理风险人口不足，默认Shadow、资格0、正式No-Go。
''')
    print('Wrote experiment_report.md and physical_observability_audit.md')


if __name__=='__main__':main()
