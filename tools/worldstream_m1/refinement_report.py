"""Summarize real frozen replays; missing validation remains explicitly Not Run."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from .io_utils import write_json
from .quality_coverage_metrics import region_metrics,identity_metrics

ROOT=Path('outputs/module1_quality_coverage_refinement')
OLD=Path('outputs/module1_quality_coverage_optimization')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def readlines(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():yield json.loads(line)


def pointkey(r):return (r['scene'],r['frame'],r['camera'],*r['pixel'])


def pointid(r):return (r['scene'],r['frame'],r['camera'],r['reference_index'])


def pct(value):return 'null' if value is None else f'{100*value:.6f}%'


def summarize():
    complete=sorted((ROOT/'traces/development').glob('*/complete.json'))
    if len(complete)!=20:raise RuntimeError(f'Refuse partial Development summary: {len(complete)}/20 scenes')
    rows=[];identities=[];points=[];failures=[];pins={}
    for p in complete:
        rows.extend(readlines(p.parent/'pixels.jsonl'));identities.extend(readlines(p.parent/'identity.jsonl'))
        points.extend(readlines(p.parent/'accepted_points.jsonl'));failures.extend(readlines(p.parent/'false_stable.jsonl'))
        for name in ('pixels.jsonl','identity.jsonl','complete.json'):
            pins[str(p.parent/name)]=sha(p.parent/name)
        for name in ('pixels.jsonl','identity.jsonl'):
            source=OLD/'traces/development'/p.parent.name/name;pins[str(source)]=sha(source)
    denominator=sum(r['method']=='selective_identity' for r in identities)
    assert denominator==5798,denominator
    regions={name:region_metrics([r for r in rows if r['method']==name]) for name in sorted({r['method'] for r in rows})}
    identity={name:identity_metrics([r for r in identities if r['method']==name],denominator) for name in sorted({r['method'] for r in identities})}
    summary=dict(status='Run',stage='Development',population='exploratory_reused_population',
        scenes=len(complete),frames=240,region=regions,identity=identity)
    write_json(ROOT/'development_summary.json',summary)
    write_json(ROOT/'baseline_replay_provenance.json',dict(files=pins,
        baseline_head='78c3a5e',baseline_execution='Frozen same-input raw rows; original old-plane masks separately replayed on all 1320 nonstartup camera pairs',
        denominator=5798,original_outputs='Archived original GPU outputs, unchanged checkpoint; no fabricated or augmented detections'))
    for filename,key in [('region_method_comparison.json','region'),('region_risk_coverage.json','region'),
                         ('identity_method_comparison.json','identity'),('identity_risk_coverage.json','identity')]:
        write_json(ROOT/filename,dict(stage='Development',population='exploratory_reused_population',methods=summary[key]))
    for key in ('scene','camera','frame'):
        groups=defaultdict(list)
        for r in rows:groups[r['method'],r[key]].append(r)
        write_json(ROOT/f'region_quality_by_{key}.json',{method:{g:region_metrics(v) for (m,g),v in groups.items() if m==method} for method in regions})
    bymethod=defaultdict(list)
    for r in points:bymethod[r['method']].append(r)
    full=bymethod['surface_consensus'];fullids={pointid(r) for r in full}
    ablations={}
    for name in ('consensus_no_split','consensus_no_components','consensus_no_withheld','consensus_keys_only','consensus_no_history','consensus_boundary_guard'):
        other=bymethod[name];ids={pointid(r) for r in other}
        ablations[name]=dict(full=regions['surface_consensus'],ablation=regions[name],
            false_points_prevented_by_full=sum(r['false'] and pointid(r) not in fullids for r in other),
            correct_points_rejected_by_full=sum(not r['false'] and pointid(r) not in fullids for r in other),
            correct_points_recovered_by_full=sum(not r['false'] and pointid(r) not in ids for r in full),
            false_points_added_by_full=sum(r['false'] and pointid(r) not in ids for r in full),
            effective_area_delta=regions['surface_consensus']['effective_stable_area_ratio']-regions[name]['effective_stable_area_ratio'],
            critical_missed_point_delta=regions['surface_consensus']['missed_critical_points']-regions[name]['missed_critical_points'],
            pairing='Exact sparse reference index / scene / frame / camera, fixed oracle; no imputed dense truth')
    oldpoints=json.loads((ROOT/'surface_true_false_comparison.json').read_text())['records']
    live={pointkey(r) for r in full}
    paired=dict(old_false=39,old_correct=155,
        old_false_prevented=sum(r['false'] and pointkey(r) not in live for r in oldpoints),
        old_false_retained=sum(r['false'] and pointkey(r) in live for r in oldpoints),
        old_correct_retained=sum(not r['false'] and pointkey(r) in live for r in oldpoints),
        old_correct_lost=sum(not r['false'] and pointkey(r) not in live for r in oldpoints),
        new_total_correct=sum(not r['false'] for r in full),new_total_false=sum(r['false'] for r in full),
        old_cases=[dict(case_id=r['case_id'],false=r['false'],new_accepted=pointkey(r) in live) for r in oldpoints])
    paired['new_correct_outside_old_accepts']=paired['new_total_correct']-paired['old_correct_retained']
    paired['new_false_outside_old_accepts']=paired['new_total_false']-paired['old_false_retained']
    write_json(ROOT/'surface_old_new_paired_support.json',paired)
    write_json(ROOT/'surface_ablation.json',dict(paired_effects=ablations,
        original_boundary_guard=dict(old_full=regions['surface_local_plane'],
            old_no_boundary=json.loads((OLD/'development_summary.json').read_text())['region']['surface_no_boundary_guard'],
            finding='Old full and ablation final decisions identical. All 39 errors are at least 20px from its measured boundary. No error prevention demonstrated.'),
        design='Base disables old boundary guard. Its effect is an explicit ablation; it was not silently stacked into the main method.',
        selection='No post-result threshold, oracle or main-method switch'))
    write_json(ROOT/'surface_failure_cases.json',dict(cases=failures,physical_cause_rule='Unknown unless independently established',
        paired_old_plane=paired,visualizations='visualizations/development/ and visualizations/old_plane_errors/'))
    new=[r for r in identities if r['method']=='temporal_object' and r['candidate']]
    base=[r for r in identities if r['method']=='temporal_identity' and r['candidate']]
    shadow=[r for r in identities if r['method']=='temporal_shadow' and r['candidate']]
    def link(r):return (r['scene'],r['source_frame'],r['frame'],r['prediction'])
    oldlinks={link(r) for r in base};shadowlinks={link(r) for r in shadow}
    added=[r for r in new if link(r) not in shadowlinks]
    write_json(ROOT/'identity_unique_instance_recovery.json',dict(old=identity['temporal_identity'],shadow=identity['temporal_shadow'],new=identity['temporal_object'],
        added_vs_same_shadow_tracking=dict(count=len(added),correct=sum(r['correct'] is True for r in added),wrong=sum(r['correct'] is False for r in added),
            unscorable=sum(r['correct'] is None for r in added),records=added),
        new_vs_old_tracking_population_note='Compare the same-shadow ablation to isolate the enhancement; old Temporal also changed ordinary tracking and thus future sources.'))
    observability=Counter();own=[]
    for r in identities:
        if r['method']!='temporal_object':continue
        for v in (r.get('measurements') or {}).get('views',[]):
            observability[v.get('reason','missing')]+=1
            if v.get('observation_model')=='translation_compensated_box_surface_ring':
                own.append(dict(scene=r['scene'],frame=r['frame'],source_frame=r['source_frame'],prediction=r['prediction'],camera=v.get('camera'),
                    correct_identity=r['correct'],candidate=r['candidate'],view=v))
    write_json(ROOT/'identity_object_observability.json',dict(view_reasons=dict(observability),object_surface_attempts=own,
        actual_supported_object_surface_views=sum(r['view']['status']=='supported' for r in own),
        ownership_truth='Not available for dense RGB tracks. Predicted-box/ring geometry is conditional evidence; background suppression on real tracks is not independently proved.',
        hybrid='Legacy supported crops retained and labeled. Surface-only ablation measures the stricter population; no relabeling of crop support as object-own support.'))
    audit=Counter();separation=Counter()
    for p in complete:
        for s in readlines(p.parent/'state.jsonl'):
            diag=s['commit'];pairs=diag['association_pairs'];geo=diag.get('geometric_tracking_pairs',pairs)
            audit['commits']+=1
            audit['duplicate_source_assignments']+=len(pairs)!=len({a for a,b in pairs})
            audit['duplicate_target_assignments']+=len(pairs)!=len({b for a,b in pairs})
            audit['duplicate_committed_ids']+=len(diag['committed_anchor_ids'])!=len(set(diag['committed_anchor_ids']))
            audit['shadow_geometry_assignment_differences']+=pairs!=geo
            audit['qualified_certifications']+=len(s['post_commit']['certified_anchor_ids'])
            for r in diag['identity_continuity_proposals']:
                separation['evaluated_edges']+=1;separation['candidate']+=r['status']=='candidate'
                separation['applied_to_tracking']+=bool(r['applied_to_tracking'])
                separation['qualified']+=bool(r['certified'])
                separation['missing_target_binding']+=not bool(r.get('target_state_version'))
    assert not any(audit[k] for k in ('duplicate_source_assignments','duplicate_target_assignments','duplicate_committed_ids','shadow_geometry_assignment_differences','qualified_certifications'))
    write_json(ROOT/'association_certification_audit.json',dict(real_replay=dict(audit),controlled_reproduction=dict(failures_before_fix=4,
        log='association_failure_reproduction.log',tests=['test_trust_refinement.py']),
        contract='Default shadow leaves geometric IDs unchanged; explicit experimental one-to-one recovery has an unqualified basis and clears prior verified provenance. Transaction fork publishes hypotheses with store atomically.'))
    write_json(ROOT/'identity_proposal_certification_separation.json',dict(real_replay=dict(separation),
        no_independent_qualification_supplied=True,formal_certifications=0,formal_FAR=None))
    # Leave-one-log-out descriptive sensitivity; no fitting, selection or new units.
    sensitivity={}
    for method in ('conservative','surface_local_plane','surface_consensus'):
        selected=[r for r in rows if r['method']==method];logs=sorted({r['log_token'] for r in selected})
        sensitivity[method]={log:region_metrics([r for r in selected if r['log_token']!=log]) for log in logs}
    write_json(ROOT/'exploratory_leave_log_out.json',dict(population='exploratory_reused_population',
        status='Run descriptive leave-one-log-out sensitivity, no fitting or selection',independent_validation=False,region=sensitivity))
    reason='No new eligible independent acquisition Log under the unchanged labeled nuScenes protocol; other local datasets have no validated original-model/reference adapter.'
    for stage in ('calibration','heldout','final_test'):
        write_json(ROOT/f'{stage}_summary.json',dict(status='Not Run',reason=reason,scene_count=0,log_count=0,identity_method=None,region_method=None))
    selection=dict(identity_method=None,region_method=None,formal_selection=False,reason=reason,
        development_selection=False,development_region_constraints=regions['surface_consensus']['risk_gates_passed'],
        above_conservative_coverage=regions['surface_consensus']['effective_stable_area_ratio']>regions['conservative']['effective_stable_area_ratio'])
    write_json(ROOT/'algorithm_selection.json',selection)
    write_json(ROOT/'formal_go_nogo.json',dict(identity='No-Go',region='No-Go',independent_evaluation='No-Go',
        qualified_certifications=0,qualified_FAR=None,identity_method=None,region_method=None,
        development_region_risk_gates=regions['surface_consensus']['risk_gates_passed'],
        development_region_coverage_gain=selection['above_conservative_coverage'],reason=reason))
    freeze=json.loads((ROOT/'development_source_freeze.json').read_text())
    write_json(ROOT/'frozen_validation_protocol.json',dict(source_sha256=freeze['files'],
        identity_method=None,region_method=None,calibration_status='Not Run',heldout_status='Not Run',final_test_status='Not Run',
        no_test_reselection=True,protocol_sha256=sha(ROOT/'experiment_protocol.json'),data_selection_sha256=sha(ROOT/'data_selection.json'),
        independence='Disjoint newly acquired labeled logs, not clips or previously exposed logs',
        minimum_zero_error_complete_identity_logs=math.ceil(math.log(.05)/math.log(.95))))
    figures(regions,identity,ablations)
    report(regions,identity,paired,ablations,dict(audit))
    print('summarized',len(complete),'scenes',len(rows),'camera rows',len(identities),'identity edges')


def figures(regions,identity,ablations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    names=['conservative','dense_original','local_r12','surface_visibility','surface_local_plane','surface_consensus']
    fig,ax=plt.subplots(figsize=(9,5))
    for n in names:
        r=regions[n]
        if r['sparse_reference_far'] is not None:
            ax.scatter(100*r['effective_stable_area_ratio'],100*r['sparse_reference_far']);ax.annotate(n,(100*r['effective_stable_area_ratio'],100*r['sparse_reference_far']),fontsize=8)
    ax.axhline(5,color='r',ls='--',label='5% FAR constraint');ax.set(xlabel='Effective Stable area (%)',ylabel='Sparse annotated FAR (%)',title='Reused Development logs; sparse support does not certify dense area')
    ax.legend();fig.tight_layout();fig.savefig(folder/'region_risk_coverage.png',dpi=160);fig.savefig(folder/'region_risk_coverage.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));names=['selective_identity','temporal_identity','temporal_shadow','temporal_object','object_surface_only']
    x=np.arange(len(names));good=np.array([identity[n]['independently_correct_proposals'] for n in names]);unknown=np.array([identity[n]['unscorable_proposals'] for n in names]);bad=np.array([identity[n]['wrong_proposals'] for n in names])
    ax.bar(x,good,label='Correct');ax.bar(x,unknown,bottom=good,label='Unscorable');ax.bar(x,bad,bottom=good+unknown,label='Wrong')
    ax.set_xticks(x);ax.set_xticklabels(names,rotation=12,fontsize=8);ax.set_ylabel('Identity proposals');ax.set_title('Formal qualified certification = 0');ax.legend();fig.tight_layout();fig.savefig(folder/'identity_coverage.png',dpi=160);plt.close(fig)
    f=json.loads((ROOT/'surface_acceptance_funnel.json').read_text())['mutually_exclusive'];names=list(f)
    fig,ax=plt.subplots(figsize=(10,5));ax.barh(names,[f[n]['candidate_pixels'] for n in names]);ax.set_xlabel('Old-plane candidate pixels, mutually exclusive primary failure');fig.tight_layout();fig.savefig(folder/'old_plane_failure_area.png',dpi=160);plt.close(fig)
    # The decision funnel excludes the diagnostic-only legacy erosion in the
    # new path. Missing stages are not silently treated as retained evidence.
    stages=defaultdict(Counter);camera_counts=defaultdict(Counter)
    for path in (ROOT/'traces/development').glob('*/pixels.jsonl'):
        for row in readlines(path):
            name=row['method']
            if name not in ('surface_local_plane','surface_consensus'):continue
            for stage,values in row.get('camera_diagnostics',{}).get('acceptance_funnel',{}).items():
                if values.get('applied',True):
                    stages[name][stage]+=values['retained'];camera_counts[name][stage]+=1
    write_json(ROOT/'surface_measured_stage_retention.json',dict(retained_pixels=stages,camera_rows_with_stage=camera_counts,
        note='Only actual measured stages; startup/early rejection have no fabricated flow. Global area denominators remain all 1440 camera images.'))
    fig,ax=plt.subplots(figsize=(11,5))
    for index,(name,values) in enumerate(stages.items()):
        labels=list(values);ax.plot(np.arange(len(labels)),list(values.values()),marker='o',label=name)
        for x,label in enumerate(labels):ax.annotate(label,(x,list(values.values())[x]),xytext=(3,8+index*12),textcoords='offset points',fontsize=6,rotation=12)
    ax.set_yscale('log');ax.set_ylabel('Measured retained pixels (log scale)');ax.set_xlabel('Actual ordered decision stage');ax.legend();fig.tight_layout();fig.savefig(folder/'surface_stage_retention.png',dpi=160);plt.close(fig)


def report(region,identity,paired,ablations,audit):
    n=region['surface_consensus'];c=region['conservative'];old=region['surface_local_plane'];ni=identity['temporal_object'];oi=identity['temporal_identity'];si=identity['temporal_shadow']
    lines=['# M1 可信物理状态与局部 Stable 覆盖率优化报告','',
        '实际基线 HEAD：`78c3a5e`。本轮只修改 M1、M1 配置、工具、测试及新的输出目录。原报告和源码快照保留，Oracle 与数值安全门槛不变。',
        '', 'Development 使用同一批 20 Scene / 240 Frame，属于 `exploratory_reused_population`，也有原模型训练曝光。基线使用已固定的同输入原始记录；旧 Local Plane 又逐帧重放，1320 个非启动相机行数值一致。',
        '', '| 区域方法 | Effective Stable | Sparse FAR | 正确/错误接受 | Critical Event Recall | Critical Point Recall | 接受 Scene/Log |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('full_refresh','conservative','dense_original','local_r12','surface_visibility','surface_local_plane','surface_consensus'):
        r=region[name];lines.append(f"| {name} | {pct(r['effective_stable_area_ratio'])} | {pct(r['sparse_reference_far'])} | {r['independently_correct_stable_support']}/{r['false_stable_points']} | {pct(r['critical_change_recall'])} | {pct(r['critical_point_recall'])} | {r['accepted_scene_support']}/{r['accepted_log_support']} |")
    lines+=['','| 身份方法 | Proposal | 可评分正确 | 错误 | 不可评分 | 唯一正确实例 | 正式认证 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('hungarian_geometry','strict_foreground','selective_identity','temporal_identity','temporal_shadow','temporal_object','object_surface_only','object_no_history','object_no_competition'):
        r=identity[name];lines.append(f"| {name} | {r['identity_proposals']} | {r['independently_correct_proposals']} | {r['wrong_proposals']} | {r['unscorable_proposals']} | {r['unique_physical_identity_recovery']} | {r['qualified_certifications']} |")
    lines+=['', '## 必答问题', '',
        '1. **核心机制。** 新区域路径按实际测量连通性形成支持，使用训练点决定多个逆深度平面假设，留出点不拟合、不排序模型；冲突支持拆分，只接受有实测对应且模型预测一致的像素。原完整邻域腐蚀改为逐像素可见支持，绝不填充 ROI。新身份路径保留有限历史，在原 crop 失败时检查运动补偿三维表面和实测外环的深度分离，并提供 surface-only 消融。',
        '2. **已确认的失败。** 旧平面 94.7% 候选面积在关键点不足单元中被拒绝；少数坏像素及混合模型会破坏整单元。新机制处理分区和选择性接受，但这不保证消除自洽错流。旧边界保护在本人口无最终作用；主方法默认关闭并保留 on/off 消融。详见逐单元、拒绝参考支持与配对消融 JSON。',
        '3. **39 个错误归因。** 可信物理主因确认 0/39，Unknown Root Cause 39/39。39 点全部有真实 RGB 可视化、平面参数、留出误差、FB、光度、纹理、位姿及独立参考。错误并非已证实的低视差：测得约 0.757–3.54°；距原边界至少约 20px。额外直接 LK 全部与 DIS 在 0.5px 内一致，仍不能预测独立参考。不能据此声称 GT 错误、混合表面或实例归属原因已确认。',
        f"4. **配对错误与正确支持。** 对旧 39 错误，新方法阻止 {paired['old_false_prevented']}、仍接受 {paired['old_false_retained']}；对旧 155 正确保留 {paired['old_correct_retained']}、损失 {paired['old_correct_lost']}。新总正确 {paired['new_total_correct']}、总错误 {paired['new_total_false']}，其中旧接受之外新增正确 {paired['new_correct_outside_old_accepts']}、新增错误 {paired['new_false_outside_old_accepts']}。不是把拒绝所有像素计作改进；完整配对结果在 `surface_old_new_paired_support.json`。",
        f"5. **覆盖。** 新 {pct(n['effective_stable_area_ratio'])}，Conservative {pct(c['effective_stable_area_ratio'])}，是否超过：{n['effective_stable_area_ratio']>c['effective_stable_area_ratio']}。所有图像像素均进入面积分母，包含启动和周期强制刷新。",
        f"6. **风险与变化召回。** 新 FAR {pct(n['sparse_reference_far'])}，其 IID 描述上界 {pct(n['far_upper95_iid_descriptive'])}；事件召回 {pct(n['critical_change_recall'])}，点召回 {pct(n['critical_point_recall'])}；最差事件 ROI 刷新 {pct(n['minimum_roi_event_refresh_ratio'])}；接受 Scene {n['accepted_scene_support']}、Log {n['accepted_log_support']}。Development 风险门槛通过：{n['risk_gates_passed']}。IID 点上界不能代替独立 Log 风险资格。",
        f"7. **身份变化。** 旧 Temporal 正确/唯一/错误/不可评分 {oi['independently_correct_proposals']}/{oi['unique_physical_identity_recovery']}/{oi['wrong_proposals']}/{oi['unscorable_proposals']}；新 {ni['independently_correct_proposals']}/{ni['unique_physical_identity_recovery']}/{ni['wrong_proposals']}/{ni['unscorable_proposals']}。同 Shadow 跟踪的原 Temporal 为 {si['independently_correct_proposals']}/{si['unique_physical_identity_recovery']}/{si['wrong_proposals']}/{si['unscorable_proposals']}。以 Shadow 对照隔离新观测贡献，避免把更改历史 ID 引起的人口差异算作算法增益。分母保持原 5798 几何候选，额外合法边另报。",
        '8. **背景轨迹。** 新路径只把满足双框运动补偿三角化且与实测环分离的轨迹作为条件对象表面支持；controlled tests 覆盖背景与前景深度相同、纯背景和错误运动。真实 RGB 缺少逐轨迹独立实例归属真值，不能宣称背景错误已被证明减少。Hybrid 保留的旧 crop 支持没有被改称对象自身；surface-only 人口单列。',
        f"9. **状态隔离。** 修复前 4 个针对性测试失败，复现覆盖关联、重复继承与缺少 Shadow API。默认 Shadow 独立记录候选及目标版本。新真实重放 {audit['commits']} 次 Commit，重复源/目标/ID 和 Shadow 改写几何关联均为 0，正式认证 0。显式 experimental 路径重新一对一分配、记录未认证来源并清除旧验证来源，失败 Commit 原子回滚。",
        '10. **独立数据。** 完整读取 train/val info（28130/6019 帧）及 850 Scene / 68 Log 元数据，不以缓存是否存在判断标签是否存在。这 68 Log 均已曝光。test 有 15 个不同 Log，但 Annotation 表 0 条，不能评分；mini 重复已有 Log。另查 16 份 DAIR/V2X metadata，完整发布的 RGB 路径实际缺失，示例有 250/251 和 91/84 张 RGB；它们确有标定和部分跟踪标注，但未建立相机时刻运动真值、畸变校正、原六相机模型输入及独立采集单位适配，不能并入现有资格。OPV2V/V2XSet 有合成 RGB，也未当作真实独立驾驶人口。',
        '11. **最少新增证据。** 需要未用于模型/M1 选择的独立采集日志、真实同步 RGB/位姿/标定、身份和几何变化参考及足够完整的接受人口。仅身份零错误要达到单侧 CP95≤5%，至少 59 个可完整评分且有接受的独立 Log，另需分离 Calibration；增加同一 Log 的帧数不增加该样本数。对自洽错流，最小机制补充应是独立同步几何或真实实例可见性观测，不能把离线评分 LiDAR 在线喂入。',
        '12. **验收。** 工程隔离与测试可核验；算法研究结果见上表。正式 Identity/Region 及独立评价均 No-Go，方法选择 null，新 Calibration/Heldout/Final 全部 Not Run，不以已曝光探索性交叉检查代替正式资格。',
        '13. **不可辨识边界。** 自洽的重复纹理错流、光度与几何均相近但实例不同、独立视差不足、来源与遮挡无法确认、框内背景、未观测的多视图冲突均可能无法仅靠当前 RGB 与预测框区分。多平面共识不是独立传感器；可测一致不能变成物理真值。',
        '14. **工程契约。** 完整 M1/M2/M3 回归与真实 GPU 状态分别见 `regression_results.json` 和 `original_model_parity.json`。未修改原 Backbone、Head、LLM、Checkpoint、模型预处理和 M2/M3 实现。GPU 未执行时必须保持 Not Run。',
        '', '## 消融与可复现性', '',
        '所有消融固定相同输入、阈值、Oracle 和分母；按独立参考索引配对，报告阻止错误、拒绝正确、面积和关键点召回变化。没有因结果更好而切换主方法。没有贡献的旧边界保护已默认关闭；其余机制的取舍需依据消融和新增独立数据，未获得资格前均为研究配置。',
        '', '原有 Local R12 的 169 个错误及 46 个低视差 / 123 个 Unknown 的历史结论保持原样，不被本轮 39 点分析覆盖。`diagnostic_freeze.json` 固定算法修改前的五份诊断；基线源码、原始行、来源摘要、命令和本轮源清单均保留。']
    (ROOT/'experiment_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':summarize()
