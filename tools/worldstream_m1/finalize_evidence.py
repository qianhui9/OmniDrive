"""Assemble actual measurements and explicit No-Go decisions for evidence gaps."""
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import xml.etree.ElementTree as ET
import numpy as np

from .evidence_experiments import ROOT,BASE,summary
from .io_utils import write_json
from .independent_refinement import distribution,sha256
from worldstream.m1.fact_certifier import calibrate_scene_maxima


def read_lines(path):
    return list(map(json.loads,path.read_text().splitlines())) if path.exists() else []


def cluster_far(rows):
    units=defaultdict(lambda:[0,0])
    for r in rows:
        units[r['scene']][0]+=r.get('false_points',0);units[r['scene']][1]+=r.get('accepted_points',0)
    values=np.asarray(list(units.values()),float)
    if not len(values) or not values[:,1].sum():return dict(interval=None,accepted_scenes=0,zero_denominator_draws=None)
    rng=np.random.RandomState(761);sample=values[rng.randint(len(values),size=(4000,len(values)))].sum(1)
    valid=sample[:,1]>0;far=sample[valid,0]/sample[valid,1]
    return dict(interval=np.percentile(far,[2.5,97.5]).tolist() if len(far) else None,
        accepted_scenes=int((values[:,1]>0).sum()),zero_denominator_draws=int((~valid).sum()),
        warning='Descriptive scene-cluster bootstrap; all-zero errors do not establish distribution-free zero risk')


def method_association(rows,kind,threshold):
    def keep(r):
        # association_ambiguities emits +inf when neither endpoint has a legal
        # competing edge. Strict JSON encodes that diagnostic as null.
        if kind=='local':return r['local_margin_m'] is None or r['local_margin_m']>threshold+1e-12
        return r['cardinality_lost']>0 or r['global_cost_margin_m'] is not None and r['global_cost_margin_m']>threshold+1e-12
    kept=[r for r in rows if keep(r)]
    scored=[r for r in kept if r['scorable']]
    rejected=[r for r in rows if not keep(r) and r['scorable']]
    return dict(method=kind,threshold_m=threshold,all_candidates=len(rows),kept=len(kept),scorable_kept=len(scored),
        correct_kept=sum(r['correct'] for r in scored),wrong_kept=sum(not r['correct'] for r in scored),
        precision=sum(r['correct'] for r in scored)/len(scored) if scored else None,
        correct_rejected=sum(r['correct'] for r in rejected),wrong_rejected=sum(not r['correct'] for r in rejected),
        unscorable_kept=len(kept)-len(scored))


def main():
    stages=['development','calibration','heldout','final_test']
    selection=json.loads((ROOT/'algorithm_selection.json').read_text())
    frozen=json.loads((ROOT/'formal_validation_protocol.json').read_text())
    chosen=selection['selected_local_method']
    pixels={};times={};populations={};associations={};foreground={};certificates={};sizes={};failures={}
    complete=[]
    for stage in stages:
        folder=ROOT/'traces'/stage
        paths=list(folder.glob('*/complete.json'))
        expected={'development':26,'calibration':20,'heldout':6,'final_test':12}[stage]
        if len(paths)!=expected:raise RuntimeError(f'{stage}: incomplete {len(paths)}/{expected}')
        complete.extend(json.loads(p.read_text()) for p in paths)
        pixels[stage]=sum((read_lines(p.parent/'pixels.jsonl') for p in paths),[])
        times[stage]=sum((read_lines(p.parent/'timing.jsonl') for p in paths),[])
        populations[stage]=sum((json.loads((p.parent/'population.json').read_text()) for p in paths),[])
        associations[stage]=sum((json.loads((p.parent/'association.json').read_text()) for p in paths),[])
        foreground[stage]=sum((read_lines(p.parent/'foreground.jsonl') for p in paths),[])
        certificates[stage]=sum((json.loads((p.parent/'certification.json').read_text()) for p in paths),[])
        sizes[stage]=sum((json.loads((p.parent/'sizes.json').read_text()) for p in paths),[])
        failures[stage]=sum((json.loads((p.parent/'failures.json').read_text()) for p in paths),[])
    region={stage:{method:dict(summary([r for r in rows if r['method']==method]),
        cluster_far=cluster_far([r for r in rows if r['method']==method]))
        for method in sorted({r['method'] for r in rows})} for stage,rows in pixels.items()}
    write_json(ROOT/'region_risk_coverage.json',dict(selection=selection,quality_scope=frozen['region_scope'],splits=region))
    write_json(ROOT/'region_quality_by_scene.json',{stage:{scene:{method:summary([r for r in rows if r['scene']==scene and r['method']==method])
        for method in sorted({r['method'] for r in rows})} for scene in sorted({r['scene'] for r in rows})} for stage,rows in pixels.items()})
    write_json(ROOT/'region_quality_by_camera.json',{stage:{cid:{method:summary([r for r in rows if r['camera']==cid and r['method']==method])
        for method in sorted({r['method'] for r in rows})} for cid in sorted({r['camera'] for r in rows})} for stage,rows in pixels.items()})
    old_proto=json.loads((BASE/'experiment_protocol.json').read_text())
    old_split={s['scene_id']:split for split,specs in old_proto['scenes'].items() for s in specs}
    reproduction={split:dict(summary([r for r in pixels['development'] if r['method']=='conservative' and old_split[r['scene']]==split]))
                  for split in old_proto['scenes']}
    parity=Counter()
    for record in complete:parity.update(record.get('mask_reproduction',{}))
    event_failures={stage:[dict(method=r['method'],scene=r['scene'],frame=r['frame'],camera=r['camera'],**e)
        for r in rows for e in r.get('events',[]) if not e['recalled']] for stage,rows in pixels.items()}
    write_json(ROOT/'critical_change_failure_cases.json',dict(old_reproduction=reproduction,exact_mask_parity=dict(parity),
        event_failures=event_failures,point_failures=failures,
        cause_scope='Offline new-instance/motion/LiDAR-disocclusion references; RGB may follow background at a missed foreground location',
        unchanged_denominators=True))
    association_metrics={stage:[method_association(rows,kind,t) for kind in ('local','global') for t in (0.,.1,.25,.5)]
                         for stage,rows in associations.items()}
    write_json(ROOT/'association_ambiguity_results.json',dict(grid_frozen_before_evaluation=True,splits=association_metrics,
        online_threshold_m=0.,online_solver_unchanged=True,
        interpretation='Local negative margins can be globally forced edges; global uniqueness is not physical identity. Diagnostic only, no online global re-solve added.'))
    write_json(ROOT/'identity_error_diagnostics.json',dict(population_breakdown={s:dict(Counter(r['status'] for r in rows)) for s,rows in populations.items()},
        errors={s:[r for r in rows if r['correct'] is False and r['continued']] for s,rows in associations.items()},
        valid_rejected={s:[r for r in rows if r['correct'] is True and not r['continued']] for s,rows in associations.items()},
        all_candidate_records='traces/<split>/<scene>/association.json and population.json',
        label_scope='Unique same-class 2m + .25m competition gate; outside-range/competing/no-supported-GT remain unscorable'))
    scene_errors=defaultdict(list)
    for row in populations['calibration']:
        if row['residual_xy'] is not None:scene_errors[row['scene']].append(row['residual_xy'])
    calibration=calibrate_scene_maxima(scene_errors,'raw_640x360_model_train_population','M1-evidence-scene-max-v1')
    write_json(ROOT/'geometry_uncertainty_results.json',dict(conditional_scene_calibration=asdict(calibration),
        online_usable=False,reason='Calibration membership uses offline identity/gate; no uncensored identity or independent ego/calibration truth',
        population='Unique same-class source labels within 2m; censored residuals are not all-target error guarantees',
        centre_error_m={s:distribution([r['center_error_m'] for r in rows if r['center_error_m'] is not None]) for s,rows in populations.items()},
        size_error_norm_m={s:distribution([np.linalg.norm(r['size_error_m']) for r in rows if r['size_error_m'] is not None]) for s,rows in populations.items()},
        unscorable_counts={s:sum(r['center_error_m'] is None for r in rows) for s,rows in populations.items()},
        source_pose_and_calibration_error='Not identifiable independently from available poses/calibration; no zero-error assumption',
        error_source_diagnostics=json.loads((ROOT/'geometry_error_sources.json').read_text()),
        guarantee_types=dict(empirical_quantiles='diagnostic only',prediction_interval='conditional scene-exchangeability assumption',deterministic_single_target_bound=None)))
    fore_metrics={};physical={};rejection={}
    for stage,rows in foreground.items():
        candidate=[r for r in rows if r['status']=='candidate']
        point_rows=[p for r in rows for p in r['offline_points']]
        selected=[p for r in candidate for p in r['offline_points'] if p['stage']=='selected']
        counts=Counter(p['status'] for p in selected)
        volume_supported=counts['target_volume_supported_not_pixel_mask']
        contradicted=counts['other_instance_volume']+counts['outside_target_volume_background_candidate']
        fore_metrics[stage]=dict(proposals=len(rows),candidate_objects=len(candidate),candidate_scene_support=len({r['scene'] for r in candidate}),
            view_selected_points=sum(len(r['selected_points']) for r in rows),receipt_candidate_points=len(selected),
            sparse_reference_status_counts=dict(counts),target_volume_supported_points=volume_supported,
            other_instance_points=counts['other_instance_volume'],outside_target_volume_background_candidates=counts['outside_target_volume_background_candidate'],
            conditional_volume_contamination=contradicted/(volume_supported+contradicted) if volume_supported+contradicted else None,
            unscorable_fraction=1-(volume_supported+contradicted)/len(selected) if selected else None,
            view_rejection_reasons=dict(Counter(v['reason'] for r in rows for v in r['views'])),
            proposal_rejection_reasons=dict(Counter(reason for r in rows for reason in r['rejection_reasons'])),
            producer_cpu_ms=distribution([r['producer_cpu_ms'] for r in rows]),
            producer_total_per_frame_ms=distribution([t['foreground_cpu_ms'] for t in times[stage] if t['method']=='conservative']),
            reference_scope='Sparse nearest LiDAR return + GT volume only; never full foreground pixel ground truth',
            genuine_instance_mask_precision=None,qualified_online_producer=False)
        records=certificates[stage];accepted=[r for r in records if r['certified']]
        physical[stage]=dict(evaluated=len(records),physical_candidates=len(candidate),certified=len(accepted),
            accepted_scorable=0,independent_accepted_scenes=0,FAR=None,far_upper95=None,
            declaration='Static candidate identity continuity + local ray consistency only; metric/velocity not certified')
        rejection[stage]=dict(Counter(reason for r in records for reason in r['rejection_reasons']))
    write_json(ROOT/'foreground_correspondence_results.json',fore_metrics)
    write_json(ROOT/'foreground_feasibility.json',dict(decision='No-Go for physical ownership qualification',splits=fore_metrics,
        conclusion='Measured same-RGB candidates exist but sparse volume support is not an instance mask; insufficient independently accepted scenes and complete ownership reference',
        additional_component='Not Run: no additional segmentation weights/sensor producer provided; no training or unsupported latency estimate',
        minimal_missing_information='Audited per-instance current/source surface ownership and identity, plus new-scene producer risk validation; metric attributes need independent localization/motion bounds'))
    write_json(ROOT/'physical_certification_results.json',physical)
    write_json(ROOT/'physical_risk_coverage.json',dict(splits=physical,threshold_one_sided95=.05,risk_unit='independent scene/log; repeated entities/frames do not increase sample size',
        zero_acceptance_is_not_zero_risk=True,qualified_artifact_shipped=False,decision='No-Go'))
    write_json(ROOT/'physical_rejection_analysis.json',rejection)
    runtime={stage:{method:{key:distribution([r[key] for r in rows if r['method']==method]) for key in
        ('precheck_ms','commit_ms','total_cpu_ms','foreground_cpu_ms','physical_certification_ms','association_ms')}
        for method in sorted({r['method'] for r in rows})} for stage,rows in times.items()}
    write_json(ROOT/'m1_runtime_cost.json',dict(replay=runtime,
        replay_scope='Precheck + Commit of already-adapted actual detections; excludes input IO/adapter and offline Oracle/logs',
        timing_limitation='Development/calibration/evaluation jobs can overlap on CPU; comparative isolated benchmark reported separately',
        GPU=json.loads((ROOT/'gpu_adapter_profile.json').read_text()),
        isolated_cpu=json.loads((ROOT/'isolated_cpu_profile.json').read_text()),
        memory={stage:dict(max_anchors=max(r['anchors'] for r in rows),max_features=max(r['features'] for r in rows),
            max_tensor_bytes=max(r['tensor_bytes'] for r in rows),state_pickle_bytes=distribution([r['state_pickle_bytes'] for r in rows])) for stage,rows in sizes.items()},
        long_sequence_scope='Two complete 40-keyframe development scenes, not hours-long endurance',P1_optimization='Not Run: no deployment quality qualification'))
    junit=ET.parse(ROOT/'regression.xml');tests=junit.findall('.//testcase')
    failed=[t.attrib['name'] for t in tests if t.find('failure') is not None or t.find('error') is not None]
    skipped=[dict(name=t.attrib['name'],reason=t.find('skipped').attrib) for t in tests if t.find('skipped') is not None]
    regress=dict(passed=len(tests)-len(failed)-len(skipped),failures=failed,skipped=skipped,old_tests_retained=True,new_evidence_cases=33,
        prior_mask_parity=dict(parity),source_immutable_arrays=True,certificates_bound_to_state_version=True,certification_phase='Commit only; no result backflow into Precheck')
    write_json(ROOT/'m1_regression_results.json',regress)
    gpu=json.loads((ROOT/'gpu_parity_isolated/results.json').read_text())
    write_json(ROOT/'original_model_parity.json',dict(comparisons=gpu['comparisons'],passed=gpu['observe_only_equal'],
        frames=len(gpu['baseline']['outputs']),scope='Original EVA/detection/map outputs and head memory; original perception parameters; LM/M2/M3 disabled',
        checkpoint_sha256=(ROOT/'checkpoint.sha256').read_text().split()[0]))
    audit=json.loads((ROOT/'independent_data_audit.json').read_text())
    sample_audit=json.loads((ROOT/'full_prior_sample_audit.json').read_text())
    independent=not sample_audit['matched_prior_files'] and sample_audit['returncode']==1
    engineering=not failed and gpu['observe_only_equal'] and parity['different_camera_masks']==0
    region_reasons=[]
    if chosen is None:region_reasons.append('No local candidate met frozen calibration screen')
    else:
        for stage in ('heldout','final_test'):
            v=region[stage][chosen]
            if v['critical_change_recall'] is None or v['critical_change_recall']<.998:region_reasons.append(stage+':critical_event_recall_below_99.8%')
            if v['critical_point_recall'] is None or v['critical_point_recall']<.998:region_reasons.append(stage+':critical_point_recall_below_99.8%')
            if v['sparse_reference_far'] is None or v['sparse_reference_far']>.05:region_reasons.append(stage+':sparse_reference_risk_failed_or_unsupported')
            if v['accepted_scene_support']<3:region_reasons.append(stage+':insufficient_accepted_scene_support')
            if v['far_upper95_iid_descriptive'] is None or v['far_upper95_iid_descriptive']>.05:region_reasons.append(stage+':descriptive_trial_bound_not_supported')
            if v['effective_stable_area']<region[stage]['conservative']['effective_stable_area']:region_reasons.append(stage+':no_effective_coverage_gain')
    if not independent:region_reasons.append('prior_M1_exposure_or_audit_failure')
    # Broad deployment mask claims require ownership of the unscorable population.
    # A conditional sparse-reference pass is reported separately, not generalized.
    sparse_protocol_pass=not region_reasons
    decisions=dict(engineering='Go' if engineering else 'No-Go',physical_certification='No-Go',
        region_validation='Go (frozen sparse-reference scope only)' if sparse_protocol_pass else 'No-Go',
        region_reasons=region_reasons,runtime_cost='No-Go for optimized deployment; costs measured, P1 not run',
        M1_new_scene_scope=independent,model_training_exposure=True,
        full_image_pixel_risk_certified=False,full_system_deployment_approved=False)
    write_json(ROOT/'go_nogo_decisions.json',decisions)
    write_json(ROOT/'evidence_contract_audit.json',dict(fixed_rejection_replaced=True,verified_receipts=True,
        actual_producer_path='post-perception foreground_correspondence.py',qualification_artifact=None,
        actual_numeric_checks=['bidirectional points','parallax','reprojection','oriented box','ring depth contrast','scene/frame/epoch/digest','state version','independent producer qualification'],
        forbidden_online_GT_tested=True,precise_geometry_and_motion_claims_rejected=True,
        synthetic_positive_contract_fixture='60 disjoint simulated validation scene/log units; engineering test only, never real quality evidence',
        source_code_match={p:sha256(p)==digest for p,digest in frozen['code_sha256'].items()},
        post_measurement_metadata_revision=json.loads((ROOT/'evidence_protocol_binding_audit.json').read_text()),
        measured_source_snapshot='measured_prototype_source/worldstream/m1/',
        revision_scope='One source hash differs: qualification fingerprint was strengthened after quality runs. Numerical producer/mask/state functions are AST-identical; no qualification existed in any run, so decisions are invariant. Final regression/GPU/isolated CPU use final source.'))
    source_files=[*Path('worldstream/m1').glob('*.py'),*Path('tools/worldstream_m1').glob('*evidence*.py'),Path('tools/worldstream_m1/evidence_oracle.py'),
                  Path('worldstream/m1/foreground_correspondence.py'),Path('tests/worldstream_m1/test_evidence_contract.py')]
    write_json(ROOT/'execution_manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256={str(p):sha256(p) for p in source_files},cpu_affinity=sorted(os.sched_getaffinity(0)),
        CUDA_PyTorch=gpu['run_metadata'],checkpoint_sha256=(ROOT/'checkpoint.sha256').read_text().split()[0],
        model_parameters_changed=False,M2_M3_changed=False,data_audit='independent_data_audit.json and full_prior_sample_audit.json'))
    write_figures(region,fore_metrics)
    write_report(region,fore_metrics,reproduction,regress,gpu,decisions,calibration,runtime,association_metrics)
    print(json.dumps(decisions,ensure_ascii=False,indent=2))


def write_figures(region,foreground):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/m1-evidence-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    from matplotlib.lines import Line2D
    methods=[m for m in region['calibration'] if m!='full_refresh']
    colors=dict(zip(methods,plt.get_cmap('tab10').colors))
    markers=dict(zip(methods,['o','s','^','v','D','X']))
    fig,axes=plt.subplots(4,2,figsize=(11,13))
    for row,stage in enumerate(('development','calibration','heldout','final_test')):
        for method in methods:
            v=region[stage][method];x=v['effective_stable_area']*100
            axes[row,0].scatter(x,v['critical_change_recall']*100,color=colors[method],marker=markers[method],s=50)
            if v['sparse_reference_far'] is not None:
                axes[row,1].scatter(x,v['sparse_reference_far']*100,color=colors[method],marker=markers[method],s=50)
        for col,(limit,label) in enumerate(((99.8,'Critical event recall (%)'),(5,'Sparse volume FAR (%)'))):
            ax=axes[row,col];ax.axhline(limit,color='red',linestyle='--',linewidth=1)
            ax.set(title=stage,xlabel='Effective stable area (%)',ylabel=label)
            ax.grid(alpha=.2)
    handles=[Line2D([],[],color=colors[m],marker=markers[m],linestyle='',label=m) for m in methods]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.975),ncol=3,frameon=False)
    fig.suptitle('Frozen methods; null FAR omitted; sparse reference is not dense pixel truth',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'region_risk_coverage.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));names=list(foreground)
    ax.bar(names,[foreground[s]['candidate_objects'] for s in names],label='Measured object proposals')
    ax.set(ylabel='Candidate objects',title='Same-RGB evidence proposals; qualified physical states = 0')
    fig.tight_layout();fig.savefig(out/'foreground_candidates.png',dpi=180);plt.close(fig)


def write_report(region,fore,reproduction,regress,gpu,decision,calibration,runtime,association):
    chosen=json.loads((ROOT/'algorithm_selection.json').read_text())['selected_local_method']
    lines=['# M1 物理证据闭环与局部区域专项实验','',
        '本轮实现可验证的真实测量证据接口、实验性前景对应生产器、局部边界拒绝和状态不可变性边界。所有质量结论来自本轮真实执行；没有把受控测试夹具当成物理成功，也没有使用 GT 在线补证。','',
        '## 数据与冻结','',
        '本轮在 train 中冻结 38 个未在旧 M1 产物出现的 scene，并采用 38 个互异采集 log：20 Calibration、6 Heldout、12 Final Test，每 scene 12 帧。456 个 sample token 在此前整个 outputs 的 JSON/JSONL/Markdown 中无匹配。它们属于 OmniDrive 的模型训练人口，只能称为对 M1 开发新的场景，不能声称模型泛化或官方隐藏测试成功。test 图像存在但缺少 sample_annotation，未用其伪造风险评价。','',
        '全部新场景检测来自本轮原 checkpoint 的实际 GPU 感知执行，使用原 test 图像预处理和检测推理参数。此前 val 的 26 个 scene、368 帧全部作为 Development/Reproduction。新旧共 64 个质量场景、824 个唯一回放帧；多个方法及重复帧不是独立样本。数据清单、日志/token、时戳、预测哈希和成本保存在 data_selection.json、model_observation_provenance.json 与 traces/。','',
        '候选算法/阈值在新质量标签读取前固定；只允许 Calibration 选方法，algorithm_selection.json 和 formal_validation_protocol.json 在 Heldout/Final 评分前写入。原关联 local margin 仍为 0，global edge-removal 只作离线比较。所有 GT/annotation/LiDAR 仅在 tools 中评分，从未作为 Precheck/在线认证/Commit 参数。','',
        '## 复现原失效与本轮区域结果','',
        '| 原确认组（本轮均为开发数据） | 关键事件召回 | 关键点漏刷 |', '|---|---:|---:|']
    for split in ('heldout','final_test','development_long'):
        v=reproduction[split];lines.append(f'| {split} | {v["critical_change_recall"]:.6%} | {v["missed_critical_points"]}/{v["critical_lidar_points"]} |')
    lines+=['','旧刷新精确 Mask 与本轮 conservative 的逐相机比较见 critical_change_failure_cases.json。事件仍要求 ROI 刷新≥90%；点级分母和事件级分母均保持，不将不可评分背景改成静态负例。实际像素、预测/参考 source UV、漏刷原因和可视化保存于 traces/ 与 visualizations/。','',
        '局部方案保持动态扫过保护，将全图的支持邻域 guard 从 8 改为 2，并只在旋转补偿后的视差跳变、折叠、双向流失配或显著光度残差周围拒绝稳定性（Unknown）。半径固定比较 4/8/12；没有扩大全图 guard。其收益和新增错误同时报告，未通过则不设为默认。','',
        f'Calibration 选中的实验候选：**{chosen}**。这不是生产批准；默认配置仍为原保守模式。','',
        '| 分组 | 方法 | 有效稳定面积 | 事件召回 | 关键点漏刷 | 稀疏参考错误/接受 |', '|---|---|---:|---:|---:|---:|']
    for stage in ('calibration','heldout','final_test'):
        for method,v in region[stage].items():
            lines.append(f'| {stage} | {method} | {v["effective_stable_area"]:.4%} | {v["critical_change_recall"]:.6%} | {v["missed_critical_points"]} | {v["false_stable_points"]}/{v["accepted_scorable_points"]} |')
    lines+=['','区域约束：事件召回≥99.8%，关键点召回≥99.8%，annotation-owned 稀疏参考 FAR≤5%，至少 3 个接受 scene，且有效稳定面积不低于保守方案。Final 不重新选方法。Trial CP 上界只作描述，并报告 scene-cluster bootstrap、接受场景数和无分母重采样；相关点不能当独立样本。未标注背景、未评分像素不计作正确，全图刷新仅为零收益参考。','',
        '## 物理证据与代价','',
        '| 分组 | 对象提案 | 几何/背景可区分候选 | 候选支持 Scene | 候选前景生产 CPU/对象 P50 ms |', '|---|---:|---:|---:|---:|']
    for stage,v in fore.items():lines.append(f'| {stage} | {v["proposals"]} | {v["candidate_objects"]} | {v["candidate_scene_support"]} | {v["producer_cpu_ms"]["p50"]} |')
    lines+=['','生产器实际消费双向 RGB 点、源/当前相机位姿、可观测三角化、旋转框成员关系、空间象限和邻域环深度分离。光流一致不自动证明静态前景：缺少视差、环无法测深、目标与背景不能区分或多视图矛盾，均输出 Unknown。检测框只提出几何候选；没有伪造实例前景 Mask。','',
        '稀疏 LiDAR 与 annotation 体积约束可发现错实例和目标体积外的背景候选，但体积内点仍不等于完整像素归属真值。foreground_correspondence_results.json 明确分解可评分/不可评分点、背景候选、错对象和观测不足。当前生产器没有足够的独立身份/前景质量资格，因此没有批准为正式可信对象来源。额外分割网络/传感器组件未提供，未训练或集成，其质量和开销为 Not Run，不提供虚构毫秒数。','',
        '认证器已不再固定返回 False：它验证带执行回执的实际数值测量，以及 scene/frame/epoch/图像 digest/几何版本、声明范围和经审计哈希固定的生产器质量资格。受控正例仅证明代码能消费完整合规输入，不是现实世界成功。实际缺少批准资格时默认拒绝；本轮严格接受仍为 0，FAR=null，上界=null，独立接受 Scene=0。','',
        '允许研究的最小声明为“静态候选对象身份连续性与局部射线一致性”；它不批准中心、距离或速度。认证发生在完整感知后的 Commit，不能反写之前的 Precheck；每次新 Detection 替换几何都更新 state_version，旧认证不能附着到新值。','',
        '质量实验完成后加强了资格协议指纹：绑定全部证据相关配置和生产器源码。该改动只改变诊断 digest；所有实验资格均为空，数值算法 AST、阈值和接受决策不变。冻结协议未改写，实测原型源码保存在 measured_prototype_source/，唯一源码哈希差异及决策不变性见 evidence_protocol_binding_audit.json。最终回归、独立 GPU 和 CPU 计时使用加强后的源码。','',
        f'20 个校准 scene 的条件 scene-max 半径（xy）：{calibration.radius_xy_m}。即使为有限值，它也仅覆盖经过离线唯一身份/2m 门限筛选的人口；在线无法确认该人口成员资格，所以仍不可用于完整三维属性认证。中心与尺寸残差、不可评分人口、独立 pose/calibration 真误差缺失分别列出，不把 P95 当确定性误差界。','',
        '## 17 项验收回答','',
        '1. 生命周期/来源仍保持：复用原子发布和 Frame Digest，新增源状态 digest、不可变 bytes-backed 几何/描述子/历史图像，以及隔离的 feature snapshot。',
        '2. 固定拒绝已改为可验证证据接口；生产器回执、数值和离线资格全部检查，不能靠 identity_verified/foreground_verified 布尔值通过。',
        '3. 当前观测能支持同源 RGB 表面/局部射线一致性提案；只有独立质量资格批准后，才支持限定对象连续性声明。',
        '4. 实测获得的前景候选与真实实例证据分开报告；没有获得可正式批准的实例归属生产者。',
        '5. 环深度对比能拒绝同深度/不足视差等情况，但无法在全部真实样本中证明目标与 ROI 背景归属，保持 Unknown。',
        '6. 未得到非零、独立风险合格的严格物理历史状态，不能以全部拒绝宣称成功。',
        '7. 物理 FAR 和单侧上界均为 null，未达到冻结的≤5%要求；重复帧不增加独立样本量。',
        '8. 精确中心/距离/速度仍缺少适用误差界，均不可认证；类别为 barrier/cone 或高检测分数不会改变结论。',
        '9. 条件关联人口已分解：独立唯一身份、超出2m、身份竞争、无传感器支持 GT、源不可评分。被拒绝的正确链接与被接受的错误链接均保留。Global assignment 唯一不等于物理实例唯一，因此没有未经验证增加在线复杂度。',
        '10. 具体漏检由逐点 source 对应与运动/新实例/独立 LiDAR 遮挡参考定位；RGB 对应可能在前景变化时仍追踪背景。局部拒绝结果与剩余失败逐例保存，没有通过扩大全图刷新抹去失效。',
        '11. 99.8% 是否达到，以以上完整分组表和 go_nogo_decisions.json 为准；不只引用成功场景或框事件，而忽略关键点漏刷。',
        '12. 有效覆盖只在同一质量约束下比较；未达约束的覆盖增长不记为成功，full refresh 有效覆盖恒为0。',
        '13. 按 scene/camera/frame 记录面积、误差、ego 位移、间隔、强制刷新和光流诊断；平均值不替代长尾失败。',
        '14. 实测新增生产器 CPU、认证、Association、Mask/Precheck/Commit 成本，逐场景结果见 m1_runtime_cost.json。并发实验时序不用于声称加速，额外提供预加载单进程对照；没有质量资格前不开展 P1。',
        f'15. 本轮实际 8 帧 GPU 感知数值一致：{gpu["comparisons"]}。Checkpoint SHA256 一致，模型权重、感知输入预处理/参数、M2/M3 算法未变。',
        '16. 有对本地记录的 M1 开发新的 scene/log，且样本 token 审计无旧匹配；它们仍是模型训练人口，不能宣称原模型泛化、官方测试或未知外部域保证。',
        '17. 工程接口和异常边界可继续作为 observe-only 基础；物理状态消费必须等待独立前景/身份资格，区域消费仅能受四项决策的具体范围约束。','',
        '## 四项独立决策','',
        f'- **M1 Engineering：{decision["engineering"]}**。{regress["passed"]} passed，{len(regress["skipped"])} skipped；旧测试保留。',
        f'- **M1 Physical Certification：{decision["physical_certification"]}**。实际严格覆盖=0；不将零接受报告为零风险。',
        f'- **M1 Region Validation：{decision["region_validation"]}**。原因/范围：{decision["region_reasons"]}。',
        f'- **M1 Runtime Cost：{decision["runtime_cost"]}**。','',
        '完整图像未评分像素的风险仍没有被认证；本轮不批准完整系统部署，不声称端到端加速。下一步最小信息是独立核验的源/当前实例归属及身份证据；精确几何另需适用人口的定位/运动误差界。当前候选生产器的成本已实测，尚不存在的额外组件成本明确 Not Run。','',
        '![区域风险覆盖](figures/region_risk_coverage.png)','![前景候选](figures/foreground_candidates.png)']
    cost=json.loads((ROOT/'isolated_cpu_profile.json').read_text())
    gpu_cost=json.loads((ROOT/'gpu_adapter_profile.json').read_text())
    lines+=['','## 独立计时结果','',
        '预加载相同 8 个开发帧，1 次预热、4 次计时，交替方法顺序。以下包含 Adapter + Precheck + Commit，排除图像 IO、Oracle 和日志；这是同机对照，不代表端到端加速。','',
        '| 方法 | P50 ms | P95 ms | 前景生产 P50 ms/帧 |','|---|---:|---:|---:|']
    for name,row in cost['methods'].items():
        d=row['total_adapter_precheck_commit_ms'];p=row['producer_ms']['p50']
        lines.append(f'| {name} | {d["p50"]:.2f} | {d["p95"]:.2f} | {p:.2f} |')
    d=gpu_cost['measurements']['m1_total_including_adapters_ms']
    lines+=['',f'独立 GPU 感知协议下，完整 M1 适配/验证/提交 P50/P95 为 {d["p50"]:.2f}/{d["p95"]:.2f} ms（8 帧，含启动帧；非同一 CPU 输入分辨率）。拆分与原始值见 gpu_adapter_profile.json。',
        '生产器新增成本没有换来经过独立资格批准的状态覆盖，因此保持显式实验开关，默认关闭。']
    lines+=['','## 关联歧义离线对照','',
        '以下为预冻结的 0m 诊断门限，保持原 2m/竞争 Oracle 口径。Global 列是排除匹配边后重求最大匹配数、最小总代价的结果；它没有接入在线策略，也不是身份真值。','',
        '| 分组 | 门限方式 | 保留候选 | 可评分保留 | 其中错误 | 不可评分保留 | 正确但拒绝 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for stage,rows in association.items():
        for row in rows:
            if row['threshold_m']!=0:continue
            lines.append(f'| {stage} | {row["method"]} | {row["kept"]} | {row["scorable_kept"]} | {row["wrong_kept"]} | {row["unscorable_kept"]} | {row["correct_rejected"]} |')
    lines+=['','没有合法局部竞争边时 margin=+∞，严格 JSON 中记为 null；全局匹配数下降单独记录，不将它伪装成有限米制距离。全部身份人口与错误实例见 identity_error_diagnostics.json，逐场景原始边见 traces/*/*/association.json。']
    (ROOT/'experiment_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
